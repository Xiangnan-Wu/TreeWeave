# TreeWeave 最小化验证实验完全操作手册（NVIDIA B300 288GB 单任务合跑版）

> **本文档定位**：面向**另一台 NVIDIA B300（Blackwell Ultra，`sm103`，单卡 288 GB HBM3e / 8×B300 共 2.3 TB 显存、公司内网无外网、离线 TensorBoard）开发机/集群**的端到端落地手册。按照本手册从上到下执行，即可完成：**B300 算子环境确认 → 模型与数据准备 → 3–5 分钟冒烟自检 → 1 个提交任务内合跑全部 3 组核心对照实验（每组含 Stage 1 + Stage 2 + 校准 + Stage 3）→ TensorBoard 曲线验收与评估**。

---

## 0. 最小化实验设计总览（针对 8×B300 288GB HBM3e 专项调优）

在投入全量 900K 数据跑顶会主表之前，最小化验证的核心目标是：**用最小的数据量（~24K 样本）和最短的卡时（8×B300 单次提交仅约 1.5–2.5 小时），一次性横向跑完 3 组核心对照实验**，直接回答论文最核心的三个问题：

| 实验编号 | 目录名 / `RUN_TAG` | 关键配置 | 验证什么核心命题 |
| :--- | :--- | :--- | :--- |
| **实验 1** | `exp1_visionweave_2d` | `--max_spatial_level 1 --max_temporal_level 0` | **VisionWeave 平面二元编织基线**（单帧 2×2 粗细二选一，省流上限 75%） |
| **实验 2** | `exp2_quadtree_3lvl` | `--max_spatial_level 2 --max_temporal_level 0` | **三层空间四叉树（无跨帧）**：单独量化 $s=2$ 对文档/GUI 留白省流的净收益 |
| **实验 3** | `exp3_treeweave_full` | `--max_spatial_level 2 --max_temporal_level 2` | **完整正交时空树（TreeWeave）**：量化跨帧二叉树对视频静止小字与背景的二次跃升 |

### B300（Blackwell Ultra `sm103`）专项适配与加速要点（代码与脚本已内置）
1. **规避 Hopper 专属 `fa3` 崩溃**：FlashAttention-3 (`fa3`) 仅支持 Hopper (`sm90`/H100)，在 B300 (`sm103`) 上强行指定会直接报错。我们已在 [offline_rollout.py](../treeweave/rollout/offline_rollout.py) 与 [serve.py](../treeweave/serving/serve.py) 中按显卡算力自动切换——检测到 `sm100/sm103`（B200/B300）时自动选用 `flashinfer`（也可手动指定 `trtllm_mha`），零报错启动。
2. **288 GB 超大显存吃满红利（默认 `GPU_ARCH=b300`）**：
   - **关闭梯度重算**（`GRADIENT_CHECKPOINTING=false`）：`Qwen3.5-4B` 在 288 GB 显存上完全无需 activation checkpointing，每步快约 **1.4×–1.6×**；
   - **大批量采样与缓存**：单卡 B300 装完 8.5 GB `Qwen3.5-4B` 还剩 ~260 GB KV Cache，默认将离线采样拉满至 `ROLLOUT_MAX_BATCH_SIZE=256, ROLLOUT_CHUNK_SIZE=1024`，Top-512 提取拉满至 `TOPK_BATCH_SIZE=16, TOPK_MAX_TOKENS=32768`；
   - **大包长 + 大微批训练**：Stage 1/2 混包提升至 `24K / 32K`，Stage 3 单卡微批提升至 `PER_DEVICE_BS=8, GRAD_ACCUM=4`（梯度累积次数减为 1/4）。
3. **自动兼容 `flash_attn` 与 B300 原生 cuDNN `sdpa`**：若你的 B300 容器暂未安装 `sm103` 版 `flash-attn`，主控脚本会自动降级到 PyTorch CUDA 12.8+/13.0 原生 cuDNN `sdpa`（精确段隔离混包，直接跑通无需卡环境）。
4. **3 组实验共享 1 次离线采样与 Top-512 缓存**：采样与 Teacher 前向只跑 1 次，后面 3 个实验 × 3 个阶段全部 0 Teacher 显存复用！

---

## 1. 需要下载什么模型（外网机下载 + 内网搬运清单）

最小化实验**只需要下载 1 个模型**（既是离线采样教师，也是 TreeWeave 初始化基座）：

| 项目 | 详情 |
| :--- | :--- |
| **模型名称** | **`Qwen3.5-4B`**（稠密多模态原生版本，**不要下纯文本版或量化版**） |
| **HuggingFace ID** | `Qwen/Qwen3.5-4B` |
| **ModelScope ID（国内快）** | `Qwen/Qwen3.5-4B` |
| **磁盘占用** | 约 **9.0 GB**（BF16 safetensors 权重 + 处理器/分词器配置） |

### 1.1 在有外网的机器上下载模型（三选一）

```bash
# 方式 A：ModelScope（国内服务器/跳板机推荐，速度最快）
pip install -U modelscope
modelscope download --model Qwen/Qwen3.5-4B --local_dir ./models/Qwen3.5-4B

# 方式 B：HuggingFace 官方（或走国内 hf-mirror）
pip install -U "huggingface_hub[cli]"
# 如需镜像：export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3.5-4B --local-dir ./models/Qwen3.5-4B
```

### 1.2 搬运到公司内网开发机后的完整性检查

把 `./models/Qwen3.5-4B` 拷贝到开发机（例如 `/data/models/Qwen3.5-4B`）后，确认以下关键文件齐全：

```bash
ls -lh /data/models/Qwen3.5-4B
# 必须包含：
# - config.json                    (含 "model_type": "qwen3_5" 与 vision_config / text_config)
# - preprocessor_config.json       (含 "patch_size": 16, "merge_size": 2)
# - video_preprocessor_config.json (含 "patch_size": 16, "merge_size": 2)
# - tokenizer.json / tokenizer_config.json
# - model.safetensors 或 model-00001-of-*.safetensors + model.safetensors.index.json
```

### 1.3 如果公司开发机连 PyPI 也没有外网（离线安装依赖包）

在有外网、同架构（Linux `x86_64` + 同 CUDA/Python 版本）的中转机上提前缓存 wheel，拷贝到开发机离线安装：

```bash
# 在有外网的中转机上下载依赖 wheel：
mkdir -p ./offline_wheels
pip download -d ./offline_wheels \
  "ms-swift>=4.6.0" "transformers>=5.9.0" "qwen-vl-utils>=0.0.14" \
  deepspeed accelerate peft safetensors pillow einops binpacking decord tensorboard \
  "vllm>=0.14.0" "flash-linear-attention>=0.2.0"

# 拷贝 offline_wheels 到内网开发机后执行：
pip install --no-index --find-links=./offline_wheels \
  ms-swift transformers qwen-vl-utils deepspeed accelerate peft \
  safetensors pillow einops binpacking decord tensorboard vllm flash-linear-attention
```

> **关于 B300 (`sm103`, CUDA 12.8+/13.0) 上的 `PyTorch` / `flash-attn` / `causal-conv1d`**：
> - 务必确保开发机底层的 PyTorch 为 **CUDA 12.8+ 或 CUDA 13.0 版本（`cu128` / `cu130`，建议 `torch>=2.7/2.8`）**，否则会报 `no kernel image is available for execution on the device (sm_103)`；
> - `flash-linear-attention` 为纯 Triton 实现，在 B300 上开箱即用；
> - 若暂未获取针对 `sm103` 编译的 `flash-attn` wheel，无需死磕编译，主脚本会自动使用 B300 原生 cuDNN `sdpa`（或手动传 `ATTN_IMPL=sdpa`），也可在有编译环境时带 `export TORCH_CUDA_ARCH_LIST="10.0;10.3"` 编译安装。

---

## 2. 需要下载什么数据（训练最小集 + 评测最小集）

### 2.1 训练数据：24K 最小化四域黄金配比（总磁盘约 25–40 GB）

为了在 24K 这样小的样本量下把 TreeWeave 的 **9 种时空粒度全部训透**，训练数据必须同时覆盖 4 类互补场景（缺任何一类都会导致对应分支激活不足）：

| 数据域 | 推荐公开数据集（HF / ModelScope ID） | 建议抽取条数 | 磁盘估算 | 为什么必须包含它？ |
| :--- | :--- | :---: | :---: | :--- |
| **① 高留白文档与图表** | `lmms-lab/DocVQA` (`DocVQA` + `InfographicVQA` 训练集)<br>或 `HuggingFaceM4/ChartQA` | **8,000 条** | ~4 GB | 含大量纯白页边距与密集小字，激活 $s=0$（细文字）与 $s=2$（大片留白 93.8% 省流）的强烈对比 |
| **② 通用自然图文** | `lmms-lab/LLaVA-OneVision-Data`（任意抽取 1–2 个通用 VQA 子集如 `vqav2` / `gqa` / `llava_next`） | **4,000 条** | ~3 GB | 保住通用场景语义表征与自然图像中的 $s=1$ 中粗粒度过渡 |
| **③ GUI 桌面截图** | `osunlp/UGround-V1-Data`（抽前 4K）<br>或 `OS-Copilot/OS-Atlas-data`（Desktop 子集） | **4,000 条** | ~4 GB | 检验大片桌面留白合并后，体积守恒注意力偏置能否保住精细点击坐标定位 |
| **④ 多帧视频** | `lmms-lab/LLaVA-Video-178K`（**只需下载 1–2 个短视频分片**，如 `0_30_s_academic_v0_1` + `0_30_s_youtube_v0_1`） | **8,000 条** | ~18 GB | `academic` 含大量网课/PPT 静止文字（激活 $(s=0,t=2)$），`youtube` 含运动前景+静止背景（激活 $(s=2,t=2)$） |
| **合计** | — | **24,000 条** | **~29 GB** | 覆盖全部 9 种时空原语，8×B300 单任务跑完 3 组实验仅需 **~1.5–2.5 小时** |

> **零下载极速冒烟选项**：在还没搬完 29 GB 真实数据之前，你可以先用我们内置的 `scripts/prepare_minimal_data.py --mode smoke` **0 秒下载**自动生成 64 张高反差文档/GUI 图 + 32 段含“静止文字+运动方块+纯色背景”的 MP4 视频，先花 5–10 分钟跑通整条管线！

### 2.2 最小化评测集（训练完后只测这 4 个最有区分度的基准）

| 评测基准 | 对应 HuggingFace ID | 样本量参考 | 用来验收什么能力 |
| :--- | :--- | :---: | :--- |
| **DocVQA** | `lmms-lab/DocVQA` (`val`) | ~5,349（快速验可先抽 500 条） | 单图留白省流：`exp2/exp3` 相对 `exp1`（16%）显著翻倍且分数不掉 |
| **ScreenSpotV2** | `OS-Copilot/ScreenSpot-v2` | 1,272 | GUI 精准点击定位 + 大片背景留白省流 |
| **VideoOCR** | 参考 VisionWeave 评测集（或抽 200 条含固定字幕/PPT 的教学视频） | ~200–500 | **杀手级验证**：验证 `exp3` 的 $(s=0,t=2)$ 在视频静态小字上比 `exp1/exp2` 省流暴涨且 OCR 零掉点 |
| **Video-MME** | `lmms-lab/Video-MME`（`short`+`medium` 子集可先抽 300 条） | 900（短片 300） | 通用视频跨帧折叠省流从 `exp1`（~50%）跃升至 `exp3`（70%+） |

### 2.3 把下载好的数据一键转为 TreeWeave 训练 JSONL

本项目要求的输入 JSONL 格式（标准 `ms-swift` 多模态格式，每行一个 JSON，**媒体路径必须是开发机上的绝对路径**）：

```json
{"messages": [{"role": "user", "content": "<image>\nWhat is the total amount shown in the invoice?"}], "images": ["/data/extracted_media/img_0001.png"]}
{"messages": [{"role": "user", "content": "<video>\nSummarize the key steps demonstrated on screen."}], "videos": ["/data/LLaVA-Video-178K/videos/clip_0001.mp4"]}
```

> 注意：不需要预先准备 `assistant` 回答字段（即便原数据集里带了 `assistant` 字段，`offline_rollout.py` 也会自动剥离并用原生 `Qwen3.5-4B` 重新生成一致的自蒸馏教师回复）。

使用内置脚本 [prepare_minimal_data.py](../scripts/prepare_minimal_data.py) 一键转换、抽样、去重并验证所有媒体文件存在：

```bash
# 选项 A：先生成 0 下载 5 分钟冒烟集（64 张图 + 32 段 MP4 视频）
python scripts/prepare_minimal_data.py \
  --mode smoke \
  --output-dir ./data/smoke \
  --num-images 64 \
  --num-videos 32

# 选项 B：把下载好的真实数据集 Parquet / JSON / JSONL 转换为 24K 最小化训练集
python scripts/prepare_minimal_data.py \
  --mode convert \
  --output-jsonl ./data/minimal_train_prompts.jsonl \
  --media-out-dir ./data/extracted_media \
  --doc-dirs /data/datasets/DocVQA /data/datasets/ChartQA \
  --general-dirs /data/datasets/LLaVA-OneVision-subset \
  --gui-dirs /data/datasets/UGround-subset \
  --video-dirs /data/datasets/LLaVA-Video-178K-subset \
  --doc-limit 8000 \
  --general-limit 4000 \
  --gui-limit 4000 \
  --video-limit 8000 \
  --seed 42
```

---

## 3. 分步骤实验路线图（在另一台开发机从 0 到跑通）

### Step 0：同步代码、安装环境与自检（约 5–10 分钟）

1. 将本地项目目录 `/Users/kris/.gemini/antigravity/scratch/treeweave` 同步到目标开发机（**不要拷贝本地 macOS 的 `.venv` 目录**）：
   ```bash
   rsync -avP --exclude='.venv' --exclude='.pytest_cache' --exclude='__pycache__' --exclude='upstream_visionweave' \
     /Users/kris/.gemini/antigravity/scratch/treeweave/ <user>@<dev_server>:/workspace/treeweave/
   ```
2. 在 B300 开发机激活已装好 **CUDA 12.8+/13.0 PyTorch** 的环境并安装依赖：
   ```bash
   cd /workspace/treeweave
   # 若可访问内网镜像源，直接运行 B300 专用环境脚本（自动检测 B300 sm103 & 288GB 显存）：
   bash scripts/00_setup_8xb300_env.sh
   ```
3. **运行内置 116 项自动化单元测试**（确认新开发机上的 PyTorch / Transformers / ms-swift / TensorBoard 全部正常）：
   ```bash
   pytest -q
   # 预期输出：116 passed, 1 skipped（若安装了 sglang 则为 117 passed）
   ```
4. **验证 B300 GPU 算力与核心加速算子就绪**：
   ```bash
   python -c '
   import torch
   print("GPU:", torch.cuda.get_device_name(0), "cap:", torch.cuda.get_device_capability(0),
         f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB", "CUDA:", torch.version.cuda)
   for pkg in ("torch", "transformers", "swift", "flash_attn", "fla", "causal_conv1d", "vllm", "tensorboard"):
       try:
           m = __import__(pkg)
           print(f"[OK] {pkg:20s} {getattr(m, \"__version__\", \"installed\")}")
       except Exception as e:
           print(f"[MISSING] {pkg:15s} {e}")
   '
   ```

---

### Step 1：先跑 3–5 分钟全链路冒烟测试（强烈推荐，零浪费算力！）

在正式提交 24K 数据任务前，先用 `GPU_ARCH=b300 PROFILE=smoke` 跑一遍 **3 个实验合并单任务**（仅 96 条自动生成数据、Stage 1=20 步 / Stage 2=20 步 / Stage 3=10 步），3–5 分钟即可把 **B300 CUDA 内核、离线采样、Top-K 缓存、9 次 `torchrun` 多卡通信、DeepSpeed ZeRO-2、路由器校准、TensorBoard 落盘** 全部真机验完：

```bash
export NATIVE_MODEL_DIR=/data/models/Qwen3.5-4B
export GPU_ARCH=b300
export PROFILE=smoke
export WORK_ROOT=./work_smoke
export TB_LOG_ROOT=./tb_logs_smoke

# 先 dry-run 看一眼全部执行命令（0 GPU，1 秒完成）：
bash scripts/run_three_experiments_one_job.sh --dry-run

# 确认路径无误后真机跑通 3-5 分钟冒烟：
bash scripts/run_three_experiments_one_job.sh
```

看到末尾输出 `[TreeWeave-Job] ALL REQUESTED EXPERIMENTS FINISHED SUCCESSFULLY! (GPU_ARCH=b300)` 且 `./work_smoke/exp{1,2,3}_*/stage3/last` 全部生成，说明 B300 开发机环境 100% 过关！

---

### Step 2 & Step 3：正式提交 24K 最小化三实验合跑任务（8×B300 约 1.5–2.5 小时）

整个任务内部会自动按顺序执行 **4 大阶段**（全部由 [run_three_experiments_one_job.sh](../scripts/run_three_experiments_one_job.sh) 统一编排）：

1. **Phase 0（共享前置，只跑 1 次，3 个实验 100% 复用）**：
   - `phase0_offline_rollout`：8 卡 `DP=8`（`max_batch_size=256`）并行调用 `vLLM`（或 `SGLang` `flashinfer`）对 24K 样本生成原生 `Qwen3.5-4B` 回复 → `./work_minimal/shared/teacher_rollouts.jsonl`（8×B300 预计耗时 ~10–15 分钟）。
   - `phase0_cache_topk_512`：8 卡 `DP=8`（`batch_size=16, max_tokens=32768`）+ Sequence Packing 预提取每个回复 Token 的 Top-512 教师 Logits → `./work_minimal/shared/topk_cache/*.safetensors`（8×B300 预计耗时 ~8–12 分钟，磁盘约 15–25 GB）。
   - **收益**：接下来 3 个实验 × 3 个阶段（共 9 次训练）全部直接读缓存，**Teacher 模型显存 = 0 GB、Teacher 前向计算量 = 0**！
2. **Experiment 1（`exp1_visionweave_2d`，`s<=1, t=0`）**：
   - `init`（<1 秒）→ `Stage 1`（800 步，`24K` 混包，无重算）→ `Stage 2`（500 步，`32K` 混包）→ `Calibrate`（目标省流 `0.38`，约 10 秒）→ `Stage 3`（250 步，`bs=8, ga=4`）。
   - 最终权重：`./work_minimal/exp1_visionweave_2d/stage3/last`
3. **Experiment 2（`exp2_quadtree_3lvl`，`s<=2, t=0`）**：
   - `init` → `Stage 1`（800 步）→ `Stage 2`（500 步）→ `Calibrate`（目标省流 `0.50`）→ `Stage 3`（250 步）。
   - 最终权重：`./work_minimal/exp2_quadtree_3lvl/stage3/last`
4. **Experiment 3（`exp3_treeweave_full`，`s<=2, t<=2`）**：
   - `init` → `Stage 1`（800 步）→ `Stage 2`（500 步）→ `Calibrate`（目标省流 `0.58`）→ `Stage 3`（250 步）。
   - 最终权重：`./work_minimal/exp3_treeweave_full/stage3/last`

---

### Step 4：通过离线 TensorBoard 实时验收训练曲线

在开发机另开一个终端启动纯离线 TensorBoard（无需外网）：

```bash
TB_LOG_ROOT=./tb_logs TB_PORT=6006 bash scripts/06_view_tensorboard.sh
```

在你自己本地电脑开 SSH 端口转发后，浏览器访问 `http://127.0.0.1:6006`：

```bash
ssh -N -L 6006:127.0.0.1:6006 <user>@<dev_server_ip>
```

**重点盯住以下 4 组健康判据曲线（3 个实验会自动叠在同一面板中对比）**：

| 阶段 | TensorBoard 指标名 | 正常健康的趋势 | 异常信号（需立即排查） |
| :--- | :--- | :--- | :--- |
| **Stage 1** | `train/distill_kl` | 从初始较高值平滑下降并趋稳（通常降到 `< 0.15 ~ 0.35`） | 持续不降或 `NaN`（检查学习率或数据空回复） |
| **Stage 2** | `train/expected_savings`、`train/distill_kl` | `expected_savings` 随步数上升并稳定在合理区间（`exp1` ~35–45%，`exp2` ~45–55%，`exp3` ~55–70%），同时 `distill_kl` 保持低位 | `expected_savings` 瞬间飙到 `0.95+` 或掉回 `0.0`（可微调 `--lambda_rate`） |
| **Stage 2** | `train/router_entropy`、`train/p_s1_mean`、`train/p_s2_mean`、`train/p_t1_mean`、`train/p_t2_mean` | 四个门控均值均落在 `0.15 ~ 0.85` 之间，未全部饱和到 `0` 或 `1` | 某个门控恒为 `0.00` 或 `1.00`（分支坍缩，可调大 `--lambda_bal` 至 `0.02`） |
| **Stage 3** | `train/actual_token_savings`、`train/distill_kl`、`train/share_s0_t2`、`train/share_s2_t0`、`train/share_s2_t2` | `distill_kl` 在前 50 步快速下降收敛；`exp2/exp3` 中 `share_s2_t0 > 0`（留白触发超粗），`exp3` 中 `share_s0_t2 > 0` 且 `share_s2_t2 > 0`（静止小字与静止背景均已激活！） | `share_s2_t0` 或 `share_s0_t2` 恒为 `0`（检查训练数据中是否真的包含留白文档与多帧视频） |

---

### Step 5：最小化评测与“是否放量跑全量 900K”的 Go / No-Go 判据

训练完成后，用 HuggingFace `.generate()` 或 SGLang 服务在 4 个最小评测集上对比 4 个模型（`Native Qwen3.5-4B`、`exp1`、`exp2`、`exp3`）：

- **快速推理/SGLang 启动检验**（以 `exp3` 为例）：
  ```bash
  python -m treeweave.serving.prepare \
    --source ./work_minimal/exp3_treeweave_full/stage3/last \
    --destination ./serve/exp3_treeweave_full

  # 单卡/双卡启动 OpenAI 兼容服务（--budget-logit-shift 可无需重训动态扫省流率！）
  python -m treeweave.serving.serve \
    --model-path ./serve/exp3_treeweave_full \
    --encoder-gpus 0 --gpus 1 --port 30000
  ```
- **Go / No-Go 验收标准（满足以下 3 条即证明 Idea 扎实，可直接扩 900K 数据跑论文全表）**：
  1. **单图留白突破（`exp2` vs `exp1`）**：在 `DocVQA` / `ScreenSpotV2` 上，`exp2` 精度与 `exp1` 持平（差距 `< 1.0` 分），但平均视觉 Token 节省率明显高出 `exp1`（预期 `+10% ~ +20%` 绝对值）；
  2. **视频正交跨帧收益（`exp3` vs `exp2` & `exp1`）**：在 `Video-MME` / `VideoOCR` 上，`exp3` 精度持平的同时，视频 Token 节省率再跃升 `+15% ~ +25%` 绝对值（突破 `65%–75%+`）；
  3. **质量-效率帕累托优势**：拨动 `--budget-logit-shift` 扫描压缩率时，`exp3` 能够在 `exp1` 撞上 75% 硬墙的位置之后继续平滑延伸至 `80%–85%+` 且无断崖崩盘。

---

## 4. 如何把三个实验合并在一个提交任务里面跑通

### 4.1 为什么直接拼接 3 个脚本容易挂？我们脚本里已解决的 5 个坑

在公司 GPU 集群上把多个训练任务塞进**同一个 Job**，如果只是简单把几个 `bash scripts/03..05` 竖着拼起来，通常会踩中以下 5 个坑——[run_three_experiments_one_job.sh](../scripts/run_three_experiments_one_job.sh) 已经**全部自动内置处理**：

1. **避免重复采样 & Teacher 显存翻倍**：在任务开头只执行 1 次 `phase0_offline_rollout` + `phase0_cache_topk_512`（`ENABLE_TOPK_CACHE=1`），3 个实验共 9 个训练阶段全部免 Teacher 前向复用。
2. **跨实验网格公平性与 Top-K 缓存复用**：默认强制设置 `TREEWEAVE_FORCE_PIXEL_FACTOR=128`（因为 `128` 同时是 `64` 和 `128` 的整数倍），确保 `exp1`（`s<=1`）、`exp2`（`s<=2, t=0`）和 `exp3`（`s<=2, t<=2`）看到**完全相同的原生视觉 Token 数**，不仅对比 100% 公平，还能零报错复用同一套 `topk_cache/`！
3. **连续 `torchrun` 端口冲突（`Address already in use`）**：单任务内会连续拉起 9 次 `torchrun`，如果共用默认 `29500` 端口，Linux 内核的 `TIME_WAIT` 经常导致下一阶段直接崩溃。脚本内置自动递增端口（`29511` → `29519`）+ 阶段间 `sleep 3` 等待显存与 NCCL 套接字完全回收。
4. **集群抢占 / 超时后的断点自动续跑（`SKIP_IF_DONE=1`）**：每完成一个子步骤会在 `./work_minimal/.markers/<step>.done` 写入完成标记。如果任务在第 5 小时（比如跑 `exp2_stage3` 时）被集群抢占或超时，**直接原样重新提交同一个任务**，会自动跳过前面所有已完成步骤（`[SKIP]`），从断点精确续跑！
5. **多实验曲线同屏对比**：每个阶段自动设置独立 `RUN_TAG="${PROFILE}_${EXP_NAME}"`，所有 TensorBoard 事件汇总到统一的 `./tb_logs/` 目录下，开 1 个 TensorBoard 就能同屏看 3 组实验全部曲线。

---

### 4.2 单任务合跑启动入口（配合公司集群任务提交工具直接调用）

所有 3 组对照实验（`Phase 0 共享离线采样 + Top-512 缓存` + `exp1` + `exp2` + `exp3`）均已封装在同一个入口脚本 [run_three_experiments_one_job.sh](../scripts/run_three_experiments_one_job.sh) 中。把下面这段标准入口命令交给公司集群的任务提交 skill 即可（把 `WORK_ROOT` / `TB_LOG_ROOT` 指向持久化共享盘路径以支持断电/抢占自动续跑）：

```bash
cd /workspace/treeweave && \
NATIVE_MODEL_DIR=/data/models/Qwen3.5-4B \
RAW_DATASET_JSONL=/data/treeweave/minimal_train_prompts.jsonl \
GPU_ARCH=b300 \
PROFILE=minimal \
EXPERIMENTS="exp1 exp2 exp3" \
ROLLOUT_BACKEND=vllm \
ENABLE_TOPK_CACHE=1 \
WORK_ROOT=/data/treeweave/work_minimal \
TB_LOG_ROOT=/data/treeweave/tb_logs \
bash scripts/run_three_experiments_one_job.sh
```

常用按需裁剪参数（只改环境变量即可）：
- **先跑 3–5 分钟冒烟验证**：改为 `PROFILE=smoke`（可省略 `RAW_DATASET_JSONL`，自动生成 96 条图文+视频冒烟数据）；
- **仅在 4 张 B300 上合跑单个完整实验（`exp3` 的 Stage 1+2+3）**：传入 `EXPERIMENTS="exp3" NPROC_PER_NODE=4 GPUS="4,5,6,7"`；
- **演练检查全部执行命令（不占卡）**：末尾追加 `--dry-run`。

---

## 5. B300 开发机常见踩坑快速排查（Troubleshooting）

| 现象 / 报错 | 原因与直接解法 |
| :--- | :--- |
| B300 报 `fa3 requires sm90` 或 `flash_attn` 报 `no kernel image for sm_103` | `fa3` 是 H100 专属，请勿手动传 `--attention-backend fa3`（默认 `auto` 已自动选 `flashinfer`）；若 `flash-attn` 报 `sm_103` 架构错，直接加 `ATTN_IMPL=sdpa` 走 B300 原生 cuDNN FlashAttention 即可 |
| 内网机器报 `ConnectionError` / 尝试连 `huggingface.co` | 确认 `NATIVE_MODEL_DIR` 指向本地已下载目录，且已设置 `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`（主合跑脚本已默认开启） |
| 极长视频下若偶发 OOM（虽然 B300 有 288 GB） | 开启 `GRADIENT_CHECKPOINTING=true` 或调小 `S3_BS=4 S3_GA=8`；确保开启 `ENABLE_TOPK_CACHE=1` |
| `vLLM` 离线采样在某 B300 镜像报兼容错 | 切换采样引擎：`ROLLOUT_BACKEND=sglang`（自动走 Blackwell `flashinfer`）或 `ROLLOUT_BACKEND=transformers` |
| 某一步想强制重跑（不跳过） | 删掉对应的 marker 即可：`rm -f ./work_minimal/.markers/exp3_treeweave_full_stage2.done`，然后原样重新执行主脚本 |
| 磁盘空间紧张，不想存 ~20 GB `topk_cache` | 传 `ENABLE_TOPK_CACHE=0`：B300 288 GB 显存同时装 Student + Teacher 绰绰有余（>200 GB 空余），且零磁盘缓存占用 |
| 离线采样速度或视频解码变慢 | 确认已安装 `decord` 或 `av`；若视频帧数过大可先将最小化验证设为 `FPS_MAX_FRAMES=32` 加速 2 倍 |
