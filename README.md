# TreeWeave (`Qwen3.5-4B` Validation & `ms-swift` 8×H100 Training Suite)

**TreeWeave** generalizes flat 2D binary visual weaving (VisionWeave, `arXiv:2610.07987`) into an **Orthogonal Spatiotemporal Quad-Binary Tree** ($\text{Spatial Quadtree } s\in\{0,1,2\} \otimes \text{Temporal Binary Tree } t\in\{0,1,2\}$, 9 native granularities up to $64\to 1$ compression), seamlessly supporting both single images ($T=1 \implies$ 3-level spatial quadtree up to $93.75\%$ whitespace savings) and videos ($T>1 \implies$ decoupled $(s=0, t=2)$ static fine OCR + $(s=2, t=2)$ static background folding).

> **快速落地 / 另一台开发机单任务合跑指南**：如果你准备把代码搬到公司内网开发机或 8×H100 集群，直接阅读完整实操手册 [docs/MINIMAL_EXPERIMENT_GUIDE.md](docs/MINIMAL_EXPERIMENT_GUIDE.md)（含模型/最小数据集下载清单、5 分钟冒烟验证、以及用 `scripts/run_three_experiments_one_job.sh` 将 3 组对照实验合并到 **1 个提交任务** 的完整方法）。

---

## Key Systems & Training Efficiency Optimizations (Tailored for 8×H100-80GB)

1. **Offline `DP=8` Teacher Rollout Sampling (`treeweave/rollout/offline_rollout.py`)**:
   - **Zero Online Sampling Waste & 100% Data Reuse**: Samples native `Qwen3.5-4B` responses **once** offline into `teacher_rollouts.jsonl`, reused across Stage 1, Stage 2, Stage 3, and all Q1–Q5 ablations.
   - **SGLang (`fa3`) / vLLM `DP=8` (`TP=1`) Linear Scaling & Subprocess `128px` Parity**: `Qwen3.5-4B` BF16 weights take only ~8.5 GB, leaving ~70 GB KV cache per H100 with zero cross-GPU NCCL overhead. On H100 (`sm90`), `--backend sglang` leverages FlashAttention-3 (`fa3`) + fused Triton GatedDeltaNet kernels for ~20–30% faster multimodal prefill; `--backend vllm` is also supported out-of-the-box. A zero-overhead `PYTHONPATH` `sitecustomize.py` import bootstrap guarantees spawned `vLLM` (`EngineCore`) and `SGLang` (`TokenizerManager`) OS child processes apply the exact `128px` / `64px` visual resize grid as training.
   - **8-GPU `DP=8` Offline Top-512 Logit Cache (`treeweave/rollout/cache_topk_logits.py`)**: Sharded `DP=8` extractor with crash-safe atomic `.safetensors` writes, `flash_attention_2`, multi-sample batching/packing, and post-truncation token-count slicing so Stage 1 / Stage 2 / Stage 3 sweeps can run with `0 GB` Teacher VRAM (`export TOPK_CACHE_DIR=./rollout_data/topk_cache`).

2. **Single-ViT-Pass Dual-Forward & `torch.no_grad()` Vision Tower**:
   - Across **all 3 stages**, the 24-layer ViT backbone (`visual.blocks`) is frozen and runs **once** per batch under `torch.no_grad()`, cutting ViT activation memory by ~95% and eliminating 100% of duplicate Teacher ViT FLOPs.
   - In Stages 1 & 2, Student's frozen `merger` + `language_model` serve directly as the Teacher (`0 GB` extra model memory).
   - In Stage 3, `_LightweightStage3Teacher` drops the duplicate ViT tower and only runs the frozen Teacher `merger` + `language_model` (`~8 GB` BF16).

3. **Active-Token Chunked Top-512 Forward KL (`treeweave/losses.py`)**:
   - `Qwen3.5-4B` has `vocab_size = 248,320`. Materializing `[1, 24576, 248320]` teacher+student logits would consume **48.8 GB** and OOM on H100.
   - TreeWeave slices `hidden_states` to supervised response positions (`shifted_labels != -100`, respecting packed sequence boundaries) **before** `lm_head`, computing Top-512 KL in `1024`-token chunks (`< 0.5 GB` peak VRAM — **97× memory reduction**).

4. **FlashAttention-2/3 Compatible Volume-Preserving Attention Compensation (`treeweave/volume_bias.py`)**:
   - `Qwen3.5-4B` uses `head_dim = 256` (the hardware limit of FlashAttention-2/3) AND QK-RMSNorm (`k_norm`, which erases input magnitude scaling).
   - TreeWeave injects $\ln(4^s 2^t) \cdot w_{\ell, h_{\mathrm{kv}}}$ into the **post-`k_norm` non-rotary key subspace (`dims 64..255`)** before KV caching, preserving `head_dim = 256` for `flash_attention_2` packed `varlen` training, `fa3`, and KV-cached decoding. Exact scalar `gamma * ln(M)` logit bias is also supported (`--volume_bias_mode exact_logit_bias`).

---

## End-to-End 8×H100 Quickstart

### 0. Environment Setup on 8×H100 Server
> **Important for Qwen3.5 Sequence Packing (`--packing true`)**: `Qwen3.5-4B` is a hybrid model with 24 `Qwen3_5GatedDeltaNet` linear-attention layers + 8 `Qwen3_5Attention` full-attention layers. Make sure to install **`flash-linear-attention` + `causal-conv1d`** alongside `flash-attn` so packed sequences execute via fused CUDA varlen kernels at full H100 speed (without them, TreeWeave automatically falls back to exact segment-by-segment execution).

```bash
bash scripts/00_setup_8xh100_env.sh
# Install SGLang (recommended on H100 `sm90` for FA3 rollout + serving) or vLLM:
# pip install "sglang[all]"   OR   pip install vllm
```

### 1. Initialize Checkpoint + Generate Reusable Offline Rollouts + Run 3-Stage Distillation
```bash
export NATIVE_MODEL_DIR=/path/to/Qwen3.5-4B
export RAW_DATASET_JSONL=/path/to/multimodal_mix.jsonl

bash scripts/01_init_qwen3_5_4b.sh
bash scripts/02_offline_rollout_8xh100.sh

# Optional: Pre-extract reusable DP=8 Teacher Top-512 `.safetensors` shards (enables 0 GB Teacher VRAM!)
# bash scripts/02b_cache_topk_8xh100.sh
# export TOPK_CACHE_DIR=./rollout_data/topk_cache

bash scripts/03_train_stage1_8xh100.sh
bash scripts/04_train_stage2_8xh100.sh
bash scripts/05_calibrate_and_stage3_8xh100.sh

# Monitor loss, KL, token savings & 4-gate router curves offline via TensorBoard:
# (Pass RUN_TAG=exp_q1 to separate ablation runs under ./tb_logs/)
bash scripts/06_view_tensorboard.sh
```

### 2. High-Throughput Disaggregated SGLang Serving (`treeweave/serving/`)
TreeWeave ships a complete disaggregated SGLang ViT+Router Encoder + Language Server integration (`zmq_to_tokenizer` transport, `POSITION_SCALE=2` exact integer MRoPE across all 9 `(s, t)` scales, `0`-token folded video slices, FP32 router heads, and post-`k_norm` `TreeWeaveAttentionVolumeAdapter` key offsets):

```bash
# Optional: Export a clean symlinked serving directory
python -m treeweave.serving.prepare \
  --source ./output/qwen3_5-4b-treeweave-stage3/last \
  --destination ./serve/qwen3_5-4b-treeweave

# Inspect disaggregated launch commands without starting servers (`--dry-run`)
python -m treeweave.serving.serve \
  --model-path ./serve/qwen3_5-4b-treeweave \
  --encoder-gpus 0,1 \
  --gpus 2,3 \
  --dry-run

# Launch live OpenAI-compatible disaggregated SGLang service (FA3 on H100)
python -m treeweave.serving.serve \
  --model-path ./serve/qwen3_5-4b-treeweave \
  --encoder-gpus 0,1 \
  --gpus 2,3 \
  --port 30000
```

You can also invoke standard `swift sft` directly via `--custom_register_path`:
```bash
TREEWEAVE_STAGE=1 swift sft \
  --custom_register_path treeweave/swift_plugin/register.py \
  --model ./checkpoints/qwen3_5-4b-treeweave-init \
  --dataset ./rollout_data/teacher_rollouts.jsonl \
  --add_version false --tuner_type full --packing true --packing_length 16384 ...
```

---

## Built-in CLI Switches for All Paper Ablations (Q1–Q5)

| Ablation Question | Flag Override |
| :--- | :--- |
| **Q1(a)**: 2D Binary (VisionWeave baseline) | `--max_spatial_level 1 --max_temporal_level 0` |
| **Q1(b)**: 3-Level Spatial Quadtree only | `--max_spatial_level 2 --max_temporal_level 0` |
| **Q1(c)**: Full Orthogonal Spatiotemporal Tree | `--max_spatial_level 2 --max_temporal_level 2` (default) |
| **Q2**: Rigid Octree (forbid `s=0` temporal fold) | `--allow_fine_temporal_merge false` |
| **Q3**: Shuffled / Heuristic Router | `model(..., routing_mode="shuffled")` or `routing_mode="heuristic"` |
| **Q4**: w/o Volume Attention Bias | `--use_volume_attn_bias false` |
| **Q5**: Single-Scale Coarse Stage 1 Warmup | `--stage1_uniform_scale "1,0"` |
