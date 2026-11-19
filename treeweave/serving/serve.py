"""Launch Qwen3.5-TreeWeave Disaggregated Encoder + Language Server via SGLang (`python -m treeweave.serving.serve`)."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from urllib.error import URLError
from urllib.request import urlopen

from .. import ARCHITECTURE, ROUTE_THRESHOLD_ENV
from .prepare import finite_float, finite_threshold, read_json
from .sglang_route import BUDGET_SHIFT_ENV, PIXEL_FACTOR_ENV, validate_serving_geometry


def gpu_list(value: str) -> List[str]:
    devices = [x.strip() for x in value.split(",") if x.strip()]
    if (
        not devices
        or any(not item.isdecimal() for item in devices)
        or len(set(devices)) != len(devices)
    ):
        raise argparse.ArgumentTypeError("use distinct numeric GPU IDs, for example 0,1")
    return devices


def positive_int(value: str) -> int:
    iv = int(value)
    if iv < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return iv


def memory_fraction(value: str) -> float:
    fv = float(value)
    if not 0.0 < fv < 1.0:
        raise argparse.ArgumentTypeError("must be strictly between 0 and 1")
    return fv


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, type=Path, help="TreeWeave checkpoint or prepared serve dir")
    p.add_argument(
        "--gpus",
        type=gpu_list,
        default=["0"],
        help="Language server GPU IDs (TP1 or TP2)",
    )
    p.add_argument(
        "--encoder-gpus",
        type=gpu_list,
        default=None,
        help="One disaggregated TreeWeave ViT+Router encoder per GPU (defaults to --gpus)",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=positive_int, default=30000)
    p.add_argument("--encoder-port", type=positive_int, default=31000)
    p.add_argument("--served-model-name", default=None)
    p.add_argument("--route-threshold", type=finite_threshold, default=None)
    p.add_argument("--budget-logit-shift", type=finite_float, default=None)
    p.add_argument("--context-length", type=positive_int, default=32768)
    p.add_argument(
        "--image-tokens",
        type=positive_int,
        default=512,
        help="Input native visual token budget per image",
    )
    p.add_argument(
        "--video-tokens",
        type=positive_int,
        default=512,
        help="Input native visual token budget per video frame",
    )
    p.add_argument("--max-frames", type=positive_int, default=64)
    p.add_argument("--mem-fraction", type=memory_fraction, default=0.65)
    p.add_argument("--encoder-mem-fraction", type=memory_fraction, default=0.10)
    p.add_argument("--max-running-requests", type=positive_int, default=8)
    p.add_argument(
        "--attention-backend",
        default="auto",
        help="SGLang attention backend: 'auto' picks 'fa3' on Hopper (H100) and 'flashinfer' on Blackwell (B200/B300); or pass explicit backend (e.g. fa3, flashinfer, trtllm_mha).",
    )
    p.add_argument("--startup-timeout", type=positive_int, default=900)
    p.add_argument("--log-dir", type=Path, default=Path("logs"))
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate checkpoint & geometry and print SGLang launch commands without starting servers",
    )
    return p


@dataclass
class Role:
    name: str
    command: List[str]
    environment: Dict[str, str]
    host: str
    port: int


def _augmented_pythonpath(existing_py_path: str) -> str:
    bootstrap_dir = str((Path(__file__).resolve().parents[1] / "rollout" / "_bootstrap").resolve())
    repo_root = str(Path(__file__).resolve().parents[2].resolve())
    parts = [p for p in existing_py_path.split(os.pathsep) if p]
    merged: List[str] = []
    for cand in [bootstrap_dir, repo_root] + parts:
        if cand not in merged:
            merged.append(cand)
    return os.pathsep.join(merged)


def build_roles(args: argparse.Namespace) -> List[Role]:
    model = args.model_path.expanduser().resolve(strict=True)
    config = read_json(model / "config.json")
    if len(args.gpus) not in (1, 2):
        raise ValueError("TreeWeave SGLang serving supports language TP1 or TP2 (--gpus 0 or --gpus 0,1).")
    if config.get("model_type") != "qwen3_5":
        raise ValueError("Only dense Qwen3.5 checkpoints are supported.")
    if config.get("architectures") != [ARCHITECTURE]:
        raise ValueError(
            f"Checkpoint architectures must be [{ARCHITECTURE!r}], got {config.get('architectures')!r}. "
            "Initialize with `treeweave-init` or export with `python -m treeweave.serving.prepare`."
        )

    tw_cfg, fine, _, coarse, pixel_factor = validate_serving_geometry(config)
    min_native_tokens = (coarse // fine) ** 2  # 16 for s<=2 (128px), 4 for s<=1 (64px)
    min_video_tokens = max(16, min_native_tokens)
    if (
        args.image_tokens < min_native_tokens
        or args.video_tokens < min_video_tokens
        or args.max_frames < 4
        or args.max_frames % 2 != 0
    ):
        raise ValueError(
            f"image-tokens >= {min_native_tokens}, video-tokens >= {min_video_tokens}, "
            "and even max-frames >= 4 are required."
        )

    for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
        proc_path = model / name
        if not proc_path.is_file():
            raise ValueError(f"Missing {name} in {model}.")
        processor = read_json(proc_path)
        if processor.get("merge_size") != 2 or processor.get("patch_size") != 16:
            raise ValueError(f"{name} requires merge_size=2 and patch_size=16.")

    threshold = args.route_threshold if args.route_threshold is not None else tw_cfg.theta_s1
    threshold = finite_threshold(threshold)
    budget_shift = (
        args.budget_logit_shift if args.budget_logit_shift is not None else tw_cfg.budget_logit_shift
    )
    budget_shift = finite_float(budget_shift)

    env = dict(os.environ)
    env["PYTHONPATH"] = _augmented_pythonpath(env.get("PYTHONPATH", ""))
    env.update(
        SGLANG_NUMA_BIND_V2="0",
        TORCH_NCCL_USE_COMM_NONBLOCKING="0",
        SGLANG_VIT_ENABLE_CUDA_GRAPH="0",
        SGLANG_RUST_SERVER="0",
        SGLANG_MM_SKIP_COMPUTE_HASH="0",
        SGLANG_EXTERNAL_MODEL_PACKAGE="treeweave.serving.models",
        SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE="treeweave.serving.processors",
        SGLANG_EXTERNAL_MM_MODEL_ARCH=ARCHITECTURE,
    )
    env[ROUTE_THRESHOLD_ENV] = str(threshold)
    env[BUDGET_SHIFT_ENV] = str(budget_shift)
    env[PIXEL_FACTOR_ENV] = str(pixel_factor)
    env.setdefault("SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT", "60")
    env.setdefault("SGLANG_ENCODER_RECV_TIMEOUT", "600")
    env.setdefault("SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_INTERVAL", "0")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MM_PER_REQUEST_TIMEOUT", "600")

    min_pixel_floor = pixel_factor * pixel_factor
    budget = {
        "image": {
            "size": {
                "shortest_edge": min_pixel_floor,
                "longest_edge": max(min_pixel_floor, args.image_tokens * 1024),
            }
        },
        "video": {
            "min_pixels": max(min_pixel_floor, args.video_tokens * 256),
            "max_pixels": max(min_pixel_floor, args.video_tokens * 1024),
            "total_pixels": max(min_pixel_floor * 4, args.video_tokens * 1024 * args.max_frames),
            "min_frames": 4,
            "max_frames": args.max_frames,
            "fps": 2.0,
        },
    }

    resolved_attn = args.attention_backend
    if resolved_attn == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                major, _ = torch.cuda.get_device_capability(0)
                resolved_attn = "fa3" if major == 9 else ("flashinfer" if major >= 10 else "")
            else:
                resolved_attn = ""
        except Exception:
            resolved_attn = ""

    common = [
        "--model-path",
        str(model),
        *(["--attention-backend", resolved_attn] if resolved_attn else []),
        "--cuda-graph-backend-prefill",
        "disabled",
        "--enforce-disable-flashinfer-allreduce-fusion",
    ]
    language = [
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tp-size",
        str(len(args.gpus)),
        "--served-model-name",
        args.served_model_name or "treeweave",
        "--context-length",
        str(args.context_length),
        "--mem-fraction-static",
        str(args.mem_fraction),
        "--tokenizer-worker-num",
        "1",
        "--mm-process-config",
        json.dumps(budget),
    ]

    encoders = args.encoder_gpus or args.gpus
    base = [
        sys.executable,
        "-m",
        "treeweave.serving.launch",
        *common,
        "--encoder-transfer-backend",
        "zmq_to_tokenizer",
    ]

    roles: List[Role] = []
    for i, gpu in enumerate(encoders):
        port = args.encoder_port + i
        enc_cmd = base + [
            "--encoder-only",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--tp-size",
            "1",
            "--mem-fraction-static",
            str(args.encoder_mem_fraction),
            "--mm-process-config",
            json.dumps({"vision_config": budget}),
        ]
        roles.append(
            Role(
                name=f"encoder-{i}",
                command=enc_cmd,
                environment=env | {"CUDA_VISIBLE_DEVICES": gpu},
                host="127.0.0.1",
                port=port,
            )
        )

    lang_cmd = (
        base
        + language
        + ["--language-only", "--encoder-urls"]
        + [f"http://127.0.0.1:{r.port}" for r in roles]
        + [
            "--encoder-bootstrap-port",
            "0",
            "--max-running-requests",
            str(args.max_running_requests),
            "--cuda-graph-max-bs",
            str(args.max_running_requests),
        ]
    )
    roles.append(
        Role(
            name="language",
            command=lang_cmd,
            environment=env | {"CUDA_VISIBLE_DEVICES": ",".join(args.gpus)},
            host=args.host,
            port=args.port,
        )
    )

    if any(not 1 <= r.port <= 65535 for r in roles) or len({r.port for r in roles}) != len(roles):
        raise ValueError("Server ports must be distinct integers in 1..65535.")
    return roles


def stop_children(children: Sequence[subprocess.Popen]) -> None:
    """Terminate owned process groups (`start_new_session=True`)."""
    for sig, timeout in ((signal.SIGTERM, 15), (signal.SIGKILL, 3)):
        for child in children:
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + timeout
        for child in children:
            try:
                child.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass


def run(roles: Sequence[Role], log_dir: Path, timeout: int) -> None:
    for role in roles:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((role.host, role.port))

    log_dir.mkdir(parents=True, exist_ok=True)
    children: List[subprocess.Popen] = []
    logs = []
    previous = {}

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, interrupted)
        for role in roles:
            log = (log_dir / f"{role.name}.log").open("a", encoding="utf-8")
            logs.append(log)
            child = subprocess.Popen(
                role.command,
                env=role.environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            children.append(child)
            deadline = time.monotonic() + timeout
            host = "127.0.0.1" if role.host == "0.0.0.0" else role.host
            while True:
                for process in children:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"Server {role.name} exited with code {process.returncode}; see {log_dir}"
                        )
                try:
                    with urlopen(f"http://{host}:{role.port}/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError, ConnectionError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{role.name} did not become healthy; see {log_dir}")
                time.sleep(1)
            print(f"{role.name} ready on port {role.port}", flush=True)
        print(f"{roles[-1].name.upper()} SERVICE READY", flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(1)
        raise RuntimeError(f"A serving subprocess exited unexpectedly; see {log_dir}")
    finally:
        stop_children(children)
        for log in logs:
            log.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = parser()
    args = p.parse_args(argv)
    try:
        roles = build_roles(args)
        if args.dry_run:
            exported_keys = {
                "CUDA_VISIBLE_DEVICES",
                ROUTE_THRESHOLD_ENV,
                BUDGET_SHIFT_ENV,
                PIXEL_FACTOR_ENV,
            }
            for role in roles:
                exports = {
                    k: v
                    for k, v in role.environment.items()
                    if k.startswith("SGLANG_EXTERNAL_") or k in exported_keys
                }
                print(
                    role.name
                    + ": "
                    + " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(exports.items()))
                    + " "
                    + shlex.join(role.command)
                )
            return
        run(roles, args.log_dir.resolve(), args.startup_timeout)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        p.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
