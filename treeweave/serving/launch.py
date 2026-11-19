"""Launch a single TreeWeave SGLang role (`--encoder-only` or `--language-only`)."""

from __future__ import annotations

import os
import sys
import warnings
from typing import Optional, Sequence

from .. import ARCHITECTURE, ROUTE_THRESHOLD_ENV
from .core_patches import (
    install_arch_allowlist,
    install_encoder_admission,
    install_encoder_decode_offload,
    install_encoder_embedding_dtype,
    install_encoder_error_cleanup,
    install_encoder_metadata,
    install_encoder_video_factor,
    install_encoder_video_timestamps,
    install_receiver_fail_closed,
    install_receiver_inflight_gate,
    install_receiver_metadata,
)
from .sglang_route import (
    BUDGET_SHIFT_ENV,
    PIXEL_FACTOR_ENV,
    budget_logit_shift_from_env,
    route_threshold_from_env,
)

_REQUIRED_TRANSFER_BACKEND = "zmq_to_tokenizer"
_FORBIDDEN_CACHE_FLAGS = ("enable_prefix_mm_cache", "enable_mm_global_cache")
_FORBIDDEN_DISPATCH_FLAG = "enable_adaptive_dispatch_to_encoder"


def _resolved(server_args, field: str):
    from sglang.srt.arg_groups.overrides import resolution_result

    return resolution_result(server_args, field)


def _check_transfer_backend(server_args, when: str) -> None:
    backend = _resolved(server_args, "encoder_transfer_backend")
    if backend != _REQUIRED_TRANSFER_BACKEND:
        raise SystemExit(
            f"treeweave.serving.launch: encoder_transfer_backend is {backend!r} ({when}), "
            f"but TreeWeave requires {_REQUIRED_TRANSFER_BACKEND!r} because routed placeholder "
            "lengths, 0-token folded video slices, and half-integer MRoPE positions are built in "
            "the tokenizer process after the ViT and SpatiotemporalTreeRouter run."
        )


def _check_caches(server_args, when: str) -> None:
    on = [f for f in _FORBIDDEN_CACHE_FLAGS if _resolved(server_args, f)]
    if on:
        flags = ", ".join("--" + f.replace("_", "-") for f in on)
        raise SystemExit(
            f"treeweave.serving.launch: {flags} is set ({when}). TreeWeave visual embeddings "
            "are dynamically tree-routed products; serve without un-keyed MM embedding caches."
        )


def _check_dispatch(server_args, when: str) -> None:
    if _resolved(server_args, _FORBIDDEN_DISPATCH_FLAG):
        raise SystemExit(
            f"treeweave.serving.launch: --enable-adaptive-dispatch-to-encoder is set ({when}). "
            "Every multimodal request must pass through the disaggregated TreeWeave encoder."
        )


def _check_env() -> None:
    expected = {
        "SGLANG_EXTERNAL_MODEL_PACKAGE": "treeweave.serving.models",
        "SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE": "treeweave.serving.processors",
        "SGLANG_EXTERNAL_MM_MODEL_ARCH": ARCHITECTURE,
    }
    wrong = {k: os.environ.get(k) for k, v in expected.items() if os.environ.get(k) != v}
    if wrong:
        raise SystemExit(
            f"treeweave.serving.launch: invalid registration env vars {wrong}; expected {expected}. "
            "Launch via `python -m treeweave.serving.serve`."
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    from . import video  # noqa: F401 - align SGLang video resize factor before checking bindings

    threshold = route_threshold_from_env()
    shift = budget_logit_shift_from_env(0.0)
    px_factor = int(os.environ.get(PIXEL_FACTOR_ENV, "128"))
    _check_env()

    install_arch_allowlist()
    install_encoder_metadata()
    install_encoder_embedding_dtype()
    install_encoder_video_timestamps()
    install_encoder_video_factor()
    install_encoder_decode_offload()
    install_encoder_admission()
    install_encoder_error_cleanup()
    install_receiver_metadata()
    install_receiver_fail_closed()
    install_receiver_inflight_gate()

    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    load_plugins()
    server_args = prepare_server_args(list(sys.argv[1:] if argv is None else argv))
    if server_args.encoder_only == server_args.language_only:
        raise SystemExit(
            f"treeweave.serving.launch requires exactly one of --encoder-only / --language-only "
            f"(got encoder_only={server_args.encoder_only}, language_only={server_args.language_only})."
        )
    _check_transfer_backend(server_args, "as given on the command line")
    _check_caches(server_args, "as given on the command line")
    _check_dispatch(server_args, "as given on the command line")
    server_args.resolve_once()
    _check_transfer_backend(server_args, "after argument resolution")
    _check_caches(server_args, "after argument resolution")
    _check_dispatch(server_args, "after argument resolution")

    role = "encoder" if server_args.encoder_only else "language"
    print(
        f"[treeweave] launching {role} role for {ARCHITECTURE} "
        f"({ROUTE_THRESHOLD_ENV}={threshold:.6g}, {BUDGET_SHIFT_ENV}={shift:+.6g}, "
        f"{PIXEL_FACTOR_ENV}={px_factor}, transfer={_resolved(server_args, 'encoder_transfer_backend')})",
        flush=True,
    )
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="sglang.launch_server")
    main()
