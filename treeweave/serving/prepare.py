"""Export a clean, symlinked SGLang serving directory from a trained `Qwen3.5-TreeWeave` checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from .. import ARCHITECTURE
from .sglang_route import validate_serving_geometry

ASSETS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "generation_config.json",
)


def read_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite_threshold(value: str | float) -> float:
    val = float(value)
    if not math.isfinite(val) or val < 0.0:
        raise argparse.ArgumentTypeError("route threshold must be finite and non-negative")
    return val


def finite_float(value: str | float) -> float:
    val = float(value)
    if not math.isfinite(val):
        raise argparse.ArgumentTypeError("value must be finite")
    return val


def prepare_serving_checkpoint(
    source: Path | str,
    destination: Path | str,
    *,
    tokenizer_source: Optional[Path | str] = None,
    route_threshold: Optional[float] = None,
    budget_logit_shift: Optional[float] = None,
) -> Path:
    """Create an SGLang-ready serving directory without mutating `source`."""
    src = Path(source).expanduser().resolve(strict=True)
    dst = Path(destination).expanduser().resolve()
    aux = Path(tokenizer_source).expanduser().resolve(strict=True) if tokenizer_source else src

    if dst.exists() or dst.is_symlink() or dst.is_relative_to(src) or dst.is_relative_to(aux):
        raise ValueError("Destination must be new and outside the source directories.")

    config = copy.deepcopy(read_json(src / "config.json"))
    if config.get("model_type") != "qwen3_5":
        raise ValueError("Only dense Qwen3.5 checkpoints are supported.")

    tw_cfg, _, _, _, pixel_factor = validate_serving_geometry(config)

    if route_threshold is not None:
        thr = finite_threshold(route_threshold)
        tw_cfg.theta_s1 = thr
        tw_cfg.theta_s2 = thr
        tw_cfg.theta_t1 = thr
        tw_cfg.theta_t2 = thr
    if budget_logit_shift is not None:
        tw_cfg.budget_logit_shift = finite_float(budget_logit_shift)

    config["treeweave"] = tw_cfg.to_dict()
    config["architectures"] = [ARCHITECTURE]
    for block in (config, config["vision_config"], config.get("text_config", {})):
        for key in ("auto_map", "_name_or_path"):
            block.pop(key, None)

    index_path = src / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = read_json(index_path)["weight_map"]
        for prefix in ("model.visual.pyramid.", "model.visual.router."):
            if not any(k.startswith(prefix) for k in weight_map):
                raise ValueError(f"Checkpoint is missing required TreeWeave weights: {prefix}*")
        shards = sorted(set(weight_map.values()))
        if any(Path(name).name != name for name in shards):
            raise ValueError("Weight shards must reside directly in the source directory.")
        weights = [index_path] + [src / name for name in shards]
    else:
        single_st = src / "model.safetensors"
        weights = [single_st]
        if single_st.is_file() and single_st.stat().st_size > 0:
            from safetensors import safe_open

            with safe_open(single_st, framework="pt", device="cpu") as sf:
                st_keys = list(sf.keys())
            for prefix in ("model.visual.pyramid.", "model.visual.router."):
                if not any(k.startswith(prefix) for k in st_keys):
                    raise ValueError(f"Checkpoint is missing required TreeWeave weights: {prefix}*")

    for wpath in weights:
        if not wpath.is_file():
            raise ValueError(f"Missing weight file: {wpath.name}")

    min_floor = pixel_factor * pixel_factor
    materialized = {"config.json": config}
    for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
        p = src / name
        if not p.is_file():
            p = aux / name
        proc = read_json(p)
        if proc.get("patch_size") != 16:
            raise ValueError(f"{name} must use patch_size=16.")
        proc["merge_size"] = 2
        proc["size"] = {"shortest_edge": min_floor, "longest_edge": 2**31}
        for key in ("min_pixels", "max_pixels", "auto_map", "_name_or_path"):
            proc.pop(key, None)
        materialized[name] = proc

    materialized["processor_config.json"] = {"processor_class": "Qwen3VLProcessor"}

    assets = []
    for name in ASSETS:
        p = aux / name
        if not p.is_file():
            p = src / name
        if p.is_file():
            assets.append(p)
    if not {"tokenizer.json", "tokenizer_config.json"}.issubset({p.name for p in assets}):
        raise ValueError("tokenizer.json and tokenizer_config.json are required; pass --tokenizer-source.")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=".treeweave-serve-", dir=dst.parent))
    try:
        for path in weights + assets:
            (tmp_dir / path.name).symlink_to(path.resolve())
        for name, value in materialized.items():
            (tmp_dir / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp_dir.rename(dst)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return dst


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, help="Trained Qwen3.5-TreeWeave checkpoint directory")
    p.add_argument("--destination", required=True, help="Output SGLang serving directory")
    p.add_argument("--tokenizer-source", help="Optional fallback tokenizer/processor directory")
    p.add_argument("--route-threshold", type=finite_threshold, default=None)
    p.add_argument("--budget-logit-shift", type=finite_float, default=None)
    args = p.parse_args(argv)
    try:
        out = prepare_serving_checkpoint(
            args.source,
            args.destination,
            tokenizer_source=args.tokenizer_source,
            route_threshold=args.route_threshold,
            budget_logit_shift=args.budget_logit_shift,
        )
        print(out)
    except (ValueError, KeyError, OSError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
