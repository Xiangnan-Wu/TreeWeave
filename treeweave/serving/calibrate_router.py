"""Post-Stage-2 Offline Router Logit Bias Calibration (`treeweave-calibrate`).

Finds the scalar logit shift `delta` applied to `router_calib_bias` so that hard tree routing
at `theta = 0.5` achieves a desired dataset-level visual token savings target (e.g. `0.55` or `0.65`),
using ONLY the Vision Tower + Router (no LLM forward needed — calibrates 2,000 samples in seconds!).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import logging
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig

from ..layout import hard_tree_route_layout
from ..modeling_treeweave import Qwen3_5TreeWeaveForConditionalGeneration, Qwen3_5TreeWeaveVisionModel

logger = logging.getLogger(__name__)


def _update_calib_bias_in_checkpoint(ckpt_dir: Path, bias_4d: torch.Tensor) -> None:
    """In-place update `model.visual.router.router_calib_bias` inside the checkpoint shard."""
    key = "model.visual.router.router_calib_bias"
    index_file = ckpt_dir / "model.safetensors.index.json"
    if index_file.is_file():
        index_data = json.loads(index_file.read_text(encoding="utf-8"))
        shard_name = index_data["weight_map"][key]
        shard_path = ckpt_dir / shard_name
    else:
        shard_path = ckpt_dir / "model.safetensors"

    tensors: Dict[str, torch.Tensor] = {}
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {"format": "pt"}
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    tensors[key] = bias_4d.detach().cpu().float().contiguous()
    save_file(tensors, str(shard_path), metadata=metadata)


@torch.no_grad()
def calibrate_router_bias(
    *,
    checkpoint_dir: Path | str,
    calibration_jsonl: Optional[Path | str] = None,
    synthetic_grids_and_queries: Optional[List[Tuple[Tuple[int, int, int], torch.Tensor]]] = None,
    target_savings: float = 0.55,
    max_samples: int = 1024,
    write_in_place: bool = True,
    device: Optional[str] = None,
) -> Tuple[float, float]:
    """Bisection-solve scalar `delta` on `router_calib_bias` for `target_savings`, returning `(best_shift, achieved_savings)`."""
    if not (0.0 <= target_savings < 0.985):
        raise ValueError(f"target_savings must be in [0.0, 0.985), got {target_savings}.")

    ckpt_path = Path(checkpoint_dir).resolve(strict=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    hf_cfg = AutoConfig.from_pretrained(ckpt_path)
    # Load ONLY vision tower if calibrating on real data, or full model on small checkpoints
    dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
    full_model = Qwen3_5TreeWeaveForConditionalGeneration.from_pretrained(ckpt_path, config=hf_cfg, dtype=dt)
    visual: Qwen3_5TreeWeaveVisionModel = full_model.model.visual.to(dev).eval()
    del full_model.model.language_model
    del full_model.lm_head

    tw = visual.tw_config
    visual.router.router_calib_bias.zero_()

    # Collect cached `(grid, router_query_cpu)` across the calibration dataset
    cached_queries: List[Tuple[Tuple[int, int, int], torch.Tensor]] = []
    if synthetic_grids_and_queries is not None:
        cached_queries = [(g, q.to(dev)) for g, q in synthetic_grids_and_queries]
    else:
        if calibration_jsonl is None:
            raise ValueError("Provide either `calibration_jsonl` or `synthetic_grids_and_queries`.")
        from swift.model import get_model_processor
        from swift.template import get_template

        from ..swift_plugin.register import register_treeweave_with_swift
        from ..swift_plugin.template import (
            QWEN3_5_TREEWEAVE_TEMPLATE_TYPE,
            install_treeweave_resize_patch,
        )

        import os

        register_treeweave_with_swift(activate_distill_trainer=False)
        forced_pf = os.environ.get("TREEWEAVE_FORCE_PIXEL_FACTOR", "").strip()
        eff_pf = int(forced_pf) if forced_pf else tw.image_resize_factor
        install_treeweave_resize_patch(eff_pf)
        _, processor = get_model_processor(str(ckpt_path), load_model=False)
        template = get_template(processor, template_type=QWEN3_5_TREEWEAVE_TEMPLATE_TYPE)
        template.set_mode("train")

        with Path(calibration_jsonl).open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= max_samples or not line.strip():
                    break
                row = json.loads(line)
                enc = template.encode(row)
                if "pixel_values" in enc and "image_grid_thw" in enc:
                    pv = enc["pixel_values"].to(dev)
                    g_thw = enc["image_grid_thw"].to(dev)
                    res = visual.forward_vit_with_taps(pv, g_thw, run_router=True, vit_no_grad=True)
                    assert res.router_query is not None
                    q_off = 0
                    for g in res.grid_thw_list:
                        nq = g[0] * (g[1] // 4) * (g[2] // 4)
                        cached_queries.append((g, res.router_query[q_off : q_off + nq].detach()))
                        q_off += nq
                if "pixel_values_videos" in enc and "video_grid_thw" in enc:
                    pvv = enc["pixel_values_videos"].to(dev)
                    vg_thw = enc["video_grid_thw"].to(dev)
                    res = visual.forward_vit_with_taps(pvv, vg_thw, run_router=True, vit_no_grad=True)
                    assert res.router_query is not None
                    q_off = 0
                    for g in res.grid_thw_list:
                        nq = g[0] * (g[1] // 4) * (g[2] // 4)
                        cached_queries.append((g, res.router_query[q_off : q_off + nq].detach()))
                        q_off += nq

    if not cached_queries:
        raise ValueError("No visual grids found in calibration samples.")

    def _eval_shift(shift: float) -> float:
        total_nat = 0
        total_out = 0
        for g, q in cached_queries:
            r_out = visual.router.score_queries(
                q,
                [g],
                budget_logit_shift=shift,
                max_spatial_level=tw.max_spatial_level,
                max_temporal_level=tw.max_temporal_level,
            )
            layout = hard_tree_route_layout(
                g,
                r_out.p_s1[0],
                r_out.p_s2[0],
                r_out.p_t1[0],
                r_out.p_t2[0],
                theta_s1=tw.theta_s1,
                theta_s2=tw.theta_s2,
                theta_t1=tw.theta_t1,
                theta_t2=tw.theta_t2,
                max_spatial_level=tw.max_spatial_level,
                max_temporal_level=tw.max_temporal_level,
                allow_fine_temporal_merge=tw.allow_fine_temporal_merge,
            )
            total_nat += layout.native_tokens
            total_out += layout.output_length
        return 1.0 - (total_out / float(max(total_nat, 1)))

    # Monotone bisection search on `shift in [-12.0, +12.0]`
    lo, hi = -12.0, 12.0
    best_shift = 0.0
    best_sav = _eval_shift(0.0)
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        sav = _eval_shift(mid)
        if abs(sav - target_savings) < abs(best_sav - target_savings):
            best_shift, best_sav = mid, sav
        if sav < target_savings:
            lo = mid
        else:
            hi = mid

    if write_in_place:
        new_bias = torch.full((4,), float(best_shift), dtype=torch.float32)
        _update_calib_bias_in_checkpoint(ckpt_path, new_bias)
        cfg_file = ckpt_path / "config.json"
        cfg_json = json.loads(cfg_file.read_text(encoding="utf-8"))
        cfg_json.setdefault("treeweave", {}).setdefault("extra", {})["calibrated_shift"] = best_shift
        cfg_json["treeweave"]["extra"]["calibrated_savings"] = best_sav
        cfg_file.write_text(json.dumps(cfg_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    logger.info(
        f"[TreeWeave Calibration] Solved shift={best_shift:+.4f} -> savings={100*best_sav:.2f}% (target={100*target_savings:.2f}%)"
    )
    return best_shift, best_sav


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--calibration-jsonl", required=True, type=Path)
    p.add_argument("--target-savings", type=float, default=0.55)
    p.add_argument("--max-samples", type=int, default=1024)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    shift, sav = calibrate_router_bias(
        checkpoint_dir=args.checkpoint,
        calibration_jsonl=args.calibration_jsonl,
        target_savings=args.target_savings,
        max_samples=args.max_samples,
        write_in_place=not args.dry_run,
    )
    print(f"Calibrated router_calib_bias={shift:+.6f} (achieved savings={100*sav:.2f}%)")


if __name__ == "__main__":
    main()
