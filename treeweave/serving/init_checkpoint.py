"""CPU-Streaming Checkpoint Initializer: Native `Qwen3.5-4B` -> `Qwen3.5-TreeWeave` without loading 4B weights."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .. import ARCHITECTURE, TreeWeaveConfig
from ..pooler import TreeWeavePyramid
from ..router import SpatiotemporalTreeRouter
from ..volume_bias import TreeWeaveAttentionVolumeAdapter

logger = logging.getLogger(__name__)

VISUAL_PREFIX = "model.visual."
DTYPES = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}
TOKENIZER_ASSETS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "generation_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "LICENSE",
    "LICENSE.txt",
    "NOTICE",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inspect_native_shards(source: Path, config: dict) -> Tuple[list[str], Dict[str, str], int, torch.dtype]:
    index_path = source / "model.safetensors.index.json"
    declared = _read_json(index_path)["weight_map"] if index_path.is_file() else None
    shards = sorted(set(declared.values())) if declared is not None else ["model.safetensors"]

    vision, text = config["vision_config"], config["text_config"]
    merged = vision["hidden_size"] * (vision["spatial_merge_size"] ** 2)
    expected_shapes = {
        "model.language_model.embed_tokens.weight": [text["vocab_size"], text["hidden_size"]],
        VISUAL_PREFIX + "patch_embed.proj.weight": [
            vision["hidden_size"],
            vision.get("in_channels", 3),
            vision.get("temporal_patch_size", 2),
            vision.get("patch_size", 16),
            vision.get("patch_size", 16),
        ],
        VISUAL_PREFIX + "merger.linear_fc1.weight": [merged, merged],
        VISUAL_PREFIX + "merger.linear_fc2.weight": [vision["out_hidden_size"], merged],
    }

    actual: Dict[str, str] = {}
    found_shapes: Dict[str, list[int]] = {}
    dtypes: Dict[str, str] = {}
    total_size = 0

    for shard in shards:
        path = source / shard
        if not path.is_file():
            raise ValueError(f"Missing weight shard: {path}")
        with safe_open(path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if name in actual:
                    raise ValueError(f"Duplicate tensor across shards: {name}")
                if name.startswith((VISUAL_PREFIX + "pyramid.", VISUAL_PREFIX + "router.")):
                    raise ValueError(f"Source checkpoint already contains TreeWeave tensors: {name}")
                actual[name] = shard
                if name in expected_shapes:
                    sl = f.get_slice(name)
                    found_shapes[name] = list(sl.get_shape())
                    dtypes[name] = sl.get_dtype()
        with path.open("rb") as raw_f:
            header_size = int.from_bytes(raw_f.read(8), "little")
        total_size += path.stat().st_size - 8 - header_size

    if declared is not None and actual != declared:
        raise ValueError("model.safetensors.index.json weight_map disagrees with shard contents.")
    for name, expected in expected_shapes.items():
        if found_shapes.get(name) != expected:
            raise ValueError(f"Incompatible or missing {name}: expected {expected}, got {found_shapes.get(name)}.")

    dtype_code = dtypes[VISUAL_PREFIX + "merger.linear_fc2.weight"]
    if dtype_code not in DTYPES:
        raise ValueError(f"Unsupported vision weight dtype: {dtype_code}")
    return shards, actual, total_size, DTYPES[dtype_code]


def _build_treeweave_tensors(
    config: dict,
    tw_cfg: TreeWeaveConfig,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    vision = config["vision_config"]
    text = config["text_config"]
    v_hidden = int(vision["hidden_size"])
    out_hidden = int(vision["out_hidden_size"])
    std = float(vision.get("initializer_range", 0.02))

    # Determine full-attention layer indices in text_config
    num_layers = int(text["num_hidden_layers"])
    layer_types = text.get("layer_types")
    if layer_types is None:
        interval = int(text.get("full_attention_interval", 4))
        layer_types = [
            "linear_attention" if bool((i + 1) % interval) else "full_attention"
            for i in range(num_layers)
        ]

    head_dim = int(text.get("head_dim", text["hidden_size"] // text["num_attention_heads"]))
    rope_params = text.get("rope_parameters") or {}
    partial_factor = float(rope_params.get("partial_rotary_factor", 0.25))
    rotary_dim = int(head_dim * partial_factor)
    num_q_heads = int(text["num_attention_heads"])
    num_kv_heads = int(text.get("num_key_value_heads", num_q_heads))

    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(tw_cfg.seed)

        pyramid = TreeWeavePyramid(v_hidden, out_hidden, fine=tw_cfg.spatial_merge_size)
        pyramid.apply_transparent_init()
        pyramid = pyramid.to(dtype=dtype)

        router = SpatiotemporalTreeRouter(
            hidden_size=v_hidden,
            depth=tw_cfg.router_cross_attn_depth,
            num_heads=tw_cfg.router_num_heads,
            mlp_ratio=tw_cfg.router_mlp_ratio,
            temporal_window_size=tw_cfg.temporal_window_size,
            use_temporal_attn=tw_cfg.router_use_temporal_attn,
        )
        router.apply_initial_weights(std=std, initial_prob=tw_cfg.initial_compression_prob)
        router = router.to(dtype=dtype)
        router.enforce_fp32_heads()

        tensors: Dict[str, torch.Tensor] = {}
        for name, val in pyramid.state_dict().items():
            tensors[f"{VISUAL_PREFIX}pyramid.{name}"] = val.contiguous()
        for name, val in router.state_dict().items():
            tensors[f"{VISUAL_PREFIX}router.{name}"] = val.contiguous()

        for layer_idx, ltype in enumerate(layer_types):
            if ltype != "full_attention":
                continue
            vol = TreeWeaveAttentionVolumeAdapter(
                num_attention_heads=num_q_heads,
                num_key_value_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                mode=tw_cfg.volume_bias_mode if tw_cfg.use_volume_attn_bias else "none",
                gamma_init=tw_cfg.volume_gamma_init,
            ).to(dtype=dtype)
            for name, val in vol.state_dict().items():
                tensors[
                    f"model.language_model.layers.{layer_idx}.self_attn.treeweave_volume_bias.{name}"
                ] = val.contiguous()

    return tensors


def initialize_treeweave_checkpoint(
    source: Path | str,
    destination: Path | str,
    *,
    seed: int = 0,
    initial_compression_prob: float = 0.1,
    route_threshold: float = 0.5,
    max_spatial_level: int = 2,
    max_temporal_level: int = 2,
    allow_fine_temporal_merge: bool = True,
    use_volume_attn_bias: bool = True,
    volume_bias_mode: str = "key_direction",
    link_base_weights: bool = False,
) -> Path:
    """Create a ready-to-train / ready-to-serve Qwen3.5-TreeWeave checkpoint from native Qwen3.5 weights."""
    source = Path(source).expanduser().resolve(strict=True)
    destination = Path(destination).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Destination already exists: {destination}")
    destination = destination.resolve()
    if destination.is_relative_to(source):
        raise ValueError("Destination must lie outside the source checkpoint directory.")
    if not (0.0 < initial_compression_prob < 1.0) or not math.isfinite(initial_compression_prob):
        raise ValueError("initial_compression_prob must be finite and strictly in (0, 1).")
    if not math.isfinite(route_threshold) or route_threshold < 0.0:
        raise ValueError("route_threshold must be finite and >= 0.")

    config = copy.deepcopy(_read_json(source / "config.json"))
    if config.get("model_type") != "qwen3_5" or config.get("architectures") != [
        "Qwen3_5ForConditionalGeneration"
    ]:
        raise ValueError("Source must be a native dense Qwen3.5 checkpoint.")
    vision = config["vision_config"]
    if vision.get("deepstack_visual_indexes"):
        raise ValueError("Deepstack checkpoints are not supported.")
    if vision.get("patch_size") != 16 or vision.get("spatial_merge_size") != 2:
        raise ValueError("Source checkpoint must have patch_size=16 and spatial_merge_size=2.")

    depth = int(vision["depth"])
    heads = int(vision["num_heads"])
    tw_cfg = TreeWeaveConfig(
        spatial_merge_size=2,
        max_spatial_level=max_spatial_level,
        max_temporal_level=max_temporal_level,
        allow_fine_temporal_merge=allow_fine_temporal_merge,
        router_cross_attn_depth=6,
        router_sample_depths=(0, depth // 2, depth),
        router_num_heads=heads,
        theta_s1=route_threshold,
        theta_s2=route_threshold,
        theta_t1=route_threshold,
        theta_t2=route_threshold,
        use_volume_attn_bias=use_volume_attn_bias,
        volume_bias_mode=volume_bias_mode,
        initial_compression_prob=initial_compression_prob,
        seed=seed,
    )

    shards, native_map, total_size, dtype = _inspect_native_shards(source, config)
    new_tensors = _build_treeweave_tensors(config, tw_cfg, dtype)
    total_size += sum(t.numel() * t.element_size() for t in new_tensors.values())

    config["architectures"] = [ARCHITECTURE]
    config["treeweave"] = tw_cfg.to_dict()
    config["dtype"] = str(dtype).removeprefix("torch.")
    config["vision_config"]["dtype"] = config["dtype"]
    for blk in (config, config["vision_config"], config["text_config"]):
        blk.pop("auto_map", None)
        blk.pop("_name_or_path", None)

    count = len(shards) + 1
    renamed = {orig: f"model-{i:05d}-of-{count:05d}.safetensors" for i, orig in enumerate(shards, 1)}
    extra_shard = f"model-{count:05d}-of-{count:05d}.safetensors"
    weight_map = {k: renamed[sh] for k, sh in native_map.items()}
    weight_map.update({k: extra_shard for k in new_tensors})

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=".treeweave-init-", dir=destination.parent))
    try:
        for orig, target in renamed.items():
            if link_base_weights:
                (tmp_dir / target).symlink_to((source / orig).resolve())
            else:
                shutil.copyfile(source / orig, tmp_dir / target)
        save_file(new_tensors, str(tmp_dir / extra_shard), metadata={"format": "pt"})

        for asset_name in TOKENIZER_ASSETS:
            src_asset = source / asset_name
            if src_asset.is_file():
                shutil.copyfile(src_asset, tmp_dir / asset_name)

        (tmp_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (tmp_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        tmp_dir.rename(destination)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    logger.info(
        "Initialized TreeWeave checkpoint at %s (+%d tensors, P_init=%.3f, threshold=%.3f)",
        destination,
        len(new_tensors),
        initial_compression_prob,
        route_threshold,
    )
    return destination


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, help="Local native Qwen/Qwen3.5-4B directory")
    p.add_argument("--destination", required=True, help="Output TreeWeave checkpoint directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--initial-compression-prob", type=float, default=0.1)
    p.add_argument("--route-threshold", type=float, default=0.5)
    p.add_argument("--max-spatial-level", type=int, default=2, choices=(1, 2))
    p.add_argument("--max-temporal-level", type=int, default=2, choices=(0, 1, 2))
    p.add_argument("--rigid-octree", action="store_true", help="Forbid s=0 temporal folding (Q2 ablation)")
    p.add_argument("--no-volume-bias", action="store_true", help="Disable volume attention bias (Q4 ablation)")
    p.add_argument("--link-base-weights", action="store_true", help="Symlink native 4B shards to save disk space")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = initialize_treeweave_checkpoint(
        args.source,
        args.destination,
        seed=args.seed,
        initial_compression_prob=args.initial_compression_prob,
        route_threshold=args.route_threshold,
        max_spatial_level=args.max_spatial_level,
        max_temporal_level=args.max_temporal_level,
        allow_fine_temporal_merge=not args.rigid_octree,
        use_volume_attn_bias=not args.no_volume_bias,
        link_base_weights=args.link_base_weights,
    )
    print(f"Created TreeWeave initialization checkpoint: {out}")


if __name__ == "__main__":
    main()
