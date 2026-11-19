# Portions derived from SGLang and VisionWeave (Apache-2.0).
# Adapted for Qwen3.5-TreeWeave 9-scale routing, 0-token folded video slices, & log-volume propagation.
"""SGLang Tokenizer-Process Multimodal Processor Building Routed Placeholders, MRoPE & Log-Volumes."""

from __future__ import annotations

import inspect
import logging
from typing import Dict, List

import torch
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.processors.base_processor import MultimodalProcessorOutput
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

from ... import ARCHITECTURE
from ..patches import set_geometry
from ..sglang_route import (
    ROUTE_KWARG,
    TreeRouteItem,
    TreeRoutePayload,
    build_routed_input_ids_and_runs,
    config_fingerprint,
    route_threshold_from_env,
    tree_routed_mrope_positions,
    validate_serving_geometry,
)

logger = logging.getLogger(__name__)

_UPSTREAM_BUILDER_MARKERS = (
    "mm_token_num = img_grid_thw[img_idx].prod() // (spatial_merge_size**2)",
    "frame_seqlen = video_grid_thw[video_idx][1:].prod().item() // (",
    'timestamp_text = f"<{curr_time:.1f} seconds>"',
    "video_tokens.extend([video_token_id] * frame_seqlen)",
    "cur_idx = mm_start_idx + 2",
)


def _assert_upstream_builder_unchanged() -> None:
    src = inspect.getsource(QwenVLImageProcessor.build_input_ids_with_timestamps)
    missing = [m for m in _UPSTREAM_BUILDER_MARKERS if m not in src]
    if missing:
        raise RuntimeError(
            f"QwenVLImageProcessor.build_input_ids_with_timestamps in this sglang version is missing: {missing}."
        )


_assert_upstream_builder_unchanged()


class Qwen3_5TreeWeaveForConditionalGeneration:
    """Name-only architecture stub so SGLang's `PROCESSOR_MAPPING` selects `TreeWeaveSGLangImageProcessor`
    without importing GPU model modules in `TokenizerManager`.
    """


assert Qwen3_5TreeWeaveForConditionalGeneration.__name__ == ARCHITECTURE

_MODALITY_NAMES = {Modality.IMAGE: "image", Modality.VIDEO: "video"}
_MODALITY_BY_NAME = {name: mod for mod, name in _MODALITY_NAMES.items()}


def _group_routes_by_modality(routes) -> Dict[Modality, List[dict]]:
    if isinstance(routes, dict):
        raise ValueError(
            "Expected flat part-ordered list of TreeRoutePayloads from SGLang's per-part metadata registry."
        )
    grouped: Dict[Modality, List[dict]] = {}
    for part_idx, raw in enumerate(routes):
        names = {item["modality"] for item in raw["items"]}
        if len(names) != 1:
            raise ValueError(
                f"Route payload part {part_idx} mixes modalities {sorted(names)}."
            )
        name = names.pop()
        mod = _MODALITY_BY_NAME.get(name)
        if mod is None:
            raise ValueError(f"Unsupported modality {name!r} in route payload part {part_idx}.")
        grouped.setdefault(mod, []).append(raw)
    return grouped


def _check_route_arity(grouped: Dict[Modality, List[dict]], img_grid_thw, video_grid_thw) -> None:
    n_items = sum(len(raw["items"]) for payloads in grouped.values() for raw in payloads)
    n_grids = sum(0 if g is None else len(g) for g in (img_grid_thw, video_grid_thw))
    if n_items != n_grids:
        raise ValueError(
            f"Encoder sent {n_grids} grid(s), but route payloads describe {n_items} media item(s)."
        )


class TreeWeaveSGLangImageProcessor(QwenVLImageProcessor):
    """Builds compacted placeholders, `POSITION_SCALE=2` MRoPE, and `treeweave_log_volumes` per item."""

    models = [Qwen3_5TreeWeaveForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        self.tw_config, self.fine, self.pool, self.coarse, self.pixel_factor = validate_serving_geometry(
            hf_config
        )
        set_geometry(hf_config)
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.threshold = route_threshold_from_env(default=self.tw_config.theta_s1)
        self.fingerprint = config_fingerprint(hf_config)
        logger.info(
            "[treeweave] processor ready: fine=%d pool=%d coarse=%d pixel_factor=%d threshold=%.6g fp=%s",
            self.fine,
            self.pool,
            self.coarse,
            self.pixel_factor,
            self.threshold,
            self.fingerprint,
        )

    def _payload_items(self, routes, modality: Modality, embeddings, grid_thw) -> List[TreeRouteItem]:
        payloads = routes.get(modality, [])
        if not payloads:
            if grid_thw is not None and len(grid_thw):
                raise ValueError(f"Encoder sent {len(grid_thw)} {modality} grid(s) without a route payload.")
            return []
        rows = int(embeddings[modality].shape[0])
        items: List[TreeRouteItem] = []
        seen = 0
        for raw in payloads:
            payload = TreeRoutePayload.from_dict(raw)
            payload.validate(
                fingerprint=self.fingerprint,
                threshold=self.threshold,
                embedding_rows=payload.output_length,
                fine=self.fine,
            )
            for item in payload.items:
                if item.modality != _MODALITY_NAMES[modality]:
                    raise ValueError(
                        f"Route payload arrived on {modality} channel with item modality {item.modality!r}."
                    )
            items.extend(payload.items)
            seen += payload.output_length
        if seen != rows:
            raise ValueError(
                f"{modality} route payloads describe {seen} tokens, encoder sent {rows} embedding rows."
            )
        if grid_thw is not None:
            grids = [tuple(int(x) for x in row) for row in grid_thw.tolist()]
            claimed = [item.grid for item in items]
            if grids != claimed:
                raise ValueError(f"{modality} grids {grids} disagree with route payload grids {claimed}.")
        return items

    def get_mm_data(self, prompt, embeddings, **kwargs):
        img_grid_thw = kwargs.get("img_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        video_timestamps = kwargs.get("video_timestamps", None)
        routes = kwargs.get(ROUTE_KWARG, None)
        if routes is None:
            raise ValueError(
                f"Encoder returned embeddings without `{ROUTE_KWARG}`; ensure encoder loaded {ARCHITECTURE}."
            )
        if kwargs.get("audio_feature_lens") is not None:
            raise ValueError("TreeWeave does not support audio input.")

        grouped = _group_routes_by_modality(routes)
        _check_route_arity(grouped, img_grid_thw, video_grid_thw)
        image_items = self._payload_items(grouped, Modality.IMAGE, embeddings, img_grid_thw)
        video_items = self._payload_items(grouped, Modality.VIDEO, embeddings, video_grid_thw)

        prompt_ids = (
            prompt if isinstance(prompt, list) else self._processor.tokenizer.encode(prompt)
        )
        built = build_routed_input_ids_and_runs(
            prompt_ids,
            image_items=image_items,
            video_items=video_items,
            video_timestamps=video_timestamps,
            encode_timestamp_fn=lambda txt: self._processor.tokenizer.encode(
                txt, add_special_tokens=False
            ),
            img_token_id=self.IM_TOKEN_ID,
            video_token_id=self.VIDEO_TOKEN_ID,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            fine=self.fine,
        )
        positions, delta = tree_routed_mrope_positions(len(built.input_ids), built.runs)

        mm_items: List[MultimodalDataItem] = []
        consumed: Dict[Modality, int] = {}
        for chunk in built.chunks:
            mod_enum = _MODALITY_BY_NAME[chunk.modality]
            e0, e1 = chunk.embedding_slice
            mm_items.append(
                MultimodalDataItem(
                    modality=mod_enum,
                    offsets=[chunk.offset],
                    precomputed_embeddings=embeddings[mod_enum][e0:e1],
                    model_specific_data={"treeweave_log_volumes": chunk.log_volumes},
                )
            )
            consumed[mod_enum] = max(consumed.get(mod_enum, 0), e1)

        for mod_enum in (Modality.IMAGE, Modality.VIDEO):
            if mod_enum in embeddings and embeddings[mod_enum] is not None:
                avail = int(embeddings[mod_enum].shape[0])
                used = consumed.get(mod_enum, 0)
                if used != avail:
                    raise ValueError(
                        f"{mod_enum} placeholders consumed {used} embedding rows, encoder sent {avail}."
                    )

        return MultimodalProcessorOutput(
            input_ids=built.input_ids,
            mm_items=mm_items,
            im_start_id=self.IM_START_TOKEN_ID,
            im_end_id=self.IM_END_TOKEN_ID,
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            audio_token_id=self.mm_tokens.audio_token_id,
            mrope_positions=torch.from_numpy(positions),
            mrope_position_delta=torch.tensor([[delta]], dtype=torch.long),
        )

    async def process_mm_data_async(self, *args, **kwargs):
        raise RuntimeError(
            f"{ARCHITECTURE} requires disaggregated serving (`--language-only` + `--encoder-urls`); "
            "local unrouted multimodal processing is disabled."
        )
