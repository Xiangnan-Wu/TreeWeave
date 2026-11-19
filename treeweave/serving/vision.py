# Portions derived from SGLang and VisionWeave (Apache-2.0).
# Adapted for Qwen3.5-TreeWeave 9-scale spatiotemporal pyramid & tree router.
"""SGLang Disaggregated Vision Tower with 9-Scale `TreeWeavePyramid` & `SpatiotemporalTreeRouter`."""

from __future__ import annotations

import inspect
import logging
from contextlib import contextmanager
from typing import List, Optional

import numpy as np
import torch
from sglang.srt.environ import envs
from sglang.srt.layers.attention.vision import prepare_vision_attention_metadata
from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel, _is_npu
from sglang.srt.server_args import get_global_server_args

from .. import TreeWeaveConfig
from ..layout import TreeRouteLayout, hard_tree_route_layout
from ..pooler import TreeWeavePyramid, patchify_to_raster
from ..router import SpatiotemporalTreeRouter
from .core_patches import publish_route
from .sglang_route import (
    TreeRouteItem,
    TreeRoutePayload,
    budget_logit_shift_from_env,
    route_threshold_from_env,
    validate_serving_geometry,
)

logger = logging.getLogger(__name__)
__all__ = ["Qwen3_5TreeWeaveSGLangVisionModel"]

_UPSTREAM_FORWARD_MARKERS = (
    "if self._use_vectorized_pos_embed(len(grid_thw_list)):",
    "pos_embeds = self.fast_pos_embed_interpolate_from_list(grid_thw_list)",
    "rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)",
    "forward_metadata = prepare_vision_attention_metadata(",
    "for layer_num, blk in enumerate(self.blocks):",
    "x = self.merger(x)",
)


def _assert_upstream_forward_unchanged() -> None:
    src = inspect.getsource(Qwen3VLMoeVisionModel.forward)
    missing = [m for m in _UPSTREAM_FORWARD_MARKERS if m not in src]
    if missing:
        raise RuntimeError(
            f"Qwen3VLMoeVisionModel.forward in this sglang version is missing expected markers: {missing}."
        )


_assert_upstream_forward_unchanged()


class Qwen3_5TreeWeaveSGLangVisionModel(Qwen3VLMoeVisionModel):
    """SGLang Vision Encoder running SGLang ViT blocks + TreeWeave 9-Scale Pyramid & FP32 Router."""

    def __init__(
        self,
        vision_config,
        *args,
        tw_config: Optional[TreeWeaveConfig] = None,
        **kwargs,
    ):
        fake_hf = {"vision_config": vision_config, "treeweave": tw_config}
        self.tw_config, self.fine, self.pool, self.coarse, self.pixel_factor = validate_serving_geometry(
            fake_hf
        )
        self.threshold = route_threshold_from_env(default=self.tw_config.theta_s1)
        self.budget_logit_shift = budget_logit_shift_from_env(default=self.tw_config.budget_logit_shift)

        if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
            raise ValueError(
                "SGLANG_VIT_ENABLE_CUDA_GRAPH is incompatible with TreeWeave dynamic routing; "
                "unset it (`python -m treeweave.serving.serve` sets SGLANG_VIT_ENABLE_CUDA_GRAPH=0)."
            )
        if get_global_server_args().mm_attention_backend == "flashinfer_cudnn":
            raise ValueError(
                "--mm-attention-backend flashinfer_cudnn pads vision sequences and is unsupported for TreeWeave."
            )

        super().__init__(vision_config, *args, **kwargs)
        if self.deepstack_visual_indexes:
            raise ValueError(
                f"TreeWeave does not support deepstack; got {self.deepstack_visual_indexes!r}."
            )

        self.router_sample_depths: List[int] = [int(d) for d in self.tw_config.router_sample_depths]
        self.pyramid = TreeWeavePyramid(
            vision_hidden_size=vision_config.hidden_size,
            llm_hidden_size=vision_config.out_hidden_size,
            fine=self.fine,
        ).to(device=self.device, dtype=self.dtype)
        self.router = SpatiotemporalTreeRouter(
            hidden_size=vision_config.hidden_size,
            depth=self.tw_config.router_cross_attn_depth,
            num_heads=self.tw_config.router_num_heads,
            mlp_ratio=self.tw_config.router_mlp_ratio,
            temporal_window_size=self.tw_config.temporal_window_size,
            use_temporal_attn=self.tw_config.router_use_temporal_attn,
        ).to(device=self.device, dtype=self.dtype)
        self.router.enforce_fp32_heads()

        self._modality: Optional[str] = None
        self.fingerprint: Optional[str] = None
        logger.info(
            "[treeweave] encoder vision tower ready: fine=%d pool=%d coarse=%d pixel_factor=%d "
            "max_s=%d max_t=%d threshold=%.6g shift=%+.6g",
            self.fine,
            self.pool,
            self.coarse,
            self.pixel_factor,
            self.tw_config.max_spatial_level,
            self.tw_config.max_temporal_level,
            self.threshold,
            self.budget_logit_shift,
        )

    @contextmanager
    def modality_ctx(self, modality: str):
        prev = self._modality
        self._modality = modality
        try:
            yield
        finally:
            self._modality = prev

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
            raise RuntimeError("SGLANG_VIT_ENABLE_CUDA_GRAPH became enabled after model initialization.")
        if self._modality is None:
            raise RuntimeError("Qwen3_5TreeWeaveSGLangVisionModel.forward called outside `modality_ctx`.")
        if self.fingerprint is None:
            raise RuntimeError("Qwen3_5TreeWeaveSGLangVisionModel.fingerprint was not initialized.")

        if isinstance(grid_thw, list):
            grid_thw_list = [(int(r[0]), int(r[1]), int(r[2])) for r in grid_thw]
            grid_np = np.array(grid_thw_list, dtype=np.int32)
        else:
            grid_thw_list = [(int(r[0]), int(r[1]), int(r[2])) for r in grid_thw.tolist()]
            grid_np = grid_thw.detach().cpu().numpy().astype(np.int32)

        for t, h, w in grid_thw_list:
            if h % self.coarse != 0 or w % self.coarse != 0:
                raise ValueError(
                    f"TreeWeave requires ViT grid divisible by coarse={self.coarse} ({self.pixel_factor}px), "
                    f"got {(t, h, w)}."
                )

        x = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
        x = self.patch_embed(x)
        if self._use_vectorized_pos_embed(len(grid_thw_list)):
            pos_embeds = self.fast_pos_embed_interpolate_vectorized(grid_thw_list)
        else:
            pos_embeds = self.fast_pos_embed_interpolate_from_list(grid_thw_list)
        x += pos_embeds
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)

        token_cu_seqlens = np.repeat(grid_np[:, 1] * grid_np[:, 2], grid_np[:, 0]).cumsum(
            axis=0, dtype=np.int32
        )
        token_cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), token_cu_seqlens])
        cu_seqlens = torch.from_numpy(token_cu_seqlens)
        cu_seqlens = cu_seqlens.to("cpu") if _is_npu else cu_seqlens.to(self.device, non_blocking=True)

        forward_metadata = prepare_vision_attention_metadata(
            cu_seqlens,
            device=self.device,
            packed_indptrs=None,
            sequence_lengths=None,
            flashinfer_max_seqlen=0,
        )

        x = x.unsqueeze(1)
        query, router_ctx = self.router.init_state(grid_thw_list, self.device, self.dtype)
        tap_cursor = 0
        if tap_cursor < len(self.router_sample_depths) and self.router_sample_depths[tap_cursor] == 0:
            rasters_0 = patchify_to_raster(x.squeeze(1), grid_thw_list, fine=self.fine)
            query = self.router.run_tap(
                query, router_ctx, rasters_0, grid_thw_list, pair_start_index=2 * tap_cursor
            )
            tap_cursor += 1

        final_rasters = None
        for block_depth, blk in enumerate(self.blocks, start=1):
            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                forward_metadata=forward_metadata,
            )
            while (
                tap_cursor < len(self.router_sample_depths)
                and self.router_sample_depths[tap_cursor] == block_depth
            ):
                rasters_d = patchify_to_raster(x.squeeze(1), grid_thw_list, fine=self.fine)
                if block_depth == len(self.blocks):
                    final_rasters = rasters_d
                query = self.router.run_tap(
                    query, router_ctx, rasters_d, grid_thw_list, pair_start_index=2 * tap_cursor
                )
                tap_cursor += 1

        if tap_cursor != len(self.router_sample_depths):
            raise RuntimeError(
                f"Only {tap_cursor}/{len(self.router_sample_depths)} router taps ran on {len(self.blocks)} ViT blocks."
            )
        if final_rasters is None:
            final_rasters = patchify_to_raster(x.squeeze(1), grid_thw_list, fine=self.fine)

        cfg = self.tw_config
        r_out = self.router.score_queries(
            query,
            grid_thw_list,
            budget_logit_shift=self.budget_logit_shift,
            max_spatial_level=cfg.max_spatial_level,
            max_temporal_level=cfg.max_temporal_level,
        )

        thr = self.threshold
        layouts: List[TreeRouteLayout] = []
        items: List[TreeRouteItem] = []
        for grid, ps1, ps2, pt1, pt2 in zip(
            grid_thw_list, r_out.p_s1, r_out.p_s2, r_out.p_t1, r_out.p_t2
        ):
            layout = hard_tree_route_layout(
                grid,
                ps1,
                ps2,
                pt1,
                pt2,
                theta_s1=thr,
                theta_s2=thr,
                theta_t1=thr,
                theta_t2=thr,
                max_spatial_level=cfg.max_spatial_level,
                max_temporal_level=cfg.max_temporal_level,
                allow_fine_temporal_merge=cfg.allow_fine_temporal_merge,
            )
            layouts.append(layout)
            items.append(TreeRouteItem.from_layout(self._modality, layout))

        pyramids = [
            self.pyramid.build_item_pyramid(
                r,
                max_spatial_level=cfg.max_spatial_level,
                max_temporal_level=cfg.max_temporal_level,
            )
            for r in final_rasters
        ]
        routed_list = self.pyramid.project_selected_nodes(pyramids, layouts, self.merger)
        routed_embeddings = (
            routed_list[0] if len(routed_list) == 1 else torch.cat(routed_list, dim=0)
        )

        payload = TreeRoutePayload(
            fingerprint=self.fingerprint,
            items=tuple(items),
            threshold=thr,
            budget_logit_shift=self.budget_logit_shift,
        )
        publish_route(payload.to_dict())
        return routed_embeddings
