"""HuggingFace / ms-swift Native Model Implementation of Qwen3.5-TreeWeave."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5VisionModel,
)
from transformers.vision_utils import (
    get_vision_attention_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

from . import ARCHITECTURE, NUM_GRANULARITIES, TreeWeaveConfig
from .layout import (
    RoutedBatchPack,
    SoftTreeResult,
    TreeRouteLayout,
    build_routed_sequence_batch,
    compute_heuristic_router_outputs,
    hard_tree_route_layout,
    sample_stochastic_tree_layout,
    shuffle_router_outputs,
    soft_tree_mix,
)
from .losses import chunked_topk_kl_distill_loss, extract_active_positions
from .pooler import TreeWeavePyramid, patchify_to_raster
from .router import RouterOutputs, SpatiotemporalTreeRouter
from .volume_bias import (
    active_token_log_volumes,
    install_volume_bias_on_language_model,
    varlen_bounds_context,
    volume_bias_context,
)

__all__ = [
    "DistillForwardOutput",
    "Qwen3_5TreeWeaveForConditionalGeneration",
    "Qwen3_5TreeWeaveModel",
    "Qwen3_5TreeWeaveVisionModel",
    "VisionTowerEncodeResult",
]


def _has_fused_qwen3_5_varlen_kernels(language_model: nn.Module) -> bool:
    """True when Qwen3_5GatedDeltaNet (`fla` + `causal_conv1d`) AND full-attention varlen
    (`flash_attention_*` OR TreeWeave's inner varlen SDPA patch) can execute a packed batch in ONE pass!
    """
    if not torch.cuda.is_available():
        return False
    first_param = next(language_model.parameters(), None)
    if first_param is not None and not first_param.is_cuda:
        return False
    has_linear_varlen = (
        importlib.util.find_spec("fla") is not None
        and importlib.util.find_spec("causal_conv1d") is not None
    )
    if not has_linear_varlen:
        return False
    attn_impl = str(getattr(language_model.config, "_attn_implementation", ""))
    if "flash" in attn_impl:
        return True
    return bool(getattr(language_model, "_treeweave_varlen_sdpa_patched", False))


def _is_truly_packed_batch(
    attention_mask: Optional[torch.Tensor],
    cu_seq_lens_q: Optional[torch.Tensor],
) -> bool:
    """Distinguish real padding-free packed rows from single-row padded batches where `ms-swift` added `[0, L]`."""
    if cu_seq_lens_q is None:
        return False
    if attention_mask is None:
        return True
    # If `attention_mask` contains any 0s (`padding_to` / left- or right-padding), the batch is NOT padding-free!
    if not bool(attention_mask.all()):
        return False
    return int(cu_seq_lens_q.numel()) > 2


def _sanitize_varlen_kwargs(
    kwargs: Dict[str, Any],
    *,
    is_packed: bool,
    attention_mask: Optional[torch.Tensor],
) -> Dict[str, Any]:
    """Strip degenerate `[0, L]` varlen kwargs when `is_packed` is False so padded batches use `attention_mask`."""
    if is_packed:
        return dict(kwargs)
    if "cu_seq_lens_q" not in kwargs and "cu_seq_lens_k" not in kwargs:
        return dict(kwargs)
    return {
        k: v
        for k, v in kwargs.items()
        if k not in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")
    }


def _maybe_install_qwen3_5_sp_linear_patch() -> None:
    """Install `ms-swift`'s `cu_seq_lens_q -> causal_conv1d(seq_idx)` patch even for standalone `from_pretrained`."""
    try:
        from swift.model.models.qwen import _patch_qwen3_5_linear_attention_sequence_parallel

        _patch_qwen3_5_linear_attention_sequence_parallel()
    except Exception:
        pass


def _run_lm_varlen_safe(
    language_model: nn.Module,
    *,
    inputs_embeds: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = None,
    token_log_volumes: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Run `Qwen3_5TextModel` in ONE fused pass on CUDA when `fla` + `causal_conv1d` are installed
    (for BOTH `flash_attention_*` AND `sdpa` via TreeWeave's inner varlen SDPA patch!), or fall back to
    exact segment-isolated execution on CPU / MPS / environments without `fla` + `causal_conv1d`.
    """
    eff_lv = token_log_volumes if token_log_volumes is not None else active_token_log_volumes()
    cu_q = kwargs.get("cu_seq_lens_q")
    has_multi_seg = isinstance(cu_q, torch.Tensor) and int(cu_q.numel()) > 2
    if has_multi_seg:
        _maybe_install_qwen3_5_sp_linear_patch()

    if cu_q is None or past_key_values is not None or _has_fused_qwen3_5_varlen_kernels(language_model):
        call_kwargs = dict(kwargs)
        if eff_lv is not None:
            call_kwargs["treeweave_token_log_volumes"] = eff_lv
        attn_impl = str(getattr(language_model.config, "_attn_implementation", "sdpa"))
        if has_multi_seg and past_key_values is None and "flash" not in attn_impl:
            # Padding-free packed sequence under varlen SDPA: pass `attention_mask=None` so `create_causal_mask`
            # stays O(1) (`None`), and pre-fetch CPU `bounds` ONCE outside the 32-layer loop!
            bounds = tuple(int(x) for x in cu_q.detach().cpu().tolist())
            with varlen_bounds_context(bounds):
                return language_model(
                    input_ids=None,
                    position_ids=position_ids,
                    attention_mask=None,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    **call_kwargs,
                )
        return language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **call_kwargs,
        )

    clean_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k", "treeweave_token_log_volumes")
    }
    bounds = cu_q.detach().cpu().tolist()
    seg_hiddens: List[torch.Tensor] = []

    for s_idx, e_idx in zip(bounds[:-1], bounds[1:]):
        s_i, e_i = int(s_idx), int(e_idx)
        if e_i <= s_i:
            continue
        seg_emb = inputs_embeds[:, s_i:e_i, :]
        seg_pos = None
        if position_ids is not None:
            seg_pos = position_ids[..., s_i:e_i]
        seg_kwargs = dict(clean_kwargs)
        if eff_lv is not None:
            seg_kwargs["treeweave_token_log_volumes"] = eff_lv[:, s_i:e_i]
        seg_out = language_model(
            input_ids=None,
            position_ids=seg_pos,
            attention_mask=None,
            past_key_values=None,
            inputs_embeds=seg_emb,
            use_cache=False,
            **seg_kwargs,
        )
        seg_hiddens.append(seg_out.last_hidden_state)

    cat_hidden = seg_hiddens[0] if len(seg_hiddens) == 1 else torch.cat(seg_hiddens, dim=1)
    return Qwen3_5ModelOutputWithPast(last_hidden_state=cat_hidden, past_key_values=None)


@dataclass
class VisionTowerEncodeResult:
    """Structured output of a single ViT + TreeWeave pass."""

    rasters: List[torch.Tensor]                # per-item `(T, H, W, d)` final ViT features
    grid_thw_list: List[Tuple[int, int, int]]  # per-item `(T, H, W)`
    router_query: Optional[torch.Tensor]       # `(total_quadrants, d)` if router ran
    native_raw_states: torch.Tensor            # `(total_patches, d)` in native merger order


@dataclass
class DistillForwardOutput:
    """Output of `Qwen3_5TreeWeaveForConditionalGeneration.forward_distill`."""

    loss: torch.Tensor
    distill_loss: torch.Tensor
    rate_loss: torch.Tensor
    balance_loss: torch.Tensor
    metrics: Dict[str, float]


@dataclass
class TreeWeaveCausalLMOutputWithPast(Qwen3_5CausalLMOutputWithPast):
    """HF ModelOutput carrying self-distillation sub-losses & routing diagnostics through DDP/DeepSpeed/FSDP."""

    distill_loss: Optional[torch.Tensor] = None
    rate_loss: Optional[torch.Tensor] = None
    balance_loss: Optional[torch.Tensor] = None
    treeweave_metrics: Optional[Dict[str, float]] = None


class Qwen3_5TreeWeaveVisionModel(Qwen3_5VisionModel):
    """Qwen3.5 Vision Encoder augmented with TreeWeave 9-Scale Pyramid & Spatiotemporal Tree Router."""

    def __init__(self, config: Qwen3_5VisionConfig, treeweave_cfg: Optional[TreeWeaveConfig] = None):
        super().__init__(config)
        self.tw_config = treeweave_cfg or TreeWeaveConfig(
            spatial_merge_size=config.spatial_merge_size,
            router_sample_depths=(0, config.depth // 2, config.depth),
            router_num_heads=config.num_heads,
        )
        sample_depths = [int(d) for d in self.tw_config.router_sample_depths]
        if len(sample_depths) != 3 or sample_depths != sorted(sample_depths):
            raise ValueError(f"router_sample_depths must be 3 sorted depths, got {sample_depths}.")
        self.router_sample_depths = sample_depths

        self.pyramid = TreeWeavePyramid(
            vision_hidden_size=config.hidden_size,
            llm_hidden_size=config.out_hidden_size,
            fine=config.spatial_merge_size,
        )
        self.router = SpatiotemporalTreeRouter(
            hidden_size=config.hidden_size,
            depth=self.tw_config.router_cross_attn_depth,
            num_heads=self.tw_config.router_num_heads,
            mlp_ratio=self.tw_config.router_mlp_ratio,
            temporal_window_size=self.tw_config.temporal_window_size,
            use_temporal_attn=self.tw_config.router_use_temporal_attn,
        )
        self.pyramid.apply_transparent_init()
        self.router.apply_initial_weights(
            std=float(getattr(config, "initializer_range", 0.02)),
            initial_prob=self.tw_config.initial_compression_prob,
        )

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "Qwen3_5TreeWeaveVisionModel.forward must not be called directly because TreeWeave emits "
            "variable-length routed visual tokens + 3D half-integer MRoPE + log-volume bias. "
            "Use `encode_hard_routed`, `encode_soft_routed`, or `Qwen3_5TreeWeaveModel.forward`."
        )

    def forward_vit_with_taps(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        *,
        run_router: bool = True,
        vit_no_grad: bool = False,
        **kwargs,
    ) -> VisionTowerEncodeResult:
        """Run the 24-layer ViT ONCE while tapping `[0, L//2, L]` for the Spatiotemporal Router."""
        grid_list = [(int(r[0]), int(r[1]), int(r[2])) for r in grid_thw.detach().cpu().tolist()]

        def _run_vit_backbone():
            interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
                grid_thw,
                num_grid_per_side=self.num_grid_per_side,
                mode=self.interpolation_mode,
                align_corners=self.interpolation_align_corners,
                spatial_merge_size=self.config.spatial_merge_size,
                kwargs=kwargs,
            )
            position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
            cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)

            x = self.patch_embed(pixel_values.to(dtype=self.dtype))
            pos_embeds = (self.pos_embed(interp_indices) * interp_weights[:, :, None]).sum(1)
            x = x + pos_embeds.to(x.dtype)
            pos_emb = self.rotary_pos_emb(x, position_ids)
            x = x.reshape(x.shape[0], -1)
            return x, cu_seqlens, max_seqlen, pos_emb

        grad_ctx = torch.no_grad() if vit_no_grad else torch.enable_grad()
        tapped_rasters: Dict[int, List[torch.Tensor]] = {}

        with grad_ctx:
            x, cu_seqlens, max_seqlen, pos_emb = _run_vit_backbone()
            if run_router and 0 in self.router_sample_depths:
                tapped_rasters[0] = patchify_to_raster(
                    x.detach() if vit_no_grad else x, grid_list, fine=self.spatial_merge_size
                )
            for depth_idx, blk in enumerate(self.blocks, start=1):
                x = blk(
                    x,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    position_embeddings=pos_emb,
                )
                if run_router and depth_idx in self.router_sample_depths:
                    tapped_rasters[depth_idx] = patchify_to_raster(
                        x.detach() if vit_no_grad else x, grid_list, fine=self.spatial_merge_size
                    )

        final_rasters = tapped_rasters.get(len(self.blocks))
        if final_rasters is None:
            final_rasters = patchify_to_raster(x, grid_list, fine=self.spatial_merge_size)

        router_q = None
        if run_router:
            router_q, router_ctx = self.router.init_state(grid_list, x.device, x.dtype)
            for tap_i, d_sample in enumerate(self.router_sample_depths):
                router_q = self.router.run_tap(
                    router_q, router_ctx, tapped_rasters[d_sample], grid_list, pair_start_index=2 * tap_i
                )

        return VisionTowerEncodeResult(
            rasters=final_rasters,
            grid_thw_list=grid_list,
            router_query=router_q,
            native_raw_states=x,
        )

    def encode_hard_routed(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        *,
        routing_mode: str = "learned",  # "learned" | "shuffled" | "heuristic" | "stochastic"
        stochastic_rng: Optional[np.random.Generator] = None,
        uniform_single_scale: Optional[Tuple[int, int]] = None,
        vit_no_grad: bool = False,
        router_no_grad: bool = False,
        return_native_states: bool = False,
        shuffle_seed: int = 0,
    ) -> Tuple[List[TreeRouteLayout], List[torch.Tensor], Optional[torch.Tensor]]:
        """Encode media items into hard-routed `(layouts, routed_embeddings, optional_native_states)`."""
        need_router = routing_mode in ("learned", "shuffled")
        enc = self.forward_vit_with_taps(
            pixel_values, grid_thw, run_router=need_router, vit_no_grad=vit_no_grad
        )
        cfg = self.tw_config

        layouts: List[TreeRouteLayout] = []
        if routing_mode == "stochastic":
            rng = stochastic_rng or np.random.default_rng()
            for grid in enc.grid_thw_list:
                layouts.append(
                    sample_stochastic_tree_layout(
                        grid,
                        rng,
                        max_spatial_level=cfg.max_spatial_level,
                        max_temporal_level=cfg.max_temporal_level,
                        allow_fine_temporal_merge=cfg.allow_fine_temporal_merge,
                        uniform_single_scale=uniform_single_scale,
                    )
                )
        else:
            r_ctx = torch.no_grad() if router_no_grad else torch.enable_grad()
            with r_ctx:
                if routing_mode == "heuristic":
                    r_out = compute_heuristic_router_outputs(enc.rasters, enc.grid_thw_list)
                else:
                    assert enc.router_query is not None
                    r_out = self.router.score_queries(
                        enc.router_query,
                        enc.grid_thw_list,
                        budget_logit_shift=cfg.budget_logit_shift,
                        max_spatial_level=cfg.max_spatial_level,
                        max_temporal_level=cfg.max_temporal_level,
                    )
                    if routing_mode == "shuffled":
                        r_out = shuffle_router_outputs(r_out, seed=shuffle_seed)

            for grid, ps1, ps2, pt1, pt2 in zip(
                enc.grid_thw_list, r_out.p_s1, r_out.p_s2, r_out.p_t1, r_out.p_t2
            ):
                layouts.append(
                    hard_tree_route_layout(
                        grid,
                        ps1,
                        ps2,
                        pt1,
                        pt2,
                        theta_s1=cfg.theta_s1,
                        theta_s2=cfg.theta_s2,
                        theta_t1=cfg.theta_t1,
                        theta_t2=cfg.theta_t2,
                        max_spatial_level=cfg.max_spatial_level,
                        max_temporal_level=cfg.max_temporal_level,
                        allow_fine_temporal_merge=cfg.allow_fine_temporal_merge,
                    )
                )

        pyramids = [
            self.pyramid.build_item_pyramid(
                r, max_spatial_level=cfg.max_spatial_level, max_temporal_level=cfg.max_temporal_level
            )
            for r in enc.rasters
        ]
        routed_embeds = self.pyramid.project_selected_nodes(pyramids, layouts, self.merger)
        native_states = enc.native_raw_states if return_native_states else None
        return layouts, routed_embeds, native_states

    def encode_soft_routed(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        *,
        rho_s1: float = 0.75,
        rho_s2: float = 0.50,
        rho_t1: float = 0.65,
        rho_t2: float = 0.50,
        vit_no_grad: bool = True,
        pooler_no_grad: bool = True,
    ) -> Tuple[SoftTreeResult, torch.Tensor]:
        """Stage 2 Soft-Tree forward returning `(SoftTreeResult, native_raw_states)`."""
        enc = self.forward_vit_with_taps(
            pixel_values, grid_thw, run_router=True, vit_no_grad=vit_no_grad
        )
        cfg = self.tw_config
        assert enc.router_query is not None
        r_out: RouterOutputs = self.router.score_queries(
            enc.router_query,
            enc.grid_thw_list,
            budget_logit_shift=0.0,
            max_spatial_level=cfg.max_spatial_level,
            max_temporal_level=cfg.max_temporal_level,
        )

        p_ctx = torch.no_grad() if pooler_no_grad else torch.enable_grad()
        with p_ctx:
            pyramids = [
                self.pyramid.build_item_pyramid(
                    r, max_spatial_level=cfg.max_spatial_level, max_temporal_level=cfg.max_temporal_level
                )
                for r in enc.rasters
            ]
            projected_scales = self.pyramid.project_all_scales(pyramids, self.merger)

        soft_res = soft_tree_mix(
            projected_scales,
            r_out,
            enc.grid_thw_list,
            rho_s1=rho_s1,
            rho_s2=rho_s2,
            rho_t1=rho_t1,
            rho_t2=rho_t2,
            max_spatial_level=cfg.max_spatial_level,
            max_temporal_level=cfg.max_temporal_level,
            allow_fine_temporal_merge=cfg.allow_fine_temporal_merge,
        )
        return soft_res, enc.native_raw_states


class Qwen3_5TreeWeaveModel(Qwen3_5Model):
    """`Qwen3_5Model` with `Qwen3_5TreeWeaveVisionModel` and Volume-Preserving Attention Bias."""

    def __init__(self, config: Qwen3_5Config):
        super().__init__(config)
        raw_tw = getattr(config, "treeweave", None)
        self.tw_config = TreeWeaveConfig.from_dict(
            raw_tw,
            vision_depth=config.vision_config.depth,
            num_heads=config.vision_config.num_heads,
        )
        # Replace standard visual tower with Qwen3_5TreeWeaveVisionModel
        self.visual = Qwen3_5TreeWeaveVisionModel(config.vision_config, treeweave_cfg=self.tw_config)

        vol_mode = self.tw_config.volume_bias_mode if self.tw_config.use_volume_attn_bias else "none"
        install_volume_bias_on_language_model(
            self.language_model,
            mode=vol_mode,
            gamma_init=self.tw_config.volume_gamma_init,
        )
        _maybe_install_qwen3_5_sp_linear_patch()
        self.last_route_layouts: Dict[str, List[TreeRouteLayout]] = {"image": [], "video": []}
        self.routed_attention_mask: Optional[torch.Tensor] = None
        self.routed_position_ids_4d: Optional[torch.Tensor] = None
        self._just_routed_prefill: bool = False

    def sync_treeweave_config(self) -> None:
        self.visual.tw_config = self.tw_config
        vol_mode = self.tw_config.volume_bias_mode if self.tw_config.use_volume_attn_bias else "none"
        install_volume_bias_on_language_model(
            self.language_model,
            mode=vol_mode,
            gamma_init=self.tw_config.volume_gamma_init,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        routing_mode: str = "learned",
        shuffle_seed: int = 0,
        **kwargs,
    ) -> Qwen3_5ModelOutputWithPast:
        if mm_encoder_outputs:
            raise ValueError(
                "Qwen3_5TreeWeaveModel does not accept pre-encoded `mm_encoder_outputs` (which bypass routing); "
                "pass raw `pixel_values` / `pixel_values_videos` instead."
            )

        is_packed = _is_truly_packed_batch(attention_mask, kwargs.get("cu_seq_lens_q"))
        lm_kwargs = _sanitize_varlen_kwargs(kwargs, is_packed=is_packed, attention_mask=attention_mask)
        text_pos_in = lm_kwargs.pop("text_position_ids", None)
        if text_pos_in is None and position_ids is not None and position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_pos_in = position_ids[0].long()
        elif (
            position_ids is not None
            and position_ids.ndim == 3
            and position_ids.shape[0] == 3
            and text_pos_in is not None
        ):
            position_ids = torch.cat((text_pos_in.to(position_ids.dtype).unsqueeze(0), position_ids), dim=0)

        # Pure text or cached decode step (no new pixel_values / pixel_values_videos)
        if pixel_values is None and pixel_values_videos is None:
            past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
            if past_len == 0:
                self.last_route_layouts = {"image": [], "video": []}
                self.routed_attention_mask = None
                self.routed_position_ids_4d = None
                self._just_routed_prefill = False
            lm_kwargs.pop("mm_token_type_ids", None)
            t_embeds = inputs_embeds if inputs_embeds is not None else self.get_input_embeddings()(input_ids)
            t_pos = position_ids
            if t_pos is None:
                t_pos = self.compute_3d_position_ids(
                    input_ids=input_ids,
                    inputs_embeds=t_embeds,
                    image_grid_thw=None,
                    video_grid_thw=None,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    mm_token_type_ids=mm_token_type_ids,
                )
            with volume_bias_context(None):
                outputs = _run_lm_varlen_safe(
                    self.language_model,
                    position_ids=t_pos,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=t_embeds,
                    use_cache=use_cache,
                    token_log_volumes=None,
                    **lm_kwargs,
                )
            return Qwen3_5ModelOutputWithPast(
                last_hidden_state=outputs.last_hidden_state,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=self.rope_deltas,
            )

        if input_ids is None or mm_token_type_ids is None:
            raise ValueError("Multimodal forward with TreeWeave requires `input_ids` and `mm_token_type_ids`.")

        base_embeds = inputs_embeds if inputs_embeds is not None else self.get_input_embeddings()(input_ids)

        img_layouts: List[TreeRouteLayout] = []
        img_embeds: List[torch.Tensor] = []
        vid_layouts: List[TreeRouteLayout] = []
        vid_embeds: List[torch.Tensor] = []

        if pixel_values is not None:
            assert image_grid_thw is not None
            img_layouts, img_embeds, _ = self.visual.encode_hard_routed(
                pixel_values,
                image_grid_thw,
                routing_mode=routing_mode,
                shuffle_seed=shuffle_seed,
            )
        if pixel_values_videos is not None:
            assert video_grid_thw is not None
            vid_layouts, vid_embeds, _ = self.visual.encode_hard_routed(
                pixel_values_videos,
                video_grid_thw,
                routing_mode=routing_mode,
                shuffle_seed=shuffle_seed,
            )

        self.last_route_layouts = {"image": img_layouts, "video": vid_layouts}

        pad_id = getattr(self.config.text_config, "pad_token_id", None) or 0
        pad_emb = self.get_input_embeddings().weight[pad_id]

        packed: RoutedBatchPack = build_routed_sequence_batch(
            input_ids=input_ids,
            base_inputs_embeds=base_embeds,
            mm_token_type_ids=mm_token_type_ids,
            image_layouts=img_layouts,
            image_routed_embeds=img_embeds,
            video_layouts=vid_layouts,
            video_routed_embeds=vid_embeds,
            attention_mask=attention_mask,
            text_position_ids=text_pos_in,
            is_packed=is_packed,
            pad_embedding=pad_emb,
            force_left_pad=not is_packed,
        )

        self.rope_deltas = packed.rope_deltas
        self.routed_attention_mask = packed.attention_mask
        self.routed_position_ids_4d = packed.position_ids_4d
        self._just_routed_prefill = True
        if is_packed and packed.cu_seq_lens_q is not None:
            lm_kwargs["cu_seq_lens_q"] = packed.cu_seq_lens_q
            lm_kwargs["cu_seq_lens_k"] = packed.cu_seq_lens_k
            lm_kwargs["max_length_q"] = packed.max_length_q
            lm_kwargs["max_length_k"] = packed.max_length_k

        eff_lv = packed.token_log_volumes if self.tw_config.use_volume_attn_bias else None
        outputs = _run_lm_varlen_safe(
            self.language_model,
            position_ids=packed.position_ids_4d,
            attention_mask=packed.attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=packed.inputs_embeds,
            use_cache=use_cache,
            token_log_volumes=eff_lv,
            **lm_kwargs,
        )

        return Qwen3_5ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


class Qwen3_5TreeWeaveForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Top-level `Qwen3_5TreeWeaveForConditionalGeneration` supporting 3-Stage Self-Distillation & Generation."""

    _keep_in_fp32_modules = [
        "head_s1",
        "head_s2",
        "head_t1",
        "head_t2",
        "bias_s1",
        "bias_s2",
        "bias_t1",
        "bias_t2",
    ]

    def __init__(self, config: Qwen3_5Config):
        super().__init__(config)
        self.model = Qwen3_5TreeWeaveModel(config)
        if config.tie_word_embeddings:
            self.tie_weights()
        self.model.visual.router.enforce_fp32_heads()

    @property
    def tw_config(self) -> TreeWeaveConfig:
        return self.model.tw_config

    @property
    def visual(self) -> Qwen3_5TreeWeaveVisionModel:
        return self.model.visual

    @property
    def language_model(self):
        return self.model.language_model

    def sync_config_to_hf(self) -> None:
        """Persist `TreeWeaveConfig` into `self.config.treeweave` before saving checkpoints."""
        self.config.treeweave = self.tw_config.to_dict()
        self.config.architectures = [ARCHITECTURE]
        self.model.sync_treeweave_config()
        self.model.visual.router.enforce_fp32_heads()

    def configure_stage_trainability(
        self,
        stage: int,
        *,
        train_volume_bias_in_stage1: bool = True,
        freeze_merger_in_stage3: bool = False,
    ) -> Dict[str, int]:
        """Configure exact parameter `requires_grad` & submodule `.eval()` flags for Stage 1, 2, or 3.

        - Stage 1 (Stochastic Pyramid Warmup): Train `visual.pyramid` (+ `treeweave_volume_bias` if enabled).
        - Stage 2 (Soft-Tree Rate-Distortion Routing): Train `visual.router` ONLY.
        - Stage 3 (Hard-Routed Alignment): Train `visual.pyramid`, `visual.merger` (unless frozen), `language_model`, `lm_head`.
        """
        self.requires_grad_(False)
        visual = self.model.visual

        if stage == 1:
            visual.pyramid.requires_grad_(True)
            enable_vol = bool(train_volume_bias_in_stage1 and self.tw_config.use_volume_attn_bias)
            for mod in self.model.language_model.modules():
                if hasattr(mod, "treeweave_volume_bias"):
                    mod.treeweave_volume_bias.configure_trainable(enable_vol)
        elif stage == 2:
            visual.router.requires_grad_(True)
        elif stage == 3:
            visual.pyramid.requires_grad_(True)
            if not freeze_merger_in_stage3:
                visual.merger.requires_grad_(True)
            self.model.language_model.requires_grad_(True)
            for mod in self.model.language_model.modules():
                if hasattr(mod, "treeweave_volume_bias"):
                    mod.treeweave_volume_bias.configure_trainable(self.tw_config.use_volume_attn_bias)
            self.lm_head.requires_grad_(True)
        else:
            raise ValueError(f"stage must be 1, 2, or 3, got {stage}.")

        visual.router.enforce_fp32_heads()
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"stage": stage, "trainable_params": trainable, "total_params": total}

    def enforce_frozen_eval_mode(self, stage: int) -> None:
        """Keep frozen sub-towers in `.eval()` mode while preserving LLM activation checkpointing in Stages 1--3."""
        visual = self.model.visual
        visual.patch_embed.eval()
        visual.pos_embed.eval()
        visual.blocks.eval()

        lm = self.model.language_model
        lm_gc = bool(getattr(lm, "gradient_checkpointing", False)) or any(
            getattr(layer, "gradient_checkpointing", False) for layer in lm.layers
        )
        if stage in (1, 2):
            visual.merger.eval()
            self.lm_head.eval()
            # In HF `transformers>=5.x`, `GradientCheckpointingLayer.__call__` checks
            # `self.gradient_checkpointing and self.training`. Since Qwen3.5 has zero
            # Dropout/BatchNorm layers (`attention_dropout == 0.0`), keeping `lm.train()`
            # when gradient checkpointing is active enables 32-layer activation checkpointing
            # on 24K-token Student forwards without introducing any stochasticity!
            if lm_gc:
                lm.train()
            else:
                lm.eval()
        if stage == 1:
            visual.router.eval()
            visual.pyramid.train()
        elif stage == 2:
            visual.pyramid.eval()
            visual.router.train()
        elif stage == 3:
            visual.router.eval()
            visual.pyramid.train()
            lm.train()

    def _ddp_zero_touch(self, stage: int) -> torch.Tensor:
        """Ensure heterogeneous image (`T=1`) vs video (`T>1`) microbatches across 8xH100 DDP/ZeRO ranks
        never leave temporal pooler/router parameters detached from the autograd graph.
        """
        acc = torch.zeros((), device=self.lm_head.weight.device, dtype=torch.float32)
        if stage in (1, 3):
            target_mods = [self.model.visual.pyramid]
            for mod in self.model.language_model.modules():
                if hasattr(mod, "treeweave_volume_bias"):
                    target_mods.append(mod.treeweave_volume_bias)
        elif stage == 2:
            target_mods = [self.model.visual.router]
        else:
            target_mods = []
        for mod in target_mods:
            for p in mod.parameters():
                if p.requires_grad:
                    acc = acc + p.reshape(-1)[0].float() * 0.0
        return acc

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        mm_encoder_outputs: Optional[dict[str, BaseModelOutputWithPooling]] = None,
        distill_stage: Optional[int] = None,
        **kwargs,
    ):
        """Standard `forward` that also routes `distill_stage in {1,2,3}` natively through DDP/DeepSpeed/FSDP."""
        if distill_stage is not None:
            assert input_ids is not None and labels is not None
            res = self.forward_distill(
                stage=int(distill_stage),
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                **kwargs,
            )
            return TreeWeaveCausalLMOutputWithPast(
                loss=res.loss,
                logits=None,
                distill_loss=res.distill_loss,
                rate_loss=res.rate_loss,
                balance_loss=res.balance_loss,
                treeweave_metrics=res.metrics,
            )
        for unused_k in ("teacher_topk_indices", "teacher_topk_logprobs", "teacher_topk_valid_mask"):
            kwargs.pop(unused_k, None)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            logits_to_keep=logits_to_keep,
            mm_encoder_outputs=mm_encoder_outputs,
            **kwargs,
        )

    def _supports_mm_encoder_outputs(self) -> bool:
        """Disable HF `GenerationMixin` pre-encoding so `pixel_values`/`pixel_values_videos` enter `TreeWeaveModel.forward`."""
        return False

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        """Synchronize `attention_mask`, `position_ids`, and `mm_token_type_ids` to the compacted KV-cache after prefill."""
        if getattr(self.model, "_just_routed_prefill", False):
            self.model._just_routed_prefill = False
            if self.model.routed_attention_mask is not None and "attention_mask" in model_kwargs:
                model_kwargs["attention_mask"] = self.model.routed_attention_mask
            if self.model.routed_position_ids_4d is not None:
                pos_sync = self.model.routed_position_ids_4d.clone()
                if self.model.rope_deltas is not None and pos_sync.shape[-1] > 0:
                    # Anchor final column's 3D MRoPE to `text_pos[-1] + rope_deltas` so `+ 1` yields `max_mrope + 1`
                    # even when a prompt ends directly on a compressed visual token without trailing text!
                    last_txt = pos_sync[0:1, :, -1:]
                    deltas = self.model.rope_deltas.to(device=pos_sync.device, dtype=pos_sync.dtype).unsqueeze(0)
                    pos_sync[1:, :, -1:] = last_txt + deltas
                model_kwargs["position_ids"] = pos_sync
                if "mm_token_type_ids" in model_kwargs and model_kwargs["mm_token_type_ids"] is not None:
                    bsz, r_len = pos_sync.shape[1], pos_sync.shape[2]
                    model_kwargs["mm_token_type_ids"] = model_kwargs["mm_token_type_ids"].new_zeros((bsz, r_len))
        return super()._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        """Ensure decode steps after compressed multimodal prefill step cleanly from `past_key_values` length."""
        cache = model_kwargs.get("past_key_values")
        past_length = cache.get_seq_length() if cache is not None else 0
        if past_length != 0 and self.model.rope_deltas is not None:
            bsz = inputs_tensor.shape[0]
            cur_len = inputs_tensor.shape[1]
            attn_mask = model_kwargs.get("attention_mask")
            if attn_mask is not None and attn_mask.shape[0] == bsz:
                pad_tokens = (attn_mask == 0).long().sum(dim=-1, keepdim=True)
                base_pos = (
                    torch.arange(past_length, past_length + cur_len, device=inputs_tensor.device)[None, :]
                    - pad_tokens
                ).clamp_min(0)
            else:
                base_pos = torch.arange(
                    past_length, past_length + cur_len, device=inputs_tensor.device
                )[None, :].expand(bsz, cur_len)

            vision_pos = base_pos.float().unsqueeze(0) + self.model.rope_deltas.to(
                device=inputs_tensor.device, dtype=torch.float32
            ).unsqueeze(0)
            vision_pos = vision_pos.expand(3, bsz, cur_len)
            return torch.cat((base_pos.float().unsqueeze(0), vision_pos), dim=0)

        return super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)

    def _prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None and past_key_values.get_seq_length() > 0 and attention_mask is not None:
            routed_mask = self.model.routed_attention_mask
            past_len = past_key_values.get_seq_length()
            if routed_mask is not None and routed_mask.shape[0] == attention_mask.shape[0]:
                # Exact post-compaction left-pad mask + all generated tokens since prefill!
                extra_decode = past_len + 1 - routed_mask.shape[1]
                if extra_decode > 0 and attention_mask.shape[1] != past_len + 1:
                    ones = routed_mask.new_ones((routed_mask.shape[0], extra_decode))
                    attention_mask = torch.cat((routed_mask.to(attention_mask.dtype), ones), dim=1)
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, **kwargs
        )
        return model_inputs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        return self._prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, **kwargs
        )

    def _run_native_teacher_hidden(
        self,
        *,
        input_ids: torch.Tensor,
        base_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        mm_token_type_ids: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
        img_native_raw: Optional[torch.Tensor],
        vid_native_raw: Optional[torch.Tensor],
        external_teacher: Optional[nn.Module],
        extra_lm_kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, nn.Module]:
        """Run Teacher LLM prefill under `torch.no_grad()` REUSING the already-computed ViT raw features!"""
        teacher_top = external_teacher if external_teacher is not None else self
        t_base = getattr(teacher_top, "model", teacher_top)
        t_merger = t_base.visual.merger
        t_lm = t_base.language_model
        t_head = teacher_top.lm_head

        was_lm_training = t_lm.training
        was_merger_training = t_merger.training
        if was_lm_training:
            t_lm.eval()
        if was_merger_training:
            t_merger.eval()

        try:
            with torch.no_grad(), volume_bias_context(None):
                t_embeds = (
                    t_base.get_input_embeddings()(input_ids)
                    if external_teacher is not None
                    else base_embeds.detach().clone()
                )
                if img_native_raw is not None:
                    img_feat = t_merger(img_native_raw.to(next(t_merger.parameters()).dtype)).to(
                        t_embeds.device, t_embeds.dtype
                    )
                    img_mask, _ = self.model.get_placeholder_mask(
                        input_ids, inputs_embeds=t_embeds, image_features=img_feat
                    )
                    t_embeds = t_embeds.masked_scatter(img_mask, img_feat)
                if vid_native_raw is not None:
                    vid_feat = t_merger(vid_native_raw.to(next(t_merger.parameters()).dtype)).to(
                        t_embeds.device, t_embeds.dtype
                    )
                    _, vid_mask = self.model.get_placeholder_mask(
                        input_ids, inputs_embeds=t_embeds, video_features=vid_feat
                    )
                    t_embeds = t_embeds.masked_scatter(vid_mask, vid_feat)

                t_pos = position_ids
                if t_pos is None:
                    t_pos = self.model.compute_3d_position_ids(
                        input_ids=input_ids,
                        inputs_embeds=t_embeds,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        attention_mask=attention_mask,
                        past_key_values=None,
                        mm_token_type_ids=mm_token_type_ids,
                    )
                t_out = _run_lm_varlen_safe(
                    t_lm,
                    position_ids=t_pos,
                    attention_mask=attention_mask,
                    inputs_embeds=t_embeds,
                    use_cache=False,
                    **extra_lm_kwargs,
                )
                return t_out.last_hidden_state, t_head
        finally:
            if was_lm_training:
                t_lm.train()
            if was_merger_training:
                t_merger.train()

    def forward_distill(
        self,
        *,
        stage: int,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        text_position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        external_teacher: Optional[nn.Module] = None,
        teacher_topk_indices: Optional[torch.Tensor] = None,
        teacher_topk_logprobs: Optional[torch.Tensor] = None,
        teacher_topk_valid_mask: Optional[torch.Tensor] = None,
        distill_topk: int = 512,
        distill_temperature: float = 1.0,
        distill_chunk_size: int = 1024,
        lambda_rate: float = 0.05,
        lambda_bal: float = 0.01,
        rho_s1: float = 0.75,
        rho_s2: float = 0.50,
        rho_t1: float = 0.65,
        rho_t2: float = 0.50,
        stochastic_rng: Optional[np.random.Generator] = None,
        stage1_uniform_scale: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> DistillForwardOutput:
        """Execute Stage 1, Stage 2, or Stage 3 Self-Distillation with a SINGLE shared ViT forward pass!"""
        has_mm = pixel_values is not None or pixel_values_videos is not None
        is_packed = _is_truly_packed_batch(attention_mask, kwargs.get("cu_seq_lens_q"))
        kwargs = _sanitize_varlen_kwargs(kwargs, is_packed=is_packed, attention_mask=attention_mask)
        if text_position_ids is None and position_ids is not None and position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0].long()
        elif position_ids is not None and position_ids.ndim == 3 and position_ids.shape[0] == 3 and text_position_ids is not None:
            position_ids = torch.cat((text_position_ids.to(position_ids.dtype).unsqueeze(0), position_ids), dim=0)

        base_embeds = self.model.get_input_embeddings()(input_ids)
        need_live_teacher = teacher_topk_indices is None or teacher_topk_logprobs is None

        # Native active positions (identical for Teacher & Stage 2 Student!)
        native_active = extract_active_positions(
            labels,
            text_position_ids=text_position_ids if is_packed else None,
        )
        metrics: Dict[str, float] = {"active_response_tokens": float(native_active.count)}

        # Pure-text batch fallback (e.g., if a text-only sample appears in a multimodal mixture)
        if not has_mm:
            with volume_bias_context(None):
                s_out = _run_lm_varlen_safe(
                    self.model.language_model,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=base_embeds,
                    use_cache=False,
                    **kwargs,
                )
            s_act = s_out.last_hidden_state[native_active.batch_indices, native_active.seq_indices]
            t_act = None
            t_head = None
            if need_live_teacher:
                t_hid, t_head = self._run_native_teacher_hidden(
                    input_ids=input_ids,
                    base_embeds=base_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=None,
                    video_grid_thw=None,
                    img_native_raw=None,
                    vid_native_raw=None,
                    external_teacher=external_teacher,
                    extra_lm_kwargs=kwargs,
                )
                t_act = t_hid[native_active.batch_indices, native_active.seq_indices]

            reduction = "sample_sqrt_len" if stage == 1 else "token_mean"
            d_loss = chunked_topk_kl_distill_loss(
                s_act,
                self.lm_head,
                sample_ids=native_active.sample_ids,
                teacher_active_hidden=t_act,
                teacher_lm_head=t_head,
                teacher_topk_indices=teacher_topk_indices,
                teacher_topk_logprobs=teacher_topk_logprobs,
                teacher_topk_valid_mask=teacher_topk_valid_mask,
                topk=distill_topk,
                temperature=distill_temperature,
                reduction=reduction,
                chunk_size=distill_chunk_size,
            )
            zero = d_loss.new_zeros(())
            return DistillForwardOutput(
                loss=d_loss + self._ddp_zero_touch(stage),
                distill_loss=d_loss.detach(),
                rate_loss=zero,
                balance_loss=zero,
                metrics=metrics,
            )

        if mm_token_type_ids is None:
            raise ValueError("Multimodal TreeWeave distillation requires `mm_token_type_ids`.")

        visual = self.model.visual

        # =====================================================================
        # STAGE 2: Differentiable Soft-Tree Mixing (100% Native Sequence Length)
        # =====================================================================
        if stage == 2:
            img_soft: Optional[SoftTreeResult] = None
            vid_soft: Optional[SoftTreeResult] = None
            img_raw = vid_raw = None

            if pixel_values is not None:
                assert image_grid_thw is not None
                img_soft, img_raw = visual.encode_soft_routed(
                    pixel_values,
                    image_grid_thw,
                    rho_s1=rho_s1,
                    rho_s2=rho_s2,
                    rho_t1=rho_t1,
                    rho_t2=rho_t2,
                    vit_no_grad=True,
                    pooler_no_grad=True,
                )
            if pixel_values_videos is not None:
                assert video_grid_thw is not None
                vid_soft, vid_raw = visual.encode_soft_routed(
                    pixel_values_videos,
                    video_grid_thw,
                    rho_s1=rho_s1,
                    rho_s2=rho_s2,
                    rho_t1=rho_t1,
                    rho_t2=rho_t2,
                    vit_no_grad=True,
                    pooler_no_grad=True,
                )

            t_act = None
            t_head = None
            if need_live_teacher:
                t_hid, t_head = self._run_native_teacher_hidden(
                    input_ids=input_ids,
                    base_embeds=base_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    img_native_raw=img_raw,
                    vid_native_raw=vid_raw,
                    external_teacher=external_teacher,
                    extra_lm_kwargs=kwargs,
                )
                t_act = t_hid[native_active.batch_indices, native_active.seq_indices]

            s_embeds = base_embeds.clone()
            rate_parts: List[torch.Tensor] = []
            bal_parts: List[torch.Tensor] = []

            if img_soft is not None:
                img_cat = torch.cat(img_soft.mixed_tokens, dim=0).to(s_embeds.device, s_embeds.dtype)
                img_mask, _ = self.model.get_placeholder_mask(
                    input_ids, inputs_embeds=s_embeds, image_features=img_cat
                )
                s_embeds = s_embeds.masked_scatter(img_mask, img_cat)
                rate_parts.append(img_soft.expected_retention)
                bal_parts.append(img_soft.balance_loss)
                for k, v in img_soft.metrics.items():
                    metrics[f"img_{k}"] = v

            if vid_soft is not None:
                vid_cat = torch.cat(vid_soft.mixed_tokens, dim=0).to(s_embeds.device, s_embeds.dtype)
                _, vid_mask = self.model.get_placeholder_mask(
                    input_ids, inputs_embeds=s_embeds, video_features=vid_cat
                )
                s_embeds = s_embeds.masked_scatter(vid_mask, vid_cat)
                rate_parts.append(vid_soft.expected_retention)
                bal_parts.append(vid_soft.balance_loss)
                for k, v in vid_soft.metrics.items():
                    metrics[f"vid_{k}"] = v

            s_pos = position_ids
            if s_pos is None:
                s_pos = self.model.compute_3d_position_ids(
                    input_ids=input_ids,
                    inputs_embeds=s_embeds,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    mm_token_type_ids=mm_token_type_ids,
                )

            with volume_bias_context(None):
                s_out = _run_lm_varlen_safe(
                    self.model.language_model,
                    position_ids=s_pos,
                    attention_mask=attention_mask,
                    inputs_embeds=s_embeds,
                    use_cache=False,
                    **kwargs,
                )
            s_act = s_out.last_hidden_state[native_active.batch_indices, native_active.seq_indices]

            d_loss = chunked_topk_kl_distill_loss(
                s_act,
                self.lm_head,
                sample_ids=native_active.sample_ids,
                teacher_active_hidden=t_act,
                teacher_lm_head=t_head,
                teacher_topk_indices=teacher_topk_indices,
                teacher_topk_logprobs=teacher_topk_logprobs,
                teacher_topk_valid_mask=teacher_topk_valid_mask,
                topk=distill_topk,
                temperature=distill_temperature,
                reduction="token_mean",
                chunk_size=distill_chunk_size,
            )
            r_val = torch.stack(rate_parts).mean()
            b_val = torch.stack(bal_parts).mean()
            total_loss = d_loss + float(lambda_rate) * r_val + float(lambda_bal) * b_val + self._ddp_zero_touch(stage)

            metrics["expected_retention"] = float(r_val.detach().item())
            metrics["expected_savings"] = float(1.0 - r_val.detach().item())
            # Also expose aggregated modality-agnostic router diagnostics alongside `img_*` / `vid_*`
            soft_sources = [s.metrics for s in (img_soft, vid_soft) if s is not None]
            for agg_key in ("router_entropy", "p_s1_mean", "p_s2_mean", "p_t1_mean", "p_t2_mean"):
                vals = [src[agg_key] for src in soft_sources if agg_key in src]
                if vals:
                    metrics[agg_key] = float(sum(vals) / len(vals))
            return DistillForwardOutput(
                loss=total_loss,
                distill_loss=d_loss.detach(),
                rate_loss=r_val.detach(),
                balance_loss=b_val.detach(),
                metrics=metrics,
            )

        # =====================================================================
        # STAGE 1 (Stochastic Pyramid) & STAGE 3 (Learned Hard Routing)
        # =====================================================================
        r_mode = "stochastic" if stage == 1 else "learned"
        img_layouts: List[TreeRouteLayout] = []
        img_routed: List[torch.Tensor] = []
        vid_layouts: List[TreeRouteLayout] = []
        vid_routed: List[torch.Tensor] = []
        img_raw = vid_raw = None

        if pixel_values is not None:
            assert image_grid_thw is not None
            img_layouts, img_routed, img_raw = visual.encode_hard_routed(
                pixel_values,
                image_grid_thw,
                routing_mode=r_mode,
                stochastic_rng=stochastic_rng,
                uniform_single_scale=stage1_uniform_scale,
                vit_no_grad=True,
                router_no_grad=True,
                return_native_states=need_live_teacher,
            )
        if pixel_values_videos is not None:
            assert video_grid_thw is not None
            vid_layouts, vid_routed, vid_raw = visual.encode_hard_routed(
                pixel_values_videos,
                video_grid_thw,
                routing_mode=r_mode,
                stochastic_rng=stochastic_rng,
                uniform_single_scale=stage1_uniform_scale,
                vit_no_grad=True,
                router_no_grad=True,
                return_native_states=need_live_teacher,
            )

        t_act = None
        t_head = None
        if need_live_teacher:
            t_hid, t_head = self._run_native_teacher_hidden(
                input_ids=input_ids,
                base_embeds=base_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                img_native_raw=img_raw,
                vid_native_raw=vid_raw,
                external_teacher=external_teacher,
                extra_lm_kwargs=kwargs,
            )
            t_act = t_hid[native_active.batch_indices, native_active.seq_indices]

        pad_id = getattr(self.config.text_config, "pad_token_id", None) or 0
        pad_emb = self.model.get_input_embeddings().weight[pad_id]

        packed = build_routed_sequence_batch(
            input_ids=input_ids,
            base_inputs_embeds=base_embeds,
            mm_token_type_ids=mm_token_type_ids,
            image_layouts=img_layouts,
            image_routed_embeds=img_routed,
            video_layouts=vid_layouts,
            video_routed_embeds=vid_routed,
            labels=labels,
            attention_mask=attention_mask,
            text_position_ids=text_position_ids,
            is_packed=is_packed,
            pad_embedding=pad_emb,
        )
        assert packed.labels is not None
        student_active = extract_active_positions(
            packed.labels,
            text_position_ids=packed.text_position_ids if is_packed else None,
        )
        if student_active.count != native_active.count:
            raise RuntimeError(
                f"Active response token mismatch after visual compaction: student={student_active.count}, native={native_active.count}."
            )

        student_lm_kwargs = dict(kwargs)
        if is_packed and packed.cu_seq_lens_q is not None:
            student_lm_kwargs["cu_seq_lens_q"] = packed.cu_seq_lens_q
            student_lm_kwargs["cu_seq_lens_k"] = packed.cu_seq_lens_k
            student_lm_kwargs["max_length_q"] = packed.max_length_q
            student_lm_kwargs["max_length_k"] = packed.max_length_k

        eff_lv = packed.token_log_volumes if self.tw_config.use_volume_attn_bias else None
        s_out = _run_lm_varlen_safe(
            self.model.language_model,
            position_ids=packed.position_ids_4d,
            attention_mask=packed.attention_mask,
            inputs_embeds=packed.inputs_embeds,
            use_cache=False,
            token_log_volumes=eff_lv,
            **student_lm_kwargs,
        )

        s_act = s_out.last_hidden_state[student_active.batch_indices, student_active.seq_indices]
        reduction = "sample_sqrt_len" if stage == 1 else "token_mean"

        d_loss = chunked_topk_kl_distill_loss(
            s_act,
            self.lm_head,
            sample_ids=student_active.sample_ids,
            teacher_active_hidden=t_act,
            teacher_lm_head=t_head,
            teacher_topk_indices=teacher_topk_indices,
            teacher_topk_logprobs=teacher_topk_logprobs,
            teacher_topk_valid_mask=teacher_topk_valid_mask,
            topk=distill_topk,
            temperature=distill_temperature,
            reduction=reduction,
            chunk_size=distill_chunk_size,
        )

        # Record actual visual token savings & 9-scale volume shares
        all_layouts = list(img_layouts) + list(vid_layouts)
        total_nat = sum(l.native_tokens for l in all_layouts)
        total_out = sum(l.output_length for l in all_layouts)
        if total_nat > 0:
            metrics["actual_token_savings"] = 1.0 - (total_out / float(total_nat))
            shares = sum(
                (l.granularity_covered_shares() * l.native_tokens for l in all_layouts),
                start=np.zeros(NUM_GRANULARITIES, dtype=np.float32),
            ) / float(total_nat)
            for s in (0, 1, 2):
                for tm in (0, 1, 2):
                    metrics[f"share_s{s}_t{tm}"] = float(shares[3 * s + tm])

        zero = d_loss.new_zeros(())
        return DistillForwardOutput(
            loss=d_loss + self._ddp_zero_touch(stage),
            distill_loss=d_loss.detach(),
            rate_loss=zero,
            balance_loss=zero,
            metrics=metrics,
        )
