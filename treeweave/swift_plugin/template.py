"""ms-swift Template & Image/Video 128px Grid-Alignment Patches for Qwen3.5-TreeWeave."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from safetensors.torch import load_file
from swift.template import register_template
from swift.template.templates.qwen import Qwen3_5Template, QwenTemplateMeta
from swift.utils import get_logger

logger = get_logger()

_PATCH_ATTR = "_treeweave_patched_factor"
_CURRENT_PIXEL_FACTOR: int = int(os.environ.get("TREEWEAVE_PIXEL_FACTOR", "128"))


def get_treeweave_pixel_factor() -> int:
    return _CURRENT_PIXEL_FACTOR


def install_treeweave_resize_patch(pixel_factor: int = 128) -> None:
    """Align `qwen_vl_utils` and `transformers` `smart_resize` to `pixel_factor` (`128` for s<=2, `64` for s<=1).

    Idempotent and safe across DataLoader / Packing worker processes (`fork` and `spawn`).
    """
    global _CURRENT_PIXEL_FACTOR
    if pixel_factor % 64 != 0 or pixel_factor <= 0:
        raise ValueError(f"TreeWeave pixel_factor must be a positive multiple of 64, got {pixel_factor}.")
    _CURRENT_PIXEL_FACTOR = int(pixel_factor)
    os.environ["TREEWEAVE_PIXEL_FACTOR"] = str(_CURRENT_PIXEL_FACTOR)
    bootstrap_dir = str((Path(__file__).resolve().parents[1] / "rollout" / "_bootstrap").resolve())
    repo_root = str(Path(__file__).resolve().parents[2].resolve())
    existing_py = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    merged_py: List[str] = []
    for cand_py in [bootstrap_dir, repo_root] + existing_py:
        if cand_py not in merged_py:
            merged_py.append(cand_py)
    os.environ["PYTHONPATH"] = os.pathsep.join(merged_py)

    def _wrap_smart_resize(orig_fn):
        if getattr(orig_fn, _PATCH_ATTR, False):
            return orig_fn

        def wrapped_smart_resize(height: int, width: int, factor: int = 32, min_pixels=None, max_pixels=None, *args, **kwargs):
            eff_factor = max(int(factor), _CURRENT_PIXEL_FACTOR)
            # Round eff_factor up to a multiple of _CURRENT_PIXEL_FACTOR if needed
            if eff_factor % _CURRENT_PIXEL_FACTOR != 0:
                eff_factor = ((eff_factor + _CURRENT_PIXEL_FACTOR - 1) // _CURRENT_PIXEL_FACTOR) * _CURRENT_PIXEL_FACTOR
            min_floor = eff_factor * eff_factor
            if min_pixels is not None:
                min_pixels = max(int(min_pixels), min_floor)
            else:
                min_pixels = min_floor
            if max_pixels is not None:
                max_pixels = max(int(max_pixels), min_pixels)
            h_bar, w_bar = orig_fn(
                height,
                width,
                factor=eff_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                *args,
                **kwargs,
            )
            return max(eff_factor, int(h_bar)), max(eff_factor, int(w_bar))

        setattr(wrapped_smart_resize, _PATCH_ATTR, True)
        return wrapped_smart_resize

    # 1) Patch qwen_vl_utils.vision_process.smart_resize (used by ms-swift replace_tag fetch_image/fetch_video)
    try:
        import qwen_vl_utils.vision_process as vp

        vp.smart_resize = _wrap_smart_resize(vp.smart_resize)
    except ImportError:
        pass

    # 2) Patch transformers qwen2_vl / qwen3_vl image & video processors
    for mod_path in (
        "transformers.models.qwen2_vl.image_processing_qwen2_vl",
        "transformers.models.qwen2_vl.image_processing_qwen2_vl_fast",
        "transformers.models.qwen2_vl.image_processing_pil_qwen2_vl",
        "transformers.models.qwen2_vl.video_processing_qwen2_vl",
        "transformers.models.qwen3_vl.video_processing_qwen3_vl",
    ):
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        fn = getattr(mod, "smart_resize", None)
        if fn is not None:
            setattr(mod, "smart_resize", _wrap_smart_resize(fn))


class Qwen3_5TreeWeaveTemplate(Qwen3_5Template):
    """Template for Qwen3.5-TreeWeave preserving raw multimodal tensors for dynamic/soft routing & packing.

    Crucial difference from `Qwen3_5Template`:
    - `Qwen3_5Template._post_encode` calls `Qwen2VLTemplate._post_encode`, which pre-scatters native visual
      embeddings into `inputs_embeds` and drops `input_ids` / `pixel_values` / `image_grid_thw` BEFORE `model.forward`!
    - `Qwen3_5TreeWeaveTemplate._post_encode` returns `inputs` intact (like `Qwen3VLTemplate`), allowing
      `Qwen3_5TreeWeaveForConditionalGeneration` to perform single-ViT teacher+student routing, sequence
      compaction, half-integer 3D MRoPE construction, and volume-preserving attention compensation inside `forward`.
    - Also forwards optional offline cached `topk_path` entries (`teacher_topk_indices`, `teacher_topk_logprobs`)
      seamlessly through both standard collation and `PackingDataset`!
    """

    support_padding_free = True

    def init_env_args(self):
        super().init_env_args()
        from .trainer import get_active_distill_config

        self.pixel_factor: int = int(os.environ.get("TREEWEAVE_PIXEL_FACTOR", str(_CURRENT_PIXEL_FACTOR)))
        self.topk_cache_dir: Optional[str] = (
            os.environ.get("TREEWEAVE_TOPK_CACHE_DIR") or get_active_distill_config().topk_cache_dir
        )
        install_treeweave_resize_patch(self.pixel_factor)
        tok = getattr(self, "tokenizer", None)
        if tok is not None and hasattr(tok, "convert_tokens_to_ids"):
            unk_id = getattr(tok, "unk_token_id", None)
            im_id = tok.convert_tokens_to_ids("<|image_pad|>")
            vid_id = tok.convert_tokens_to_ids("<|video_pad|>")
            if isinstance(im_id, int) and im_id >= 0 and im_id != unk_id:
                self.image_token_id = im_id
            if isinstance(vid_id, int) and vid_id >= 0 and vid_id != unk_id:
                self.video_token_id = vid_id

    @staticmethod
    def _extract_active_supervised_targets(labels: Any) -> Optional[List[int]]:
        """Extract post-truncation causal supervised target token IDs `labels[1:][labels[1:] != -100]`."""
        if labels is None:
            return None
        lbl = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
        if lbl.numel() < 2:
            return []
        nxt = lbl[1:]
        return nxt[nxt != -100].tolist()

    @classmethod
    def _count_active_supervised_tokens(cls, labels: Any) -> Optional[int]:
        """Count post-truncation causal supervised targets `(labels[1:] != -100).sum()`."""
        targets = cls._extract_active_supervised_targets(labels)
        return None if targets is None else len(targets)

    def encode(self, inputs, return_template_inputs: bool = False, return_length: bool = False) -> Dict[str, Any]:
        install_treeweave_resize_patch(getattr(self, "pixel_factor", _CURRENT_PIXEL_FACTOR))
        resolved_topk_path: Optional[str] = None
        if isinstance(inputs, dict):
            if inputs.get("topk_cache_path"):
                resolved_topk_path = str(inputs["topk_cache_path"])
            else:
                from .trainer import get_active_distill_config

                cache_dir = (
                    getattr(self, "topk_cache_dir", None)
                    or os.environ.get("TREEWEAVE_TOPK_CACHE_DIR")
                    or get_active_distill_config().topk_cache_dir
                )
                if cache_dir:
                    from ..rollout.offline_rollout import compute_sample_uuid, compute_topk_cache_uuid

                    candidates: List[str] = []
                    if inputs.get("_topk_cache_uuid"):
                        candidates.append(str(inputs["_topk_cache_uuid"]))
                    candidates.append(compute_topk_cache_uuid(inputs))
                    if inputs.get("_rollout_uuid"):
                        candidates.append(str(inputs["_rollout_uuid"]))
                    candidates.append(compute_sample_uuid(inputs))
                    for uid in dict.fromkeys(candidates):
                        cand = Path(cache_dir) / f"{uid}.safetensors"
                        if cand.is_file():
                            resolved_topk_path = str(cand)
                            break

        encoded = super().encode(
            inputs, return_template_inputs=return_template_inputs, return_length=return_length
        )
        if resolved_topk_path and isinstance(encoded, dict) and "input_ids" in encoded:
            encoded["topk_cache_path"] = resolved_topk_path
            active_targets = self._extract_active_supervised_targets(encoded.get("labels"))
            if active_targets is not None:
                encoded["topk_active_tokens"] = len(active_targets)
                encoded["topk_active_targets"] = active_targets
                encoded["topk_truncation_side"] = str(
                    getattr(self, "truncation_strategy", "right") or "right"
                )
        return encoded

    def _encode(self, inputs) -> Dict[str, Any]:
        install_treeweave_resize_patch(getattr(self, "pixel_factor", _CURRENT_PIXEL_FACTOR))
        return super()._encode(inputs)

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.padding_free and self.sequence_parallel_size <= 1 and not self.transformers_5_9:
            raise RuntimeError(
                f"Qwen3.5-TreeWeave packing requires transformers>=5.9.0 (found {self.transformers_version})."
            )
        return inputs

    @staticmethod
    def _find_contiguous_subarray(haystack: torch.Tensor, needle: torch.Tensor) -> Optional[int]:
        """Return first offset `s` where `haystack[s : s + len(needle)] == needle`, or `None`."""
        n_h = int(haystack.shape[0])
        n_n = int(needle.shape[0])
        if n_n == 0:
            return 0
        if n_n > n_h:
            return None
        for s in range(n_h - n_n + 1):
            if bool(torch.equal(haystack[s : s + n_n], needle)):
                return s
        return None

    @classmethod
    def _load_offline_topk_list(
        cls,
        paths: List[str],
        expected_counts: Optional[List[Optional[int]]] = None,
        truncation_sides: Optional[List[str]] = None,
        expected_targets: Optional[List[Optional[List[int]]]] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not paths:
            return None
        counts = expected_counts if expected_counts is not None else [None] * len(paths)
        sides = truncation_sides if truncation_sides is not None else ["right"] * len(paths)
        targets_list = expected_targets if expected_targets is not None else [None] * len(paths)
        if len(counts) != len(paths) or len(sides) != len(paths) or len(targets_list) != len(paths):
            raise ValueError("Mismatched metadata lengths in `_load_offline_topk_list`.")

        idx_parts: List[torch.Tensor] = []
        lp_parts: List[torch.Tensor] = []
        tgt_parts: List[torch.Tensor] = []
        all_have_targets = True
        k_width: Optional[int] = None
        fallback_k: int = 1

        for p, exp_cnt, side, exp_tgt in zip(paths, counts, sides, targets_list):
            if not p or not Path(p).is_file():
                return None
            tensors = load_file(p, device="cpu")
            idx = tensors["topk_indices"]
            lp = tensors["topk_logprobs"]
            tgt = tensors.get("target_labels")
            if idx.ndim != 2 or lp.ndim != 2 or idx.shape != lp.shape:
                raise ValueError(
                    f"Invalid offline Top-K tensor shapes in {p!r}: "
                    f"indices={tuple(idx.shape)}, logprobs={tuple(lp.shape)}."
                )
            if tgt is not None and (tgt.ndim != 1 or int(tgt.shape[0]) != int(idx.shape[0])):
                raise ValueError(
                    f"Invalid offline Top-K `target_labels` shape in {p!r}: "
                    f"target_labels={tuple(tgt.shape)} vs indices={tuple(idx.shape)}."
                )
            if idx.shape[1] > 0:
                fallback_k = int(idx.shape[1])

            if exp_cnt is not None:
                if exp_cnt < 0 or exp_cnt > idx.shape[0]:
                    raise ValueError(
                        f"Offline Top-K cache {p!r} has {idx.shape[0]} active tokens, fewer than "
                        f"post-truncation sample active count {exp_cnt}."
                    )
                if exp_cnt == 0:
                    idx = idx[:0]
                    lp = lp[:0]
                    if tgt is not None:
                        tgt = tgt[:0]
                elif tgt is not None and exp_tgt is not None and len(exp_tgt) == exp_cnt:
                    tgt_long = tgt.to(torch.long)
                    needle = torch.tensor(exp_tgt, dtype=torch.long)
                    if side == "left":
                        s_off = int(idx.shape[0]) - exp_cnt
                        if not bool(torch.equal(tgt_long[s_off:], needle)):
                            s_off_alt = cls._find_contiguous_subarray(tgt_long, needle)
                            if s_off_alt is None:
                                raise ValueError(
                                    f"Offline Top-K `target_labels` in {p!r} do not match Student supervised `labels`."
                                )
                            s_off = s_off_alt
                    else:
                        # Handles both `loss_scale='default'` (s_off=0) and `default+ignore_empty_think`
                        # (s_off = empty `<think>\n\n</think>\n\n` prefix length) + right-truncation!
                        s_off_found = cls._find_contiguous_subarray(tgt_long, needle)
                        if s_off_found is None:
                            raise ValueError(
                                f"Offline Top-K `target_labels` in {p!r} do not match Student supervised `labels`."
                            )
                        s_off = s_off_found
                    idx = idx[s_off : s_off + exp_cnt]
                    lp = lp[s_off : s_off + exp_cnt]
                    tgt = tgt[s_off : s_off + exp_cnt]
                elif exp_cnt < idx.shape[0]:
                    if side == "left":
                        idx = idx[-exp_cnt:]
                        lp = lp[-exp_cnt:]
                        if tgt is not None:
                            tgt = tgt[-exp_cnt:]
                    else:
                        idx = idx[:exp_cnt]
                        lp = lp[:exp_cnt]
                        if tgt is not None:
                            tgt = tgt[:exp_cnt]

            if idx.shape[0] > 0:
                if k_width is None:
                    k_width = int(idx.shape[1])
                elif int(idx.shape[1]) != k_width:
                    raise ValueError(
                        f"Inconsistent Top-K width across cached shards: expected K={k_width}, got K={idx.shape[1]} in {p!r}."
                    )
            idx_parts.append(idx)
            lp_parts.append(lp)
            if tgt is not None:
                tgt_parts.append(tgt.to(torch.long))
            else:
                all_have_targets = False

        eff_k = k_width if k_width is not None else fallback_k
        norm_idx = [
            x if x.shape[0] > 0 or x.shape[1] == eff_k else torch.zeros((0, eff_k), dtype=x.dtype)
            for x in idx_parts
        ]
        norm_lp = [
            x if x.shape[0] > 0 or x.shape[1] == eff_k else torch.zeros((0, eff_k), dtype=x.dtype)
            for x in lp_parts
        ]
        out: Dict[str, torch.Tensor] = {
            "teacher_topk_indices": torch.cat(norm_idx, dim=0),
            "teacher_topk_logprobs": torch.cat(norm_lp, dim=0),
        }
        if all_have_targets and len(tgt_parts) == len(paths):
            out["_teacher_target_labels"] = torch.cat(tgt_parts, dim=0)
        return out

    def packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
        default_side = str(getattr(self, "truncation_strategy", "right") or "right")
        clean_row: List[Dict[str, Any]] = []
        topk_paths: List[Optional[str]] = []
        topk_counts: List[Optional[int]] = []
        topk_sides: List[str] = []
        topk_targets: List[Optional[List[int]]] = []
        for r_orig in row:
            r = dict(r_orig)
            if "input_ids" in r:
                if "length" not in r:
                    r["length"] = len(r["input_ids"])
                if self.requires_mm_token_type_ids and "mm_token_type_ids" not in r:
                    r["mm_token_type_ids"] = torch.zeros(len(r["input_ids"]), dtype=torch.int64)
            p = r.pop("topk_cache_path", None)
            cnt = r.pop("topk_active_tokens", None)
            tgt = r.pop("topk_active_targets", None)
            side = str(r.pop("topk_truncation_side", None) or default_side)
            if (cnt is None or tgt is None) and "labels" in r:
                extracted = self._extract_active_supervised_targets(r.get("labels"))
                if extracted is not None:
                    if cnt is None:
                        cnt = len(extracted)
                    if tgt is None and len(extracted) == cnt:
                        tgt = extracted
            topk_paths.append(p)
            topk_counts.append(cnt)
            topk_sides.append(side)
            topk_targets.append(tgt)
            clean_row.append(r)
        packed = super().packing_row(clean_row)
        if topk_paths and all(p is not None for p in topk_paths):
            packed["_topk_cache_paths"] = [str(p) for p in topk_paths]
            packed["_topk_active_tokens"] = topk_counts
            packed["_topk_truncation_sides"] = topk_sides
            packed["_topk_active_targets"] = topk_targets
        return packed

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        default_side = str(getattr(self, "truncation_strategy", "right") or "right")
        clean_batch: List[Dict[str, Any]] = []
        collected_paths: List[str] = []
        collected_counts: List[Optional[int]] = []
        collected_sides: List[str] = []
        collected_targets: List[Optional[List[int]]] = []
        all_have_topk = len(batch) > 0
        for item_orig in batch:
            item = dict(item_orig)
            if "_topk_cache_paths" in item:
                sub_paths = item.pop("_topk_cache_paths")
                sub_counts = item.pop("_topk_active_tokens", [None] * len(sub_paths))
                sub_sides = item.pop("_topk_truncation_sides", [default_side] * len(sub_paths))
                sub_targets = item.pop("_topk_active_targets", [None] * len(sub_paths))
                collected_paths.extend(sub_paths)
                collected_counts.extend(sub_counts)
                collected_sides.extend(sub_sides)
                collected_targets.extend(sub_targets)
            elif "topk_cache_path" in item:
                p = item.pop("topk_cache_path")
                cnt = item.pop("topk_active_tokens", None)
                tgt = item.pop("topk_active_targets", None)
                side = str(item.pop("topk_truncation_side", None) or default_side)
                if (cnt is None or tgt is None) and "labels" in item:
                    extracted = self._extract_active_supervised_targets(item.get("labels"))
                    if extracted is not None:
                        if cnt is None:
                            cnt = len(extracted)
                        if tgt is None and len(extracted) == cnt:
                            tgt = extracted
                collected_paths.append(p)
                collected_counts.append(cnt)
                collected_sides.append(side)
                collected_targets.append(tgt)
            else:
                item.pop("_topk_active_tokens", None)
                item.pop("_topk_truncation_sides", None)
                item.pop("_topk_active_targets", None)
                item.pop("topk_active_tokens", None)
                item.pop("topk_truncation_side", None)
                item.pop("topk_active_targets", None)
                all_have_topk = False
            clean_batch.append(item)

        res = super()._data_collator(clean_batch, padding_to=padding_to)
        if not self.padding_free and "attention_mask" in res and res["attention_mask"] is not None:
            if not bool(res["attention_mask"].all()):
                for k in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"):
                    res.pop(k, None)
        if all_have_topk and collected_paths:
            loaded = self._load_offline_topk_list(
                collected_paths, collected_counts, collected_sides, collected_targets
            )
            if loaded is not None:
                cached_targets = loaded.pop("_teacher_target_labels", None)
                if isinstance(res.get("labels"), torch.Tensor):
                    from ..losses import extract_active_positions

                    tpos = res.get("text_position_ids")
                    pos_ids = res.get("position_ids")
                    if (
                        tpos is None
                        and isinstance(pos_ids, torch.Tensor)
                        and pos_ids.ndim == 3
                        and pos_ids.shape[0] == 4
                    ):
                        tpos = pos_ids[0].long()
                    active_pos = extract_active_positions(
                        res["labels"],
                        text_position_ids=tpos if self.padding_free else None,
                    )
                    collated_active = active_pos.count
                    cached_active = int(loaded["teacher_topk_indices"].shape[0])
                    if cached_active != collated_active:
                        raise ValueError(
                            f"Collated batch has {collated_active} active supervised tokens, "
                            f"but sliced offline Top-K cache produced {cached_active}."
                        )
                    if cached_targets is not None and not torch.equal(
                        cached_targets.long(), active_pos.target_labels.detach().cpu().long()
                    ):
                        raise ValueError(
                            "Offline Top-K `target_labels` do not match Student supervised `labels` "
                            "after truncation/packing; Teacher Top-K cache is misaligned with this sample."
                        )
                res.update(loaded)
        return res


QWEN3_5_TREEWEAVE_TEMPLATE_TYPE = "qwen3_5_treeweave"


def register_treeweave_template() -> None:
    from swift.template import TEMPLATE_MAPPING

    if QWEN3_5_TREEWEAVE_TEMPLATE_TYPE in TEMPLATE_MAPPING:
        return
    register_template(
        QwenTemplateMeta(
            QWEN3_5_TREEWEAVE_TEMPLATE_TYPE,
            template_cls=Qwen3_5TreeWeaveTemplate,
            default_system=None,
            thinking_prefix="<think>\n",
            non_thinking_prefix="<think>\n\n</think>\n\n",
            agent_template="qwen3_5",
            is_thinking=True,
        )
    )


register_treeweave_template()
