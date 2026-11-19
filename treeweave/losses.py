"""Memory-Efficient Active-Token Chunked Top-K Forward KL Self-Distillation Loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "ActivePositions",
    "chunked_topk_kl_distill_loss",
    "compute_teacher_topk_from_hidden",
    "extract_active_positions",
]


@dataclass
class ActivePositions:
    """1-to-1 aligned supervised prediction positions inside a padded or packed batch."""

    batch_indices: torch.Tensor     # `(N_active,)` long
    seq_indices: torch.Tensor       # `(N_active,)` long (`t` predicting `labels[:, t + 1]`)
    sample_ids: torch.Tensor        # `(N_active,)` long (`0..M-1` across batch or packed sub-samples)
    target_labels: torch.Tensor     # `(N_active,)` long (`labels[b, t + 1] != -100`)

    @property
    def count(self) -> int:
        return int(self.batch_indices.shape[0])


def extract_active_positions(
    labels: torch.Tensor,
    text_position_ids: Optional[torch.Tensor] = None,
) -> ActivePositions:
    """Extract causal prediction indices `t` where `labels[:, t + 1] != -100`.

    In packed mode (`text_position_ids` resets to `0` at every sub-sample start), we also enforce
    `text_position_ids[:, t + 1] != 0` so the last token of sample `m` never predicts token 0 of sample `m + 1`,
    and assign exact per-sub-sample IDs `sample_ids` for Stage 1 `sqrt(|y_m|)` loss weighting.
    """
    if labels.ndim != 2 or labels.shape[1] < 2:
        empty = torch.zeros((0,), dtype=torch.long, device=labels.device)
        return ActivePositions(empty, empty, empty, empty)

    next_labels = labels[:, 1:]
    valid = next_labels != -100

    if text_position_ids is not None and text_position_ids.shape == labels.shape:
        not_new_sample = text_position_ids[:, 1:] != 0
        valid = valid & not_new_sample
        # Global sample ID across all rows and packed segments
        is_start = (text_position_ids == 0).long()
        flat_sample_ids = (is_start.reshape(-1).cumsum(dim=0) - 1).clamp_min(0).view_as(labels)
        next_sample_ids = flat_sample_ids[:, 1:]
    else:
        bsz, seq_len_m1 = next_labels.shape
        next_sample_ids = torch.arange(bsz, device=labels.device, dtype=torch.long)[:, None].expand(
            bsz, seq_len_m1
        )

    b_idx, s_idx = torch.nonzero(valid, as_tuple=True)
    samp_ids = next_sample_ids[b_idx, s_idx]
    targets = next_labels[b_idx, s_idx]
    return ActivePositions(
        batch_indices=b_idx,
        seq_indices=s_idx,
        sample_ids=samp_ids,
        target_labels=targets,
    )


@torch.no_grad()
def compute_teacher_topk_from_hidden(
    teacher_active_hidden: torch.Tensor,
    teacher_lm_head: nn.Module,
    topk: int = 512,
    temperature: float = 1.0,
    chunk_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute renormalized Teacher Top-K `(topk_indices int32, topk_logprobs float16)` in memory-bounded chunks."""
    n_tok = teacher_active_hidden.shape[0]
    if n_tok == 0:
        dev = teacher_active_hidden.device
        vocab_sz = getattr(teacher_lm_head, "out_features", None)
        if vocab_sz is None and hasattr(teacher_lm_head, "weight"):
            vocab_sz = int(teacher_lm_head.weight.shape[0])
        k_empty = min(int(topk), int(vocab_sz)) if vocab_sz is not None else max(1, int(topk))
        return (
            torch.zeros((0, k_empty), dtype=torch.int32, device=dev),
            torch.zeros((0, k_empty), dtype=torch.float16, device=dev),
        )

    idx_chunks = []
    lp_chunks = []
    for start in range(0, n_tok, chunk_size):
        end = min(start + chunk_size, n_tok)
        logits = teacher_lm_head(teacher_active_hidden[start:end]).float()
        if temperature != 1.0:
            logits = logits / float(temperature)
        k_eff = min(int(topk), logits.shape[-1])
        vals, idx = torch.topk(logits, k=k_eff, dim=-1)
        renorm_lp = F.log_softmax(vals, dim=-1)
        idx_chunks.append(idx.to(torch.int32))
        lp_chunks.append(renorm_lp.to(torch.float16))

    return torch.cat(idx_chunks, dim=0), torch.cat(lp_chunks, dim=0)


def chunked_topk_kl_distill_loss(
    student_active_hidden: torch.Tensor,
    student_lm_head: nn.Module,
    *,
    sample_ids: Optional[torch.Tensor] = None,
    teacher_active_hidden: Optional[torch.Tensor] = None,
    teacher_lm_head: Optional[nn.Module] = None,
    teacher_topk_indices: Optional[torch.Tensor] = None,
    teacher_topk_logprobs: Optional[torch.Tensor] = None,
    topk: int = 512,
    temperature: float = 1.0,
    reduction: str = "token_mean",
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Compute Forward KL `KL(p_hat_T || p_hat_S)` over Teacher's Top-K vocabulary entries in chunks.

    Peak memory on Qwen3.5 (`vocab_size = 248,320`) for `chunk_size = 1024` is `< 0.5 GB` instead of `48 GB`!

    Args:
        reduction:
            - `"token_mean"`: Global mean over all supervised response tokens (VisionWeave Stages 2 & 3).
            - `"sample_sqrt_len"`: Weights each sample's mean token KL by `sqrt(|y_m|)` (VisionWeave Stage 1).
    """
    n_tok = student_active_hidden.shape[0]
    if (teacher_topk_indices is None) != (teacher_topk_logprobs is None):
        raise ValueError(
            "Both `teacher_topk_indices` and `teacher_topk_logprobs` must be provided together."
        )
    use_offline_topk = teacher_topk_indices is not None and teacher_topk_logprobs is not None
    if not use_offline_topk and (teacher_active_hidden is None or teacher_lm_head is None):
        raise ValueError(
            "Provide either `(teacher_topk_indices, teacher_topk_logprobs)` or `(teacher_active_hidden, teacher_lm_head)`."
        )
    if use_offline_topk:
        assert teacher_topk_indices is not None and teacher_topk_logprobs is not None
        if teacher_topk_indices.ndim != 2 or teacher_topk_logprobs.ndim != 2:
            raise ValueError(
                f"Offline Teacher Top-K tensors must be 2D `(N_active, K)`, got "
                f"indices={tuple(teacher_topk_indices.shape)}, logprobs={tuple(teacher_topk_logprobs.shape)}."
            )
        if (
            teacher_topk_indices.shape[0] != n_tok
            or teacher_topk_logprobs.shape[0] != n_tok
            or teacher_topk_indices.shape[1] != teacher_topk_logprobs.shape[1]
        ):
            raise ValueError(
                f"Active token count / shape mismatch between Student ({n_tok}) and offline Teacher Top-K "
                f"(indices={tuple(teacher_topk_indices.shape)}, logprobs={tuple(teacher_topk_logprobs.shape)})."
            )
        if n_tok > 0 and teacher_topk_indices.shape[1] == 0:
            raise ValueError("Offline Teacher Top-K tensors must have K >= 1 columns when N_active > 0.")
    else:
        assert teacher_active_hidden is not None
        if teacher_active_hidden.shape[0] != n_tok:
            raise ValueError(
                f"Active token count mismatch between Student ({n_tok}) and Teacher ({teacher_active_hidden.shape[0]})."
            )

    if n_tok == 0:
        # Touch student_lm_head with 0 multiplier so DDP/ZeRO sees no unused-parameter hang on empty batches
        dummy = student_active_hidden.sum() * 0.0
        for p in student_lm_head.parameters():
            if p.requires_grad:
                dummy = dummy + p.sum() * 0.0
                break
        return dummy

    per_token_kl = []
    for start in range(0, n_tok, chunk_size):
        end = min(start + chunk_size, n_tok)
        s_logits = student_lm_head(student_active_hidden[start:end]).float()
        if temperature != 1.0:
            s_logits = s_logits / float(temperature)

        if use_offline_topk:
            t_idx = teacher_topk_indices[start:end].to(device=s_logits.device, dtype=torch.long)
            t_vals = teacher_topk_logprobs[start:end].to(device=s_logits.device, dtype=torch.float32)
            k_eff = min(int(topk), t_idx.shape[-1], s_logits.shape[-1])
            if k_eff < t_idx.shape[-1]:
                t_idx = t_idx[:, :k_eff]
                t_vals = t_vals[:, :k_eff]
            # Re-normalize over `k_eff` in case cached topk > requested topk or unnormalized logits were stored
            t_log_p = F.log_softmax(t_vals, dim=-1)
        else:
            with torch.no_grad():
                t_logits = teacher_lm_head(teacher_active_hidden[start:end]).float()
                if temperature != 1.0:
                    t_logits = t_logits / float(temperature)
                k_eff = min(int(topk), t_logits.shape[-1])
                t_top_vals, t_idx = torch.topk(t_logits, k=k_eff, dim=-1)
                t_log_p = F.log_softmax(t_top_vals, dim=-1)

        s_selected = torch.gather(s_logits, dim=-1, index=t_idx)
        s_log_p = F.log_softmax(s_selected, dim=-1)
        t_p = t_log_p.exp()
        kl_chunk = (t_p * (t_log_p - s_log_p)).sum(dim=-1)
        per_token_kl.append(kl_chunk)

    token_kls = torch.cat(per_token_kl, dim=0)

    if reduction == "token_mean" or sample_ids is None:
        return token_kls.mean()

    if reduction == "sample_sqrt_len":
        unique_ids, inverse = torch.unique(sample_ids.to(token_kls.device), sorted=True, return_inverse=True)
        num_samples = unique_ids.shape[0]
        counts = torch.zeros(num_samples, dtype=torch.float32, device=token_kls.device)
        sums = torch.zeros(num_samples, dtype=torch.float32, device=token_kls.device)
        ones = torch.ones_like(token_kls, dtype=torch.float32)
        counts.scatter_add_(0, inverse, ones)
        sums.scatter_add_(0, inverse, token_kls)
        sample_means = sums / counts.clamp_min(1.0)
        weights = torch.sqrt(counts.clamp_min(1.0))
        return (weights * sample_means).sum() / weights.sum().clamp_min(1.0)

    raise ValueError(f"Unsupported reduction: {reduction!r}.")
