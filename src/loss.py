"""src/loss.py -- masked BCE on eligible candidate pairs.

The loss only ever scores *eligible* candidate pairs (canonical AU/UA/GC/CG/GU/UG
with ``j - i >= min_sep``), mirroring the prior project's protocol and RiNALMo's
decode geometry.  Pairs that no decoder could ever emit must not receive
gradient: a model that spends capacity ranking non-canonical pairs is optimising
a quantity that is never evaluated.

``masked_pair_bce`` accepts a single ``(L, L)`` matrix or a padded batch
``(B, L, L)``; in both cases only the strict upper triangle of eligible,
non-padded positions contributes.  An RNA with zero eligible pairs returns
``None`` so callers can skip it instead of averaging in a fake zero.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .decode import MIN_SEP, candidate_mask


def upper_triangle_mask(length: int, device=None, dtype=torch.bool) -> torch.Tensor:
    return torch.triu(
        torch.ones((length, length), device=device, dtype=dtype), diagonal=1
    ).to(torch.bool)


def candidate_loss_mask(
    sequence: str, min_sep: int = MIN_SEP, device=None
) -> torch.Tensor:
    """Boolean ``(L, L)`` mask of eligible pairs (upper triangle only)."""
    mask = candidate_mask(sequence, min_sep=min_sep)
    mask = np.triu(mask, 1)
    return torch.from_numpy(mask).to(device=device)


def masked_pair_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    eligible: torch.Tensor,
    *,
    pair_mask: torch.Tensor | None = None,
    pos_weight: float | torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor | None:
    """BCE-with-logits restricted to eligible candidate pairs.

    Parameters
    ----------
    logits, target:
        ``(L, L)`` or ``(B, L, L)``; targets are 0/1 floats.
    eligible:
        Same shape as ``target`` (bool); symmetric candidate geometry is fine,
        only its strict upper triangle is used.
    pair_mask:
        Optional bool mask for padded batches (``True`` == both positions real).
    pos_weight:
        Optional positive-class weight passed to ``binary_cross_entropy_with_logits``.

    Returns ``None`` when no eligible pair exists, so the caller can skip an
    empty RNA rather than silently learning "predict nothing".
    """
    if logits.shape != target.shape:
        raise ValueError(f"logits {tuple(logits.shape)} != target {tuple(target.shape)}")
    if eligible.shape != target.shape:
        raise ValueError(f"eligible {tuple(eligible.shape)} != target {tuple(target.shape)}")

    length = logits.shape[-1]
    mask = eligible.to(torch.bool) & upper_triangle_mask(length, device=logits.device)
    if pair_mask is not None:
        mask = mask & pair_mask.to(torch.bool)
    if not mask.any():
        return None

    selected_logits = logits[mask].float()
    selected_target = target[mask].float()
    if not torch.isfinite(selected_logits).all():
        raise FloatingPointError("non-finite logits inside the eligible-pair mask")
    weight = None
    if pos_weight is not None:
        weight = (
            pos_weight.to(device=logits.device, dtype=torch.float32)
            if isinstance(pos_weight, torch.Tensor)
            else torch.tensor(float(pos_weight), device=logits.device)
        )
    loss = F.binary_cross_entropy_with_logits(
        selected_logits, selected_target, pos_weight=weight, reduction=reduction
    )
    return loss


def full_upper_triangle_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    pair_mask: torch.Tensor | None = None,
    pos_weight: float | torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor | None:
    """BCE over the **whole** upper triangle (official RiNALMo objective).

    The official fine-tuning code computes
    ``BCEWithLogitsLoss(logits[upper_tri], sec_struct[upper_tri])`` for every
    position, including non-canonical pairs and pairs that decoding will never
    emit.  This is deliberately different from :func:`masked_pair_bce` and the
    two objectives are selectable per config (``train.loss``).
    """
    if logits.shape != target.shape:
        raise ValueError(f"logits {tuple(logits.shape)} != target {tuple(target.shape)}")
    length = logits.shape[-1]
    mask = upper_triangle_mask(length, device=logits.device)
    if pair_mask is not None:
        mask = mask & pair_mask.to(torch.bool)
    if not mask.any():
        return None
    selected_logits = logits[mask].float()
    if not torch.isfinite(selected_logits).all():
        raise FloatingPointError("non-finite logits in the upper-triangle loss")
    weight = None
    if pos_weight is not None:
        weight = (
            pos_weight.to(device=logits.device, dtype=torch.float32)
            if isinstance(pos_weight, torch.Tensor)
            else torch.tensor(float(pos_weight), device=logits.device)
        )
    return F.binary_cross_entropy_with_logits(
        selected_logits, target[mask].float(), pos_weight=weight, reduction=reduction
    )


def pair_bce_from_batch(
    logits: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    *,
    pos_weight: float | torch.Tensor | None = None,
    mode: str = "eligible_candidates",
) -> torch.Tensor | None:
    """Convenience wrapper for ``collate_batch`` output.

    ``mode``:
        * ``eligible_candidates`` (default, our B3 protocol) -- only pairs a
          decoder could emit (canonical + ``min_sep``);
        * ``full_upper_triangle`` / ``official`` -- the official RiNALMo
          objective, every upper-triangle position.
    """
    mode = str(mode).lower()
    if mode in ("eligible_candidates", "eligible", "masked", "candidates"):
        return masked_pair_bce(
            logits,
            batch["target"],
            batch["eligible"],
            pair_mask=batch.get("pair_mask"),
            pos_weight=pos_weight,
        )
    if mode in ("full_upper_triangle", "official", "full"):
        return full_upper_triangle_bce(
            logits,
            batch["target"],
            pair_mask=batch.get("pair_mask"),
            pos_weight=pos_weight,
        )
    raise ValueError(f"unknown loss mode {mode!r}")


class PairBCELoss(torch.nn.Module):
    """Module form (keeps ``pos_weight`` on the right device and the mode fixed)."""

    def __init__(self, pos_weight: float | None = None, mode: str = "eligible_candidates"):
        super().__init__()
        self.mode = str(mode)
        self.register_buffer(
            "pos_weight",
            None if pos_weight is None else torch.tensor(float(pos_weight)),
            persistent=False,
        )

    def forward(self, logits: torch.Tensor, batch: Mapping[str, torch.Tensor]):
        return pair_bce_from_batch(
            logits,
            batch,
            pos_weight=self.pos_weight if self.pos_weight is not None else None,
            mode=self.mode,
        )


def count_eligible(sequence: str, min_sep: int = MIN_SEP) -> int:
    return int(np.triu(candidate_mask(sequence, min_sep=min_sep), 1).sum())


def positive_fraction(
    target: torch.Tensor, eligible: torch.Tensor, pair_mask: torch.Tensor | None = None
) -> float:
    """Share of eligible pairs that are true base pairs (diagnostic)."""
    length = target.shape[-1]
    mask = eligible.to(torch.bool) & upper_triangle_mask(length, device=target.device)
    if pair_mask is not None:
        mask = mask & pair_mask.to(torch.bool)
    total = int(mask.sum())
    if total == 0:
        return 0.0
    return float(target[mask].sum().item()) / total
