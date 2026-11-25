"""src/official.py -- official RiNALMo fine-tuned checkpoint: load, decode, score.

This module exists for one scientific reason: *before* training anything we must
reproduce the published RiNALMo fine-tuned numbers with **our** evaluation code.
Two independent metric conventions are therefore computed side by side:

``official_prf_from_pairs`` / ``official_prf_from_matrices``
    Verbatim re-implementation of ``rinalmo.utils.sec_struct.ss_precision`` /
    ``ss_recall`` / ``ss_f1`` (the published convention): *precision* relaxes the
    **target** by one position, *recall* relaxes the **prediction**.  The official
    trainer macro-averages these per-RNA values over the split.

``src.metrics.pair_prf`` / ``flexible_prf``
    Our own exact and one-to-one shift-1 metrics, used for every model we train
    so that all rows of a table are computed by the same code.

Checkpoint layout (Zenodo ``rinalmo_giga_ss_*_ft.pt``) is the state dict of the
official Lightning wrapper: keys ``lm.*``, ``pred_head.*`` and a scalar
``threshold``.  :class:`OfficialSecStructModel` keeps the same attribute names
(``lm``, ``pred_head``) so weights load 1:1, and mirrors the wrapper's
``load_state_dict`` override that pops ``threshold``.

``prob_mat_to_sec_struct`` mutates its input in place and applies the canonical +
sharp-loop masks before greedy cleaning; our :func:`official_decode` copies the
probability matrix first and reproduces the same rule.  ``tests/test_official.py``
checks both the decoder and the metric replica against the upstream package when
it is importable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .model import BackboneUnavailableError, ensure_rinalmo

Pair = Tuple[int, int]
Triple = Tuple[float, float, float]

CANONICAL_PAIRS = frozenset({"AU", "UA", "GC", "CG", "GU", "UG"})
SHARP_LOOP_DIST_THRESHOLD = 4  # official: offsets 1..3 are forbidden
VALID_BASES = frozenset("ACGU")
EPSILON = 1e-5  # official ss_f1 epsilon


def normalize_sequence(sequence: str) -> str:
    return sequence.strip().upper().replace("T", "U")


# ---------------------------------------------------------------------------
# Official decoder (re-implementation of prob_mat_to_sec_struct)
# ---------------------------------------------------------------------------
def _sharp_loop_mask(length: int) -> np.ndarray:
    mask = np.eye(length, k=0, dtype=bool)
    for offset in range(1, SHARP_LOOP_DIST_THRESHOLD):
        mask = mask | np.eye(length, k=offset, dtype=bool) | np.eye(length, k=-offset, dtype=bool)
    return mask


def _canonical_pairs_mask(sequence: str) -> np.ndarray:
    sequence = normalize_sequence(sequence)
    mask = np.zeros((len(sequence), len(sequence)), dtype=bool)
    for i, base_i in enumerate(sequence):
        for j, base_j in enumerate(sequence):
            if f"{base_i}{base_j}" in CANONICAL_PAIRS:
                mask[i, j] = True
    return mask


def _clean_sec_struct(sec_struct: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    clean = np.copy(sec_struct)
    remaining = np.copy(probabilities)
    remaining[sec_struct < 1] = 0.0
    while np.sum(remaining > 0.0) > 0:
        i, j = np.unravel_index(np.argmax(remaining, axis=None), remaining.shape)
        remaining[i, :] = remaining[j, :] = 0.0
        clean[i, :] = clean[j, :] = 0
        remaining[:, i] = remaining[:, j] = 0.0
        clean[:, i] = clean[:, j] = 0
        clean[i, j] = clean[j, i] = 1
    return clean


def official_decode_probs(
    probabilities: np.ndarray,
    sequence: str,
    threshold: float = 0.5,
    *,
    min_sep: int = SHARP_LOOP_DIST_THRESHOLD,
    allow_nc_pairs: bool = False,
) -> np.ndarray:
    """Official ``prob_mat_to_sec_struct`` on a *copy*; returns a 0/1 matrix."""
    sequence = normalize_sequence(sequence)
    probabilities = np.array(probabilities, dtype=np.float64, copy=True)
    length = probabilities.shape[-1]
    if probabilities.shape != (length, length):
        raise ValueError(f"probability matrix must be square, got {probabilities.shape}")
    if len(sequence) != length:
        raise ValueError(f"sequence length {len(sequence)} != matrix {length}")
    if not np.isfinite(probabilities).all():
        raise ValueError("non-finite probabilities")
    allowed = ~np.eye(length, dtype=bool)
    if min_sep > 1:
        allowed &= ~_sharp_loop_mask(length)
    if not allow_nc_pairs:
        allowed &= _canonical_pairs_mask(sequence)
    probabilities[~allowed] = 0.0
    sec_struct = np.greater(probabilities, threshold).astype(np.int64)
    return _clean_sec_struct(sec_struct, probabilities)


def matrix_to_pairs(matrix: np.ndarray) -> List[Pair]:
    indices = np.argwhere(np.triu(matrix, 1) > 0)
    return [(int(i), int(j)) for i, j in indices]


def official_decode(
    probabilities: np.ndarray,
    sequence: str,
    threshold: float = 0.5,
    *,
    min_sep: int = SHARP_LOOP_DIST_THRESHOLD,
    allow_nc_pairs: bool = False,
) -> List[Pair]:
    matrix = official_decode_probs(
        probabilities,
        sequence,
        threshold,
        min_sep=min_sep,
        allow_nc_pairs=allow_nc_pairs,
    )
    return matrix_to_pairs(matrix)


# ---------------------------------------------------------------------------
# Official metric replica (rinalmo.utils.sec_struct ss_precision/ss_recall/ss_f1)
# ---------------------------------------------------------------------------
def relax_ss(matrix: np.ndarray, radius: int = 1) -> np.ndarray:
    """Official ``_relax_ss``: dilate a 0/1 matrix in the Manhattan ball.

    The upstream implementation adds four rolls (left/right/up/down) to the
    original matrix, i.e. a cell becomes positive when a positive cell is within
    **one step in one coordinate** (centre + 4 orthogonal neighbours for
    ``radius=1``), *not* the full 3x3 Chebyshev neighbourhood.  Reproduced
    exactly here; ``radius > 1`` extends it to the Manhattan ball of that radius.
    """
    matrix = np.asarray(matrix)
    if radius < 1:
        return matrix.copy()
    padded = np.pad(matrix, ((radius, radius), (radius, radius)), mode="constant")
    relaxed = padded.copy()
    for step in range(1, radius + 1):
        relaxed = relaxed + np.roll(padded, shift=step, axis=-1)
        relaxed = relaxed + np.roll(padded, shift=-step, axis=-1)
        relaxed = relaxed + np.roll(padded, shift=step, axis=-2)
        relaxed = relaxed + np.roll(padded, shift=-step, axis=-2)
    relaxed = relaxed[radius:-radius, radius:-radius]
    return np.clip(relaxed, 0, 1)


def pairs_to_matrix(pairs: Iterable[Pair], length: int) -> np.ndarray:
    matrix = np.zeros((length, length), dtype=np.float64)
    for pair in pairs:
        i, j = int(pair[0]), int(pair[1])
        if i > j:
            i, j = j, i
        if not (0 <= i < j < length):
            raise ValueError(f"pair ({i}, {j}) outside length {length}")
        matrix[i, j] = matrix[j, i] = 1.0
    return matrix


def _binary_precision(target: np.ndarray, prediction: np.ndarray) -> float:
    predicted_positive = int(prediction.sum())
    if predicted_positive == 0:
        return 0.0
    true_positive = int(np.logical_and(target, prediction).sum())
    return true_positive / predicted_positive


def _binary_recall(target: np.ndarray, prediction: np.ndarray) -> float:
    actual_positive = int(target.sum())
    if actual_positive == 0:
        return 0.0
    true_positive = int(np.logical_and(target, prediction).sum())
    return true_positive / actual_positive


def official_prf_from_matrices(
    target_matrix: np.ndarray, prediction_matrix: np.ndarray, radius: int = 1
) -> Triple:
    """Official precision/recall/F1 on upper triangles of two 0/1 matrices.

    Exactly ``ss_precision(target, pred)`` (target relaxed) and
    ``ss_recall(target, pred)`` (prediction relaxed), with sklearn's binary
    ``zero_division=0`` behaviour, and ``ss_f1``'s ``EPSILON`` guard.
    """
    target = np.asarray(target_matrix, dtype=np.float64)
    prediction = np.asarray(prediction_matrix, dtype=np.float64)
    if target.shape != prediction.shape:
        raise ValueError(f"shape mismatch {target.shape} vs {prediction.shape}")
    length = target.shape[0]
    upper = np.triu_indices(length, k=1)
    target_upper = target[upper]
    prediction_upper = prediction[upper]
    relaxed_target = relax_ss(target, radius=radius)[upper]
    relaxed_prediction = relax_ss(prediction, radius=radius)[upper]
    precision = _binary_precision(relaxed_target, prediction_upper)
    recall = _binary_recall(target_upper, relaxed_prediction)
    if precision + recall < EPSILON:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def official_prf_from_pairs(
    reference: Iterable[Pair], predicted: Iterable[Pair], length: int, radius: int = 1
) -> Triple:
    return official_prf_from_matrices(
        pairs_to_matrix(reference, length), pairs_to_matrix(predicted, length), radius=radius
    )


def macro_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


# ---------------------------------------------------------------------------
# Official model
# ---------------------------------------------------------------------------
def make_official_config(
    lm_config: str = "giga",
    *,
    embed_dim: int | None = None,
    num_blocks: int | None = None,
    num_heads: int | None = None,
    use_flash_attn: bool | None = None,
):
    """Official ``model_config`` with optional architecture overrides.

    ``embed_dim``/``num_blocks``/``num_heads`` exist for local CPU tests with a
    tiny randomly initialised model; production runs use the untouched official
    ``giga`` config.
    """
    ensure_rinalmo()
    from rinalmo.config import model_config  # type: ignore

    config = model_config(lm_config)
    if embed_dim is not None:
        config.globals.embed_dim = int(embed_dim)
        config.model.embedding.embedding_dim = int(embed_dim)
        config.model.transformer.embed_dim = int(embed_dim)
        config.model.lm_mask_head.embed_dim = int(embed_dim)
    if num_blocks is not None:
        config.model.transformer.num_blocks = int(num_blocks)
    if num_heads is not None:
        config.model.transformer.num_heads = int(num_heads)
    if use_flash_attn is not None:
        config.model.transformer.use_flash_attn = bool(use_flash_attn)
    return config


class OfficialSecStructModel(nn.Module):
    """Official RiNALMo + ``SecStructPredictionHead`` with checkpoint plumbing.

    Attribute names (``lm``, ``pred_head``) match the official Lightning wrapper
    ``SecStructPredictionWrapper`` so a Zenodo checkpoint loads verbatim.
    """

    def __init__(
        self,
        lm_config: str = "giga",
        num_blocks: int = 2,
        *,
        config=None,
        extra_paths: Iterable[str | Path] | None = None,
    ):
        super().__init__()
        ensure_rinalmo(extra_paths)
        from rinalmo.model.downstream import SecStructPredictionHead  # type: ignore
        from rinalmo.model.model import RiNALMo  # type: ignore

        if config is None:
            config = make_official_config(lm_config)
        self.lm = RiNALMo(config)
        embed_dim = int(self.lm.config.model.transformer.embed_dim)
        self.pred_head = SecStructPredictionHead(embed_dim, num_blocks=int(num_blocks))
        self.threshold = 0.5
        self.load_report: Dict[str, Any] = {}

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        representation = self.lm(tokens)["representation"]
        logits = self.pred_head(representation[..., 1:-1, :]).squeeze(-1)
        return logits

    # -- checkpoint --------------------------------------------------------
    def load_official_checkpoint(
        self, path: str | Path, strict: bool = True
    ) -> Dict[str, Any]:
        """Load a Zenodo official checkpoint (``lm.*`` / ``pred_head.*`` / threshold)."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, Mapping) and "state_dict" in payload:
            payload = payload["state_dict"]
        if not isinstance(payload, Mapping):
            raise TypeError(f"{path}: expected a state dict, got {type(payload)!r}")
        state_dict = dict(payload)
        if state_dict and all(key.startswith("module.") for key in state_dict):
            state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}
        threshold = state_dict.pop("threshold", None)
        report = self.load_state_dict(state_dict, strict=bool(strict))
        if threshold is not None:
            self.threshold = float(threshold)
        self.load_report = {
            "source": str(path),
            "threshold": self.threshold,
            "threshold_from_checkpoint": threshold is not None,
            "missing_keys": list(report.missing_keys),
            "unexpected_keys": list(report.unexpected_keys),
            "strict": bool(strict),
        }
        return self.load_report

    # The official wrapper overrides ``load_state_dict`` to accept a ``threshold``
    # entry; mirror that so callers can use the same code path.
    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        state_dict = dict(state_dict)
        if "threshold" in state_dict:
            self.threshold = float(state_dict.pop("threshold"))
        return super().load_state_dict(state_dict, strict=strict)

    def on_save_checkpoint_payload(self) -> Dict[str, Any]:
        state = dict(self.state_dict())
        state["threshold"] = self.threshold
        return state


def official_checkpoint_threshold(path: str | Path) -> float | None:
    """Read just the ``threshold`` scalar from an official checkpoint (cheap)."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping) and "state_dict" in payload:
        payload = payload["state_dict"]
    if isinstance(payload, Mapping) and "threshold" in payload:
        return float(payload["threshold"])
    return None


def build_tiny_official_model(extra_paths: Iterable[str | Path] | None = None) -> OfficialSecStructModel:
    """Randomly initialised CPU-sized official model (tests, no weights needed)."""
    config = make_official_config(
        "micro", embed_dim=64, num_blocks=2, num_heads=4, use_flash_attn=False
    )
    return OfficialSecStructModel(config=config, num_blocks=2, extra_paths=extra_paths)
