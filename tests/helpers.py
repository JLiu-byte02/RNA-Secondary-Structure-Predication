"""Shared synthetic-data helpers for the test suite (no dependency on data/)."""
from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")]


def synthetic_record(
    index: int,
    length: int,
    rng: random.Random,
    *,
    split: str = "TR0",
    family: str = "fam0",
) -> dict:
    """Random nested (pseudoknot-free) RNA whose pairs are all canonical and >= 4 apart."""
    bases: list[str | None] = [None] * length
    pairs: list[list[int]] = []
    i, j = 0, length - 1
    while j - i >= 4:
        if rng.random() < 0.55:
            base_i, base_j = CANONICAL[rng.randrange(len(CANONICAL))]
            bases[i], bases[j] = base_i, base_j
            pairs.append([i, j])
            i += 1
            j -= 1
        elif rng.random() < 0.5:
            i += 1
        else:
            j -= 1
    sequence = "".join(base if base is not None else rng.choice("ACGU") for base in bases)
    return {
        "id": f"rna{index:04d}",
        "sequence": sequence,
        "pairs": pairs,
        "length": length,
        "split": split,
        "family": family,
        "source_id": f"synth{index:04d}",
    }


def synthetic_records(
    count: int = 8,
    *,
    seed: int = 0,
    length_range: Sequence[int] = (24, 44),
    split: str = "TR0",
    families: Sequence[str] = ("famA", "famB"),
) -> list[dict]:
    rng = random.Random(seed)
    records = []
    for index in range(count):
        length = rng.randint(length_range[0], length_range[1])
        records.append(
            synthetic_record(
                index,
                length,
                rng,
                split=split,
                family=families[index % len(families)],
            )
        )
    return records


def write_jsonl_gz(path: str | Path, records: Iterable[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def write_thermo_npz(
    directory: str | Path,
    sequence: str,
    *,
    seed: int = 0,
    keys: tuple[str, str] = ("P", "Q"),
) -> Path:
    """Write a BPP/stack cache in the prior project's format, keyed by sha256(sequence)."""
    import hashlib

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    length = len(sequence)
    rng = np.random.default_rng(seed)
    upper_bpp = np.triu(rng.random((length, length), dtype=np.float32), 1)
    upper_stack = np.triu(rng.random((length, length), dtype=np.float32), 1)
    bpp = upper_bpp + upper_bpp.T
    stack = upper_stack + upper_stack.T
    path = directory / f"{hashlib.sha256(sequence.encode('ascii')).hexdigest()}.npz"
    np.savez(path, **{keys[0]: bpp, keys[1]: stack})
    return path


def write_ct(path: str | Path, sequence: str, pairs: Iterable[Sequence[int]]) -> Path:
    """Minimal CT writer (0-based pairs -> 1-based CT columns)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partner = [0] * (len(sequence) + 1)
    for i, j in pairs:
        partner[i + 1] = j + 1
        partner[j + 1] = i + 1
    lines = [f"{len(sequence)}\t{path.stem}"]
    for index in range(len(sequence)):
        lines.append(
            f"{index + 1}\t{sequence[index]}\t{index}\t{index + 2}\t{partner[index + 1]}\t{index + 1}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_stub_checkpoint(
    path: str | Path,
    *,
    seed: int = 0,
    threshold: float = 0.2,
    epoch: int = 1,
    best_score: float = 0.5,
    embed_dim: int = 64,
    num_layers: int = 1,
    num_heads: int = 4,
    conv_dim: int = 8,
    num_blocks: int = 1,
    data_cfg: dict | None = None,
    drop_key: str | None = None,
    extra_key: bool = False,
    signature_override: str | None = None,
) -> dict:
    """Write a small stub-model checkpoint compatible with evaluate.py / soup.py.

    Parameters exist to build deliberately incompatible checkpoints in tests.
    """
    import torch

    from src.model import build_model, model_signature

    model_cfg = {
        "backbone": {
            "type": "stub",
            "embed_dim": embed_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
        },
        "head": {
            "type": "outer_product",
            "kwargs": {"conv_dim": conv_dim, "num_blocks": num_blocks},
        },
        "channels": [],
    }
    torch.manual_seed(seed)
    model = build_model(model_cfg)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.01 * seed)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if drop_key is not None:
        state.pop(drop_key)
    if extra_key:
        state["unexpected.tensor"] = torch.zeros(1)
    payload = {
        "format_version": 1,
        "model": state,
        "cfg": {
            "model": model_cfg,
            "data": data_cfg
            or {
                "min_sep": 4,
                "strict_sequence": False,
                "strict_pairs": False,
                "require_eligible": False,
                "use_rinalmo_alphabet": False,
            },
        },
        "epoch": epoch,
        "threshold": threshold,
        "best_score": best_score,
        "best_epoch": epoch,
        "stale": 0,
        "history": [],
        "model_signature": signature_override or model_signature(model),
        "data_sha256": {"train": "stub"},
        "code_sha256": {"combined_sha256": "stub"},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload
