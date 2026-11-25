#!/usr/bin/env python
"""scripts/soup.py -- average compatible training checkpoints into a weight soup.

Given per-epoch checkpoints from one run (``train.save_every_epoch: true`` writes
``epoch{N}.pt``, weights only), average their floating-point tensors::

    python scripts/soup.py \
        --checkpoints runs/X/epoch10.pt runs/X/epoch11.pt runs/X/epoch12.pt \
        --out runs/X/soup.pt

What is guaranteed (and unit-tested):

* every checkpoint must have the **same state-dict keys and shapes**; a mismatch
  aborts with the offending keys before touching the output;
* non-floating (integer/bool) buffers are not averaged: they must be *identical*
  across sources and are copied through;
* the architecture fingerprint (``model_signature`` and ``cfg.model``) must agree
  across sources, otherwise the average is meaningless and the script refuses;
* the output is a normal training checkpoint (``format_version``, ``model``,
  ``cfg``, ``threshold``, ...) so ``scripts/evaluate.py`` consumes it unchanged;
* provenance (every source path + sha256 + epoch + threshold, weights, output
  sha256) is written to ``soup.json`` next to the output.

Threshold: a soup does not re-select its decoding threshold (that must happen on
the validation split only). By default it inherits the threshold of the source
with the best ``best_score``; override with ``--threshold`` after a VL0 sweep.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# IO helpers (kept local; scripts/ is not a package)
# ---------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_torch(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


# ---------------------------------------------------------------------------
# Checkpoint handling
# ---------------------------------------------------------------------------
def expand_checkpoints(values: Sequence[str | Path]) -> List[Path]:
    """Expand globs, de-duplicate and sort checkpoint paths."""
    paths: List[Path] = []
    for value in values:
        text = str(value)
        if any(char in text for char in "*?["):
            matches = sorted(Path(match) for match in glob.glob(text))
            if not matches:
                raise FileNotFoundError(f"checkpoint glob matched nothing: {value}")
            paths.extend(matches)
        else:
            path = Path(value)
            if not path.exists():
                raise FileNotFoundError(f"checkpoint not found: {path}")
            paths.append(path)
    resolved = sorted({path.resolve() for path in paths}, key=str)
    if not resolved:
        raise FileNotFoundError("no checkpoints given")
    return resolved


def load_checkpoint_payload(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise ValueError(f"{path}: not a training checkpoint (missing 'model')")
    model = payload["model"]
    if not isinstance(model, Mapping) or not model:
        raise ValueError(f"{path}: 'model' is not a non-empty state dict")
    return dict(payload)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def validate_compatible(
    paths: Sequence[Path], payloads: Sequence[Mapping[str, Any]]
) -> Dict[str, List[str]]:
    """Assert identical keys/shapes/dtypes/architecture; raise with details."""
    if len(payloads) < 1:
        raise ValueError("no checkpoints to validate")
    reference = payloads[0]["model"]
    reference_keys = list(reference.keys())
    float_keys: List[str] = []
    nonfloat_keys: List[str] = []
    for key in reference_keys:
        tensor = reference[key]
        if not torch.is_tensor(tensor):
            raise ValueError(f"{paths[0]}: state_dict entry {key!r} is not a tensor")
        (float_keys if tensor.is_floating_point() else nonfloat_keys).append(key)

    for index, (path, payload) in enumerate(zip(paths, payloads)):
        model = payload["model"]
        keys = list(model.keys())
        missing = [key for key in reference_keys if key not in model]
        unexpected = [key for key in keys if key not in reference]
        if missing or unexpected:
            raise ValueError(
                f"checkpoint {index} ({path}) has mismatched keys: "
                f"missing={missing[:10]}{'...' if len(missing) > 10 else ''} "
                f"unexpected={unexpected[:10]}{'...' if len(unexpected) > 10 else ''}"
            )
        for key in reference_keys:
            left, right = reference[key], model[key]
            if tuple(left.shape) != tuple(right.shape):
                raise ValueError(
                    f"checkpoint {index} ({path}) shape mismatch for {key!r}: "
                    f"{tuple(left.shape)} vs {tuple(right.shape)}"
                )
            if left.dtype != right.dtype:
                raise ValueError(
                    f"checkpoint {index} ({path}) dtype mismatch for {key!r}: "
                    f"{left.dtype} vs {right.dtype}"
                )
            if key in nonfloat_keys and not torch.equal(left, right):
                raise ValueError(
                    f"checkpoint {index} ({path}) non-float buffer {key!r} differs; "
                    "refusing to copy an ambiguous value"
                )
    # architecture fingerprint must agree (missing is tolerated for old ckpts)
    signatures = {
        str(item.get("model_signature"))
        for item in payloads
        if item.get("model_signature") is not None
    }
    if len(signatures) > 1:
        raise ValueError(f"checkpoints have different model_signature: {sorted(signatures)}")
    configs = {
        _canonical_json((item.get("cfg") or {}).get("model"))
        for item in payloads
        if (item.get("cfg") or {}).get("model") is not None
    }
    if len(configs) > 1:
        raise ValueError("checkpoints have different cfg.model (architecture) sections")
    return {"float_keys": float_keys, "nonfloat_keys": nonfloat_keys}


def normalize_weights(weights: Sequence[float] | None, count: int) -> List[float]:
    if weights is None:
        return [1.0 / count] * count
    values = [float(weight) for weight in weights]
    if len(values) != count:
        raise ValueError(f"--weights has {len(values)} entries for {count} checkpoints")
    if any(value < 0 for value in values):
        raise ValueError("--weights must be non-negative")
    total = sum(values)
    if total <= 0:
        raise ValueError("--weights sum to zero")
    return [value / total for value in values]


def average_checkpoints(
    paths: Sequence[str | Path],
    *,
    weights: Sequence[float] | None = None,
    threshold: float | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return ``(soup_payload, manifest)``; validates every source first."""
    resolved = [Path(path).resolve() for path in paths]
    payloads = [load_checkpoint_payload(path) for path in resolved]
    groups = validate_compatible(resolved, payloads)
    normalized = normalize_weights(weights, len(payloads))
    reference = payloads[0]

    averaged: Dict[str, torch.Tensor] = {}
    for key in groups["float_keys"]:
        accumulator = payloads[0]["model"][key].to(torch.float64) * normalized[0]
        for weight, payload in zip(normalized[1:], payloads[1:]):
            accumulator = accumulator + payload["model"][key].to(torch.float64) * weight
        averaged[key] = accumulator.to(payloads[0]["model"][key].dtype)
    for key in groups["nonfloat_keys"]:
        averaged[key] = payloads[0]["model"][key].clone()

    # threshold: explicit > best source (highest best_score) > first payload
    sources = []
    for path, payload in zip(resolved, payloads):
        sources.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "epoch": payload.get("epoch"),
                "threshold": payload.get("threshold"),
                "best_score": payload.get("best_score"),
            }
        )
    if threshold is not None:
        selected_threshold = float(threshold)
        threshold_source = "explicit --threshold"
    else:
        scored = [
            (source, payload)
            for source, payload in zip(sources, payloads)
            if payload.get("best_score") is not None
        ]
        if scored:
            best_source, best_payload = max(scored, key=lambda item: float(item[1]["best_score"]))
            selected_threshold = float(best_payload.get("threshold", 0.5))
            threshold_source = f"best source {best_source['path']}"
        else:
            selected_threshold = float(reference.get("threshold", 0.5))
            threshold_source = f"first source {sources[0]['path']}"

    cfg = reference.get("cfg")
    soup = {
        "method": "weighted_average" if weights is not None else "average",
        "weights": normalized,
        "num_sources": len(sources),
        "sources": sources,
        "averaged_tensors": len(groups["float_keys"]),
        "copied_nonfloat_tensors": len(groups["nonfloat_keys"]),
        "threshold_source": threshold_source,
    }
    payload = {
        "format_version": 1,
        "model": averaged,
        "cfg": cfg,
        "epoch": None,
        "global_step": None,
        "threshold": selected_threshold,
        "best_score": max(
            (float(item["best_score"]) for item in payloads if item.get("best_score") is not None),
            default=None,
        ),
        "best_epoch": None,
        "stale": 0,
        "history": [],
        "model_signature": reference.get("model_signature"),
        "data_sha256": reference.get("data_sha256"),
        "code_sha256": reference.get("code_sha256"),
        "soup": soup,
    }
    manifest = {
        **soup,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__)),
        "model_signature": reference.get("model_signature"),
        "threshold": selected_threshold,
        "output": None,  # filled by the caller after writing
    }
    return payload, manifest


def write_soup(
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    out: str | Path,
    manifest_path: str | Path | None = None,
) -> Tuple[Path, Path, Dict[str, Any]]:
    out = Path(out)
    manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else out.with_name(out.stem + ".json")
    )
    save_torch(out, payload)
    finished = dict(manifest)
    finished["output"] = {
        "path": str(out.resolve()),
        "sha256": sha256_file(out),
        "num_keys": len(payload["model"]),
    }
    save_json(manifest_path, finished)
    return out, manifest_path, finished


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="checkpoint paths or globs (e.g. runs/X/epoch10.pt runs/X/epoch11.pt)",
    )
    parser.add_argument("--out", required=True, help="output soup checkpoint, e.g. runs/X/soup.pt")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="optional per-checkpoint weights (normalised); default uniform",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="decoding threshold to store; default inherits the best source checkpoint's",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="provenance JSON path (default: next to --out, same stem .json)",
    )
    args = parser.parse_args(argv)
    try:
        paths = expand_checkpoints(args.checkpoints)
        payload, manifest = average_checkpoints(
            paths, weights=args.weights, threshold=args.threshold
        )
        out, manifest_path, finished = write_soup(
            payload, manifest, args.out, args.manifest
        )
    except (ValueError, FileNotFoundError) as error:
        print(f"SOUP_FAILED: {error}", file=sys.stderr, flush=True)
        return 2
    summary = {
        "out": str(out),
        "manifest": str(manifest_path),
        "num_sources": finished["num_sources"],
        "averaged_tensors": finished["averaged_tensors"],
        "copied_nonfloat_tensors": finished["copied_nonfloat_tensors"],
        "threshold": finished["threshold"],
        "threshold_source": finished["threshold_source"],
        "sha256": finished["output"]["sha256"],
    }
    print("SOUP_COMPLETED " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
