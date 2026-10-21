from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time
from collections import Counter, defaultdict

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPLIT_KEYS = {"fit-train": "fit_train", "family-dev": "family_dev", "VL0": "VL0"}
PARAMS = {
    "temperature": 37.0,
    "dangles": 2,
    "noLP": False,
    "uniq_ML": True,
    "compute_bpp": True,
    "energy_parameters": "RNA_Turner2004",
    "min_loop_size": 3,
    "mfe_for_pf_scaling": True,
}
CANONICAL = {"AU", "UA", "CG", "GC", "GU", "UG"}
TOLERANCE = 1e-6
PRETRAINED_REFERENCE_SHA256 = "cd93c3f21eb3e767373c9491192686b5846247bd1110693e453c1dd0f321c0db"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_hash(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def pairs_from_text(text, length):
    if not text or str(text).strip() in {".", "[]", "nan", "None"}:
        return set()
    pairs = set()
    for token in str(text).split(";"):
        first, second = token.strip().split("-")
        left, right = int(first) - 1, int(second) - 1
        if not 0 <= left < right < length:
            raise ValueError(f"invalid 1-based annotation pair {token!r}")
        pairs.add((left, right))
    return pairs


def split_ids(value):
    if isinstance(value, dict):
        if "ids" in value:
            value = value["ids"]
        elif "sample_ids" in value:
            value = value["sample_ids"]
        else:
            value = list(value)
    if not isinstance(value, list):
        raise ValueError("split entry must contain a list of RNA IDs")
    return [str(item["sample_id"] if isinstance(item, dict) else item) for item in value]


def load_selected(root, requested, manifest_path):
    split_path = root / "data/processed/split_ids.json"
    with split_path.open(encoding="utf-8") as handle:
        available = json.load(handle)
    assignments = defaultdict(list)
    splits = {}
    for name in requested:
        key = SPLIT_KEYS[name]
        if key not in available:
            raise ValueError(f"split_ids.json lacks {key!r}")
        identifiers = split_ids(available[key])
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"duplicate RNA IDs within {name}")
        splits[name] = identifiers
        for sample_id in identifiers:
            assignments[sample_id].append(name)
    selected = {}
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"sample_id", "sequence", "original_pairs", "eligible_pairs", "family_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        for row in reader:
            sample_id = str(row["sample_id"])
            if sample_id not in assignments:
                continue
            if sample_id in selected:
                raise ValueError(f"duplicate manifest sample_id {sample_id}")
            sequence = row["sequence"].strip().upper().replace("T", "U")
            if not sequence or set(sequence) - set("ACGU"):
                raise ValueError(f"{sample_id}: sequence must contain only A,C,G,U")
            row["sequence"] = sequence
            row["computed_sequence_hash"] = sequence_hash(sequence)
            row["selected_splits"] = assignments[sample_id]
            if row.get("length") and int(row["length"]) != len(sequence):
                raise ValueError(f"{sample_id}: sequence/manifest length mismatch")
            selected[sample_id] = row
    missing_ids = set(assignments) - set(selected)
    if missing_ids:
        raise ValueError(f"selected IDs missing from manifest: {sorted(missing_ids)[:20]}")
    return selected, splits, split_path


def known_family(value):
    return str(value).strip() not in {"", ".", "NA", "NaN", "nan", "None", "unknown", "unannotated", "-1"}


def find_embeddings(root, override, requested):
    directory = Path(override).resolve() if override else root / "embeddings"
    if directory.is_file():
        if len(requested) != 1:
            raise ValueError("a single --embeddings file requires exactly one requested split")
        return {requested[0]: directory}
    paths = {name: directory / f"{SPLIT_KEYS[name]}.pt" for name in requested}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"embedding caches missing: {missing}")
    return paths


def embedding_mapping(payload):
    if not isinstance(payload, dict):
        raise ValueError("embedding cache must be a dictionary")
    for key in ("embeddings", "embedding_cache", "representations"):
        if key in payload and isinstance(payload[key], dict):
            return payload[key]
    return payload


def embedding_array(value):
    if isinstance(value, dict):
        for key in ("embedding", "embeddings", "representation", "representations", "rnafm_embedding"):
            if key in value:
                value = value[key]
                break
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def audit(root, selected, splits, manifest_path, split_path, embedding_paths, version):
    started = time.perf_counter()
    errors = []
    counts = {name: len(identifiers) for name, identifiers in splits.items()}
    for name, expected in {"fit-train": 3299, "family-dev": 1068, "VL0": 1286}.items():
        if name in counts and counts[name] != expected:
            errors.append(f"{name} count {counts[name]} != expected {expected}")
    family_sets = {}
    hash_sets = {}
    family_counts = Counter()
    for name, identifiers in splits.items():
        family_sets[name] = {selected[sample_id]["family_id"] for sample_id in identifiers if known_family(selected[sample_id]["family_id"])}
        hash_sets[name] = {selected[sample_id]["computed_sequence_hash"] for sample_id in identifiers}
        family_counts.update((name, selected[sample_id]["family_id"]) for sample_id in identifiers)
    if "family-dev" in family_sets and len(family_sets["family-dev"]) != 292:
        errors.append(f"family-dev known families {len(family_sets['family-dev'])} != expected 292")
    overlap = {}
    if "fit-train" in splits and "family-dev" in splits:
        overlap = {
            "sample_ids": sorted(set(splits["fit-train"]) & set(splits["family-dev"])),
            "family_ids": sorted(family_sets["fit-train"] & family_sets["family-dev"]),
            "sequence_hashes": sorted(hash_sets["fit-train"] & hash_sets["family-dev"]),
        }
        for category, values in overlap.items():
            if values:
                errors.append(f"fit-train/family-dev {category} overlap: {len(values)}")
    import torch
    embedding_failures = []
    embedding_hashes = {}
    for name, embedding_path in embedding_paths.items():
        payload = torch.load(embedding_path, map_location="cpu", weights_only=False)
        embeddings = embedding_mapping(payload)
        embedding_hashes[name] = file_hash(embedding_path)
        for sample_id in splits[name]:
            row = selected[sample_id]
            lookup = next((key for key in (sample_id, row.get("source_id"), row["computed_sequence_hash"]) if key in embeddings), None)
            if lookup is None:
                embedding_failures.append({"sample_id": sample_id, "split": name, "reason": "embedding missing"})
                continue
            try:
                array = embedding_array(embeddings[lookup])
                expected = (len(row["sequence"]), 1280)
                if array.shape != expected or not np.isfinite(array).all():
                    raise ValueError(f"expected finite {expected} nucleotide-only array, got {array.shape}")
            except Exception as exc:
                embedding_failures.append({"sample_id": sample_id, "split": name, "reason": str(exc)})
        del payload, embeddings
    if embedding_failures:
        errors.append(f"embedding audit failed for {len(embedding_failures)} selected RNAs")
    sys.path.insert(0, str(root))
    candidate_pairs = importlib.import_module("src.decoder").candidate_pairs
    label_totals = Counter()
    label_counts = []
    for sample_id, row in selected.items():
        sequence = row["sequence"]
        original = pairs_from_text(row["original_pairs"], len(sequence))
        eligible = pairs_from_text(row["eligible_pairs"], len(sequence))
        candidates = {tuple(map(int, pair)) for pair in candidate_pairs(sequence)}
        if not eligible.issubset(original):
            errors.append(f"{sample_id}: eligible pairs are not a subset of original truth")
        if not eligible.issubset(candidates):
            errors.append(f"{sample_id}: eligible pairs are not a subset of decoder candidates")
        unreachable = original - candidates
        label_totals.update(original_pairs=len(original), eligible_pairs=len(eligible), decoder_unreachable_original_pairs=len(unreachable))
        label_counts.append({"sample_id": sample_id, "original_pairs": len(original), "eligible_pairs": len(eligible), "decoder_unreachable_original_pairs": len(unreachable)})
    provenance_files = {}
    provenance_directory = root / "outputs/thermo_fast/provenance"
    for path in sorted((root / "outputs").rglob("*.json")):
        if path.name in {"cache_reuse.json", "environment_provenance.json"} or path.name.startswith("embedding_completion"):
            with path.open(encoding="utf-8") as handle:
                provenance_files[str(path.relative_to(root))] = {"sha256": file_hash(path), "metadata": json.load(handle)}
    report = {
        "counts": counts,
        "expected_counts": {"fit-train": 3299, "family-dev": 1068, "VL0": 1286},
        "expected_known_family_reference": 292,
        "known_family_counts": {name: len(values) for name, values in family_sets.items()},
        "known_families_selected_union": len(set().union(*family_sets.values())),
        "fit_family_overlap": overlap,
        "unique_sequences": {name: len(values) for name, values in hash_sets.items()},
        "embedding_paths": {name: str(path) for name, path in embedding_paths.items()},
        "embedding_sha256": embedding_hashes,
        "embedding_dimension": 1280,
        "embedding_special_tokens": "No CLS/EOS: every array length equals nucleotide sequence length",
        "embedding_failures": embedding_failures,
        "label_totals": dict(label_totals),
        "decoder_unreachable_original_truth_fraction": label_totals["decoder_unreachable_original_pairs"] / max(1, label_totals["original_pairs"]),
        "per_rna_label_counts": label_counts,
        "manifest_sha256": file_hash(manifest_path),
        "split_ids_sha256": file_hash(split_path),
        "code_sha256": file_hash(__file__),
        "viennarna_version": version,
        "params": PARAMS,
        "metadata_provenance": provenance_files,
        "preflight": json.loads((provenance_directory / "preflight.json").read_text()),
        "pretrained_reference_sha256": PRETRAINED_REFERENCE_SHA256,
        "pretrained_reference_hash_found_in_metadata": any(PRETRAINED_REFERENCE_SHA256 in json.dumps(value["metadata"]) for value in provenance_files.values()),
        "supervised_initialization": "No old supervised checkpoint is loaded. All ablation modes must start from the same random initialization for each seed.",
        "label_policy": "Original truth is retained unchanged; eligible pairs and decoder-unreachable original pairs are reported separately.",
        "errors": errors,
        "passed": not errors,
        "duration_seconds": time.perf_counter() - started,
    }
    write_json(provenance_directory / "split_audit.json", report)
    write_json(root / "outputs/thermo_fast/split_audit.json", report)
    write_json(provenance_directory / "selected_ids.json", splits)
    provenance_directory.mkdir(parents=True, exist_ok=True)
    with (provenance_directory / "selected_id_manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        columns = ["sample_id", "source_id", "sequence_hash", "length", "working_split", "family_id", "annotation_status", "original_pairs", "eligible_pairs"]
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for sample_id, row in selected.items():
            output = {key: row.get(key, "") for key in columns}
            output.update(sample_id=sample_id, sequence_hash=row["computed_sequence_hash"], length=len(row["sequence"]), working_split=",".join(row["selected_splits"]))
            writer.writerow(output)
    with (provenance_directory / "family_census.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["working_split", "family_id", "rna_count"])
        for (name, family), count in sorted(family_counts.items()):
            writer.writerow([name, family, count])
    if errors:
        raise RuntimeError("split audit failed: " + "; ".join(errors))
    return report


class Ep(ctypes.Structure):
    _fields_ = [("i", ctypes.c_int), ("j", ctypes.c_int), ("p", ctypes.c_float), ("type", ctypes.c_int)]


def stack_probabilities(fold_compound, cutoff, RNA):
    if hasattr(fold_compound, "stack_prob"):
        for entry in fold_compound.stack_prob(cutoff):
            if entry.i == 0:
                break
            yield int(entry.i), int(entry.j), float(entry.p)
        return
    if sys.platform != "linux":
        raise RuntimeError("ViennaRNA 2.6.x C stack-probability bridge requires Linux with the same libc allocator")
    library = ctypes.CDLL(RNA._RNA.__file__)
    compute = library.vrna_stack_prob
    compute.argtypes = [ctypes.c_void_p, ctypes.c_double]
    compute.restype = ctypes.POINTER(Ep)
    release_library = library if hasattr(library, "free") else ctypes.CDLL(None)
    release = release_library.free
    release.argtypes = [ctypes.c_void_p]
    release.restype = None
    pointer = compute(ctypes.c_void_p(int(fold_compound.this)), cutoff)
    if not pointer:
        raise RuntimeError("vrna_stack_prob returned a null pointer")
    try:
        position = 0
        maximum = fold_compound.length * (fold_compound.length - 1) // 2
        while position <= maximum:
            entry = pointer[position]
            if entry.i == 0:
                break
            yield int(entry.i), int(entry.j), float(entry.p)
            position += 1
        else:
            raise RuntimeError("vrna_stack_prob list lacks its i=0 terminator")
    finally:
        release(ctypes.cast(pointer, ctypes.c_void_p))


def valid_pair(sequence, left, right):
    return right - left >= 4 and sequence[left] + sequence[right] in CANONICAL


def validate_matrices(sequence, pairing, stacking):
    length = len(sequence)
    if pairing.shape != (length, length) or stacking.shape != (length, length):
        raise ValueError("thermodynamic matrix shape mismatch")
    if not np.isfinite(pairing).all() or not np.isfinite(stacking).all():
        raise ValueError("nonfinite thermodynamic probability")
    if min(float(pairing.min()), float(stacking.min())) < -TOLERANCE:
        raise ValueError("negative thermodynamic probability")
    if max(float(pairing.max()), float(stacking.max())) > 1.0 + TOLERANCE:
        raise ValueError("thermodynamic probability above one")
    if np.any(np.tril(pairing) != 0) or np.any(np.tril(stacking) != 0):
        raise ValueError("P and Q must be strictly upper triangular")
    for left, right in zip(*np.nonzero(pairing)):
        if not valid_pair(sequence, left, right):
            raise ValueError(f"invalid base-pair geometry at {(int(left), int(right))}")
    for left, right in zip(*np.nonzero(stacking)):
        if left + 1 >= right - 1 or not valid_pair(sequence, left, right) or not valid_pair(sequence, left + 1, right - 1):
            raise ValueError(f"invalid stack geometry at {(int(left), int(right))}")
        bound = min(pairing[left, right], pairing[left + 1, right - 1])
        if stacking[left, right] > bound + TOLERANCE:
            raise ValueError(f"Q exceeds both-pair marginal bound at {(int(left), int(right))}: {stacking[left, right]} > {bound}")
    if np.max(pairing.sum(axis=0) + pairing.sum(axis=1)) > 1.0 + TOLERANCE:
        raise ValueError("base-pair marginal row sum exceeds one")


def cache_one(task):
    sequence, identifiers, destination, cutoff, features, version = task
    started = time.perf_counter()
    result = {"sample_ids": identifiers, "sequence_hash": sequence_hash(sequence), "length": len(sequence), "cache_path": str(destination)}
    try:
        import RNA
        destination = Path(destination)
        encoded_params = json.dumps(PARAMS, sort_keys=True)
        if destination.exists():
            with np.load(destination, allow_pickle=False) as stored:
                if str(stored["sequence_hash"]) != result["sequence_hash"] or str(stored["viennarna_version"]) != version or str(stored["params_json"]) != encoded_params or float(stored["stack_cutoff"]) != cutoff or not set(features).issubset(set(stored["features"].tolist())):
                    raise ValueError("existing cache provenance differs from requested settings")
                validate_matrices(sequence, stored["P"], stored["Q"])
            result.update(status="success", reused=True, pf_calls=0, cache_sha256=file_hash(destination))
            return result
        RNA.params_load_RNA_Turner2004()
        model = RNA.md()
        model.temperature = 37.0
        model.dangles = 2
        model.noLP = 0
        model.uniq_ML = 1
        model.compute_bpp = 1
        model.min_loop_size = 3
        fold_compound = RNA.fold_compound(sequence, model)
        _, minimum_energy = fold_compound.mfe()
        if not math.isfinite(minimum_energy) or minimum_energy >= RNA.INF / 100.0:
            raise RuntimeError("MFE scaling failed")
        fold_compound.exp_params_rescale(minimum_energy)
        _, ensemble_energy = fold_compound.pf()
        if not math.isfinite(ensemble_energy) or ensemble_energy >= RNA.INF / 100.0:
            raise RuntimeError("partition function failed or returned INF/100 sentinel")
        raw = np.asarray(fold_compound.bpp(), dtype=np.float64)
        length = len(sequence)
        if raw.shape != (length + 1, length + 1):
            raise ValueError(f"bpp() shape {raw.shape} != {(length + 1, length + 1)}")
        pairing = np.triu(raw[1:, 1:], 1)
        stacking = np.zeros_like(pairing)
        if "stack" in features:
            for outer_left, outer_right, probability in stack_probabilities(fold_compound, cutoff, RNA):
                left, right = outer_left - 1, outer_right - 1
                if not 0 <= left < right < length:
                    raise ValueError(f"stack_prob returned out-of-range 1-based pair {(outer_left, outer_right)}")
                if stacking[left, right] != 0:
                    raise ValueError("stack_prob returned a duplicate outer pair")
                stacking[left, right] = probability
        validate_matrices(sequence, pairing, stacking)
        pairing = pairing.astype(np.float32)
        stacking = stacking.astype(np.float32)
        validate_matrices(sequence, pairing, stacking)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent, suffix=".npz", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, P=pairing, Q=stacking, sample_ids=np.asarray(identifiers), sequence=np.asarray(sequence), sequence_hash=np.asarray(result["sequence_hash"]), viennarna_version=np.asarray(version), params_json=np.asarray(encoded_params), stack_cutoff=np.asarray(cutoff), features=np.asarray(features), ensemble_energy=np.asarray(ensemble_energy), pf_calls=np.asarray(1))
            os.replace(temporary, destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        result.update(status="success", reused=False, pf_calls=1, ensemble_energy=float(ensemble_energy), cache_sha256=file_hash(destination))
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        result["duration_seconds"] = time.perf_counter() - started
    return result


def main():
    parser = argparse.ArgumentParser(description="Audit RNA ThermoFusion inputs and cache genuine same-PF base-pair/stack probabilities.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--split", default="fit-train,family-dev")
    parser.add_argument("--features", default="bpp,stack")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cutoff", type=float, default=1e-8)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    requested = [part.strip() for part in arguments.split.split(",") if part.strip()]
    features = [part.strip() for part in arguments.features.split(",") if part.strip()]
    if not requested or set(requested) - set(SPLIT_KEYS):
        parser.error("--split permits only fit-train,family-dev,VL0")
    if not features or set(features) - {"bpp", "stack"} or "bpp" not in features:
        parser.error("--features must include bpp; stack is optional")
    if arguments.workers < 1 or not math.isfinite(arguments.cutoff) or not 0 <= arguments.cutoff < 1:
        parser.error("workers must be positive and cutoff must be finite in [0,1)")
    import RNA
    version = str(RNA.__version__)
    if not version.startswith("2.6."):
        raise RuntimeError(f"README specifies ViennaRNA 2.6.x; installed {version}")
    manifest_path = arguments.manifest.resolve() if arguments.manifest else root / "data/processed/manifest.tsv"
    try:
        selected, splits, split_path = load_selected(root, requested, manifest_path)
        embedding_paths = find_embeddings(root, arguments.embeddings, requested)
        audit_report = audit(root, selected, splits, manifest_path, split_path, embedding_paths, version)
    except Exception as exc:
        if not str(exc).startswith("split audit failed:"):
            failure_report = {"passed": False, "errors": [f"{type(exc).__name__}: {exc}"], "requested_splits": requested, "viennarna_version": version, "params": PARAMS}
            write_json(root / "outputs/thermo_fast/provenance/split_audit.json", failure_report)
            write_json(root / "outputs/thermo_fast/split_audit.json", failure_report)
        raise
    if arguments.audit_only:
        print(json.dumps({"audit": "passed", "counts": audit_report["counts"]}))
        return 0
    started = time.perf_counter()
    cache_directory = root / "outputs/thermo_fast/thermo_cache"
    grouped = defaultdict(list)
    for sample_id, row in selected.items():
        grouped[row["sequence"]].append(sample_id)
    tasks = [(sequence, identifiers, cache_directory / f"{sequence_hash(sequence)}.npz", arguments.cutoff, features, version) for sequence, identifiers in grouped.items()]
    successes = []
    failures = []
    with multiprocessing.get_context("spawn").Pool(arguments.workers) as pool:
        for result in pool.imap_unordered(cache_one, tasks, chunksize=1):
            (successes if result["status"] == "success" else failures).append(result)
            if result["status"] == "failed":
                print(json.dumps(result), file=sys.stderr, flush=True)
            elif (len(successes) + len(failures)) % 100 == 0:
                print(f"thermo cache: {len(successes)} successful sequences, {len(failures)} failed", flush=True)
    mapping = {}
    for result in successes + failures:
        for sample_id in result["sample_ids"]:
            mapping[sample_id] = {**result, "cache_path": str(Path(result["cache_path"]).relative_to(root)), "splits": selected[sample_id]["selected_splits"], "split": ",".join(selected[sample_id]["selected_splits"])}
    shared_cohort = {name: [sample_id for sample_id in identifiers if mapping[sample_id]["status"] == "success"] for name, identifiers in splits.items()}
    manifest = {
        "viennarna_version": version,
        "params": PARAMS,
        "features": features,
        "stack_cutoff": arguments.cutoff,
        "matrix_layout": "P and Q are L x L float32 strictly upper triangular; Q[i,j] is joint outer (i,j) and inner (i+1,j-1) probability, zero-based",
        "stack_api": "native fold_compound.stack_prob when available, otherwise same-library C vrna_stack_prob bridge",
        "stack_api_source": "https://github.com/ViennaRNA/ViennaRNA/blob/v2.6.4/src/ViennaRNA/equilibrium_probs.c",
        "bpp_api_source": "https://github.com/ViennaRNA/ViennaRNA/blob/v2.6.4/interfaces/part_func.i",
        "successes": sorted(successes, key=lambda value: value["sequence_hash"]),
        "failures": sorted(failures, key=lambda value: value["sequence_hash"]),
        "samples": mapping,
        "coverage": {name: {"requested": len(identifiers), "successful": len(shared_cohort[name]), "failed": len(identifiers) - len(shared_cohort[name])} for name, identifiers in splits.items()},
        "shared_successful_cohort": shared_cohort,
        "cohort_policy": "Every training and evaluation mode uses exactly these same successful RNA IDs; no mode-specific sample dropping.",
        "label_policy": audit_report["label_policy"],
        "initialization_policy": audit_report["supervised_initialization"],
        "code_sha256": file_hash(__file__),
        "input_manifest_sha256": audit_report["manifest_sha256"],
        "split_ids_sha256": audit_report["split_ids_sha256"],
        "embedding_sha256": audit_report["embedding_sha256"],
        "pretrained_reference_sha256": audit_report["pretrained_reference_sha256"],
        "pretrained_reference_hash_found_in_metadata": audit_report["pretrained_reference_hash_found_in_metadata"],
        "duration_seconds": time.perf_counter() - started,
    }
    write_json(cache_directory / "manifest.json", manifest)
    write_json(root / "outputs/thermo_fast/thermo_cache_manifest.json", manifest)
    write_json(root / "outputs/thermo_fast/provenance/shared_successful_ids.json", shared_cohort)
    print(json.dumps({"successful_unique_sequences": len(successes), "failed_unique_sequences": len(failures), "coverage": manifest["coverage"]}))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
