#!/usr/bin/env python
"""Generate ViennaRNA thermodynamic features (BPP, MFE, ensemble) for canonical datasets.

Input : one or more .jsonl.gz files (records with id/sequence) or a directory of them.
Output: per-sequence npz cache keyed by sha256(sequence), plus manifest.tsv.

Usage (server, env with ViennaRNA 2.6.x):
  python scripts/cache_thermo.py --inputs data/datasets/official --out data/thermo/official \
      --workers 32 --min-len 10 --max-len 4000
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import RNA
except ImportError:  # pragma: no cover
    RNA = None

TEMPERATURE = 37.0
DANGLES = 2
NO_LP = False
UNIQ_ML = True


def seq_hash(seq: str) -> str:
    return hashlib.sha256(seq.encode("ascii")).hexdigest()


def iter_records(paths):
    seen = set()
    for path in paths:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                h = seq_hash(rec["sequence"])
                if h in seen:
                    continue
                seen.add(h)
                yield h, rec["sequence"]
    return seen


def compute_one(task):
    h, seq, out_dir = task
    t0 = time.time()
    try:
        md = RNA.md()
        md.temperature = TEMPERATURE
        md.dangles = DANGLES
        md.noLP = 0 if not NO_LP else 1
        md.uniq_ML = 1 if UNIQ_ML else 0
        fc = RNA.fold_compound(seq, md)
        mfe_dbn, mfe = fc.mfe()
        fc.exp_params_rescale(mfe)
        pf_dbn, pf_energy = fc.pf()
        bpp = np.asarray(fc.bpp(), dtype=np.float32)  # (L+1, L+1), 1-based boundary
        if bpp.shape != (len(seq) + 1, len(seq) + 1):
            raise RuntimeError(f"unexpected bpp shape {bpp.shape} for L={len(seq)}")
        bpp = bpp[1:, 1:].astype(np.float16)
        centroid_dbn, centroid_energy = fc.centroid()
        mea_dbn, mea_score = fc.MEA(1.0)
        out = dict(
            sha256=h,
            length=len(seq),
            mfe_dbn=mfe_dbn,
            mfe_energy=float(mfe),
            pf_energy=float(pf_energy),
            centroid_dbn=centroid_dbn,
            centroid_energy=float(centroid_energy),
            mea_dbn=mea_dbn,
            mea_score=float(mea_score),
            bpp=bpp,
        )
        target = out_dir / f"{h}.npz"
        tmp = target.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **out)
        os.replace(tmp, target)
        return dict(sha256=h, length=len(seq), ok=True, seconds=round(time.time() - t0, 3),
                    path=str(target.relative_to(out_dir)))
    except Exception as exc:  # noqa: BLE001
        return dict(sha256=h, length=len(seq), ok=False, seconds=round(time.time() - t0, 3),
                    error=f"{type(exc).__name__}: {exc}")


def collect_paths(inputs):
    paths = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.jsonl.gz")) + sorted(p.rglob("*.jsonl")))
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(p)
    if not paths:
        raise SystemExit("no input jsonl files found")
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-len", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N sequences")
    args = ap.parse_args()
    if RNA is None:
        raise SystemExit("ViennaRNA (RNA) not importable in this env")

    paths = collect_paths(args.inputs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    skipped = 0
    for h, seq in iter_records(paths):
        if not (args.min_len <= len(seq) <= args.max_len):
            skipped += 1
            continue
        records.append((h, seq))
        if args.limit and len(records) >= args.limit:
            break
    print(f"inputs={len(paths)} unique sequences={len(records)} skipped_by_length={skipped}", flush=True)

    todo = [(h, seq, out_dir) for h, seq in records
            if not (out_dir / f"{h}.npz").exists()]
    print(f"already_cached={len(records) - len(todo)} pending={len(todo)}", flush=True)

    t0 = time.time()
    results = []
    if args.workers > 1 and len(todo) > 1:
        ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(compute_one, todo, chunksize=8), 1):
                results.append(res)
                if i % 200 == 0 or i == len(todo):
                    ok = sum(r["ok"] for r in results)
                    print(f"progress {i}/{len(todo)} ok={ok} elapsed={time.time()-t0:.0f}s", flush=True)
    else:
        for i, task in enumerate(todo, 1):
            results.append(compute_one(task))
            if i % 50 == 0 or i == len(todo):
                print(f"progress {i}/{len(todo)} elapsed={time.time()-t0:.0f}s", flush=True)

    manifest = out_dir / "manifest.tsv"
    with manifest.open("w", encoding="utf-8") as fh:
        fh.write("sha256\tlength\tok\tpath\tseconds\terror\n")
        for r in results:
            fh.write(f"{r['sha256']}\t{r['length']}\t{int(r['ok'])}\t{r.get('path','')}\t{r['seconds']}\t{r.get('error','')}\n")

    failures = [r for r in results if not r["ok"]]
    summary = dict(
        inputs=[str(p) for p in paths], n_unique=len(records), n_new=len(todo),
        n_ok=sum(r["ok"] for r in results), n_failed=len(failures),
        skipped_by_length=skipped, workers=args.workers,
        temperature=TEMPERATURE, dangles=DANGLES, noLP=NO_LP, uniq_ML=UNIQ_ML,
        seconds=round(time.time() - t0, 1),
        failures=failures[:50],
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_unique", "n_new", "n_ok", "n_failed", "seconds")}, indent=2))
    if failures:
        print(f"FAILED {len(failures)} sequences (first: {failures[0]})", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
