#!/usr/bin/env python
"""Single-sequence inference: FASTA / raw sequence -> dot-bracket + pairs.

Thin wrapper around the tested batch pipeline (`scripts/cache_thermo.py` for the
ViennaRNA BPP cache and `scripts/evaluate.py` for model inference + decoding),
so the decoding rule is exactly the one used in all reported numbers.

Examples
--------
    source activate.sh
    python scripts/predict.py --sequence GGGCUUCGGCCC \\
        --checkpoint checkpoints/soup_tRNA.pt --out out/query
    python scripts/predict.py --fasta queries.fa \\
        --checkpoint checkpoints/soup_tRNA.pt --out out/queries

The checkpoint must be one of ours (``best.pt`` / ``soup.pt`` / ``epochN.pt``);
official Zenodo checkpoints use ``scripts/eval_official.py`` instead.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def read_sequences(args) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    if args.sequence:
        seq = "".join(args.sequence.split()).upper().replace("T", "U")
        records.append(("query", seq))
    if args.fasta:
        name, chunks = None, []
        for line in Path(args.fasta).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(chunks).upper().replace("T", "U")))
                name = line[1:].split()[0] or f"seq{len(records)}"
                chunks = []
            else:
                chunks.append(line)
        if name is not None:
            records.append((name, "".join(chunks).upper().replace("T", "U")))
    if not records:
        raise SystemExit("provide --sequence or --fasta")
    return records


def dot_bracket(length: int, pairs) -> str:
    chars = ["."] * length
    for i, j in pairs:
        i, j = int(i), int(j)
        if 0 <= i < length and 0 <= j < length:
            chars[i] = "("
            chars[j] = ")"
    return "".join(chars)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", default=None, help="raw RNA sequence (T accepted)")
    ap.add_argument("--fasta", default=None, help="FASTA file with one or more sequences")
    ap.add_argument("--checkpoint", required=True, help="our checkpoint (.pt)")
    ap.add_argument("--config", default=None, help="optional training YAML override")
    ap.add_argument("--out", default="out/predict", help="output directory")
    ap.add_argument("--threshold", type=float, default=None, help="default: checkpoint threshold")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--thermo-dir", default=None, help="override data.thermo_dir from the checkpoint")
    args = ap.parse_args()

    records = read_sequences(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="rna_predict_"))
    jsonl = tmp / "query.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for name, seq in records:
            handle.write(json.dumps({
                "id": name, "family": "query", "length": len(seq),
                "pairs": [], "sequence": seq, "source_id": f"query:{name}", "split": "query",
            }) + "\n")

    # 1) physics cache for the query sequences
    thermo_dir = args.thermo_dir
    cache_cmd = [PY, str(ROOT / "scripts" / "cache_thermo.py"), "--inputs", str(jsonl)]
    if thermo_dir:
        cache_cmd += ["--out", thermo_dir]
    else:
        # infer the thermo dir from the checkpoint config via evaluate.py's resolver
        cache_cmd += ["--out", str(ROOT / "data" / "thermo" / "all")]
    subprocess.run(cache_cmd, check=True)

    # 2) model inference (batch pipeline, includes canonical + sharp-loop decoding)
    eval_cmd = [PY, str(ROOT / "scripts" / "evaluate.py"),
                "--checkpoint", args.checkpoint,
                "--split-file", f"query={jsonl}",
                "--out", str(tmp / "eval"), "--device", args.device]
    if args.config:
        eval_cmd += ["--config", args.config]
    if args.threshold is not None:
        eval_cmd += ["--threshold", str(args.threshold)]
    if thermo_dir:
        eval_cmd += ["--thermo-dir", thermo_dir]
    subprocess.run(eval_cmd, check=True)

    # 3) collect dot-bracket + pairs
    pred = tmp / "eval" / "predictions_query.jsonl"
    results = []
    with pred.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            seq = next(s for n, s in records if n == row["id"])
            results.append({
                "id": row["id"], "length": row["length"], "threshold": row["threshold"],
                "sequence": seq, "dot_bracket": dot_bracket(row["length"], row["predicted_pairs"]),
                "n_pairs": len(row["predicted_pairs"]),
                "pairs": row["predicted_pairs"],
            })
    (out_dir / "predictions.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "predictions.txt").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(f'>{row["id"]} len={row["length"]} threshold={row["threshold"]}\n')
            handle.write(row["sequence"] + "\n")
            handle.write(row["dot_bracket"] + "\n")
    for row in results:
        print(f'>{row["id"]} len={row["length"]} n_pairs={row["n_pairs"]}')
        print(row["dot_bracket"])
    print(f"wrote {out_dir/'predictions.txt'} and {out_dir/'predictions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
