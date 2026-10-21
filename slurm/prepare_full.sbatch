#!/bin/bash
#SBATCH --job-name=full-prepare
#SBATCH --partition=g-3090-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/full-prepare-%j.out
#SBATCH --error=logs/full-prepare-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ./activate.sh
python -u tests/test_full_protocol.py
python -u tests/test_contract.py
python -u tests/test_portability.py
python -u scripts/prepare_full_embeddings.py
python -u scripts/prepare_full_cache.py
for mode in E0_no_thermo E1_bpp E2_bpp_stack; do
  python -u scripts/run_full_tr0.py --mode "$mode" --seed 17 --smoke
done
python - <<'PY'
import json
from pathlib import Path
out=Path('outputs/full_tr0')
runs=[json.loads((out/'smoke'/m/'seed17/summary.json').read_text()) for m in ('E0_no_thermo','E1_bpp','E2_bpp_stack')]
assert len({r['initialization_hash'] for r in runs})==1
assert all(r['gradients_verified'] and r['encoder_frozen'] for r in runs)
(out/'preparation_complete.json').write_text(json.dumps(dict(passed=True,smoke_runs=runs),indent=2))
PY
