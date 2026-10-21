#!/bin/bash
#SBATCH --job-name=long-prepare
#SBATCH --partition=g-3090-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/long-prepare-%j.out
#SBATCH --error=logs/long-prepare-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ./activate.sh
python tests/test_long_protocol.py
python tests/test_long_models.py
for mode in B0 E0_no_thermo E1_bpp E2_bpp_stack; do
  python -u scripts/run_long_epochs.py --mode "$mode" --seed 17 --smoke
done
python - <<'PY'
import json
from pathlib import Path
out=Path('outputs/long_epochs')
rows=[json.loads((out/'smoke'/m/'seed17/summary.json').read_text()) for m in ('B0','E0_no_thermo','E1_bpp','E2_bpp_stack')]
assert len({r['head_initialization_hash'] for r in rows})==1
assert all(r['gradients_verified'] and r['encoder_frozen'] and r['epochs_completed']==2 for r in rows)
(out/'preparation_complete.json').write_text(json.dumps(dict(passed=True,runs=rows),indent=2))
PY
