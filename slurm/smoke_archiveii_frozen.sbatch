#!/bin/bash
#SBATCH --job-name=arch-frozen-smoke
#SBATCH --partition=g-3090-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/archiveii-frozen-smoke-%j.out
#SBATCH --error=logs/archiveii-frozen-smoke-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ./activate.sh
python tests/test_archiveii_frozen.py
python tests/test_long_models.py
for mode in E0_no_thermo E1_bpp E2_bpp_stack; do
  python -u scripts/run_archiveii_frozen.py --family 5s --mode "$mode" --seed 17 --smoke
done
python scripts/smoke_archiveii_frozen.py
