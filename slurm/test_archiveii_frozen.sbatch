#!/bin/bash
#SBATCH --job-name=arch-frozen-test
#SBATCH --partition=g-3090-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/archiveii-frozen-test-%j.out
#SBATCH --error=logs/archiveii-frozen-test-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ./activate.sh
for seed in 17 23 41; do
  for mode in E0_no_thermo E1_bpp E2_bpp_stack; do
    python -u scripts/eval_archiveii_frozen.py --family "$1" --mode "$mode" --seed "$seed"
  done
done
