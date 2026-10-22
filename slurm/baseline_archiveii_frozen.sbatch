#!/bin/bash
#SBATCH --job-name=arch-frozen-baseline
#SBATCH --partition=g-3090-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/archiveii-frozen-baseline-%j.out
#SBATCH --error=logs/archiveii-frozen-baseline-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ./activate.sh
python -u scripts/eval_archiveii_frozen.py --family "$1" --baseline
