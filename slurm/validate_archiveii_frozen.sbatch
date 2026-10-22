#!/bin/bash
#SBATCH --job-name=arch-frozen-verify
#SBATCH --partition=g-3090-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/archiveii-frozen-verify-%j.out
#SBATCH --error=logs/archiveii-frozen-verify-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ./activate.sh
python -u scripts/eval_archiveii_frozen.py --finalize
