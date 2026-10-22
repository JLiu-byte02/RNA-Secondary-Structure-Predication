#!/bin/bash
#SBATCH --job-name=arch-frozen-embed
#SBATCH --partition=g-3090-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/archiveii-frozen-embed-%j.out
#SBATCH --error=logs/archiveii-frozen-embed-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ./activate.sh
python -u scripts/prepare_archiveii_frozen.py
