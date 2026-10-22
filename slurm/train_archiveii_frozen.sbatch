#!/bin/bash
#SBATCH --job-name=arch-frozen-train
#SBATCH --partition=g-3090-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/archiveii-frozen-train-%j.out
#SBATCH --error=logs/archiveii-frozen-train-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ./activate.sh
python -u scripts/run_archiveii_frozen.py --family "$1" --mode "$2" --seed "$3"
