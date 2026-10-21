#!/bin/bash
#SBATCH --job-name=long-train
#SBATCH --partition=g-3090-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/long-train-%j.out
#SBATCH --error=logs/long-train-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ./activate.sh
python -u scripts/run_long_epochs.py --mode "$1" --seed "$2"
