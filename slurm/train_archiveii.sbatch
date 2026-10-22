#!/bin/bash
#SBATCH --job-name=archiveii-train
#SBATCH --partition=g-3090-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/archiveii-train-%j.out
#SBATCH --error=logs/archiveii-train-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ./activate.sh
.venv_archiveii/bin/python -u scripts/run_archiveii.py --family "$1" --mode "$2" --seed "$3"
