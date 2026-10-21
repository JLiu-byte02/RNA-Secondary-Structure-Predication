#!/bin/bash
#SBATCH --job-name=thermo-train
#SBATCH --partition=g-3090-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ./activate.sh
python -u scripts/run_thermo_fast.py --mode "$1" --seed "$2"
