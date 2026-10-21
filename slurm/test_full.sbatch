#!/bin/bash
#SBATCH --job-name=full-test
#SBATCH --partition=g-3090-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/full-test-%j.out
#SBATCH --error=logs/full-test-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ./activate.sh
python -u scripts/eval_full_tr0.py --mode "$1" --seed "$2"
