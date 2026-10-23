#!/bin/bash
#SBATCH --job-name=da-matched-cache
#SBATCH --partition=g-3090-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=logs/da-matched-%j.out
#SBATCH --error=logs/da-matched-%j.err
set -euo pipefail
source activate.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -u scripts/run_direction_a_matched.py cache "$@"
