#!/bin/bash
#SBATCH --job-name=da-stage2-cpu
#SBATCH --partition=c-80-255
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/da-stage2-%j.out
#SBATCH --error=logs/da-stage2-%j.err
set -euo pipefail
source activate.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u scripts/run_direction_a_matched.py "$@"
