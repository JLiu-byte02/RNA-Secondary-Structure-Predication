#!/bin/bash
#SBATCH --job-name=da-utility-cpu
#SBATCH --partition=c-80-255
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/da-utility-cpu-%j.out
#SBATCH --error=logs/da-utility-cpu-%j.err
set -euo pipefail
source activate.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 RNA_DIRECTION_A_DEVICE=cpu
python -u scripts/run_direction_a_utility.py "$@"
