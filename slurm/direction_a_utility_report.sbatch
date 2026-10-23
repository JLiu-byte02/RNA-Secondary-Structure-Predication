#!/bin/bash
#SBATCH --job-name=da-utility-report
#SBATCH --partition=c-80-255
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/da-utility-report-%j.out
#SBATCH --error=logs/da-utility-report-%j.err
set -euo pipefail
source activate.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u scripts/run_direction_a_utility.py report
