#!/bin/bash
#SBATCH --job-name=da-matched-audit
#SBATCH --partition=c-80-255
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/da-matched-audit-%j.out
#SBATCH --error=logs/da-matched-audit-%j.err
set -euo pipefail
source activate.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u scripts/audit_direction_a_matched.py
