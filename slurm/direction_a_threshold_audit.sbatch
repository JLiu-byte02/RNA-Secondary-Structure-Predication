#!/bin/bash
#SBATCH --job-name=da-threshold-audit
#SBATCH --partition=c-80-255
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/da-threshold-%j.out
#SBATCH --error=logs/da-threshold-%j.err
set -euo pipefail
source activate.sh
export TMPDIR="$PWD/.tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/audit_direction_a_thresholds.py
