#!/bin/bash
#SBATCH --job-name=da-diagnostics
#SBATCH --partition=c-80-255
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/da-diagnostics-%j.out
#SBATCH --error=logs/da-diagnostics-%j.err
set -euo pipefail
source activate.sh
export TMPDIR="$PWD/.tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/analyze_direction_a_errors.py
if [[ "${1:-}" != "--analysis-only" ]]; then
  python scripts/plot_direction_a.py
  python scripts/plot_direction_a_risk.py
fi
