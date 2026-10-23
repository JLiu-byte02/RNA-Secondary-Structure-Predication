#!/bin/bash
#SBATCH --job-name=da-base
#SBATCH --partition=g-3090-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/da-base-%j.out
#SBATCH --error=logs/da-base-%j.err
set -euo pipefail
source activate.sh
export TMPDIR="$PWD/.tmp"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
if [[ " $* " == *" --smoke "* ]]; then python scripts/smoke_direction_a.py; fi
python scripts/run_direction_a_base.py "$@"
