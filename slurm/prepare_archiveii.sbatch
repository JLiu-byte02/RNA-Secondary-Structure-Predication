#!/bin/bash
#SBATCH --job-name=archiveii-prepare
#SBATCH --partition=g-3090-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/archiveii-prepare-%j.out
#SBATCH --error=logs/archiveii-prepare-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ./activate.sh
PYTHON=.venv_archiveii/bin/python
"$PYTHON" setup/check_archiveii_environment.py
"$PYTHON" tests/test_archiveii_metrics.py
"$PYTHON" scripts/archiveii_data.py --cache --workers 8
"$PYTHON" -u scripts/smoke_archiveii.py
"$PYTHON" -u tests/test_archiveii_training.py
