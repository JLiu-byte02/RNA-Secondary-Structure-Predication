#!/bin/bash
#SBATCH --job-name=archiveii-trainer-check
#SBATCH --partition=g-3090-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/archiveii-trainer-check-%j.out
#SBATCH --error=logs/archiveii-trainer-check-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ./activate.sh
.venv_archiveii/bin/python setup/check_archiveii_environment.py
.venv_archiveii/bin/python -u tests/test_archiveii_training.py
