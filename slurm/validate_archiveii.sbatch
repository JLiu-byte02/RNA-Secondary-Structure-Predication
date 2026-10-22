#!/bin/bash
#SBATCH --job-name=archiveii-validate
#SBATCH --partition=g-3090-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/archiveii-validate-%j.out
#SBATCH --error=logs/archiveii-validate-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ./activate.sh
.venv_archiveii/bin/python -u scripts/report_archiveii.py --validate
