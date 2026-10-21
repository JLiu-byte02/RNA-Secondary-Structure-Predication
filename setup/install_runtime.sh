#!/bin/bash
set -euo pipefail
PROJECT_DIR="$HOME/projects/RNA_ThermoFusion"
PY="$PROJECT_DIR/.venv/bin/python"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
trap 'rc=$?; printf "%s\n" "$rc" > "$PROJECT_DIR/logs/install.exit"' EXIT
"$PY" -m pip install --disable-pip-version-check --progress-bar off 'pip==24.3.1' 'setuptools==75.6.0' 'wheel==0.45.1'
printf '\nSTAGE scientific_libraries\n'
"$PY" -m pip install --disable-pip-version-check --progress-bar off --only-binary=:all: -c "$PROJECT_DIR/setup/constraints.txt" -r "$PROJECT_DIR/setup/requirements.txt"
printf '\nSTAGE pytorch_cuda118\n'
"$PY" -m pip install --disable-pip-version-check --progress-bar off --only-binary=:all: 'torch==2.1.2' --index-url https://download.pytorch.org/whl/cu118 -c "$PROJECT_DIR/setup/constraints.txt"
printf '\nSTAGE flash_attention\n'
"$PY" -m pip install --disable-pip-version-check --progress-bar off --no-deps "$PROJECT_DIR/setup/wheels/flash_attn-2.3.2+cu118torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
printf '\nSTAGE rinalmo\n'
RINALMO_COMMIT=$(tr -d '\r\n' < "$PROJECT_DIR/setup/rinalmo_commit.txt")
if [ ! -d "$PROJECT_DIR/vendor/RiNALMo/.git" ]; then
    git clone --no-checkout https://github.com/lbcb-sci/RiNALMo.git "$PROJECT_DIR/vendor/RiNALMo"
fi
git --git-dir="$PROJECT_DIR/vendor/RiNALMo/.git" --work-tree="$PROJECT_DIR/vendor/RiNALMo" fetch origin "$RINALMO_COMMIT"
git --git-dir="$PROJECT_DIR/vendor/RiNALMo/.git" --work-tree="$PROJECT_DIR/vendor/RiNALMo" checkout --detach "$RINALMO_COMMIT"
"$PY" -m pip install --disable-pip-version-check --progress-bar off --no-deps --no-build-isolation "$PROJECT_DIR/vendor/RiNALMo"
"$PY" -m pip check
"$PY" -m pip freeze > "$PROJECT_DIR/setup/requirements.lock.txt"
printf '\nINSTALL_COMPLETE\n'
