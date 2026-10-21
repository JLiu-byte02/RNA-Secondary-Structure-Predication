#!/bin/bash
set -euo pipefail
PROJECT_DIR="$HOME/projects/RNA_ThermoFusion"
PY="$PROJECT_DIR/.venv/bin/python"
WHEEL="$PROJECT_DIR/setup/wheels/torch-2.1.2+cu118-cp311-cp311-linux_x86_64.whl"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
trap 'rc=$?; printf "%s\n" "$rc" > "$PROJECT_DIR/logs/finish_install.exit"' EXIT
if [ ! -f "$WHEEL" ]; then
    "$PY" "$PROJECT_DIR/setup/download_torch.py"
fi
(cd "$PROJECT_DIR/setup/wheels" && sha256sum -c ../torch_wheel_sha256.txt)
printf '\nSTAGE install_torch\n'
"$PY" -m pip install --disable-pip-version-check --progress-bar off --only-binary=:all: "$WHEEL" --index-url https://download.pytorch.org/whl/cu118 -c "$PROJECT_DIR/setup/constraints.txt"
printf '\nSTAGE flash_attention\n'
"$PY" -m pip install --disable-pip-version-check --progress-bar off --no-deps "$PROJECT_DIR/setup/wheels/flash_attn-2.3.2+cu118torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
printf '\nSTAGE rinalmo\n'
RINALMO_COMMIT=$(tr -d '\r\n' < "$PROJECT_DIR/setup/rinalmo_commit.txt")
if [ ! -d "$PROJECT_DIR/vendor/RiNALMo/.git" ]; then git clone --no-checkout https://github.com/lbcb-sci/RiNALMo.git "$PROJECT_DIR/vendor/RiNALMo"; fi
git --git-dir="$PROJECT_DIR/vendor/RiNALMo/.git" --work-tree="$PROJECT_DIR/vendor/RiNALMo" fetch origin "$RINALMO_COMMIT"
git --git-dir="$PROJECT_DIR/vendor/RiNALMo/.git" --work-tree="$PROJECT_DIR/vendor/RiNALMo" checkout --detach "$RINALMO_COMMIT"
"$PY" -m pip install --disable-pip-version-check --progress-bar off --no-deps --no-build-isolation "$PROJECT_DIR/vendor/RiNALMo"
"$PY" -m pip check
"$PY" -m pip freeze > "$PROJECT_DIR/setup/requirements.lock.txt"
source "$PROJECT_DIR/activate.sh"
"$PY" "$PROJECT_DIR/setup/check_environment.py" > "$PROJECT_DIR/logs/environment_validation.log" 2>&1
printf '\nINSTALL_AND_CPU_VALIDATION_COMPLETE\n'
