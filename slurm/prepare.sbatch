#!/bin/bash
#SBATCH --job-name=thermo-prepare
#SBATCH --partition=g-3090-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/prepare-%j.out
#SBATCH --error=logs/prepare-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ./activate.sh
mkdir -p tmp .tmp outputs/thermo_fast/provenance
python - <<'PY'
import hashlib,json,os,sys
from pathlib import Path
root=Path.cwd();out=root/'outputs/thermo_fast'
files=[p for folder in ('src','scripts') for p in (root/folder).glob('*.py')]
files.append(root/'configs/thermo_fast.yaml')
code={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(out/'provenance/implementation_manifest.json').write_text(json.dumps(dict(code_sha256=code,
 adaptation='Relocated original FAST-v1 source; unchanged B0 modules verified against source_snapshot.json.',
 frozen_encoder='Fresh official generic pretrained RiNALMo-Giga, no supervised weights.',
 selection='family-dev only; full original truth; identical random initialization per seed.'),indent=2))
gpu=json.loads((root/'setup/gpu_validation.json').read_text())
(out/'environment_audit.json').write_text(json.dumps(dict(python=sys.version,executable=sys.executable,
 project_realpath=str(root),gpu_validation_record=gpu,slurm_job_id=os.environ.get('SLURM_JOB_ID')),indent=2))
PY
python -u scripts/prepare_embeddings.py
python -u scripts/preflight.py
python -u tests/test_contract.py
python -u tests/test_portability.py
python -u scripts/cache_thermo.py --split fit-train,family-dev,VL0 --features bpp,stack --workers 8
for mode in E0_no_thermo E1_bpp E2_bpp_stack; do
  python -u scripts/run_thermo_fast.py --mode "$mode" --seed 17 --smoke
done
python - <<'PY'
import json
from pathlib import Path
r=Path('outputs/thermo_fast')
v=[json.loads((r/'smoke'/m/'seed17/summary.json').read_text()) for m in ('E0_no_thermo','E1_bpp','E2_bpp_stack')]
assert len({x['initialization_hash'] for x in v})==1
assert all(x['gradients_verified'] for x in v)
(r/'smoke_summary.json').write_text(json.dumps(dict(passed=True,runs=v),indent=2))
PY
echo PREPARATION_COMPLETE
