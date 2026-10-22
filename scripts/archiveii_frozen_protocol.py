"""Frozen-E ArchiveII experiment identity and validation-only selection."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODES=('E0_no_thermo','E1_bpp','E2_bpp_stack')
SEEDS=(17,23,41)
FAMILIES=('5s','16s','23s','tRNA','srp','tmRNA','RNaseP','grp1','telomerase')
PRETRAINED_SHA='cd93c3f21eb3e767373c9491192686b5846247bd1110693e453c1dd0f321c0db'

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def validate_config(config):
    expected=dict(protocol='ARCHIVEII-FROZEN-E-LOFO-v2',families=list(FAMILIES),seeds=list(SEEDS),
        modes=list(MODES),max_epochs=50,early_stopping=False,initial_lr=.0001,
        weight_decay=.0001,batch_size=1,gradient_accumulation=4,clip_grad_norm=1.,optimizer='AdamW',
        encoder='public_generic_frozen_RiNALMo_Giga',thresholds=[i/10 for i in range(1,10)])
    for key,value in expected.items():
        if config.get(key)!=value:raise ValueError('Frozen-E protocol changed: '+key)

def select_epoch(rows,budget=50):
    if [r['epoch'] for r in rows]!=list(range(1,len(rows)+1)) or len(rows)!=budget:
        raise ValueError('Require the complete fixed training trajectory')
    return max(rows,key=lambda row:row['score'])

def shared_threshold(curves):
    if len(curves)!=3 or any(set(c)!=set(curves[0]) for c in curves):
        raise ValueError('Require all three paired validation curves')
    return max(sorted(curves[0]),key=lambda t:sum(c[t] for c in curves)/3)

def code_identity():
    names=['scripts/'+name+'.py' for name in (
        'archiveii_frozen_protocol','archiveii_frozen_data','archiveii_frozen_metrics',
        'prepare_archiveii_frozen','run_archiveii_frozen',
        '_common','run_thermo_fast','cache_thermo','run_long_epochs','long_models','long_metrics',
        'long_protocol','full_protocol')]
    names+=['configs/archiveii_frozen.yaml','third_party/rinalmo_frozen_reference/sec_struct.py']
    names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'src').glob('*.py'))]
    files={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    return dict(sha256=digest(files),files=files)
