"""Prospective roles, identities and resource decisions for Direction A."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/direction_a_v1'
CONFIG=ROOT/'configs/direction_a.json'
FAMILIES=('5s','16s','23s','tRNA','srp','tmRNA','RNaseP','grp1','telomerase')

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def config():return json.loads(CONFIG.read_text())
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def family_blocks(families):
    ordered=sorted(set(families),key=lambda f:(digest(f),f))
    assert len(ordered)>=2
    return ordered[::2],ordered[1::2]

def group_calibration(records,ids,fraction):
    assert 0<fraction<1 and len(ids)==len(set(ids))
    groups={}
    for sid in sorted(ids):
        row=records[sid];sequence=row['sequence'].upper().replace('T','U')
        entry=groups.setdefault(sequence,dict(family=row['family'],ids=[]))
        if entry['family']!=row['family']:raise ValueError('Cross-family identical sequence')
        entry['ids'].append(sid)
    fit=[];cal=[]
    for family in sorted({r['family'] for r in groups.values()}):
        selected=sorted((seq for seq,row in groups.items() if row['family']==family),key=lambda seq:(digest(seq),seq))
        if len(selected)<2:raise ValueError('Need at least two unique sequences per calibration family')
        n=min(len(selected)-1,max(1,math.ceil(len(selected)*fraction)))
        for index,seq in enumerate(selected):(cal if index<n else fit).extend(groups[seq]['ids'])
    return sorted(fit),sorted(cal)

def pilot_roles(records,allowed_ids,outer,development):
    assert development!=outer
    allowed=sorted(s for s in allowed_ids if records[s]['family']!=outer)
    a,b=family_blocks({records[s]['family'] for s in allowed}-{development})
    roles=dict(outer=outer,development_family=development,
        block_a=sorted(s for s in allowed if records[s]['family'] in a),
        block_b=sorted(s for s in allowed if records[s]['family'] in b),
        development=sorted(s for s in allowed if records[s]['family']==development))
    roles['correction_fit'],roles['calibration']=group_calibration(records,roles['block_a']+roles['block_b'],.2)
    validate_roles(records,roles);return roles

def validate_roles(records,roles):
    a,b,dev=(set(roles[k]) for k in ('block_a','block_b','development'))
    assert not(a&b or a&dev or b&dev)
    fa,fb,fd=({records[s]['family'] for s in ids} for ids in (a,b,dev))
    assert not(fa&fb or fa&fd or fb&fd)
    assert roles['outer'] not in fa|fb|fd
    assert fd=={roles['development_family']}
    fit,cal=(set(roles[k]) for k in ('correction_fit','calibration'))
    assert fit|cal==a|b and not fit&cal
    assert not({records[s]['sequence'] for s in fit}&{records[s]['sequence'] for s in cal})
    return True

def source_identity(stage):
    files=['scripts/direction_a_protocol.py','scripts/direction_a_data.py','configs/direction_a.json',
           'scripts/archiveii_frozen_data.py','scripts/run_archiveii_frozen.py','scripts/run_thermo_fast.py',
           'scripts/run_long_epochs.py','scripts/long_models.py','scripts/long_metrics.py',
           'scripts/_common.py','scripts/cache_thermo.py','scripts/full_protocol.py','scripts/long_protocol.py']
    files += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'src').glob('*.py'))]
    if stage=='base':files+=['scripts/run_direction_a_base.py']
    elif stage=='correction':files+=['scripts/direction_a_models.py','scripts/run_direction_a_correction.py','scripts/direction_a_evaluate.py','scripts/archiveii_frozen_metrics.py']
    else:raise ValueError(stage)
    hashes={name:file_sha(ROOT/name) for name in sorted(set(files))}
    return dict(stage=stage,files=hashes,sha256=digest(hashes))

def cache_key(sid):return hashlib.sha256(sid.encode()).hexdigest()
