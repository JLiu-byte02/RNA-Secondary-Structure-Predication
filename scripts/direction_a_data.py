"""Read historical generic inputs; write only the new Direction A namespace."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.direction_a_protocol import ROOT,OUT,config,digest,pilot_roles,write_json,file_sha
from scripts.archiveii_frozen_data import load_manifest,OUT as HISTORICAL
from scripts.run_archiveii_frozen import input_identity
from scripts.run_thermo_fast import build_records

def prepare_roles():
    manifest=load_manifest();cfg=config();records=manifest['records']
    inputs=input_identity()
    assert inputs==json.loads((HISTORICAL/'experiment_identity.json').read_text())['inputs'], 'Historical inputs changed'
    result=dict(protocol=cfg['protocol'],historical_inputs=inputs,
                config_sha256=file_sha(ROOT/'configs/direction_a.json'),cases={})
    for family in cfg['pilot_development_families']:
        roles=pilot_roles(records,manifest['folds'][cfg['pilot_outer_excluded']]['train'],cfg['pilot_outer_excluded'],family)
        result['cases'][family]=roles
    result['sha256']=digest(result)
    path=OUT/'pilot_roles.json'
    if path.exists():assert json.loads(path.read_text())==result
    else:write_json(path,result)
    return result

def load_records(ids,with_embeddings=False):
    ids=sorted(ids);assert len(ids)==len(set(ids))
    manifest=load_manifest();physical=json.loads((HISTORICAL/'physical_manifest.json').read_text())
    assert physical['passed']
    features={sid:r for r in physical['samples'] for sid in r['sample_ids']}
    converted={};canonical={'AU','UA','CG','GC','GU','UG'}
    for sid in ids:
        row=manifest['records'][sid];seq=row['sequence'];pairs=row['pairs']
        encode=lambda ps:';'.join(f'{a+1}-{b+1}' for a,b in ps)
        converted[sid]=dict(sequence=seq,original_pairs=encode(pairs),family_id=row['family'],
            eligible_pairs=encode([(a,b) for a,b in pairs if b-a>=4 and seq[a]+seq[b] in canonical]))
    records=build_records(ROOT,ids,converted,features)
    if not with_embeddings:return records
    path=ROOT/'embeddings/archiveii_frozen/all.pt'
    expected=json.loads((HISTORICAL/'embedding_provenance.json').read_text())['embeddings_sha256']
    assert file_sha(path)==expected
    all_values=torch.load(path,map_location='cpu',weights_only=False)
    embeddings={sid:all_values[sid] for sid in ids}
    for sid,value in embeddings.items():
        assert value.shape==(len(records[sid]['sequence']),1280) and not value.requires_grad and torch.isfinite(value).all()
    return records,embeddings

def roles(case):return prepare_roles()['cases'][case]

if __name__=='__main__':
    r=prepare_roles();print(json.dumps({f:{k:len(v) for k,v in row.items() if isinstance(v,list)} for f,row in r['cases'].items()}))
