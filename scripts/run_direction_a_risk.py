"""The sole bounded revision: calibrate decoded edit risk, reusing v1 weights."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.direction_a_protocol import ROOT,OUT,config,digest,file_sha,source_identity,write_json
from scripts.direction_a_data import prepare_roles,load_records
from scripts.direction_a_models import LocalCorrection
from scripts.direction_a_risk import family_metrics,choose_policy
from scripts.direction_a_evaluate import probabilities,score,edits,metric_row,aggregate
from scripts.archiveii_frozen_metrics import greedy_all_thresholds
from scripts.run_direction_a_correction import load_bases,infer
from scripts.run_thermo_fast import state_hash
from scripts.report_direction_a import verify_reference
from scripts._common import set_seed

RISK_OUT=OUT/'revision1'
POLICIES=('risk_fixed_mix','risk_global_gate','risk_local_gate')

def revision_source():
    names=['scripts/run_direction_a_risk.py','scripts/direction_a_risk.py','configs/direction_a_revision1.json']
    return dict(v1=source_identity('correction'),revision_files={n:file_sha(ROOT/n) for n in names},
                official_reference_sha256=verify_reference(ROOT/'third_party/rinalmo_frozen_reference/sec_struct.py'))

def parent(case):
    directory=OUT/'pilot'/case/'correction_seed17';path=directory/'choices_sealed.json';row=json.loads(path.read_text())
    assert row['identity']['code']==source_identity('correction') and row['identity']['case']==case and row['identity']['seed']==17 and not row['identity']['smoke']
    return directory,row,file_sha(path)

def chosen_logits(case,method,ids,records,base,device):
    directory,sealed,_=parent(case);path=directory/method/'selected.pt';entry=sealed['learned'][method]
    assert file_sha(path)==entry['selected_sha256'];state=torch.load(path,map_location=device,weights_only=False)
    assert state['identity']==entry['identity'];model=LocalCorrection(method).to(device);model.load_state_dict(state['model'])
    before=state_hash(model.state_dict());values,_=infer(model,ids,records,base,device)
    assert state_hash(model.state_dict())==before,'A reused corrector was modified'
    return values

def identity(case):
    _,p,sha=parent(case);manifest=prepare_roles()
    return dict(source=revision_source(),case=case,seed=17,v1_choices_sha256=sha,
                roles_sha256=manifest['sha256'],parents=p['identity']['parents'],training='none')

def read_seal(case):
    path=RISK_OUT/case/'choices_sealed.json';row=json.loads(path.read_text());claimed=row.pop('sha256')
    assert digest(row)==claimed and row['identity']==identity(case),'Revision seal or source changed'
    row['sha256']=claimed;return row

def decode_values(values,ids,records,threshold):
    return {s:greedy_all_thresholds(probabilities(values[s]),records[s]['sequence'],[threshold])[threshold] for s in ids}

def risk_of_predictions(predictions,ids,records,base_predictions):
    return family_metrics([dict(family=records[s]['family'],f1=score(records[s]['reference'],predictions[s],len(records[s]['sequence']))['f1'],
                harm=edits(base_predictions[s],predictions[s],records[s]['reference'])['harmful_rate']) for s in ids])

def prepare(case):
    set_seed(17)
    cfg=config();rev=json.loads((ROOT/'configs/direction_a_revision1.json').read_text());manifest=prepare_roles()
    directory=RISK_OUT/case;directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (directory/'choices_sealed.json').exists():read_seal(case);print('RISK_ALREADY_SEALED',case);return
        ident=identity(case);ids=manifest['cases'][case]['calibration'];records=load_records(ids)
        tensors,parents=load_bases(case,17,records);assert parents==ident['parents']
        base={s:v.numpy() for s,v in tensors.items()};_,old,_=parent(case)
        base_threshold=old['baselines']['base']['threshold'];base_predictions=decode_values(base,ids,records,base_threshold)
        physical={s:np.log(np.clip(records[s]['physical'][0].numpy(),1e-4,1-1e-4)/(1-np.clip(records[s]['physical'][0].numpy(),1e-4,1-1e-4))) for s in ids}
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported();device=torch.device('cuda');started=time.time()
        outputs={m:chosen_logits(case,m,ids,records,tensors,device) for m in ('residual','global_gate','local_gate')}
        residual_predictions=decode_values(outputs['residual'],ids,records,old['learned']['residual']['calibration']['threshold'])
        residual_risk=risk_of_predictions(residual_predictions,ids,records,base_predictions)
        mix=old['baselines']['fixed_mix'];alpha=mix['alpha'];fixed_values={s:(1-alpha)*base[s]+alpha*physical[s] for s in ids}
        fixed_risk=risk_of_predictions(decode_values(fixed_values,ids,records,mix['threshold']),ids,records,base_predictions)
        budget=min(residual_risk['harm'],fixed_risk['harm']);choices={};profiles={}
        for policy,expert in [('risk_fixed_mix',physical),('risk_global_gate',outputs['global_gate']),('risk_local_gate',outputs['local_gate'])]:
            grid=[]
            for beta in rev['betas']:
                by_threshold={t:[] for t in cfg['thresholds']}
                for sid in ids:
                    r=records[sid];value=base[sid]+beta*(expert[sid]-base[sid])
                    for threshold,pred in greedy_all_thresholds(probabilities(value),r['sequence'],cfg['thresholds']).items():
                        by_threshold[threshold].append(dict(family=r['family'],f1=score(r['reference'],pred,len(r['sequence']))['f1'],
                            harm=edits(base_predictions[sid],pred,r['reference'])['harmful_rate']))
                grid.extend(dict(beta=beta,threshold=t,**family_metrics(rows)) for t,rows in by_threshold.items())
            profiles[policy]=grid;choices[policy]=choose_policy(grid,budget)
            print('RISK_CALIBRATION_POLICY',case,policy,json.dumps(choices[policy]),flush=True)
        assert ident==identity(case)
        sealed=dict(identity=ident,calibration_ids=ids,base_threshold=base_threshold,budget=budget,
            risk_references=dict(residual=residual_risk,fixed_mix=fixed_risk),choices=choices,profiles=profiles,
            calibration_seconds=time.time()-started,slurm_job_id=os.environ.get('SLURM_JOB_ID'),sealed_at=time.time())
        sealed['sha256']=digest(sealed);write_json(directory/'choices_sealed.json',sealed);print('RISK_CHOICES_SEALED',case,flush=True)

def evaluate(case):
    set_seed(17)
    for name in config()['pilot_development_families']:read_seal(name)
    sealed=read_seal(case);manifest=prepare_roles();ids=manifest['cases'][case]['development'];records=load_records(ids)
    tensors,parents=load_bases(case,17,records,development=True);assert parents==sealed['identity']['parents']
    base={s:v.numpy() for s,v in tensors.items()};base_predictions=decode_values(base,ids,records,sealed['base_threshold'])
    physical={s:np.log(np.clip(records[s]['physical'][0].numpy(),1e-4,1-1e-4)/(1-np.clip(records[s]['physical'][0].numpy(),1e-4,1-1e-4))) for s in ids}
    device=torch.device('cuda');outputs={m:chosen_logits(case,m,ids,records,tensors,device) for m in ('global_gate','local_gate')}
    predictions={};metrics={}
    for policy,expert in [('risk_fixed_mix',physical),('risk_global_gate',outputs['global_gate']),('risk_local_gate',outputs['local_gate'])]:
        choice=sealed['choices'][policy];assert choice['harm']<=sealed['budget']+1e-12;rows=[];common=[]
        for sid in ids:
            r=records[sid];value=base[sid]+choice['beta']*(expert[sid]-base[sid])
            decoded=greedy_all_thresholds(probabilities(value),r['sequence'],sorted(set([choice['threshold'],sealed['base_threshold']])))
            rows.append(metric_row(sid,r,decoded[choice['threshold']],base_predictions[sid]))
            common.append(metric_row(sid,r,decoded[sealed['base_threshold']],base_predictions[sid]))
        predictions[policy]=dict(selected=rows,common_base_threshold=common)
        metrics[policy]=dict(selected=aggregate(rows),common_base_threshold=aggregate(common),choice=choice)
    assert sealed['identity']==identity(case)
    directory=RISK_OUT/case;write_json(directory/'development_predictions.json',predictions)
    write_json(directory/'development_summary.json',dict(completed=True,case=case,seed=17,n=len(ids),identity=sealed['identity'],methods=metrics,
        choices_sha256=file_sha(directory/'choices_sealed.json'),predictions_sha256=file_sha(directory/'development_predictions.json'),
        official_verification='pending separate CPU audit',slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    print('RISK_DEVELOPMENT_INFERENCE_COMPLETE',case,len(ids),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','evaluate'));parser.add_argument('--case',choices=('srp','tmRNA'),required=True)
    args=parser.parse_args();(prepare if args.action=='prepare' else evaluate)(args.case)
