"""Post-hoc, uniformly calibrated frozen checkpoints; no model retraining."""
import argparse,json,os,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.direction_a_protocol import ROOT,OUT,config,digest,file_sha,write_json,source_identity,cache_key
from scripts.direction_a_data import prepare_roles,load_records
from scripts.direction_a_stage2 import thresholds,choose
from scripts.direction_a_models import METHODS
from scripts.run_direction_a_correction import load_bases
from scripts.run_direction_a_risk import parent,chosen_logits
from scripts.direction_a_evaluate import probabilities,score,edits,metric_row,aggregate
from scripts.archiveii_frozen_metrics import greedy_all_thresholds
from scripts.report_direction_a import verify_reference,verify_records,write_csv

STAGE=ROOT/'outputs/direction_a_stage2/matched'
STRENGTHS=(0.,.25,.5,.75,1.)
POLICIES=('base','p_only','fixed_mix')+METHODS

def source():
    names=['scripts/run_direction_a_matched.py','scripts/direction_a_stage2.py']
    return dict(v1=source_identity('correction'),files={n:file_sha(ROOT/n) for n in names},official=verify_reference(ROOT/'third_party/rinalmo_frozen_reference/sec_struct.py'))

def cache_identity(case,role):
    roles=prepare_roles();_,p,sha=parent(case)
    return dict(case=case,role=role,roles_sha256=roles['sha256'],ids=roles['cases'][case][role],source=source(),parent_choices_sha256=sha,parents=p['identity']['parents'])

def read_cache(case,role):
    directory=STAGE/case/role;row=json.loads((directory/'cache.json').read_text())
    assert row['completed'] and row['identity']==cache_identity(case,role)
    assert set(row['files'])==set(row['identity']['ids'])
    return directory,row

def cached(directory,row,sid):
    path=directory/(cache_key(sid)+'.npz');assert file_sha(path)==row['files'][sid]
    with np.load(path,allow_pickle=False) as handle:result={k:handle[k].copy() for k in handle.files}
    assert set(result)==set(('base',)+METHODS)
    assert all(np.isfinite(v).all() and v.shape==result['base'].shape for v in result.values())
    return result

def cache(case):
    assert torch.cuda.is_available();torch.set_num_threads(2);device=torch.device('cuda')
    for role in ('calibration','development'):
        ident=cache_identity(case,role);directory=STAGE/case/role;directory.mkdir(parents=True,exist_ok=True)
        if (directory/'cache.json').exists():read_cache(case,role);continue
        ids=ident['ids'];records=load_records(ids);base,parents=load_bases(case,17,records,development=role=='development');assert parents==ident['parents']
        outputs={m:chosen_logits(case,m,ids,records,base,device) for m in METHODS};files={}
        for sid in ids:
            path=directory/(cache_key(sid)+'.npz');temporary=path.with_suffix('.tmp.npz')
            values=dict(base=base[sid].numpy(),**{m:outputs[m][sid] for m in METHODS})
            np.savez_compressed(temporary,**values);temporary.replace(path);files[sid]=file_sha(path)
        assert ident==cache_identity(case,role)
        write_json(directory/'cache.json',dict(completed=True,identity=ident,files=files,job_id=os.environ.get('SLURM_JOB_ID')))
        print('MATCHED_CACHE_COMPLETE',case,role,len(ids),flush=True)

def physical(record):
    p=np.clip(record['physical'][0].numpy(),1e-4,1-1e-4);return np.log(p/(1-p))

def mix(values,method,beta,p):
    if method=='base':return values['base']
    expert=p if method in ('p_only','fixed_mix') else values[method]
    return values['base']+beta*(expert-values['base'])

def calibrate(case):
    directory,row=read_cache(case,'calibration');ids=row['identity']['ids'];records=load_records(ids);ts=thresholds(config()['thresholds'])
    profiles={m:defaultdict(lambda:defaultdict(list)) for m in POLICIES}
    # First choose the one common reference, independent of every candidate.
    first=defaultdict(lambda:defaultdict(list))
    for sid in ids:
        r=records[sid];values=cached(directory,row,sid)
        for t,pred in greedy_all_thresholds(probabilities(values['base']),r['sequence'],ts).items():first[t][r['family']].append(score(r['reference'],pred,len(r['sequence']))['f1'])
    reference=choose([dict(beta=0.,threshold=t,f1=float(np.mean([np.mean(v) for v in families.values()])),harm=0.) for t,families in first.items()])
    for index,sid in enumerate(ids):
        r=records[sid];values=cached(directory,row,sid);p=physical(r)
        base_pred=greedy_all_thresholds(probabilities(values['base']),r['sequence'],[reference['threshold']])[reference['threshold']]
        for method in POLICIES:
            strengths=(0.,) if method=='base' else (1.,) if method=='p_only' else STRENGTHS
            for beta in strengths:
                for t,pred in greedy_all_thresholds(probabilities(mix(values,method,beta,p)),r['sequence'],ts).items():
                    profiles[method][beta,t][r['family']].append((score(r['reference'],pred,len(r['sequence']))['f1'],edits(base_pred,pred,r['reference'])['harmful_rate']))
        if (index+1)%100==0:print('MATCHED_CALIBRATE',case,index+1,len(ids),flush=True)
    grids={m:[dict(beta=b,threshold=t,f1=float(np.mean([np.mean([r[0] for r in v]) for v in families.values()])),
                   harm=float(np.mean([np.mean([r[1] for r in v]) for v in families.values()]))) for (b,t),families in rows.items()] for m,rows in profiles.items()}
    unconstrained={m:choose(rows) for m,rows in grids.items()};unconstrained['base']=reference;budget=unconstrained['fixed_mix']['harm']
    choices=dict(unconstrained);choices.update({'risk_'+m:choose(grids[m],budget) for m in ('fixed_mix',)+METHODS})
    assert all(c['harm']<=budget+1e-12 for m,c in choices.items() if m.startswith('risk_'))
    assert row['identity']==cache_identity(case,'calibration')
    sealed=dict(source=source(),case=case,calibration_cache_sha256=file_sha(directory/'cache.json'),base_reference=reference,
                budget=budget,thresholds=ts,choices=choices,profiles=grids,uses_development_scores=False,sealed_unix=time.time())
    sealed['sha256']=digest(sealed);path=STAGE/case/'choices_sealed.json'
    if path.exists():raise FileExistsError('Do not overwrite an existing selection seal')
    write_json(path,sealed);print('MATCHED_CHOICES_SEALED',case,flush=True)

def read_seal(case):
    path=STAGE/case/'choices_sealed.json';row=json.loads(path.read_text());claimed=row.pop('sha256')
    assert digest(row)==claimed and row['source']==source() and not row['uses_development_scores']
    assert row['calibration_cache_sha256']==file_sha(STAGE/case/'calibration/cache.json')
    row['sha256']=claimed;return row

def evaluate(case):
    for other in config()['pilot_development_families']:read_seal(other)
    seal=read_seal(case);directory,row=read_cache(case,'development');ids=row['identity']['ids'];records=load_records(ids)
    predictions={m:[] for m in seal['choices']}
    for sid in ids:
        r=records[sid];values=cached(directory,row,sid);p=physical(r)
        base_pred=greedy_all_thresholds(probabilities(values['base']),r['sequence'],[seal['base_reference']['threshold']])[seal['base_reference']['threshold']]
        for name,choice in seal['choices'].items():
            method=name.removeprefix('risk_');value=mix(values,method,choice['beta'],p)
            pred=greedy_all_thresholds(probabilities(value),r['sequence'],[choice['threshold']])[choice['threshold']]
            predictions[name].append(metric_row(sid,r,pred,base_pred))
    assert seal==read_seal(case)
    output=STAGE/case;write_json(output/'predictions.json',predictions)
    write_json(output/'summary.json',dict(completed=True,case=case,n=len(ids),methods={m:aggregate(v) for m,v in predictions.items()},
        choices_sha256=file_sha(output/'choices_sealed.json'),predictions_sha256=file_sha(output/'predictions.json'),source=source(),official_verification='pending'))
    print('MATCHED_EVALUATION_COMPLETE',case,len(ids),len(predictions),flush=True)

def report():
    rows=[];verified=0;sources={}
    for case in config()['pilot_development_families']:
        seal=read_seal(case);p=STAGE/case;summary=json.loads((p/'summary.json').read_text());predictions=json.loads((p/'predictions.json').read_text())
        assert summary['completed'] and summary['source']==source() and file_sha(p/'predictions.json')==summary['predictions_sha256']
        assert file_sha(p/'choices_sealed.json')==summary['choices_sha256'];expected=set(prepare_roles()['cases'][case]['development'])
        base={r['sample_id']:r['prediction'] for r in predictions['base']}
        records=load_records(sorted(expected))
        for method,values in predictions.items():
            assert {r['sample_id'] for r in values}==expected and len(values)==len(expected)
            verify_records(values);verified+=len(values)
            rebuilt=[metric_row(r['sample_id'],records[r['sample_id']],r['prediction'],base[r['sample_id']]) for r in values]
            for old,new in zip(values,rebuilt):
                for key in ('exact_f1','flexible_f1','helpful','harmful','harmful_rate'):assert abs(old[key]-new[key])<1e-10
            actual=aggregate(values);assert actual==summary['methods'][method]
            rows.append(dict(case=case,method=method,n=len(values),**actual['equal_family'],threshold=seal['choices'][method]['threshold'],beta=seal['choices'][method]['beta']))
        sources[case]=dict(summary_sha256=file_sha(p/'summary.json'),prediction_sha256=file_sha(p/'predictions.json'),choices_sha256=file_sha(p/'choices_sealed.json'))
    target=ROOT/'reports/direction_a_stage2_2026-10-23';target.mkdir(exist_ok=True)
    write_csv(target/'matched_metrics.csv',rows)
    write_json(target/'matched_verification.json',dict(passed=True,official_predictions_verified=verified,source=source(),artifacts=sources,
        scope='Post-hoc uniformly retuned frozen checkpoints, same base reference. Original negative promotion not overwritten.'))
    print('MATCHED_REPORT_COMPLETE',verified,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('cache','calibrate','evaluate','report'));parser.add_argument('--case',choices=('srp','tmRNA'))
    args=parser.parse_args();(report() if args.action=='report' else globals()[args.action](args.case))
