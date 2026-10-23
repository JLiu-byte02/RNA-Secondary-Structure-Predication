"""Cross-fitted correction fitting and separately sealed development evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.direction_a_protocol import OUT,config,digest,file_sha,source_identity,write_json
from scripts.direction_a_data import prepare_roles,load_records
from scripts.direction_a_models import METHODS,LocalCorrection,correction_loss
from scripts.direction_a_evaluate import probabilities,calibrate,metric_row,aggregate,physical_structure,official_verify
from scripts.archiveii_frozen_metrics import greedy_all_thresholds
from scripts.run_thermo_fast import save_checkpoint,state_hash
from scripts._common import set_seed

def update_best(best,model,row):
    if best is None or row['calibration']['equal_family_exact_f1']>best['calibration']['equal_family_exact_f1']:
        return dict(epoch=row['epoch'],calibration=row['calibration'],model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    return best

def load_resume(path,identity,model,optimizer,device):
    old=torch.load(path,map_location=device,weights_only=False);assert old['identity']==identity,'Resume identity changed'
    rows=old['rows'];assert [r['epoch'] for r in rows]==list(range(1,len(rows)+1))
    model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer'])
    torch.set_rng_state(old['torch_rng'].cpu())
    if device.type=='cuda':torch.cuda.set_rng_state_all([r.cpu() for r in old['cuda_rng']])
    return rows,old['best']

def load_bases(case,seed,records,development=False):
    manifest=prepare_roles();roles=manifest['cases'][case];summaries={};values={}
    for part in ('a','b','full'):
        directory=OUT/'pilot'/case/f'base_{part}_seed{seed}';path=directory/'summary.json'
        row=json.loads(path.read_text());identity=row['identity']
        assert row['completed'] and row['epochs']==config()['base_epochs'] and not identity['smoke']
        assert identity['code']==source_identity('base') and identity['inputs']==manifest['historical_inputs']
        assert identity['roles_sha256']==manifest['sha256'] and identity['case']==case and identity['part']==part and identity['seed']==seed
        train=roles['block_a'] if part=='a' else roles['block_b'] if part=='b' else sorted(roles['block_a']+roles['block_b'])
        predict=roles['block_b'] if part=='a' else roles['block_a'] if part=='b' else roles['development']
        assert row['train_ids']==train and row['prediction_ids']==predict and set(row['artifacts'])==set(predict)
        assert identity['train_ids_sha256']==digest(train) and identity['predict_ids_sha256']==digest(predict)
        assert file_sha(directory/'final.pt')==row['checkpoint_sha256'] and file_sha(directory/'epochs.jsonl')==row['log_sha256']
        summaries[part]=file_sha(path)
        if (part=='full')!=development:continue
        for sid,item in row['artifacts'].items():
            if sid not in records:continue
            path=ROOT/item['path'];assert file_sha(path)==item['sha256']
            with np.load(path,allow_pickle=False) as data:value=data['base'].copy()
            r=records[sid];length=len(r['sequence'])
            assert value.shape==(length,length) and np.isfinite(value).all() and np.allclose(value,value.T,atol=1e-5)
            assert sid not in values;values[sid]=torch.from_numpy(value)
    assert set(values)==set(records),'Cache coverage mismatch'
    return values,summaries

def tensors(sid,records,base,device):
    r=records[sid]
    return base[sid].to(device),r['physical'][0].to(device),r['valid'].to(device)

def train_one_epoch(model,optimizer,ids,records,base,seed,epoch,cfg,device,base_threshold):
    model.train();order=list(ids);random.Random(seed*1000+epoch).shuffle(order)
    optimizer.zero_grad(set_to_none=True);accumulation=cfg['gradient_accumulation'];micro=0;losses=[];bces=[];harms=[]
    if device.type=='cuda':torch.cuda.synchronize()
    started=time.perf_counter();verified=False
    for sid in order:
        r=records[sid];s,p,v=tensors(sid,records,base,device);out,_=model(s,p,v)
        loss,terms=correction_loss(out,s,r['cand_i'].to(device),r['cand_j'].to(device),r['labels'].to(device),base_threshold,
                                   cfg['harm_lambda'] if model.method=='safe_local_gate' else 0.)
        assert torch.isfinite(loss);(loss/accumulation).backward();micro+=1
        if not verified:
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().max()>0 for g in grads)
            assert all(not b.requires_grad for b in base.values());verified=True
        losses.append(float(loss.detach()));bces.append(terms['bce']);harms.append(terms['harm'])
        if micro==accumulation:
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad_norm'],error_if_nonfinite=True)
            optimizer.step();optimizer.zero_grad(set_to_none=True);micro=0
    if micro:
        for p in model.parameters():
            if p.grad is not None:p.grad.mul_(accumulation/micro)
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad_norm'],error_if_nonfinite=True)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
    if device.type=='cuda':torch.cuda.synchronize()
    assert len(losses)==len(ids)
    return dict(mean_loss=float(np.mean(losses)),mean_bce=float(np.mean(bces)),mean_harm_surrogate=float(np.mean(harms)),
                training_seconds=time.perf_counter()-started,n_steps=len(losses),gradients_verified=verified)

def infer(model,ids,records,base,device):
    model.eval();values={};diagnostics={}
    with torch.no_grad():
        for sid in ids:
            s,p,v=tensors(sid,records,base,device);value,gate=model(s,p,v)
            arr=value.float().cpu().numpy();assert np.isfinite(arr).all();values[sid]=arr
            g=gate[v].float();diagnostics[sid]=dict(gate_mean=float(g.mean()),gate_std=float(g.std(unbiased=False)))
    return values,diagnostics

def fit_method(method,fit,cal,records,base,cfg,seed,identity,directory,device,maximum,base_threshold):
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'summary.json').exists():
        summary=json.loads((directory/'summary.json').read_text());assert summary['identity']==identity
        assert file_sha(directory/'selected.pt')==summary['selected_sha256'];return summary
    set_seed(seed);model=LocalCorrection(method).to(device);initial=state_hash(model.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['initial_lr'],weight_decay=cfg['weight_decay'])
    rows=[];best=None;last=directory/'last.pt'
    if last.exists():rows,best=load_resume(last,identity,model,optimizer,device)
    assert len(rows)<=maximum
    for epoch in range(len(rows)+1,maximum+1):
        training=train_one_epoch(model,optimizer,fit,records,base,seed,epoch,cfg,device,base_threshold)
        values,gates=infer(model,cal,records,base,device)
        calibration=calibrate(records,cal,{s:probabilities(v) for s,v in values.items()},cfg['thresholds'])
        val_losses=[]
        for sid,arr in values.items():
            r=records[sid];x=arr[r['cand_i'].numpy(),r['cand_j'].numpy()];y=r['labels'].numpy()
            val_losses.append(float(np.mean(np.logaddexp(0,x)-y*x)))
        row=dict(epoch=epoch,calibration=calibration,calibration_bce=float(np.mean(val_losses)),
                 calibration_gate_mean=float(np.mean([g['gate_mean'] for g in gates.values()])),**training)
        rows.append(row);best=update_best(best,model,row)
        save_checkpoint(last,dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),rows=rows,best=best,
                                  torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []))
        write_json(directory/'status.json',dict(status='training',epoch=epoch,maximum=maximum,best_epoch=best['epoch']))
        print('DIRECTION_A_CORRECTION_EPOCH',identity['case'],method,epoch,round(calibration['equal_family_exact_f1'],6),flush=True)
    assert [r['epoch'] for r in rows]==list(range(1,maximum+1))
    save_checkpoint(directory/'selected.pt',dict(identity=identity,**best))
    (directory/'epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summary=dict(completed=True,identity=identity,epochs=maximum,selected_epoch=best['epoch'],calibration=best['calibration'],
                 initialization_hash=initial,selected_sha256=file_sha(directory/'selected.pt'),log_sha256=file_sha(directory/'epochs.jsonl'),
                 training_seconds=sum(r['training_seconds'] for r in rows),slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    write_json(directory/'summary.json',summary);return summary

def baseline_choices(records,cal,base,cfg):
    base_values={s:base[s].numpy() for s in cal}
    physical={s:records[s]['physical'][0].numpy() for s in cal}
    choices=dict(base=calibrate(records,cal,{s:probabilities(v) for s,v in base_values.items()},cfg['thresholds']),
                 p_only=calibrate(records,cal,physical,cfg['thresholds']),mfe=dict(fixed=True),mea=dict(fixed=True,gamma=1.))
    mixes=[]
    for alpha in cfg['mixture_alphas']:
        mixed={s:probabilities((1-alpha)*base_values[s]+alpha*np.log(np.clip(physical[s],1e-4,1-1e-4)/(1-np.clip(physical[s],1e-4,1-1e-4)))) for s in cal}
        mixes.append(dict(alpha=alpha,**calibrate(records,cal,mixed,cfg['thresholds'])))
    choices['fixed_mix']=max(mixes,key=lambda r:r['equal_family_exact_f1']);choices['mixture_grid']=mixes
    return choices

def fit(args):
    cfg=config();manifest=prepare_roles();roles=manifest['cases'][args.case]
    fit_ids=roles['correction_fit'][:8] if args.smoke else roles['correction_fit']
    cal=roles['calibration'][:4] if args.smoke else roles['calibration']
    records=load_records(sorted(fit_ids+cal));base,parents=load_bases(args.case,args.seed,records)
    identity=dict(code=source_identity('correction'),parents=parents,roles_sha256=manifest['sha256'],case=args.case,seed=args.seed,
                  smoke=args.smoke,fit_sha256=digest(fit_ids),calibration_sha256=digest(cal))
    directory=OUT/('smoke_correction' if args.smoke else 'pilot')/args.case/f'correction_seed{args.seed}';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (directory/'choices_sealed.json').exists():
            sealed=json.loads((directory/'choices_sealed.json').read_text());assert sealed['identity']==identity
            for method in METHODS:assert file_sha(directory/method/'selected.pt')==sealed['learned'][method]['selected_sha256']
            print('CORRECTION_ALREADY_SEALED',args.case);return
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported();device=torch.device('cuda')
        write_json(directory/'identity.json',identity)
        choices=baseline_choices(records,cal,base,cfg);learned={}
        for method in METHODS:
            learned[method]=fit_method(method,fit_ids,cal,records,base,cfg,args.seed,dict(method=method,**identity),directory/method,device,
                                       2 if args.smoke else cfg['pilot_correction_epochs'],choices['base']['threshold'])
        sealed=dict(identity=identity,baselines=choices,learned=learned,fit_ids=fit_ids,calibration_ids=cal,
                    selection='equal-family exact F1 on calibration only; earliest epoch/lower threshold ties',sealed_at=time.time())
        write_json(directory/'choices_sealed.json',sealed);print('DIRECTION_A_CHOICES_SEALED',args.case,flush=True)

def evaluate(args):
    cfg=config();manifest=prepare_roles();roles=manifest['cases'][args.case]
    stage='smoke_correction' if args.smoke else 'pilot'
    # The two development cases cannot be scored until every model choice is sealed.
    for case in ([args.case] if args.smoke else cfg['pilot_development_families']):
        path=OUT/stage/case/f'correction_seed{args.seed}'/'choices_sealed.json'
        sealed_check=json.loads(path.read_text());assert sealed_check['identity']['code']==source_identity('correction')
    directory=OUT/stage/args.case/f'correction_seed{args.seed}';seal_path=directory/'choices_sealed.json'
    sealed=json.loads(seal_path.read_text());assert sealed['identity']['roles_sha256']==manifest['sha256']
    ids=roles['development'][:4] if args.smoke else roles['development'];records=load_records(ids)
    base,parents=load_bases(args.case,args.seed,records,development=True);assert sealed['identity']['parents']==parents
    threshold=sealed['baselines']['base']['threshold'];base_probs={s:probabilities(v.numpy()) for s,v in base.items()}
    base_predictions={s:greedy_all_thresholds(base_probs[s],records[s]['sequence'],[threshold])[threshold] for s in ids}
    results={};rows_by_method={};device=torch.device('cuda')
    physical={s:records[s]['physical'][0].numpy() for s in ids}
    for method in ('base','p_only','fixed_mix',*METHODS):
        choice=sealed['baselines'][method] if method in sealed['baselines'] else sealed['learned'][method]['calibration']
        diagnostics={}
        if method=='base':probs=base_probs
        elif method=='p_only':probs=physical
        elif method=='fixed_mix':
            alpha=choice['alpha'];probs={s:probabilities((1-alpha)*base[s].numpy()+alpha*np.log(np.clip(physical[s],1e-4,1-1e-4)/(1-np.clip(physical[s],1e-4,1-1e-4)))) for s in ids}
        else:
            path=directory/method/'selected.pt';assert file_sha(path)==sealed['learned'][method]['selected_sha256']
            state=torch.load(path,map_location=device,weights_only=False);assert state['identity']==sealed['learned'][method]['identity']
            model=LocalCorrection(method).to(device);model.load_state_dict(state['model'])
            values,diagnostics=infer(model,ids,records,base,device);probs={s:probabilities(v) for s,v in values.items()}
        rows=[];common=[]
        for sid in ids:
            predictions=greedy_all_thresholds(probs[sid],records[sid]['sequence'],sorted(set([threshold,choice['threshold']])))
            row=metric_row(sid,records[sid],predictions[choice['threshold']],base_predictions[sid]);row.update(diagnostics.get(sid,{}));rows.append(row)
            common.append(metric_row(sid,records[sid],predictions[threshold],base_predictions[sid]))
        results[method]=dict(selected=aggregate(rows),common_base_threshold=aggregate(common),official_verification=official_verify(rows),choice=choice)
        rows_by_method[method]=dict(selected=rows,common_base_threshold=common)
    # Sequence-only physical baselines never see labels; fixed gamma is not tuned on development.
    physical_cache={}
    for sid in ids:
        sequence=records[sid]['sequence']
        if sequence not in physical_cache:physical_cache[sequence]=physical_structure(sequence)
    for method in ('mfe','mea'):
        rows=[metric_row(s,records[s],physical_cache[records[s]['sequence']][method],base_predictions[s]) for s in ids]
        results[method]=dict(selected=aggregate(rows),official_verification=official_verify(rows),choice=sealed['baselines'][method]);rows_by_method[method]=dict(selected=rows)
    predictions_path=directory/'development_predictions.json';write_json(predictions_path,rows_by_method)
    write_json(directory/'development_summary.json',dict(case=args.case,seed=args.seed,smoke=args.smoke,identity=sealed['identity'],
        choices_sha256=file_sha(seal_path),predictions_sha256=file_sha(predictions_path),n=len(ids),methods=results,
        evaluation_role='pseudo-OOD development only, no outer-family test',completed=True,slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    print('DIRECTION_A_DEVELOPMENT_COMPLETE',args.case,len(ids),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('fit','evaluate'));parser.add_argument('--case',choices=('srp','tmRNA'),required=True)
    parser.add_argument('--seed',type=int,default=17);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    (fit if args.action=='fit' else evaluate)(args)
