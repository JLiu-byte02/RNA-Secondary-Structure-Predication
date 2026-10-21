"""Train a fixed 50-epoch trajectory, preserving the previous stage read-only."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts._common import load_manifest,load_split,set_seed
from scripts.cache_thermo import file_hash
from scripts.run_thermo_fast import digest,input_hash,state_hash,save_json,save_checkpoint,build_records,model_inputs
from scripts.long_protocol import MODES,SEEDS,BUDGETS,validate_long_config,select_epoch
from scripts.long_models import make_model
from scripts.long_metrics import evaluate_diagnostics
from src.losses import pair_bce_loss
OUT=ROOT/'outputs/long_epochs'
RUNTIME_FILES=[f'scripts/{n}.py' for n in ('_common','cache_thermo','full_protocol','run_thermo_fast',
    'long_protocol','long_models','long_metrics','run_long_epochs')]

def code_identity():
    files=RUNTIME_FILES+[str(p.relative_to(ROOT)) for p in sorted((ROOT/'src').glob('*.py'))]
    hashes={p:file_hash(ROOT/p) for p in files}
    return dict(sha256=digest(hashes),files=hashes)

def synchronize(device):
    if device.type=='cuda':torch.cuda.synchronize()

def train_epoch(model,optimizer,ids,embeddings,records,seed,epoch,config,device,progress=None):
    model.train();order=list(ids);random.Random(seed*1000+epoch).shuffle(order)
    optimizer.zero_grad(set_to_none=True);total=count=micro=0;verified=False
    accumulation=config['gradient_accumulation']
    synchronize(device);started=time.perf_counter()
    for step,sid in enumerate(order,1):
        r=records[sid]
        if not r['cand_i'].numel():continue
        logits=model(*model_inputs(r,embeddings[sid],device))
        loss=pair_bce_loss(logits,r['cand_i'].to(device),r['cand_j'].to(device),r['labels'].to(device))
        if loss is None or not torch.isfinite(loss):raise FloatingPointError('Invalid training loss: '+sid)
        (loss/accumulation).backward()
        if not verified:
            groups=[model.head]
            if hasattr(model,'correction'):groups.append(model.correction)
            for group in groups:
                grads=[p.grad for p in group.parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().max()>0 for g in grads):
                    raise RuntimeError('Missing, zero or nonfinite gradients')
            verified=True
        total+=loss.item();count+=1;micro+=1
        if micro==accumulation:
            torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_grad_norm'],error_if_nonfinite=True)
            optimizer.step();optimizer.zero_grad(set_to_none=True);micro=0
        if progress and step%500==0:progress(step,total/count,time.perf_counter()-started)
    if micro:
        for parameter in model.parameters():
            if parameter.grad is not None:parameter.grad.mul_(accumulation/micro)
        torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_grad_norm'],error_if_nonfinite=True)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
    if not count:raise RuntimeError('No training candidates')
    synchronize(device)
    return dict(mean_pair_loss=total/count,n_steps=count,training_seconds=time.perf_counter()-started,gradients_verified=verified)

def epoch_record(epoch,training,validation,origin):
    thresholds=sorted(map(float,validation['threshold_metrics']))
    threshold=max(thresholds,key=lambda t:validation['threshold_metrics'][str(t)]['rna_macro_exact_f1'])
    return dict(epoch=epoch,score=validation['threshold_metrics'][str(threshold)]['rna_macro_exact_f1'],
                threshold=threshold,validation=validation,origin=origin,**training)

def train(args):
    config=yaml.safe_load((ROOT/'configs/long_epochs.yaml').read_text());validate_long_config(config)
    configuration=digest(config);code=code_identity()
    directory=OUT/('smoke' if args.smoke else '')/args.mode/f'seed{args.seed}'
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.run.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        status=dict(status='loading',mode=args.mode,seed=args.seed,config_hash=configuration,code_hash=code['sha256'])
        try:_train(args,config,code,directory,status)
        except Exception:
            status.update(status='failed',error=traceback.format_exc());save_json(directory/'status.json',status);raise

def _train(args,config,code,directory,status):
    started=time.perf_counter();seed=args.seed;mode=args.mode
    previous=ROOT/'outputs/full_tr0'
    audit=json.loads((previous/'split_audit.json').read_text())
    cache=json.loads((previous/'thermo_cache_manifest.json').read_text())
    if not audit['passed'] or cache['failed_unique_sequences']:raise RuntimeError('Full cohort audit incomplete')
    inputs=input_hash(ROOT,audit,cache)
    if (directory/'summary.json').exists():
        old=json.loads((directory/'summary.json').read_text())
        if old['config_hash']!=status['config_hash'] or old['code_hash']!=code['sha256'] or old['input_provenance_hash']!=inputs:
            raise RuntimeError('Completed trajectory identity changed')
        print('ALREADY_COMPLETED',mode,seed,flush=True);return
    record_path=directory/'code_identity.json'
    if record_path.exists() and json.loads(record_path.read_text())!=code:raise RuntimeError('Refuse source changes during a trajectory')
    save_json(record_path,code);save_json(directory/'config.json',config)
    for path in code['files']:
        destination=directory/'source_snapshot'/path;destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():shutil.copy2(ROOT/path,destination)
    save_json(directory/'status.json',status)
    set_seed(seed)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('Require CUDA BF16')
    device=torch.device('cuda');model=make_model(mode).to(device)
    initialization=state_hash(model.state_dict());head_initialization=state_hash(model.head.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['initial_lr'],weight_decay=config['weight_decay'])
    manifest,split=load_manifest(ROOT),load_split(ROOT)
    train_ids=split['TR0'][:32] if args.smoke else split['TR0']
    validation_ids=split['VL0'][:16] if args.smoke else split['VL0']
    if any(cache['samples'].get(sid,{}).get('status')!='success' for sid in train_ids+validation_ids):
        raise RuntimeError('Every training and validation sample is required')
    records=build_records(ROOT,train_ids+validation_ids,manifest,cache['samples'])
    embeddings={}
    for name,ids in (('TR0',train_ids),('VL0',validation_ids)):
        path=ROOT/'embeddings/full'/f'{name}.pt'
        if file_hash(path)!=audit['embedding_sha256'][name]:raise RuntimeError('Embedding input changed: '+name)
        loaded=torch.load(path,map_location='cpu',weights_only=False)
        for sid in ids:
            value=loaded[sid]
            if value.requires_grad or value.shape!=(len(manifest[sid]['sequence']),1280):raise ValueError('Invalid frozen embedding')
            embeddings[sid]=value
        del loaded
    status.update(status='running',initialization_hash=initialization,head_initialization_hash=head_initialization,
        parameter_count=sum(p.numel() for p in model.parameters()),train_count=len(train_ids),validation_count=len(validation_ids),
        input_provenance_hash=inputs,encoder_frozen=True,pid=os.getpid())
    save_json(directory/'status.json',status)
    metadata=dict(config_hash=status['config_hash'],code_hash=code['sha256'],input_provenance_hash=inputs,
        initialization_hash=initialization,head_initialization_hash=head_initialization,mode=mode,seed=seed,
        train_ids_hash=digest(train_ids),validation_ids_hash=digest(validation_ids))
    log_path=directory/'epochs.jsonl';rows=[];start=1;imported=0;parent={}
    def checkpoint(epoch):
        return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),**metadata)
    if (directory/'last.pt').exists():
        state=torch.load(directory/'last.pt',map_location=device,weights_only=False)
        if any(state.get(k)!=v for k,v in metadata.items()):raise RuntimeError('Resume identity mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        start=state['epoch']+1
        rows=[json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        rows=[r for r in rows if r['epoch']<start]
        if [r['epoch'] for r in rows]!=list(range(1,start)):raise RuntimeError('Incomplete resume log')
        parent=json.loads((directory/'parent_provenance.json').read_text());imported=parent['imported_prefix_epochs']
        log_path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    elif mode!='B0' and not args.smoke:
        origin=previous/mode/f'seed{seed}'
        summary=json.loads((origin/'summary.json').read_text());old_config=json.loads((origin/'config.json').read_text())
        old_code=json.loads((origin/'code_identity.json').read_text())['files']
        for path,sha in code['files'].items():
            if path in old_code and file_hash(ROOT/path)!=old_code[path]:raise RuntimeError('Inherited scientific code changed: '+path)
        for key in ('initial_lr','weight_decay','gradient_accumulation','batch_size','clip_grad_norm'):
            if config[key]!=old_config[key]:raise RuntimeError('Inherited optimizer protocol changed: '+key)
        if summary['initialization_hash']!=initialization or summary['input_provenance_hash']!=inputs:
            raise RuntimeError('Inherited trajectory input or initialization mismatch')
        original_rows=[json.loads(line) for line in (origin/'train_log.jsonl').read_text().splitlines() if line.strip()]
        imported=summary['epochs_completed']
        if len(original_rows)!=imported or imported not in (9,10):raise RuntimeError('Unexpected inherited prefix')
        parent=dict(imported_prefix_epochs=imported,origin=str(origin.relative_to(ROOT)),
            last_checkpoint_sha256=file_hash(origin/'last.pt'),summary_sha256=file_hash(origin/'summary.json'),
            optimizer_continued=True,imported_epochs={})
        with log_path.open('w') as log:
            for old in original_rows:
                epoch=old['epoch'];path=origin/f'epoch{epoch}.pt'
                state=torch.load(path,map_location=device,weights_only=False);model.load_state_dict(state['model'])
                validation=evaluate_diagnostics(model,validation_ids,embeddings,records,config['thresholds'],device)
                for t,metrics in old['threshold_metrics'].items():
                    if abs(validation['threshold_metrics'][t]['rna_macro_exact_f1']-metrics['rna_macro_exact_f1'])>1e-12:
                        raise RuntimeError('Inherited validation curve changed at epoch '+str(epoch))
                training=dict(mean_pair_loss=old['mean_pair_loss'],n_steps=old['n_steps'],
                    training_seconds=old['training_seconds'],gradients_verified=True)
                row=epoch_record(epoch,training,validation,'verified_previous_stage');rows.append(row)
                log.write(json.dumps(row)+'\n');log.flush()
                shutil.copy2(path,directory/path.name);parent['imported_epochs'][str(epoch)]=file_hash(path)
                status.update(phase='backfill_validation_diagnostics',epoch=epoch,step=len(train_ids));save_json(directory/'status.json',status)
        state=torch.load(origin/'last.pt',map_location=device,weights_only=False)
        if state['epoch']!=imported:raise RuntimeError('Must continue last epoch, not best checkpoint')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        start=imported+1;save_json(directory/'parent_provenance.json',parent)
        save_checkpoint(directory/'last.pt',checkpoint(imported))
    else:parent=dict(imported_prefix_epochs=0,origin=None,optimizer_continued=False)
    save_json(directory/'parent_provenance.json',parent)
    max_epochs=2 if args.smoke else config['max_epochs']
    print('TRAJECTORY_START',mode,seed,'next_epoch',start,'max_epochs',max_epochs,flush=True)
    with log_path.open('a') as log:
        for epoch in range(start,max_epochs+1):
            def progress(step,loss,seconds):
                status.update(phase='training',epoch=epoch,step=step,training_mean_loss=loss,epoch_seconds=seconds)
                save_json(directory/'status.json',status)
                print('TRAIN',mode,seed,epoch,step,'/',len(train_ids),'loss',round(loss,6),flush=True)
            training=train_epoch(model,optimizer,train_ids,embeddings,records,seed,epoch,config,device,progress)
            status.update(phase='validation',epoch=epoch,step=len(train_ids));save_json(directory/'status.json',status)
            validation=evaluate_diagnostics(model,validation_ids,embeddings,records,config['thresholds'],device)
            row=epoch_record(epoch,training,validation,'new_stage');rows.append(row)
            save_checkpoint(directory/f'epoch{epoch}.pt',checkpoint(epoch))
            log.write(json.dumps(row)+'\n');log.flush();os.fsync(log.fileno())
            save_checkpoint(directory/'last.pt',checkpoint(epoch))
            best=select_epoch(rows,epoch)
            status.update(phase='epoch_complete',epoch=epoch,best_epoch=best['epoch'],best_VL0_f1=best['score'],
                last_VL0_f1=row['score'],validation_bce=validation['mean_pair_bce'])
            save_json(directory/'status.json',status)
            print('EPOCH_COMPLETE',mode,seed,epoch,'loss',training['mean_pair_loss'],'VL0_F1',row['score'],'best_epoch',best['epoch'],flush=True)
    if len(rows)!=max_epochs:raise RuntimeError('Incomplete fixed-horizon trajectory')
    selections={str(b):dict(epoch=select_epoch(rows,b)['epoch'],score=select_epoch(rows,b)['score'],
        threshold=select_epoch(rows,b)['threshold']) for b in BUDGETS if b<=max_epochs}
    summary=dict(status='completed',**metadata,parameter_count=status['parameter_count'],encoder_frozen=True,
        epochs_completed=max_epochs,imported_prefix_epochs=imported,train_count=len(train_ids),validation_count=len(validation_ids),
        early_stopping=False,gradients_verified=all(r['gradients_verified'] for r in rows),
        budget_selections=selections,training_seconds=sum(r['training_seconds'] for r in rows),
        validation_seconds=sum(r['validation']['seconds'] for r in rows),
        new_training_seconds=sum(r['training_seconds'] for r in rows if r['origin']=='new_stage'),
        new_stage_wall_seconds=time.perf_counter()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated(),test_scores_used=False)
    save_json(directory/'summary.json',summary);save_json(directory/'status.json',summary)
    print('TRAJECTORY_COMPLETED',mode,seed,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=MODES,required=True)
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True);parser.add_argument('--smoke',action='store_true')
    train(parser.parse_args())
