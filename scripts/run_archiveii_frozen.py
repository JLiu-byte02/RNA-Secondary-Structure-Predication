"""Unchanged frozen-E numerical training on each official ArchiveII family fold."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import yaml
from scripts._common import set_seed
from scripts.run_thermo_fast import ThermoFusion,build_records,save_checkpoint,state_hash
from scripts.run_long_epochs import train_epoch,epoch_record
from scripts.long_metrics import evaluate_diagnostics
from scripts.archiveii_frozen_data import OUT,load_manifest,normalized_sequence,sha,write_json
from scripts.archiveii_frozen_protocol import MODES,SEEDS,FAMILIES,validate_config,code_identity,select_epoch,digest

def input_identity():
    names=('data_manifest.json.gz','physical_manifest.json','embedding_provenance.json')
    records={name:sha(OUT/name) for name in names}
    embedding=json.loads((OUT/'embedding_provenance.json').read_text())
    assert embedding['passed'] and embedding['frozen'] and not embedding['supervised_weights_loaded']
    records['embeddings']=embedding['embeddings_sha256']
    return dict(files=records,sha256=digest(records))

def load_inputs(family,roles,limit=None):
    manifest=load_manifest();assert manifest['passed']
    physical=json.loads((OUT/'physical_manifest.json').read_text());assert physical['passed']
    assert physical['data_manifest_sha256']==sha(OUT/'data_manifest.json.gz')
    features={sid:row for row in physical['samples'] for sid in row['sample_ids']}
    ids={role:manifest['folds'][family][role][:limit] if limit else manifest['folds'][family][role] for role in roles}
    identifiers=sum(ids.values(),[]);assert len(identifiers)==len(set(identifiers))
    converted={}
    canonical={'AU','UA','CG','GC','GU','UG'}
    for sid in identifiers:
        row=manifest['records'][sid];seq=normalized_sequence(row['sequence'])
        assert not set(seq)-set('ACGU')
        pairs=[tuple(p) for p in row['pairs']]
        eligible=[(a,b) for a,b in pairs if b-a>=4 and seq[a]+seq[b] in canonical]
        encode=lambda value:';'.join(f'{a+1}-{b+1}' for a,b in value)
        converted[sid]=dict(sequence=seq,original_pairs=encode(pairs),eligible_pairs=encode(eligible),family_id=row['family'])
    records=build_records(ROOT,identifiers,converted,features)
    path=ROOT/'embeddings/archiveii_frozen/all.pt'
    expected=json.loads((OUT/'embedding_provenance.json').read_text())['embeddings_sha256']
    assert sha(path)==expected,'Embedding content changed'
    all_embeddings=torch.load(path,map_location='cpu',weights_only=False)
    embeddings={sid:all_embeddings[sid] for sid in identifiers}
    for sid,value in embeddings.items():
        assert value.shape==(len(records[sid]['sequence']),1280) and not value.requires_grad
        assert torch.isfinite(value).all()
    return ids,records,embeddings

def train(args):
    config=yaml.safe_load((ROOT/'configs/archiveii_frozen.yaml').read_text());validate_config(config)
    code=code_identity();inputs=input_identity()
    if not args.smoke:
        frozen=json.loads((OUT/'experiment_identity.json').read_text())
        assert frozen['training_source']==code and frozen['inputs']==inputs,'Frozen experiment changed'
    directory=OUT/('smoke/'+code['sha256'][:12] if args.smoke else 'runs')/args.family/args.mode/f'seed{args.seed}'
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=dict(status='loading',family=args.family,mode=args.mode,seed=args.seed,pid=os.getpid(),
            code_hash=code['sha256'],input_hash=inputs['sha256'],config_hash=digest(config))
        try:_train(args,config,code,inputs,directory,status)
        except Exception:
            status.update(status='failed',error=traceback.format_exc());write_json(directory/'status.json',status);raise

def _train(args,config,code,inputs,directory,status):
    if (directory/'summary.json').exists():
        summary=json.loads((directory/'summary.json').read_text())
        assert summary['code_hash']==code['sha256'] and summary['input_hash']==inputs['sha256']
        assert summary['config_hash']==digest(config) and summary['completed']
        assert sha(directory/'best.pt')==summary['best_checkpoint_sha256']
        print('ALREADY_COMPLETED',args.family,args.mode,args.seed,flush=True);return
    write_json(directory/'status.json',status)
    set_seed(args.seed)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('Require CUDA BF16')
    device=torch.device('cuda');model=ThermoFusion(args.mode).to(device)
    initial=state_hash(model.state_dict());head_initial=state_hash(model.head.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['initial_lr'],weight_decay=config['weight_decay'])
    ids,records,embeddings=load_inputs(args.family,('train','valid'),32 if args.smoke else None)
    manifest=load_manifest()
    assert all(manifest['records'][sid]['family']!=args.family for role in ids for sid in ids[role])
    metadata=dict(family=args.family,mode=args.mode,seed=args.seed,code_hash=code['sha256'],
        input_hash=inputs['sha256'],config_hash=digest(config),initialization_hash=initial,
        head_initialization_hash=head_initial,train_ids_hash=digest(ids['train']),validation_ids_hash=digest(ids['valid']))
    source_path=directory/'code_identity.json'
    if source_path.exists():assert json.loads(source_path.read_text())==code
    else:
        write_json(source_path,code);write_json(directory/'config.json',config)
        for name in code['files']:
            target=directory/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    start=1;rows=[];log_path=directory/'epochs.jsonl'
    def checkpoint(epoch):
        return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),**metadata)
    if (directory/'last.pt').exists():
        saved=torch.load(directory/'last.pt',map_location=device,weights_only=False)
        assert all(saved.get(key)==value for key,value in metadata.items()),'Resume identity mismatch'
        text=log_path.read_text() if log_path.exists() else '';rows=[]
        lines=text.splitlines()
        for index,line in enumerate(lines):
            if not line.strip():continue
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:
                if index!=len(lines)-1 or text.endswith('\n'):raise
        retained=min(saved['epoch'],rows[-1]['epoch'] if rows else 0)
        if retained!=saved['epoch']:
            write_json(directory/'incomplete_epoch_recovery.json',dict(saved_epoch=saved['epoch'],
                retained_complete_epoch=retained,reason='Resume only an atomically saved checkpoint with a complete validation log row.'))
            saved=torch.load(directory/f'epoch{retained}.pt',map_location=device,weights_only=False) if retained else None
        if saved:
            assert saved['epoch']==retained and all(saved.get(key)==value for key,value in metadata.items())
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
        start=retained+1
        rows=[row for row in rows if row['epoch']<start]
        assert [row['epoch'] for row in rows]==list(range(1,start))
        log_path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    maximum=2 if args.smoke else config['max_epochs']
    status.update(status='running',train_count=len(ids['train']),validation_count=len(ids['valid']),
        encoder_frozen=True,head_initialization_hash=head_initial,parameter_count=sum(p.numel() for p in model.parameters()))
    write_json(directory/'status.json',status)
    process_started=time.time()
    with log_path.open('a') as log:
        for epoch in range(start,maximum+1):
            def progress(step,loss,seconds):
                status.update(epoch=epoch,phase='training',step=step,training_bce=loss,epoch_seconds=seconds)
                write_json(directory/'status.json',status)
            status.update(epoch=epoch,phase='training',step=0);write_json(directory/'status.json',status)
            training=train_epoch(model,optimizer,ids['train'],embeddings,records,args.seed,epoch,config,device,progress)
            status.update(phase='validation');write_json(directory/'status.json',status)
            validation=evaluate_diagnostics(model,ids['valid'],embeddings,records,config['thresholds'],device)
            row=epoch_record(epoch,training,validation,'fresh_ArchiveII_frozen_E');rows.append(row)
            save_checkpoint(directory/f'epoch{epoch}.pt',checkpoint(epoch))
            save_checkpoint(directory/'last.pt',checkpoint(epoch))
            log.write(json.dumps(row)+'\n');log.flush()
            print('ARCHIVEII_FROZEN_EPOCH',args.family,args.mode,args.seed,epoch,'val_exact_F1',round(row['score'],6),flush=True)
    best=select_epoch(rows,maximum)
    shutil.copy2(directory/f"epoch{best['epoch']}.pt",directory/'best.pt')
    summary=dict(completed=True,**metadata,encoder_frozen=True,epochs_completed=maximum,
        parameter_count=sum(p.numel() for p in model.parameters()),train_count=len(ids['train']),validation_count=len(ids['valid']),
        best_epoch=best['epoch'],best_score=best['score'],best_validation=best['validation'],
        best_checkpoint_sha256=sha(directory/'best.pt'),epochs_sha256=sha(log_path),
        training_seconds=sum(row['training_seconds'] for row in rows),
        validation_seconds=sum(row['validation']['seconds'] for row in rows),
        gradients_verified=all(row['gradients_verified'] for row in rows),
        last_process_wall_seconds=time.time()-process_started,slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    write_json(directory/'summary.json',summary);status.update(status='completed',epoch=maximum,best_epoch=best['epoch'])
    write_json(directory/'status.json',status)
    print('ARCHIVEII_FROZEN_TRAIN_COMPLETE',args.family,args.mode,args.seed,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--family',choices=FAMILIES,required=True)
    parser.add_argument('--mode',choices=MODES,required=True);parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--smoke',action='store_true');train(parser.parse_args())
