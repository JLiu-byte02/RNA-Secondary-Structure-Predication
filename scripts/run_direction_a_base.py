"""Fresh fixed-horizon base fits and family-out-of-fold logit caches."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.direction_a_protocol import OUT,config,digest,file_sha,source_identity,write_json,cache_key
from scripts.direction_a_data import prepare_roles,load_records
from scripts._common import set_seed
from scripts.long_models import B0
from scripts.run_long_epochs import train_epoch
from scripts.run_thermo_fast import save_checkpoint,state_hash,model_inputs

def run(args):
    cfg=config();role_manifest=prepare_roles();roles=role_manifest['cases'][args.case]
    train=roles['block_a'] if args.part=='a' else roles['block_b'] if args.part=='b' else sorted(roles['block_a']+roles['block_b'])
    predict=roles['block_b'] if args.part=='a' else roles['block_a'] if args.part=='b' else roles['development']
    if args.smoke:train=train[:8];predict=predict[:4]
    identity=dict(code=source_identity('base'),inputs=role_manifest['historical_inputs'],roles_sha256=role_manifest['sha256'],
                  case=args.case,part=args.part,seed=args.seed,train_ids_sha256=digest(train),predict_ids_sha256=digest(predict),smoke=args.smoke)
    directory=OUT/('smoke' if args.smoke else 'pilot')/args.case/f'base_{args.part}_seed{args.seed}'
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (directory/'summary.json').exists():
            old=json.loads((directory/'summary.json').read_text());assert old['identity']==identity
            assert old['checkpoint_sha256']==file_sha(directory/'final.pt');print('BASE_ALREADY_COMPLETE',args.case,args.part);return
        write_json(directory/'status.json',dict(status='loading',identity=identity,pid=os.getpid()))
        set_seed(args.seed);assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        device=torch.device('cuda');model=B0().to(device);initial=state_hash(model.state_dict())
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['initial_lr'],weight_decay=cfg['weight_decay'])
        records,embeddings=load_records(sorted(train+predict),True)
        rows=[];checkpoint=directory/'last.pt'
        if checkpoint.exists():
            old=torch.load(checkpoint,map_location=device,weights_only=False);assert old['identity']==identity
            model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer']);rows=old['rows']
            torch.set_rng_state(old['torch_rng'].cpu());torch.cuda.set_rng_state_all([t.cpu() for t in old['cuda_rng']])
        maximum=2 if args.smoke else cfg['base_epochs'];start=time.time()
        for epoch in range(len(rows)+1,maximum+1):
            row=dict(epoch=epoch,**train_epoch(model,optimizer,train,embeddings,records,args.seed,epoch,cfg,device))
            assert row['n_steps']==len(train);rows.append(row)
            save_checkpoint(checkpoint,dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),rows=rows,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),initialization_hash=initial))
            write_json(directory/'status.json',dict(status='training',epoch=epoch,maximum=maximum,n=len(train)))
            print('DIRECTION_A_BASE_EPOCH',args.case,args.part,args.seed,epoch,round(row['mean_pair_loss'],6),flush=True)
        assert [r['epoch'] for r in rows]==list(range(1,maximum+1))
        save_checkpoint(directory/'final.pt',dict(identity=identity,model=model.state_dict(),epochs=maximum,initialization_hash=initial))
        (directory/'epochs.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        cache=directory/'logits';cache.mkdir(exist_ok=True);model.eval();artifacts={}
        with torch.no_grad():
            for sid in predict:
                value=model(*model_inputs(records[sid],embeddings[sid],device)).float().cpu().numpy()
                assert np.isfinite(value).all();path=cache/f'{cache_key(sid)}.npz'
                temporary=path.with_suffix('.tmp.npz');np.savez_compressed(temporary,base=value);temporary.replace(path)
                artifacts[sid]=dict(path=path.relative_to(ROOT).as_posix(),sha256=file_sha(path))
        summary=dict(completed=True,identity=identity,epochs=maximum,train_n=len(train),prediction_n=len(predict),
            train_ids=train,prediction_ids=predict,initialization_hash=initial,checkpoint_sha256=file_sha(directory/'final.pt'),
            log_sha256=file_sha(directory/'epochs.jsonl'),artifacts=artifacts,
            training_seconds=sum(r['training_seconds'] for r in rows),last_process_seconds=time.time()-start,slurm_job_id=os.environ.get('SLURM_JOB_ID'))
        write_json(directory/'summary.json',summary);write_json(directory/'status.json',dict(status='completed',epochs=maximum))
        print('DIRECTION_A_BASE_COMPLETE',args.case,args.part,len(train),len(predict),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=('srp','tmRNA'),required=True)
    p.add_argument('--part',choices=('a','b','full'),required=True);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--smoke',action='store_true');run(p.parse_args())
