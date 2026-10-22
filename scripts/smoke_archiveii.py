"""Verify FP16 gradients, actual 512nt capacity, official unfreezing and head parity."""
import gc
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'third_party/rinalmo_upstream'))
import numpy as np
import torch
import lightning.pytorch as pl
from rinalmo.data.alphabet import Alphabet
from rinalmo.utils.finetune_callback import GradualUnfreezing
from rinalmo.utils.sec_struct import prob_mat_to_sec_struct,ss_f1
from scripts.archiveii_data import OUT,load_manifest,write_json
from scripts.archiveii_models import FamilyWrapper
from scripts.archiveii_metrics import greedy_all_thresholds,pair_set,sparse_prf
from scripts.run_archiveii import FamilyDataset,MODES,tensor_hash,code_identity

def main():
    manifest=load_manifest();maximum=max(manifest['records'].values(),key=lambda r:r['length'])
    rows=[]
    for mode in MODES:
        started=time.time();pl.seed_everything(17,workers=True)
        model=FamilyWrapper(mode);model.load_pretrained_lm_weights(ROOT/'weights/rinalmo_giga_pretrained.pt')
        head_hash=tensor_hash(model.pred_head.state_dict())
        alphabet=Alphabet(**model.lm.config['alphabet'])
        family=next(f for f,v in manifest['folds'].items() if maximum['id'] in v['train'])
        dataset=FamilyDataset(manifest,family,'train',alphabet)
        ids,seq,tokens,truth,physical=dataset[dataset.ids.index(maximum['id'])]
        model=model.cuda().train();tokens=tokens[None].cuda();truth=truth[None].cuda();physical=physical[None].cuda()
        callback=GradualUnfreezing(str(ROOT/'third_party/rinalmo_reference/giga_sec_struct_ft.yaml'))
        callback.freeze_before_training(model)
        optimizer=model.configure_optimizers()['optimizer'];phases=[]
        assert not any(p.requires_grad for p in model.lm.parameters())
        for epoch in (0,3,6,9,12):
            if epoch:callback.finetune_function(model,epoch,optimizer)
            names=[n for n,p in model.lm.named_parameters() if p.requires_grad]
            block_ids=sorted({int(n.split('.')[2]) for n in names if n.startswith('transformer.blocks.')})
            expected=list(range(33-3*(epoch//3),33)) if epoch else []
            assert block_ids==expected,(epoch,block_ids,expected)
            phases.append(dict(epoch=epoch,trainable_blocks=block_ids,optimizer_groups=len(optimizer.param_groups)))
        torch.cuda.reset_peak_memory_stats();before=model.pred_head.head.out.weight.detach().clone()
        # This small capacity probe uses a conservative scaler; formal Trainer uses its default scaler.
        scaler=torch.cuda.amp.GradScaler(init_scale=1024)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            logits=model(tokens,physical,[seq]);mask=torch.triu(torch.ones(len(seq),len(seq),device='cuda',dtype=torch.bool),1)
            loss=model.loss(logits[...,mask],truth[...,mask])
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in model.lm.parameters() if p.requires_grad)
        scaler.step(optimizer);scaler.update()
        assert not torch.equal(before,model.pred_head.head.out.weight.detach())
        probs=torch.sigmoid(logits[0]).detach().cpu().numpy();target=truth[0].cpu().numpy()
        thresholds=[i/100 for i in range(1,30)];fast=greedy_all_thresholds(probs,seq,thresholds)
        for t in thresholds:
            reference=prob_mat_to_sec_struct(probs.copy(),seq,t)
            assert fast[t]==pair_set(reference)
            assert abs(sparse_prf(pair_set(target),fast[t],len(seq))[2]-ss_f1(target,reference))<1e-15
        rows.append(dict(mode=mode,seed=17,maximum_sequence_length=len(seq),rna_id=ids,
            head_initialization_hash=head_hash,phases=phases,finite_nonzero_encoder_gradients=True,
            head_updated=True,official_threshold_parity_checks=29,fp16_loss=float(loss.detach()),
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,seconds=time.time()-started))
        print('ARCHIVEII_SMOKE_PASSED',mode,len(seq),rows[-1]['peak_allocated_GiB'],flush=True)
        del loss,logits,optimizer,model,scaler,dataset,tokens,truth,physical,before
        gc.collect();torch.cuda.empty_cache()
    assert len({r['head_initialization_hash'] for r in rows})==1
    write_json(OUT/'preparation_complete.json',dict(passed=True,runs=rows,head_initialization_identical=True,training_source=code_identity(),
        note='Smoke updates are discarded. Formal runs reinitialize and use the official Lightning FP16 scaler.'))

if __name__=='__main__':main()
