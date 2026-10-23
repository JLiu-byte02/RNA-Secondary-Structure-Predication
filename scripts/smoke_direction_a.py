"""Real CUDA finite-gradient and frozen-score checks on longest eligible RNA."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.direction_a_data import prepare_roles,load_records
from scripts.direction_a_models import METHODS,LocalCorrection,correction_loss
from scripts.direction_a_protocol import OUT,write_json
from scripts.archiveii_frozen_data import load_manifest

def run():
    roles=prepare_roles()['cases']['srp'];manifest=load_manifest()
    sid=max(roles['correction_fit'],key=lambda s:len(manifest['records'][s]['sequence']))
    r=load_records([sid])[sid];device='cuda';assert torch.cuda.is_bf16_supported()
    p=r['physical'][0].to(device);valid=r['valid'].to(device)
    base=torch.zeros_like(p,requires_grad=True);results={}
    for method in METHODS:
        model=LocalCorrection(method).to(device);out,gate=model(base,p,valid)
        loss,terms=correction_loss(out,base,r['cand_i'].to(device),r['cand_j'].to(device),r['labels'].to(device),.5,.1)
        loss.backward();assert base.grad is None
        grads=[v.grad for v in model.parameters() if v.grad is not None]
        assert torch.isfinite(out).all() and all(torch.isfinite(g).all() for g in grads) and any(g.abs().max()>0 for g in grads)
        results[method]=dict(loss=float(loss),terms=terms,parameters=sum(p.numel() for p in model.parameters()))
    write_json(OUT/'smoke/cuda_longest.json',dict(passed=True,sample_id=sid,length=len(r['sequence']),results=results))
    print(json.dumps(dict(passed=True,length=len(r['sequence']),methods=list(results))))
if __name__=='__main__':run()
