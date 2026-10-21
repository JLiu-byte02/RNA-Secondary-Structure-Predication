"""Per-epoch validation diagnostics, with full-truth structured F1 as primary."""
import time
import torch
from src.decoder import decode_multi
from src.losses import pair_bce_loss
from src.metrics import pair_prf,rna_macro
from scripts.run_thermo_fast import model_inputs

@torch.no_grad()
def evaluate_diagnostics(model,ids,embeddings,records,thresholds,device):
    model.eval()
    if device.type=='cuda':torch.cuda.synchronize()
    started=time.perf_counter(); losses=[];counts=[0,0,0,0]
    scores={t:[] for t in thresholds}
    for sid in ids:
        record=records[sid]
        logits=model(*model_inputs(record,embeddings[sid],device))
        if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite validation: '+sid)
        ci,cj=record['cand_i'].to(device),record['cand_j'].to(device)
        labels=record['labels'].to(device)
        if ci.numel():
            loss=pair_bce_loss(logits,ci,cj,labels);losses.append(loss.item())
            pred=logits[ci,cj]>0;truth=labels>0.5
            for k,value in enumerate(((pred&truth).sum(),(pred&~truth).sum(),(~pred&~truth).sum(),(~pred&truth).sum())):
                counts[k]+=int(value)
        predictions=decode_multi(logits.cpu().numpy(),record['sequence'],thresholds)
        for t,pairs in predictions.items():scores[t].append(pair_prf(record['original'],pairs))
    if not losses:raise RuntimeError('No validation candidates')
    tp,fp,tn,fn=counts
    if device.type=='cuda':torch.cuda.synchronize()
    return dict(n=len(ids),mean_pair_bce=sum(losses)/len(losses),
        candidate_accuracy=(tp+tn)/sum(counts),candidate_balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)),
        candidate_confusion=dict(tp=tp,fp=fp,tn=tn,fn=fn),
        threshold_metrics={str(t):dict(zip(('precision','recall','rna_macro_exact_f1'),rna_macro(v))) for t,v in scores.items()},
        seconds=time.perf_counter()-started)
