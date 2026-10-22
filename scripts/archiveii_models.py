"""Original E heads with online, gradually unfrozen RiNALMo representations."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'third_party/rinalmo_upstream'))
from src.pair_head import PairHead
from scripts.run_thermo_fast import CorrectionCNN
from scripts.archiveii_metrics import validation_scores,pair_set
from rinalmo.utils.sec_struct import prob_mat_to_sec_struct,ss_precision,ss_recall,ss_f1

spec=importlib.util.spec_from_file_location('rinalmo_official_secondary',ROOT/'third_party/rinalmo_reference/train_sec_struct_prediction.py')
official=importlib.util.module_from_spec(spec);sys.modules[spec.name]=official;spec.loader.exec_module(official)

class ProtocolPairHead(PairHead):
    def forward(self,embedding):
        # Respect the enclosing official FP16-mixed context; old BF16 runs are untouched.
        return self._forward(embedding)

class FamilyEHead(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode;self.head=ProtocolPairHead();self.correction=CorrectionCNN()

    def forward(self,embedding,physical,sequences):
        assert embedding.shape[0]==1
        raw=self.head(embedding)[0];initial=raw.float();length=raw.shape[0]
        seq=sequences[0].replace('T','U')
        alphabet={'A':0,'C':1,'G':2,'U':3}
        nts=torch.tensor([alphabet.get(n,4) for n in seq],device=raw.device)
        a,b=nts[:,None],nts[None,:]
        canonical=((a==0)&(b==3))|((a==3)&(b==0))|((a==1)&(b==2))|((a==2)&(b==1))|((a==2)&(b==3))|((a==3)&(b==2))
        positions=torch.arange(length,device=raw.device,dtype=torch.float32)
        separation=(positions[:,None]-positions[None,:]).abs()
        distance=separation/length;valid=canonical&(separation>=4)
        zero=torch.zeros_like(initial)
        bpp=physical[0,0].float()*valid if self.mode!='E0_no_thermo' else zero
        stack=physical[0,1].float()*valid if self.mode=='E2_bpp_stack' else zero
        residual=self.correction(torch.stack((initial,bpp,stack,distance,valid.float()))).float()
        corrected=initial+residual;corrected=(corrected+corrected.T)/2
        # No hard candidate mask before the official all-upper-triangle BCE.
        return corrected.to(raw.dtype).unsqueeze(0)

class AuditMixin:
    prediction_stream=None
    def _update_eval_step_outputs(self,logits,sec_struct_true,ss_ids,seqs,thresholds):
        probs=torch.sigmoid(logits)
        if probs.dtype==torch.bfloat16:probs=probs.type(torch.float16)
        probs=probs.detach().cpu().numpy();truth=sec_struct_true.detach().cpu().numpy()
        for index in range(len(ss_ids)):
            if not self.trainer.testing:
                scores=validation_scores(probs[index],seqs[index],truth[index],thresholds)
                for threshold,(p,r,f) in scores.items():
                    for name,value in (('precision',p),('recall',r),('f1',f)):self._eval_step_outputs[name][threshold].append(value)
            else:
                assert len(thresholds)==1
                threshold=thresholds[0]
                pred=prob_mat_to_sec_struct(probs[index].copy(),seqs[index],threshold)
                p=ss_precision(truth[index],pred);r=ss_recall(truth[index],pred);f=ss_f1(truth[index],pred)
                for name,value in (('precision',p),('recall',r),('f1',f)):self._eval_step_outputs[name][threshold].append(value)
                record=dict(rna_id=ss_ids[index],sequence=seqs[index],threshold=float(threshold),
                    true_pairs=sorted([int(a),int(b)] for a,b in pair_set(truth[index])),
                    predicted_pairs=sorted([int(a),int(b)] for a,b in pair_set(pred)),
                    precision=float(p),recall=float(r),f1=float(f),
                    exact_precision=float(ss_precision(truth[index],pred,False)),
                    exact_recall=float(ss_recall(truth[index],pred,False)),exact_f1=float(ss_f1(truth[index],pred,False)))
                if self.prediction_stream:
                    import json
                    self.prediction_stream.write(json.dumps(record)+'\n');self.prediction_stream.flush()

class AuditedOfficialWrapper(AuditMixin,official.SecStructPredictionWrapper):pass

class FamilyWrapper(AuditedOfficialWrapper):
    def __init__(self,mode,lr=1e-4):
        super().__init__(lm_config='giga',lr=lr)
        self.pred_head=FamilyEHead(mode)

    def forward(self,tokens,physical,sequences):
        embedding=self.lm(tokens)['representation'][...,1:-1,:]
        return self.pred_head(embedding,physical,sequences)

    def _common_step(self,batch,batch_idx,log_prefix):
        ss_ids,seqs,tokens,truth,physical=batch
        logits=self(tokens,physical,seqs);length=truth.shape[1]
        mask=torch.triu(torch.ones(length,length,dtype=torch.bool,device=logits.device),diagonal=1)
        loss=self.loss(logits[...,mask],truth[...,mask])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite official upper-triangle BCE')
        self.log(f'{log_prefix}/loss',loss,batch_size=1)
        return loss,logits

    def _eval_step(self,batch,batch_idx,thresholds,log_prefix='eval'):
        ids,seqs,_,truth,_=batch
        _,logits=self._common_step(batch,batch_idx,log_prefix)
        if self.trainer.testing or (self.trainer.current_epoch+1)%self.tune_threshold_every_n_epoch==0:
            self._update_eval_step_outputs(logits,truth,ids,seqs,thresholds)
