"""Bounded physical correction; a frozen base is an input, never a parameter."""
import torch
from torch import nn
import torch.nn.functional as F

METHODS=('residual','global_gate','local_gate','safe_local_gate')

def physical_entropy(p):
    p=p.detach().float().clamp(0,1)
    if (p.sum(-1)>1.001).any():raise ValueError('Physical probability mass exceeds one')
    unpaired=(1-p.sum(-1)).clamp(0,1)
    return -(p*p.clamp_min(1e-12).log()).sum(-1)-unpaired*unpaired.clamp_min(1e-12).log()

class LocalCorrection(nn.Module):
    def __init__(self,method):
        super().__init__();assert method in METHODS;self.method=method
        self.layers=nn.Sequential(nn.Conv2d(7,32,3,padding=1),nn.GELU(),
                                  nn.Conv2d(32,32,3,padding=1),nn.GELU(),nn.Conv2d(32,1,1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.constant_(self.layers[-1].bias,0 if method=='residual' else -4.)
    def forward(self,base,p,valid):
        base=base.detach().float();p=p.detach().float();valid=valid.detach()
        entropy=physical_entropy(p);length=base.shape[0]
        assert base.shape==p.shape==valid.shape and torch.allclose(base,base.T,atol=1e-5)
        physical=torch.logit(p.clamp(1e-4,1-1e-4))
        position=torch.arange(length,device=base.device).float()
        distance=(position[:,None]-position[None,:]).abs()/length
        features=torch.stack((base.clamp(-12,12)/8,physical/8,
            entropy[:,None].expand_as(p),entropy[None,:].expand_as(p),
            (base.sigmoid()-p).abs(),distance,valid.float()))
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=base.is_cuda):
            raw=self.layers(features.unsqueeze(0))[0,0].float()
        raw=(raw+raw.T)/2
        if self.method=='residual':out=base+raw;gate=torch.zeros_like(raw)
        else:
            if self.method=='global_gate':raw=((raw*valid).sum()/valid.sum().clamp_min(1)).expand_as(raw)
            gate=raw.sigmoid();out=base+gate*(physical-base)
        return out.masked_fill(~valid,-100.),gate.masked_fill(~valid,0.)

def correction_loss(logits,base,ci,cj,labels,threshold,harm_lambda):
    score=logits[ci,cj];original=base.detach()[ci,cj]
    bce=F.binary_cross_entropy_with_logits(score.float(),labels.float())
    correct=(original.sigmoid()>=threshold)==(labels>.5)
    change=(score.sigmoid()-original.sigmoid()).abs()
    terms=[change[correct & mask].mean() for mask in (labels>.5,labels<=.5) if (correct & mask).any()]
    harm=torch.stack(terms).mean() if terms else score.sum()*0
    return bce+harm_lambda*harm,dict(bce=float(bce.detach()),harm=float(harm.detach()))
