import torch
import torch.nn.functional as F

def pair_bce_loss(logits,cand_i,cand_j,labels,pos_weight=None):
 if cand_i.numel()==0:return None
 s=logits[cand_i,cand_j]
 assert torch.isfinite(s).all()
 return F.binary_cross_entropy_with_logits(s.float(),labels.float(),pos_weight=pos_weight)
def values(logits,pairs):
 ij=torch.tensor(pairs,device=logits.device,dtype=torch.long)
 return logits[ij[:,0],ij[:,1]].float()
def stem_score(logits,pairs):return values(logits,pairs).mean() if pairs else None

def ranking_loss(logits,anchors_pos,anchors_neg,margin=1.0):
 anchors=[]
 for pos,negs in zip(anchors_pos,anchors_neg):
  terms=[F.relu(margin-stem_score(logits,pos)+stem_score(logits,neg)) for neg in negs if neg]
  if terms:anchors.append(torch.stack(terms).mean())
 return torch.stack(anchors).mean() if anchors else None

def hard_pair_bce_loss(logits,anchors_pos,anchors_neg):
 anchors=[]
 for pos,negs in zip(anchors_pos,anchors_neg):
  terms=[]
  p=values(logits,pos);lp=F.binary_cross_entropy_with_logits(p,torch.ones_like(p))
  for neg in negs:
   if neg:
    n=values(logits,neg);ln=F.binary_cross_entropy_with_logits(n,torch.zeros_like(n))
    terms.append(.5*lp+.5*ln)
  if terms:anchors.append(torch.stack(terms).mean())
 return torch.stack(anchors).mean() if anchors else None

def combine(pair_loss,aux_loss,lam):return (pair_loss,0.) if aux_loss is None else (pair_loss+lam*aux_loss,float(aux_loss.detach()))
