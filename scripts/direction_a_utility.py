"""Matching-safe edit inference. Truth appears only in the training-target helper."""
import math
from scripts.direction_a_edit_packets import valid_matching

FEATURES=('n_add','n_remove','n_edit','max_stem','span_mean','span_min','span_max','endpoint_fraction',
          'base_add','base_remove','P_add','P_remove','base_signed','P_signed','entropy','cross_add','cross_remove','base_density')

def apply_packets(base,packets,accept):
    if len(packets)!=len(accept):raise ValueError('Acceptance coverage differs')
    prediction=set(map(tuple,base))
    for packet,keep in zip(packets,accept):
        if keep:prediction.difference_update(map(tuple,packet['removed']));prediction.update(map(tuple,packet['added']))
    if not valid_matching(prediction):raise ValueError('An edit subset violates one-partner matching')
    return prediction

def target(packet,truth):
    truth=set(map(tuple,truth));add=set(map(tuple,packet['added']));remove=set(map(tuple,packet['removed']))
    return (len(add&truth)-len(remove&truth))/(len(add)+len(remove))

def predict_changes(packet,normalized):
    na,nr=len(packet['added']),len(packet['removed']);change=max(-nr,min(na,float(normalized)*(na+nr)))
    return change,na-nr-change

def features(packet,base,logits,physical,entropy):
    added=list(map(tuple,packet['added']));removed=list(map(tuple,packet['removed']));all_pairs=added+removed
    if not all_pairs:raise ValueError('Empty edit packet')
    length=len(logits);n=len(all_pairs);base=set(map(tuple,base))
    mean=lambda v:sum(v)/len(v) if v else 0.
    prob=lambda a,b:1./(1.+math.exp(-max(-100.,min(100.,float(logits[a][b])))))
    getp=lambda a,b:float(physical[a][b])
    spans=[(b-a)/length for a,b in all_pairs];ends={v for p in all_pairs for v in p}
    def stem_max(pairs):
        pairs=set(pairs);maximum=0
        for a,b in pairs:
            if (a-1,b+1) in pairs:continue
            size=0
            while (a+size,b-size) in pairs:size+=1
            maximum=max(maximum,size)
        return maximum
    def crossings(pairs):return sum(a<c<b<d or c<a<d<b for a,b in pairs for c,d in base)/(max(1,len(base))*max(1,len(pairs)))
    return [len(added)/20,len(removed)/20,n/20,max(stem_max(added),stem_max(removed))/20,
            mean(spans),min(spans),max(spans),len(ends)/length,mean([prob(*p) for p in added]),mean([prob(*p) for p in removed]),
            mean([getp(*p) for p in added]),mean([getp(*p) for p in removed]),
            (sum(prob(*p) for p in added)-sum(prob(*p) for p in removed))/n,
            (sum(getp(*p) for p in added)-sum(getp(*p) for p in removed))/n,
            mean([float(entropy[i]) for i in ends])/max(1.,math.log(length)),crossings(added),crossings(removed),len(base)/length]
