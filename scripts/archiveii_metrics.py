"""Equivalent sparse acceleration for the official validation threshold search.

Final test scores are independently computed with the unmodified official functions.
Equal-probability ties follow the official full-matrix np.argmax row-major order.
"""
import numpy as np

CANONICAL={'AU','UA','CG','GC','GU','UG'}

def pair_set(matrix):return set(zip(*np.where(np.triu(matrix,1)>0)))

def relax_pairs(pairs,length):
    return {(a+di,b+dj) for a,b in pairs for di,dj in ((0,0),(1,0),(-1,0),(0,1),(0,-1))
        if 0<=a+di<b+dj<length}

def sparse_prf(target,pred,length,flexible=True):
    precision_hits=len(pred & (relax_pairs(target,length) if flexible else target))
    recall_hits=len(target & (relax_pairs(pred,length) if flexible else pred))
    precision=precision_hits/len(pred) if pred else 0.
    recall=recall_hits/len(target) if target else 0.
    f1=2*precision*recall/(precision+recall) if precision+recall>=1e-5 else 0.
    return precision,recall,f1

def greedy_all_thresholds(probs,sequence,thresholds):
    length=len(sequence);seq=sequence.replace('T','U')
    assert probs.shape==(length,length) and np.isfinite(probs).all()
    assert np.all(np.isclose(probs,probs.T))
    i,j=np.triu_indices(length,k=4)
    alphabet={'A':0,'C':1,'G':2,'U':3}
    encoded=np.asarray([alphabet.get(n,4) for n in seq],dtype=np.int8)
    table=np.zeros((5,5),dtype=bool)
    for pair in CANONICAL:table[alphabet[pair[0]],alphabet[pair[1]]]=True
    keep=table[encoded[i],encoded[j]]
    values=probs[i,j];keep &= values>min(thresholds)
    i,j,values=i[keep],j[keep],values[keep]
    order=np.lexsort((j,i,-values))
    occupied=set();accepted=[]
    for index in order:
        left,right=int(i[index]),int(j[index])
        if left in occupied or right in occupied:continue
        occupied.update((left,right));accepted.append((left,right,float(values[index])))
    # Array comparison preserves NumPy's source-dtype threshold rounding.
    accepted_values=np.asarray([v for _,_,v in accepted],dtype=probs.dtype)
    return {t:{(a,b) for (a,b,_),keep in zip(accepted,np.greater(accepted_values,t)) if keep} for t in thresholds}

def validation_scores(probs,sequence,target,thresholds):
    truth=pair_set(target)
    return {threshold:sparse_prf(truth,pred,len(sequence))
        for threshold,pred in greedy_all_thresholds(probs,sequence,thresholds).items()}
