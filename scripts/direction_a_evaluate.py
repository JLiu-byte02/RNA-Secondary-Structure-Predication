"""Full-truth metrics, calibration and auditable structural-edit accounting."""
from collections import defaultdict
import numpy as np
from scripts.archiveii_frozen_metrics import sparse_prf,greedy_all_thresholds

def score(truth,pred,length,flexible=False):
    return dict(zip(('precision','recall','f1'),sparse_prf(set(truth),set(pred),length,flexible)))

def edits(base,pred,truth):
    base,pred,truth=map(set,(base,pred,truth));added=pred-base;removed=base-pred
    helpful=len(added&truth)+len(removed-truth)
    harmful=len(added-truth)+len(removed&truth)
    return dict(added=len(added),removed=len(removed),helpful=helpful,harmful=harmful,
                harmful_rate=harmful/(helpful+harmful) if helpful+harmful else 0.,
                net_correct_change=helpful-harmful)

def select_threshold(scores):
    threshold=max(sorted(scores),key=lambda t:scores[t]);return threshold,scores[threshold]

def aggregate(rows):
    families=defaultdict(list)
    for row in rows:families[row['family']].append(row)
    keys=('exact_f1','exact_precision','exact_recall','flexible_f1','harmful_rate','helpful','harmful')
    per_family={f:{k:float(np.mean([r[k] for r in values])) for k in keys if k in values[0]} for f,values in families.items()}
    return dict(n=len(rows),families=per_family,equal_family={k:float(np.mean([r[k] for r in per_family.values()])) for k in next(iter(per_family.values()))},
                rna_macro={k:float(np.mean([r[k] for r in rows])) for k in keys if k in rows[0]})

def metric_row(sid,record,pred,base=None):
    truth=set(record['reference']);length=len(record['sequence']);pred=set(pred)
    row=dict(sample_id=sid,family=record['family'],length=length,prediction=sorted(pred),reference=sorted(truth))
    for prefix,flexible in [('exact',False),('flexible',True)]:
        row.update({prefix+'_'+k:v for k,v in score(truth,pred,length,flexible).items()})
    if base is not None:row.update(edits(base,pred,truth))
    return row

def probabilities(logits):
    return 1./(1.+np.exp(-np.asarray(logits,dtype=np.float64).clip(-100,100)))

def calibrate(records,ids,probabilities_by_id,thresholds):
    family_sums={t:defaultdict(list) for t in thresholds}
    for sid in ids:
        r=records[sid]
        for t,pred in greedy_all_thresholds(probabilities_by_id[sid],r['sequence'],thresholds).items():
            family_sums[t][r['family']].append(sparse_prf(set(r['reference']),pred,len(r['sequence']),False)[2])
    scores={t:float(np.mean([np.mean(v) for v in families.values()])) for t,families in family_sums.items()}
    threshold,value=select_threshold(scores)
    return dict(threshold=threshold,equal_family_exact_f1=value,threshold_scores=scores)

def physical_structure(sequence):
    """ViennaRNA2.6.4 Turner2004/37C MFE and fixed gamma=1 MEA, no truth access."""
    import RNA
    RNA.params_load_RNA_Turner2004();md=RNA.md();md.temperature=37.;md.dangles=2;md.noLP=False;md.uniq_ML=True
    md.compute_bpp=True;md.min_loop_size=3
    fc=RNA.fold_compound(sequence,md);mfe,energy=fc.mfe();fc.exp_params_rescale(energy);fc.pf();mea,value=fc.MEA(1.)
    def parse(text):
        stack=[];pairs=[]
        for i,c in enumerate(text):
            if c=='(':stack.append(i)
            elif c==')':pairs.append((stack.pop(),i))
        assert not stack
        return sorted(pairs)
    return dict(mfe=parse(mfe),mea=parse(mea),mfe_energy=float(energy),mea_value=float(value))

def official_verify(rows):
    from third_party.rinalmo_frozen_reference.sec_struct import ss_precision,ss_recall,ss_f1
    for row in rows:
        truth=np.zeros((row['length'],row['length']));pred=np.zeros_like(truth)
        for a,b in row['reference']:truth[a,b]=truth[b,a]=1
        for a,b in row['prediction']:pred[a,b]=pred[b,a]=1
        for prefix,flexible in [('exact',False),('flexible',True)]:
            for key,fn in [('precision',ss_precision),('recall',ss_recall),('f1',ss_f1)]:
                assert abs(row[prefix+'_'+key]-fn(truth,pred,allow_flexible_pairings=flexible))<1e-10
    return dict(passed=True,n=len(rows))
