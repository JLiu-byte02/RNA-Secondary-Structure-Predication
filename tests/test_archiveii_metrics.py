"""Parity tests target tied FP16 probabilities, strict thresholds and relaxed metrics."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.archiveii_metrics import greedy_all_thresholds,pair_set,sparse_prf
spec=importlib.util.spec_from_file_location('official_ss',ROOT/'third_party/rinalmo_reference/sec_struct_reference.py')
official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)

def main():
    rng=np.random.default_rng(731)
    thresholds=[i/100 for i in range(1,30)]
    checked=0
    for index in range(70):
        length=int(rng.integers(5,90));seq=''.join(rng.choice(list('ACGUTN'),length))
        probs=np.triu(rng.random((length,length)),1);probs+=probs.T
        if index%2:probs=(np.round(probs*20)/20).astype(np.float16)
        target=np.triu((rng.random((length,length))<.025).astype(int),1);target+=target.T
        fast=greedy_all_thresholds(probs,seq,thresholds)
        for t in thresholds:
            pred=official.prob_mat_to_sec_struct(probs.copy(),seq,t)
            assert fast[t]==pair_set(pred),(index,t)
            for flexible in (True,False):
                result=sparse_prf(pair_set(target),fast[t],length,flexible)
                ref=(official.ss_precision(target,pred,flexible),official.ss_recall(target,pred,flexible),official.ss_f1(target,pred,flexible))
                assert np.allclose(result,ref,rtol=0,atol=1e-15),(result,ref)
            checked+=1
    # Flexible scores are intentionally distinct from exact scores.
    assert sparse_prf({(1,8)},{(1,9)},12,True)==(1.,1.,1.)
    assert sparse_prf({(1,8)},{(1,9)},12,False)==(0.,0.,0.)
    print('OFFICIAL_DECODER_AND_METRIC_PARITY_PASSED',checked,'threshold comparisons')

if __name__=='__main__':main()
