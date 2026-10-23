import unittest
import numpy as np
from scripts.direction_a_evaluate import score,edits,select_threshold
from scripts.archiveii_frozen_metrics import greedy_all_thresholds,pair_set
from third_party.rinalmo_frozen_reference.sec_struct import prob_mat_to_sec_struct,ss_precision as precision,ss_recall as recall,ss_f1 as f1

class EvaluationTests(unittest.TestCase):
    def test_sparse_official_parity_with_ties_and_empty(self):
        rng=np.random.default_rng(17)
        for length in (9,24):
            seq=('ACGU'*20)[:length];p=rng.choice([.1,.3,.5,.8],size=(length,length));p=np.triu(p,1);p+=p.T
            target={(0,length-1)};matrix=np.zeros_like(p)
            for a,b in target:matrix[a,b]=matrix[b,a]=1
            for t,pred in greedy_all_thresholds(p,seq,[.1,.5,.95]).items():
                official=prob_mat_to_sec_struct(p,seq,t)
                self.assertEqual(pred,pair_set(official))
                for flexible in (False,True):
                    measured=score(target,pred,length,flexible)
                    for key,fn in [('precision',precision),('recall',recall),('f1',f1)]:
                        self.assertAlmostEqual(measured[key],fn(matrix,official,allow_flexible_pairings=flexible))
        self.assertEqual(score(set(),set(),9)['f1'],0)
    def test_edit_accounting_and_tie_selection(self):
        result=edits({(0,5),(1,6)},{(0,5),(2,7)},{(1,6),(2,7)})
        self.assertEqual(result['helpful'],1);self.assertEqual(result['harmful'],1)
        self.assertEqual(result['net_correct_change'],0)
        self.assertEqual(select_threshold({.5:.2,.1:.2})[0],.1)

if __name__=='__main__':unittest.main()
