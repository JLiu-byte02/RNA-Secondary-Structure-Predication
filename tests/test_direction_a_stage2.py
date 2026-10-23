import importlib
import importlib.util
import unittest

class StageTwoTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.direction_a_stage2'),'Matched-stage policy helpers missing')
        return importlib.import_module('scripts.direction_a_stage2')
    def test_exact_ties_prefer_less_harm_then_identity_strength(self):
        m=self.module();rows=[dict(f1=.7,harm=.2,beta=.5,threshold=.1),dict(f1=.7,harm=.1,beta=1.,threshold=.2),dict(f1=.7,harm=.1,beta=0.,threshold=.3)]
        self.assertEqual(m.choose(rows)['beta'],0.)
        self.assertEqual(m.choose(rows,budget=.15)['beta'],0.)
    def test_feasibility_and_nonfinite_fail_closed(self):
        m=self.module()
        with self.assertRaises(ValueError):m.choose([dict(f1=.8,harm=.2,beta=.5,threshold=.1)],budget=.1)
        with self.assertRaises(ValueError):m.choose([dict(f1=float('nan'),harm=0.,beta=0.,threshold=.1)])
    def test_uniform_grid_contains_legacy_and_low_thresholds(self):
        m=self.module();grid=m.thresholds([.05,.1,.95])
        self.assertTrue({.0005,.01,.02,.05,.1,.95}<=set(grid))
        self.assertEqual(grid,sorted(set(grid)))
    def test_family_chain_strict_exclusion_and_determinism(self):
        m=self.module();records={f+str(i):dict(family=f,sequence=f+str(i)) for f in ('a','b','c','d','e','f','g','dev','5s') for i in range(3)}
        old=dict(block_a=[k for k,v in records.items() if v['family'] in ('a','b','c','d')],block_b=[k for k,v in records.items() if v['family'] in ('e','f','g')],development=[k for k,v in records.items() if v['family']=='dev'])
        role=m.utility_roles(records,old,'dev')
        self.assertEqual(role,m.utility_roles(records,old,'dev'))
        groups=[{records[k]['family'] for k in role[n]} for n in ('base_train','proposal_fit','utility_fit','development')]
        self.assertEqual(list(map(len,groups)),[3,2,2,1]);self.assertFalse(any('5s' in g for g in groups))
        self.assertFalse(any(a&b for i,a in enumerate(groups) for b in groups[i+1:]))
        self.assertEqual({records[k]['sequence'] for k in role['utility_fit']} & {records[k]['sequence'] for k in role['utility_cal']},set())

if __name__=='__main__':unittest.main()
