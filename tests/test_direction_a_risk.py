import importlib
import importlib.util
import unittest

class RiskCalibrationTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.direction_a_risk'),'Risk calibration is missing')
        return importlib.import_module('scripts.direction_a_risk')
    def test_risk_budget_rejects_higher_f1_but_unsafe_policy(self):
        m=self.module();profiles=[dict(beta=0.,threshold=.1,f1=.5,harm=0.),dict(beta=.5,threshold=.2,f1=.7,harm=.15),dict(beta=1.,threshold=.1,f1=.8,harm=.3)]
        self.assertEqual(m.choose_policy(profiles,.2)['beta'],.5)
    def test_exact_ties_prefer_less_harm_then_smaller_beta(self):
        m=self.module();profiles=[dict(beta=1.,threshold=.2,f1=.7,harm=.1),dict(beta=.5,threshold=.3,f1=.7,harm=.1),dict(beta=0.,threshold=.1,f1=.7,harm=.2)]
        self.assertEqual(m.choose_policy(profiles,.2)['beta'],.5)
    def test_family_risk_is_not_confused_with_pooled_edit_rate(self):
        m=self.module();rows=[dict(family='a',f1=.8,harm=.1),dict(family='a',f1=.8,harm=.1),dict(family='b',f1=.2,harm=.7)]
        result=m.family_metrics(rows);self.assertAlmostEqual(result['f1'],.5);self.assertAlmostEqual(result['harm'],.4)
    def test_revision_cannot_promote_by_weakening_simple_baseline(self):
        m=self.module();rows={case:dict(candidate_f1=.62,risk_baseline_f1=.60,original_baseline_f1=.64,candidate_harm=.1,residual_harm=.2) for case in ('a','b')}
        cfg=dict(promotion_mean_exact_gain=.005,promotion_max_family_loss=.01)
        self.assertFalse(m.revision_decision(rows,cfg)['promote'])
        for r in rows.values():r['original_baseline_f1']=.60
        self.assertTrue(m.revision_decision(rows,cfg)['promote'])
    def test_extrapolation_is_rejected_by_bounded_policy(self):
        m=self.module()
        with self.assertRaises(AssertionError):m.choose_policy([dict(beta=1.2,threshold=.1,f1=.8,harm=0.)],.2)

if __name__=='__main__':unittest.main()
