import importlib
import importlib.util
import unittest

class BudgetTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.direction_a_control'),'Budget controller is missing')
        return importlib.import_module('scripts.direction_a_control')
    def test_running_job_reserves_full_remaining_walltime(self):
        m=self.module();jobs=[dict(job_id='1',wall_seconds=3600,gpus=1),dict(job_id='2',wall_seconds=3600,gpus=1)]
        accounting={'1':dict(state='COMPLETED',elapsed=1800),'2':dict(state='RUNNING',elapsed=600)}
        decision=m.budget_decision(jobs,accounting,3600,2.,4)
        self.assertFalse(decision['allowed']);self.assertEqual(decision['charged_seconds'],2400)
        self.assertEqual(decision['reserved_remaining_seconds'],3000)
    def test_unknown_accounting_fails_closed_and_global_concurrency(self):
        m=self.module();jobs=[dict(job_id='1',wall_seconds=3600,gpus=1)]
        with self.assertRaises(ValueError):m.budget_decision(jobs,{},3600,8.,4)
        decision=m.budget_decision(jobs,{'1':dict(state='RUNNING',elapsed=30)},60,8.,1)
        self.assertFalse(decision['allowed'])
    def test_promotion_requires_gain_and_no_family_regression(self):
        m=self.module();cfg=dict(promotion_mean_exact_gain=.005,promotion_max_family_loss=.01)
        def summaries(f1s):
            return {f:dict(methods={name:dict(selected=dict(equal_family=dict(exact_f1=value,harmful_rate=.1))) for name,value in values.items()}) for f,values in f1s.items()}
        data=summaries({'srp':dict(fixed_mix=.6,residual=.5,local_gate=.62,safe_local_gate=.62),
                        'tmRNA':dict(fixed_mix=.6,residual=.5,local_gate=.58,safe_local_gate=.60)})
        result=m.promotion(data,cfg)
        self.assertFalse(result['candidates']['local_gate']['promote']);self.assertTrue(result['candidates']['safe_local_gate']['promote'])
        self.assertEqual(result['winner'],'safe_local_gate')
    def test_submission_receipt_reserves_job_before_sacct_catches_up(self):
        m=self.module();ledger=dict(smoke=[dict(job_id='3',wall_seconds=600,gpus=1)],pilot=[],core=[])
        result=m.resource_status(ledger,{'3':dict(state='SUBMITTED',elapsed=0)})
        self.assertEqual(result['phases']['smoke']['reserved_remaining_seconds'],600)
        self.assertEqual(result['phases']['smoke']['active'],1)

if __name__=='__main__':unittest.main()
