import importlib
import importlib.util
import unittest
import json
import tempfile
from pathlib import Path
from scripts.direction_a_protocol import config,file_sha,write_json,ROOT,source_identity

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
    def test_cpu_only_report_does_not_consume_gpu_hours(self):
        m=self.module();jobs=[dict(job_id='4',wall_seconds=1200,gpus=0)]
        result=m.budget_decision(jobs,{'4':dict(state='RUNNING',elapsed=300)},0,8.,4)
        self.assertEqual(result['charged_seconds'],0);self.assertEqual(result['reserved_remaining_seconds'],0)
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
    def test_core_admission_rejects_stale_prediction_artifact(self):
        m=self.module();gate=getattr(m,'validate_promotion',None)
        self.assertTrue(callable(gate),'Core admission does not validate promotion artifacts')
        cfg=config()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_json(root/'pilot_roles.json',dict(sha256='roles'))
            summaries={};sources={}
            for case in cfg['pilot_development_families']:
                directory=root/'pilot'/case/'correction_seed17';directory.mkdir(parents=True)
                identity=dict(code=source_identity('correction'),roles_sha256='roles')
                write_json(directory/'choices_sealed.json',dict(identity=identity))
                write_json(directory/'development_predictions.json',{})
                methods={name:dict(selected=dict(equal_family=dict(exact_f1=value,harmful_rate=.1)),official_verification=dict(passed=True)) for name,value in dict(fixed_mix=.6,residual=.5,local_gate=.62,safe_local_gate=.61).items()}
                summary=dict(completed=True,smoke=False,case=case,identity=identity,methods=methods,
                    choices_sha256=file_sha(directory/'choices_sealed.json'),predictions_sha256=file_sha(directory/'development_predictions.json'))
                write_json(directory/'development_summary.json',summary);summaries[case]=summary;sources[case]=file_sha(directory/'development_summary.json')
            path=root/'pilot/promotion.json'
            write_json(path,dict(source_summaries=sources,config_sha256=file_sha(ROOT/'configs/direction_a.json'),
                role_manifest_sha256=file_sha(root/'pilot_roles.json'),**m.promotion(summaries,cfg)))
            self.assertTrue(gate(path,cfg))
            (root/'pilot/srp/correction_seed17/development_predictions.json').write_text('{"replaced":true}')
            with self.assertRaises(AssertionError):gate(path,cfg)

if __name__=='__main__':unittest.main()
