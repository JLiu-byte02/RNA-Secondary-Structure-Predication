import importlib
import unittest
import tempfile
from pathlib import Path
from scripts.direction_a_protocol import ROOT

class ReportTests(unittest.TestCase):
    def test_reference_binding_needs_no_git_on_cpu_node(self):
        module=importlib.import_module('scripts.report_direction_a');verify=getattr(module,'verify_reference',None)
        self.assertTrue(callable(verify),'CPU audit still depends on a Git executable')
        self.assertEqual(len(verify(ROOT/'third_party/rinalmo_frozen_reference/sec_struct.py')),64)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'changed.py';path.write_text('changed reference')
            with self.assertRaises(AssertionError):verify(path)
    def test_parallel_official_audit_preserves_counts_and_rejects_bad_metric(self):
        module=importlib.import_module('scripts.report_direction_a');verify=getattr(module,'verify_records',None)
        self.assertTrue(callable(verify),'Independent audit has no bounded CPU parallel verification')
        row=dict(length=9,reference=[(0,8)],prediction=[(0,8)],exact_precision=1.,exact_recall=1.,exact_f1=1.,
                 flexible_precision=1.,flexible_recall=1.,flexible_f1=1.)
        self.assertEqual(verify([row]*64,workers=2),dict(passed=True,n=64))
        bad=dict(row,exact_f1=.5)
        with self.assertRaises(AssertionError):verify([row]*63+[bad],workers=2)

if __name__=='__main__':unittest.main()
