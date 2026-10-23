import importlib,importlib.util,json,tempfile,unittest
from pathlib import Path

class AuditRowsTests(unittest.TestCase):
    def m(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.direction_a_audit_rows'))
        return importlib.import_module('scripts.direction_a_audit_rows')
    def test_json_roundtrip_normalizes_both_prediction_and_reference_base(self):
        m=self.m();row=json.loads(json.dumps(dict(prediction=[(0,5)],base=[(1,7)])))
        def original(sid,record,pred,base):return set(pred),set(base)
        result=m.normalizing_metric_row(original)('x',{},row['prediction'],row['base'])
        self.assertEqual(result,({(0,5)},{(1,7)}))
    def test_changed_executable_dependency_fails_even_with_old_cache_seal(self):
        m=self.m()
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'loader.py';path.write_text('original')
            manifest={'loader.py':hashlib.sha256(path.read_bytes()).hexdigest()};m.verify_dependencies(Path(d),manifest)
            path.write_text('changed')
            with self.assertRaises(AssertionError):m.verify_dependencies(Path(d),manifest)

if __name__=='__main__':unittest.main()
