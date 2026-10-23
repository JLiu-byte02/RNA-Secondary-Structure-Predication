import importlib,importlib.util,unittest

class RuntimeTests(unittest.TestCase):
    def test_cpu_requires_consistent_fp32_base_recomputation(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.direction_a_runtime'))
        m=importlib.import_module('scripts.direction_a_runtime');r=m.execution_spec('cpu')
        self.assertEqual(r['device'],'cpu');self.assertEqual(r['base_logits'],'recompute_fp32_all_roles')
        self.assertEqual(r['threads'],2)
        with self.assertRaises(ValueError):m.execution_spec('automatic')

if __name__=='__main__':unittest.main()
