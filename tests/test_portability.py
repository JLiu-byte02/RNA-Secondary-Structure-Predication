"""Relocation and freshly generated embedding provenance are required on HPC."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('preflight', ROOT / 'scripts/preflight.py')
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


class PortableInputs(unittest.TestCase):
    def test_uses_project_local_weight_at_relocated_root(self):
        self.assertTrue(hasattr(preflight, 'resolve_weight'), 'Preflight still depends on the old server')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'weights').mkdir()
            weight = root / 'weights/rinalmo_giga_pretrained.pt'
            weight.write_bytes(b'unit-test-weight')
            self.assertEqual(preflight.resolve_weight(root), weight)

    def test_preflight_rejects_modified_reused_b0_module(self):
        self.assertTrue(hasattr(preflight, 'validate_source'), 'Original B0 source provenance must survive relocation')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'src').mkdir()
            module = root / 'src/pair_head.py'
            module.write_text('changed')
            errors, records = preflight.validate_source(root, {'src/pair_head.py': 'incorrect-hash'})
            self.assertTrue(errors)
            self.assertFalse(records[0]['unchanged'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
