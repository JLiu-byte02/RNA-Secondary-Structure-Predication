import importlib.util
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def module(name):
    path = ROOT / 'scripts' / (name + '.py')
    assert path.exists(), f'Required implementation is missing: {path.name}'
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Contracts(unittest.TestCase):
    def test_stack_is_joint_and_outer_pair_is_zero_based(self):
        cache = module('cache_thermo')
        import RNA
        import numpy as np
        sequence = 'GGGAAACCC'
        md = RNA.md()
        md.uniq_ML = 1
        md.dangles = 2
        fc = RNA.fold_compound(sequence, md)
        fc.pf()
        p = np.asarray(fc.bpp())[1:, 1:]
        q = np.zeros_like(p)
        for i, j, value in cache.stack_probabilities(fc, 0., RNA):
            q[i-1, j-1] = value
        cache.validate_matrices(sequence, p, q)
        structures = fc.subopt(10000)
        rt = RNA.exp_param(md).kT / 1000.
        z = outer = inner = both = 0.
        for structure in structures:
            w = math.exp(-structure.energy / rt)
            pt = RNA.ptable(structure.structure)
            a, b = pt[1] == 9, pt[2] == 8
            z += w
            outer += w*a
            inner += w*b
            both += w*a*b
        self.assertAlmostEqual(p[0, 8], outer/z, delta=0.003)
        self.assertAlmostEqual(p[1, 7], inner/z, delta=0.003)
        self.assertAlmostEqual(q[0, 8], both/z, delta=0.003)
        self.assertGreater(abs(q[0, 8]-p[0, 8]*p[1, 7]), 0.01)

    def test_equal_initialization_symmetry_and_both_gradients(self):
        run = module('run_thermo_fast')
        import torch
        from scripts._common import set_seed
        hashes = []
        for mode in run.MODES:
            set_seed(17)
            model = run.ThermoFusion(mode)
            hashes.append(run.state_hash(model.state_dict()))
            embedding = torch.randn(9, 1280)
            physical = torch.rand(2, 9, 9)
            valid = torch.ones(9, 9, dtype=torch.bool)
            logits, initial, residual = model(embedding, physical, torch.zeros(9, 9), valid, True)
            self.assertTrue(torch.equal(logits, initial))
            self.assertTrue(torch.equal(logits, logits.T))
            self.assertEqual(residual.abs().max().item(), 0)
            logits.square().mean().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.parameters()))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.correction.parameters()))
            self.assertFalse(embedding.requires_grad)
        self.assertEqual(len(set(hashes)), 1)

    def test_full_truth_and_equal_family_metric(self):
        module('run_thermo_fast')
        from src.metrics import pair_prf, family_macro
        self.assertAlmostEqual(pair_prf({(0,8),(1,7)}, {(0,8)})[2], 2/3)
        self.assertAlmostEqual(family_macro([('a',1.),('a',1.),('b',0.),('unknown',0.)]), .5)
        self.assertEqual(pair_prf(set(),set())[2],1.)


from test_regressions import Regression

if __name__ == '__main__':
    if '--root' in sys.argv:
        pos = sys.argv.index('--root'); del sys.argv[pos:pos+2]
    if '--root' in sys.argv:
        pos = sys.argv.index('--root'); del sys.argv[pos:pos+2]
    unittest.main(verbosity=2)
