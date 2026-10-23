"""State and selection failures here would invalidate expensive experiments."""
import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
import torch
from scripts.direction_a_models import LocalCorrection

class RunnerTests(unittest.TestCase):
    def runner(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.run_direction_a_correction'),'Correction runner is not implemented')
        return importlib.import_module('scripts.run_direction_a_correction')
    def test_selection_keeps_earliest_epoch_on_equal_score(self):
        module=self.runner();model=LocalCorrection('local_gate')
        first=dict(epoch=1,calibration=dict(equal_family_exact_f1=.6,threshold=.2))
        best=module.update_best(None,model,first)
        later=dict(epoch=2,calibration=dict(equal_family_exact_f1=.6,threshold=.1))
        self.assertEqual(module.update_best(best,model,later)['epoch'],1)
    def test_resume_rejects_changed_identity_and_restores_optimizer(self):
        module=self.runner();torch.set_num_threads(1)
        model=LocalCorrection('local_gate');optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        sum(p.sum() for p in model.parameters()).backward();optimizer.step()
        identity=dict(seed=17,source='abc')
        state=dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),rows=[dict(epoch=1)],best=None,
                   torch_rng=torch.get_rng_state(),cuda_rng=[])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'last.pt';torch.save(state,path)
            other=LocalCorrection('local_gate');opt=torch.optim.AdamW(other.parameters(),lr=.2)
            with self.assertRaises(AssertionError):module.load_resume(path,dict(seed=23,source='abc'),other,opt,torch.device('cpu'))
            rows,best=module.load_resume(path,identity,other,opt,torch.device('cpu'))
            self.assertEqual(rows,[dict(epoch=1)]);self.assertEqual(opt.param_groups[0]['lr'],.001)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(model.parameters(),other.parameters())))

if __name__=='__main__':unittest.main()
