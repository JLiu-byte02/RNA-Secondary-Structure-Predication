"""Family leakage, unchanged selection and official metric/decoder equivalence."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import yaml
from scripts.archiveii_frozen_protocol import validate_config,select_epoch,shared_threshold,FAMILIES
from scripts.archiveii_frozen_metrics import sparse_prf,pair_set
from src.decoder import decode_multi
spec=importlib.util.spec_from_file_location('archiveii_metric_test_reference',ROOT/'third_party/rinalmo_frozen_reference/sec_struct.py')
official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)

class ProtocolTests(unittest.TestCase):
    def test_namespace_package_source_provenance(self):
        from scripts import prepare_archiveii_frozen
        self.assertTrue(hasattr(prepare_archiveii_frozen,'rinalmo_source_files'),'Source provenance must support the installed namespace package')
        files=prepare_archiveii_frozen.rinalmo_source_files()
        self.assertIn('config.py',files)
        self.assertIn('model/model.py',files)
        self.assertTrue(all(len(value)==64 for value in files.values()))
    def test_frozen_training_is_unchanged(self):
        current=yaml.safe_load((ROOT/'configs/archiveii_frozen.yaml').read_text())
        previous=yaml.safe_load((ROOT/'configs/long_epochs.yaml').read_text())
        validate_config(current)
        for key in ('seeds','max_epochs','early_stopping','encoder','initial_lr','optimizer','weight_decay',
            'batch_size','gradient_accumulation','clip_grad_norm','thresholds'):
            self.assertEqual(current[key],previous[key],key)
        altered=dict(current,encoder='fine_tuned');self.assertRaises(ValueError,validate_config,altered)
    def test_complete_validation_only_selection(self):
        rows=[dict(epoch=i,score=1 if i in (11,12) else .5) for i in range(1,51)]
        self.assertEqual(select_epoch(rows)['epoch'],11)
        self.assertRaises(ValueError,select_epoch,rows[:-1])
        self.assertEqual(shared_threshold([{.1:.2,.2:.2}]*3),.1)
        self.assertRaises(ValueError,shared_threshold,[{.1:.2}]*2)
    def test_metric_equivalence(self):
        rng=np.random.default_rng(20261022)
        for length in (8,21,64,135):
            for _ in range(8):
                target=np.zeros((length,length),dtype=np.int8);pred=np.zeros_like(target)
                for matrix in (target,pred):
                    for a,b in zip(rng.integers(0,length,20),rng.integers(0,length,20)):
                        if a!=b:matrix[a,b]=matrix[b,a]=1
                for flexible in (True,False):
                    actual=sparse_prf(pair_set(target),pair_set(pred),length,flexible)
                    expected=(official.ss_precision(target,pred,flexible),official.ss_recall(target,pred,flexible),official.ss_f1(target,pred,flexible))
                    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
        # Official empty/empty convention is zero; report evaluator keeps it.
        self.assertEqual(sparse_prf(set(),set(),8),(0.,0.,0.))
    def test_original_E_decoder_matches_official_rules(self):
        rng=np.random.default_rng(17)
        for length in (28,93):
            sequence=''.join(rng.choice(list('ACGU'),length))
            for tied in (False,True):
                scores=rng.normal(size=(length,length));scores=(scores+scores.T)/2
                if tied:scores=np.round(scores,0)
                thresholds=[.1,.2,.5,.9]
                decoded=decode_multi(scores,sequence,thresholds)
                probs=1/(1+np.exp(-scores.astype(np.float64)))
                for threshold in thresholds:
                    expected=pair_set(official.prob_mat_to_sec_struct(probs.copy(),sequence,threshold))
                    self.assertEqual(set(decoded[threshold]),expected)

if __name__=='__main__':unittest.main()
