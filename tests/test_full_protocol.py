"""Full-stage contracts: disjoint cohorts, validation-only selection, fair seeds."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.full_protocol import audit_cohorts, shared_threshold, validate_config


class FullProtocol(unittest.TestCase):
    def test_sequence_overlap_is_rejected_even_with_distinct_ids(self):
        splits = {'TR0':['a'], 'VL0':['b'], 'TS0':['c'], 'bprna_new':['d']}
        rows = {i:{'sequence':s} for i,s in zip('abcd',['ACGU','ACGU','AAAA','CCCC'])}
        with self.assertRaisesRegex(ValueError, 'sequence overlap'):
            audit_cohorts(rows, splits, expected_counts=False)

    def test_all_seed_validation_curves_choose_one_threshold(self):
        curves = [{.1:.6,.2:.8},{.1:.9,.2:.6},{.1:.8,.2:1.0}]
        self.assertEqual(shared_threshold(curves), .2)
        self.assertEqual(shared_threshold([{.2:.8,.1:.8}]*3), .1)

    def test_test_cohort_cannot_select_checkpoints(self):
        base = dict(train_split='TR0',select_split='VL0',threshold_split='VL0',
                    seeds=[17,23,41],test_splits=['TS0','bprna_new'])
        validate_config(base)
        for key in ('select_split','threshold_split'):
            with self.assertRaises(ValueError):
                validate_config(dict(base, **{key:'TS0'}))
        with self.assertRaises(ValueError):
            validate_config(dict(base, seeds=[17]))

if __name__ == '__main__':
    unittest.main(verbosity=2)
