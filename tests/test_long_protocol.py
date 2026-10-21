import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.long_protocol import select_epoch, validate_long_config


class LongProtocol(unittest.TestCase):
    def test_budget_cannot_see_future_epochs(self):
        rows=[dict(epoch=e,score=s) for e,s in [(1,.5),(2,.6),(3,.6),(4,.9)]]
        self.assertEqual(select_epoch(rows,3)['epoch'],2)
        self.assertEqual(select_epoch(rows,4)['epoch'],4)

    def test_missing_or_duplicate_epochs_rejected(self):
        for rows in ([dict(epoch=1,score=.5),dict(epoch=3,score=.6)],
                     [dict(epoch=1,score=.5),dict(epoch=1,score=.6)]):
            with self.assertRaises(ValueError):select_epoch(rows,2)

    def test_fixed_50_epochs_and_validation_only_selection(self):
        config=dict(max_epochs=50,early_stopping=False,budgets=[10,20,30,40,50],
            train_split='TR0',select_split='VL0',threshold_split='VL0',
            seeds=[17,23,41],test_splits=['TS0','bprna_new'])
        validate_long_config(config)
        for field,value in [('early_stopping',True),('max_epochs',10),('select_split','TS0')]:
            with self.assertRaises(ValueError):validate_long_config(dict(config,**{field:value}))

if __name__=='__main__':unittest.main(verbosity=2)
