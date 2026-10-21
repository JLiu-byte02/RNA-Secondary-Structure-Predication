"""Synthetic temporary fixtures test sealing only; never scientific observations."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import scripts.eval_long_epochs as study
from scripts._common import load_split,set_seed
from scripts.long_models import make_model
from scripts.run_thermo_fast import digest,state_hash
from scripts.run_long_epochs import code_identity


class LongSelection(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory(dir=study.ROOT/'.tmp');self.previous=study.OUT;study.OUT=Path(self.folder.name)
        split=load_split(study.ROOT);code=code_identity()['sha256']
        for mode in study.MODES:
            for seed in study.SEEDS:
                directory=study.OUT/mode/f'seed{seed}';directory.mkdir(parents=True)
                set_seed(seed);model=make_model(mode)
                rows=[]
                for epoch in range(1,51):
                    score=.3 if epoch==1 else (.9 if epoch==49 else .6)
                    curve={str(i/10):dict(rna_macro_exact_f1=score if i==2 else score-.1) for i in range(1,10)}
                    rows.append(dict(epoch=epoch,score=score,mean_pair_loss=.2,
                        validation=dict(mean_pair_bce=.2,threshold_metrics=curve)))
                (directory/'epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                for epoch in (2,49):torch.save(dict(model=model.state_dict(),mode=mode,seed=seed,epoch=epoch),directory/f'epoch{epoch}.pt')
                summary=dict(status='completed',epochs_completed=50,early_stopping=False,
                    train_count=len(split['TR0']),validation_count=len(split['VL0']),
                    train_ids_hash=digest(split['TR0']),validation_ids_hash=digest(split['VL0']),
                    encoder_frozen=True,gradients_verified=True,test_scores_used=False,
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    head_initialization_hash=state_hash(model.head.state_dict()),config_hash='synthetic-fixture',
                    input_provenance_hash='synthetic-fixture',code_hash=code)
                (directory/'summary.json').write_text(json.dumps(summary))

    def tearDown(self):study.OUT=self.previous;self.folder.cleanup()

    def test_sixty_choices_sealed_without_future_epoch_leak_and_idempotent(self):
        with contextlib.redirect_stdout(io.StringIO()):study.lock()
        path=study.OUT/'selection_lock.json';first=path.read_bytes();record=json.loads(first)
        self.assertEqual(len(record['choices']),60)
        self.assertFalse(record['test_scores_used'])
        self.assertTrue(all(v['epoch']==(49 if v['budget']==50 else 2) and v['threshold']==.2 for v in record['choices'].values()))
        with contextlib.redirect_stdout(io.StringIO()):study.lock()
        self.assertEqual(path.read_bytes(),first)

    def test_mismatched_baseline_head_initialization_rejected(self):
        path=study.OUT/'B0/seed17/summary.json';summary=json.loads(path.read_text())
        summary['head_initialization_hash']='modified';path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError,'Paired experiment identities differ'):study.lock()

if __name__=='__main__':unittest.main(verbosity=2)
