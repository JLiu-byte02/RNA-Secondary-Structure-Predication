"""Exercise all 15 Lightning phases on tiny non-held-out fixtures; discard weights."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'third_party/rinalmo_upstream'))
import torch
import lightning.pytorch as pl
from rinalmo.data.alphabet import Alphabet
from rinalmo.utils.finetune_callback import GradualUnfreezing
from scripts.run_archiveii import FamilyWrapper,FamilyDataModule,EpochAudit,code_identity
from scripts.archiveii_data import OUT,write_json

class FixtureDataModule(FamilyDataModule):
    def setup(self,stage=None):
        super().setup(stage)
        # Exercise test hooks on existing validation RNAs, never the held-out 5s family.
        self.test=self.valid

def main():
    pl.seed_everything(17,workers=True)
    model=FamilyWrapper('E0_no_thermo');model.load_pretrained_lm_weights(ROOT/'weights/rinalmo_giga_pretrained.pt')
    directory=OUT/'smoke/lightning';directory.mkdir(parents=True,exist_ok=True)
    (directory/'epochs.jsonl').unlink(missing_ok=True)
    data=FixtureDataModule('5s',Alphabet(**model.lm.config['alphabet']))
    trainer=pl.Trainer(accelerator='gpu',devices=1,max_epochs=15,precision='16-mixed',logger=False,
        default_root_dir=directory,enable_progress_bar=False,enable_checkpointing=False,
        limit_train_batches=2,limit_val_batches=2,limit_test_batches=2,num_sanity_val_steps=2,
        callbacks=[GradualUnfreezing(str(ROOT/'third_party/rinalmo_reference/giga_sec_struct_ft.yaml')),EpochAudit(directory/'epochs.jsonl')])
    trainer.fit(model,datamodule=data)
    rows=[json.loads(s) for s in (directory/'epochs.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows]==list(range(1,16))
    assert all(r['trainable_lm_parameters']==0 for r in rows[:3])
    for r in rows:
        names=r['trainable_lm_parameter_names'];phase=(r['epoch']-1)//3
        blocks=sorted({int(n.split('.')[2]) for n in names if n.startswith('transformer.blocks.')})
        assert blocks==list(range(33-phase*3,33))
        assert r['training_seconds']>0 and r['epoch_wall_seconds']>=r['training_seconds']
    with (directory/'fixture_predictions.jsonl').open('w') as stream:
        model.prediction_stream=stream;trainer.test(model,datamodule=data);model.prediction_stream=None
    records=[json.loads(s) for s in (directory/'fixture_predictions.jsonl').read_text().splitlines()]
    assert len(records)==2 and all(not r['rna_id'].startswith('5s_') for r in records)
    preparation=json.loads((OUT/'preparation_complete.json').read_text())
    current=code_identity()
    for name in ('scripts/archiveii_models.py','src/pair_head.py','scripts/cache_thermo.py','scripts/archiveii_metrics.py'):
        assert preparation['training_source']['files'][name]==current['files'][name]
    preparation['capacity_smoke_training_source']=preparation['training_source']
    preparation['lightning_fixture_job_id']=os.environ.get('SLURM_JOB_ID')
    preparation['lightning_15_phase_fixture_passed']=True
    preparation['fixture_test_uses_nonheldout_validation_RNAs']=True
    preparation['training_source']=current
    write_json(OUT/'preparation_complete.json',preparation)
    print('LIGHTNING_ALL_15_PHASES_AND_EVALUATION_HOOKS_PASSED',flush=True)

if __name__=='__main__':main()
