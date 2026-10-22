"""RiNALMo-protocol ArchiveII fold training and official-checkpoint evaluation."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'third_party/rinalmo_upstream'))
sys.path.insert(0,str(ROOT/'setup'))
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback,ModelCheckpoint
from torch.utils.data import Dataset,DataLoader
from rinalmo.data.alphabet import Alphabet
from rinalmo.data.downstream.secondary_structure.datamodule import SecondaryStructureDataModule
from rinalmo.utils.finetune_callback import GradualUnfreezing
from rinalmo.utils.sec_struct import parse_sec_struct_file
from scripts.archiveii_data import OUT,FAMILIES,load_manifest,feature_path,sha,write_json
from scripts.archiveii_models import FamilyWrapper,AuditedOfficialWrapper

MODES=('E0_no_thermo','E1_bpp','E2_bpp_stack')
SEEDS=(17,23,41)
RUNTIME_FILES=('scripts/archiveii_data.py','scripts/archiveii_metrics.py','scripts/archiveii_models.py',
    'scripts/run_archiveii.py','src/pair_head.py','scripts/run_thermo_fast.py','scripts/cache_thermo.py',
    'configs/archiveii_ood.yaml','third_party/rinalmo_reference/train_sec_struct_prediction.py',
    'third_party/rinalmo_reference/giga_sec_struct_ft.yaml','setup/check_archiveii_environment.py',
    'setup/archiveii_official_source_manifest.json')

def code_identity():
    files={name:sha(ROOT/name) for name in RUNTIME_FILES}
    return dict(files=files,sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest())

def tensor_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class FamilyDataset(Dataset):
    def __init__(self,manifest,family,role,alphabet):
        self.records=manifest['records'];self.ids=manifest['folds'][family][role];self.alphabet=alphabet;self.feature_cache={}
    def __len__(self):return len(self.ids)
    def __getitem__(self,index):
        row=self.records[self.ids[index]]
        seq,target=parse_sec_struct_file(ROOT/row['ct_path'])
        assert seq==row['sequence']
        key=str(feature_path(seq))
        if key not in self.feature_cache:
            with np.load(key,allow_pickle=False) as f:
                p=f['P'];q=f['Q'];physical=np.stack((p+p.T,q+q.T))
            # Bounded worker-local cache; source .npz files remain immutable.
            if len(self.feature_cache)>=128:self.feature_cache.pop(next(iter(self.feature_cache)))
            self.feature_cache[key]=torch.from_numpy(physical)
        return row['id'],seq,torch.tensor(self.alphabet.encode(seq),dtype=torch.int64),torch.tensor(target),self.feature_cache[key]

class FamilyDataModule(pl.LightningDataModule):
    def __init__(self,family,alphabet):super().__init__();self.family=family;self.alphabet=alphabet
    def setup(self,stage=None):
        manifest=load_manifest()
        for role in ('train','valid','test'):setattr(self,role,FamilyDataset(manifest,self.family,role,self.alphabet))
    def train_dataloader(self):return DataLoader(self.train,batch_size=1,shuffle=True,num_workers=0)
    def val_dataloader(self):return DataLoader(self.valid,batch_size=1,num_workers=0)
    def test_dataloader(self):return DataLoader(self.test,batch_size=1,num_workers=0)

class EpochAudit(Callback):
    def __init__(self,path):self.path=path;self.losses=[];self.started=None;self.train_seconds=0
    def on_train_epoch_start(self,trainer,model):
        self.started=time.perf_counter();self.losses=[]
        names=[n for n,p in model.lm.named_parameters() if p.requires_grad]
        groups=[float(g['lr']) for g in trainer.optimizers[0].param_groups]
        self.phase=dict(trainable_lm_parameters=sum(p.numel() for p in model.lm.parameters() if p.requires_grad),
            trainable_lm_parameter_names=names,optimizer_group_lrs_at_epoch_start=groups)
    def on_train_batch_end(self,trainer,model,outputs,batch,batch_idx):
        value=outputs['loss'] if isinstance(outputs,dict) else outputs
        self.losses.append(float(value.detach()))
    def on_validation_start(self,trainer,model):
        if trainer.sanity_checking:return
        torch.cuda.synchronize();self.train_seconds=time.perf_counter()-self.started
    def on_validation_end(self,trainer,model):
        if trainer.sanity_checking:return
        scores={str(t):float(sum(v)/len(v)) for t,v in model._eval_step_outputs['f1'].items()}
        row=dict(epoch=trainer.current_epoch+1,global_step=trainer.global_step,
            train_bce_online=float(np.mean(self.losses)),validation_bce=float(trainer.callback_metrics['val/loss']),
            validation_f1=scores[str(model.threshold)],threshold=float(model.threshold),threshold_f1=scores,
            training_seconds=self.train_seconds,epoch_wall_seconds=time.perf_counter()-self.started,**self.phase)
        previous=[json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []
        previous=[v for v in previous if v['epoch']<row['epoch']]
        assert len(previous)==row['epoch']-1,'Missing epoch history'
        tmp=self.path.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(v)+'\n' for v in previous+[row]));tmp.replace(self.path)
        print('ARCHIVEII_EPOCH',row['epoch'],'F1',round(row['validation_f1'],6),'threshold',model.threshold,flush=True)

def train(args,directory):
    from check_archiveii_environment import verify_official_sources
    identity=code_identity();frozen=json.loads((OUT/'experiment_identity.json').read_text())
    assert frozen['training_source']==identity,'Numerical source changed after protocol freeze'
    assert frozen['data_manifest_sha256']==sha(OUT/'data_manifest.json.gz')
    assert frozen['physical_manifest_sha256']==sha(OUT/'physical_manifest.json')
    assert verify_official_sources()==frozen['official_sources']
    pretrained=ROOT/'weights/rinalmo_giga_pretrained.pt'
    assert sha(pretrained)==frozen['pretrained_sha256']
    # The official script trains an uninterrupted trajectory. Do not silently resume
    # partial gradual-unfreezing optimizer groups with reset requires_grad flags.
    prior=[p for p in directory.iterdir() if p.name not in ('.run.lock','previous_attempts')]
    if prior:
        archive=directory/'previous_attempts'/str(time.time_ns());archive.mkdir(parents=True)
        for path in prior:
            assert not path.is_symlink();path.rename(archive/path.name)
        write_json(archive/'restart_reason.json',dict(reason='Incomplete trajectory preserved; restart all 15 epochs from the same public pretrained weights and seed, following the official uninterrupted recipe.'))
    pl.seed_everything(args.seed,workers=True)
    model=FamilyWrapper(args.mode,lr=1e-4);model.load_pretrained_lm_weights(pretrained)
    initial=tensor_hash(model.pred_head.state_dict())
    write_json(directory/'initialization.json',dict(seed=args.seed,mode=args.mode,head_initialization_hash=initial,
        pretrained_sha256=frozen['pretrained_sha256'],code_hash=identity['sha256'],data_manifest_sha256=frozen['data_manifest_sha256']))
    datamodule=FamilyDataModule(args.family,Alphabet(**model.lm.config['alphabet']))
    callbacks=[GradualUnfreezing(str(ROOT/'third_party/rinalmo_reference/giga_sec_struct_ft.yaml')),
        EpochAudit(directory/'epochs.jsonl'),ModelCheckpoint(dirpath=directory,filename='last_epoch',save_top_k=0,save_last=True)]
    trainer=pl.Trainer(accelerator='gpu',devices=1,max_epochs=15,default_root_dir=directory,callbacks=callbacks,
        precision='16-mixed',logger=False,log_every_n_steps=50,enable_progress_bar=False)
    started=time.time()
    trainer.fit(model,datamodule=datamodule)
    epochs=[json.loads(line) for line in (directory/'epochs.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in epochs]==list(range(1,16))
    final=directory/'final.pt'
    if not final.exists():
        state=dict(model.state_dict());state['threshold']=float(model.threshold)
        tmp=final.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(final)
    lock=dict(mode=args.mode,family=args.family,seed=args.seed,epoch=15,threshold=float(model.threshold),
        checkpoint_sha256=sha(final),code_hash=identity['sha256'],data_manifest_sha256=frozen['data_manifest_sha256'],
        selection='Final 15th epoch and its validation-tuned threshold; same as the official training script.',test_scores_used=False)
    if (directory/'selection_lock.json').exists():assert json.loads((directory/'selection_lock.json').read_text())==lock
    else:write_json(directory/'selection_lock.json',lock)
    with (directory/'predictions.jsonl.tmp').open('w') as stream:
        model.prediction_stream=stream;metrics=trainer.test(model,datamodule=datamodule)[0];model.prediction_stream=None
    (directory/'predictions.jsonl.tmp').replace(directory/'predictions.jsonl')
    predictions=[json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
    macro={field:float(np.mean([r[field] for r in predictions])) for field in ('precision','recall','f1','exact_precision','exact_recall','exact_f1')}
    write_json(directory/'summary.json',dict(completed=True,mode=args.mode,family=args.family,seed=args.seed,
        epochs_completed=15,metrics=metrics,rna_macro=macro,n=len(predictions),threshold=float(model.threshold),head_initialization_hash=initial,
        code_hash=identity['sha256'],selection_lock_sha256=sha(directory/'selection_lock.json'),
        predictions_sha256=sha(directory/'predictions.jsonl'),checkpoint_sha256=lock['checkpoint_sha256'],
        last_process_wall_seconds=time.time()-started,source='Actual full-fold online RiNALMo gradual fine-tuning.'))

def baseline(args,directory):
    from check_archiveii_environment import verify_official_sources
    assert verify_official_sources()['passed']
    path=ROOT/'weights/archiveii'/f'rinalmo_giga_ss_archiveII-{args.family}_ft.pt'
    provenance=json.loads(Path(str(path)+'.provenance.json').read_text());assert sha(path)==provenance['sha256']
    model=AuditedOfficialWrapper(lm_config='giga')
    model.load_state_dict(torch.load(path,map_location='cpu'))
    datamodule=SecondaryStructureDataModule(ROOT/'data',Alphabet(**model.lm.config['alphabet']),dataset='archiveII_'+args.family)
    trainer=pl.Trainer(accelerator='gpu',devices=1,default_root_dir=directory,precision='16-mixed',logger=False,enable_progress_bar=False)
    with (directory/'predictions.jsonl.tmp').open('w') as stream:
        model.prediction_stream=stream;metrics=trainer.test(model,datamodule=datamodule)[0];model.prediction_stream=None
    (directory/'predictions.jsonl.tmp').replace(directory/'predictions.jsonl')
    predictions=[json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
    macro={field:float(np.mean([r[field] for r in predictions])) for field in ('precision','recall','f1','exact_precision','exact_recall','exact_f1')}
    write_json(directory/'summary.json',dict(completed=True,family=args.family,mode='RiNALMo_official',metrics=metrics,rna_macro=macro,n=len(predictions),
        threshold=float(model.threshold),checkpoint=provenance,predictions_sha256=sha(directory/'predictions.jsonl'),
        source='Re-evaluation of a public official family-specific fine-tuned checkpoint; not a new multi-seed baseline.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=FAMILIES,required=True);p.add_argument('--mode',choices=MODES,default=MODES[0]);p.add_argument('--seed',type=int,choices=SEEDS,default=17);p.add_argument('--baseline',action='store_true');a=p.parse_args()
    directory=OUT/('baseline' if a.baseline else 'runs')/a.family
    if not a.baseline:directory=directory/a.mode/f'seed{a.seed}'
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (directory/'summary.json').exists():
            summary=json.loads((directory/'summary.json').read_text());assert summary['completed']
            assert sha(directory/'predictions.jsonl')==summary['predictions_sha256']
            if not a.baseline:assert summary['code_hash']==code_identity()['sha256']
            print('ALREADY_COMPLETED',directory)
        elif a.baseline:baseline(a,directory)
        else:train(a,directory)
