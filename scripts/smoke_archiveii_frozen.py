"""Accept only a complete real preparation, unchanged gradients and paired starts."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.archiveii_frozen_protocol import MODES,code_identity,PRETRAINED_SHA
from scripts.archiveii_frozen_data import OUT,FAMILIES,load_manifest,sha,write_json
from scripts.run_archiveii_frozen import load_inputs,input_identity
from scripts.run_thermo_fast import ThermoFusion,model_inputs
from src.losses import pair_bce_loss

def main():
    identity=code_identity();manifest=load_manifest();assert manifest['passed']
    physical=json.loads((OUT/'physical_manifest.json').read_text());assert physical['passed']
    embedding=json.loads((OUT/'embedding_provenance.json').read_text());assert embedding['passed']
    assert embedding['n']==3865 and embedding['weight_sha256']==PRETRAINED_SHA
    samples={sid:row for row in physical['samples'] for sid in row['sample_ids']}
    assert set(samples)==set(manifest['records']) and all(row['status']=='success' for row in samples.values())
    for family in FAMILIES:
        fold=manifest['folds'][family];roles={}
        for role in ('train','valid','test'):
            for sid in fold[role]:
                assert sid not in roles;roles[sid]=role
                assert (manifest['records'][sid]['family']==family)==(role=='test')
        assert set(roles)==set(manifest['records'])
    runs=[json.loads((OUT/'smoke'/identity['sha256'][:12]/'5s'/mode/'seed17/summary.json').read_text()) for mode in MODES]
    assert all(row['completed'] and row['epochs_completed']==2 and row['encoder_frozen'] and row['gradients_verified'] for row in runs)
    assert all(row['parameter_count']==186306 for row in runs)
    assert len({row['initialization_hash'] for row in runs})==1
    assert len({row['head_initialization_hash'] for row in runs})==1
    # Explicitly exercise the longest real training RNA, without test-family labels.
    ids,records,embeddings=load_inputs('5s',('train',))
    sid=max(ids['train'],key=lambda sid:len(records[sid]['sequence']))
    model=ThermoFusion(MODES[-1]).cuda();record=records[sid]
    logits=model(*model_inputs(record,embeddings[sid],torch.device('cuda')))
    loss=pair_bce_loss(logits,record['cand_i'].cuda(),record['cand_j'].cuda(),record['labels'].cuda());loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().max()>0 for p in model.correction.parameters())
    write_json(OUT/'preparation_complete.json',dict(passed=True,training_source=identity,inputs=input_identity(),
        smoke_runs=runs,longest_training_rna=dict(id=sid,length=len(record['sequence']),loss=float(loss),
        peak_cuda_bytes=torch.cuda.max_memory_allocated()),encoder_frozen=True,
        torch=torch.__version__,gpu=torch.cuda.get_device_name(0)))
    print('ARCHIVEII_FROZEN_PREPARATION_PASSED',flush=True)

if __name__=='__main__':main()
