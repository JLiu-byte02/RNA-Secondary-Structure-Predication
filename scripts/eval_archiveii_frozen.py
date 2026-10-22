"""Seal validation-only E selections, evaluate held-out families, verify predictions."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from scripts._common import set_seed
from scripts.run_thermo_fast import ThermoFusion,model_inputs
from src.decoder import decode_multi
from scripts.archiveii_frozen_data import OUT,FAMILIES,load_manifest,normalized_sequence,sha,write_json
from scripts.archiveii_frozen_protocol import MODES,SEEDS,code_identity,select_epoch,shared_threshold,digest
from scripts.archiveii_frozen_metrics import sparse_prf,pair_set
from scripts.run_archiveii_frozen import load_inputs,input_identity

spec=importlib.util.spec_from_file_location('archiveii_official_reference',ROOT/'third_party/rinalmo_frozen_reference/sec_struct.py')
official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
FIELDS=('precision','recall','f1','exact_precision','exact_recall','exact_f1')

def average(rows):
    return {key:sum(row[key] for row in rows)/len(rows) for key in FIELDS}

def seal():
    identity=json.loads((OUT/'experiment_identity.json').read_text())
    assert identity['training_source']==code_identity() and identity['inputs']==input_identity()
    choices={};summaries=[]
    for family in FAMILIES:
        for mode in MODES:
            trajectories=[]
            for seed in SEEDS:
                directory=OUT/'runs'/family/mode/f'seed{seed}'
                summary=json.loads((directory/'summary.json').read_text())
                rows=[json.loads(line) for line in (directory/'epochs.jsonl').read_text().splitlines()]
                assert summary['completed'] and summary['epochs_completed']==50 and summary['encoder_frozen']
                assert summary['code_hash']==identity['training_source']['sha256']
                assert summary['input_hash']==identity['inputs']['sha256']
                assert sha(directory/'epochs.jsonl')==summary['epochs_sha256']
                assert sha(directory/'best.pt')==summary['best_checkpoint_sha256']
                selected=select_epoch(rows)
                assert selected['epoch']==summary['best_epoch'] and selected['score']==summary['best_score']
                assert all(row['gradients_verified'] for row in rows)
                curves={float(t):v['rna_macro_exact_f1'] for t,v in selected['validation']['threshold_metrics'].items()}
                trajectories.append((seed,summary,curves));summaries.append(summary)
            threshold=shared_threshold([r[2] for r in trajectories])
            for seed,summary,curve in trajectories:
                choices[f'{family}/{mode}/seed{seed}']=dict(family=family,mode=mode,seed=seed,
                    epoch=summary['best_epoch'],threshold=threshold,validation_exact_f1=curve[threshold],
                    checkpoint_sha256=summary['best_checkpoint_sha256'])
        for seed in SEEDS:
            matched=[s for s in summaries if s['family']==family and s['seed']==seed]
            assert len(matched)==3 and len({s['head_initialization_hash'] for s in matched})==1
            assert len({s['initialization_hash'] for s in matched})==1
    record=dict(passed=True,selection_split='official_fold_valid',test_scores_used=False,
        selected_trajectories=81,choices=choices,code_hash=identity['training_source']['sha256'],
        input_hash=identity['inputs']['sha256'],rule='Unchanged frozen-E exact-F1 epoch selection; shared three-seed validation threshold.',
        experiment_identity_sha256=sha(OUT/'experiment_identity.json'),evaluation_source_sha256=sha(Path(__file__)))
    path=OUT/'selection_lock.json'
    if path.exists():assert json.loads(path.read_text())==record
    else:write_json(path,record)
    print('ARCHIVEII_FROZEN_SELECTIONS_SEALED',len(choices),flush=True)

def prediction_record(sid,family,sequence,truth,pred,threshold):
    p,r,f=sparse_prf(truth,pred,len(sequence),True)
    ep,er,ef=sparse_prf(truth,pred,len(sequence),False)
    return dict(rna_id=sid,family=family,sequence=sequence,threshold=float(threshold),
        true_pairs=[[int(a),int(b)] for a,b in sorted(truth)],predicted_pairs=[[int(a),int(b)] for a,b in sorted(pred)],
        precision=p,recall=r,f1=f,exact_precision=ep,exact_recall=er,exact_f1=ef)

def evaluate_e(family,mode,seed):
    lock=json.loads((OUT/'selection_lock.json').read_text());assert lock['passed'] and not lock['test_scores_used']
    assert lock['code_hash']==code_identity()['sha256'] and lock['input_hash']==input_identity()['sha256']
    assert lock['evaluation_source_sha256']==sha(Path(__file__))
    choice=lock['choices'][f'{family}/{mode}/seed{seed}'];directory=OUT/'runs'/family/mode/f'seed{seed}'
    path=directory/'best.pt';assert sha(path)==choice['checkpoint_sha256']
    output=OUT/'tests'/family/mode/f'seed{seed}';output.mkdir(parents=True,exist_ok=True)
    if (output/'summary.json').exists():
        summary=json.loads((output/'summary.json').read_text())
        assert summary['selection_lock_sha256']==sha(OUT/'selection_lock.json')
        assert summary['predictions_sha256']==sha(output/'predictions.jsonl');return
    set_seed(seed);model=ThermoFusion(mode).cuda().eval()
    checkpoint=torch.load(path,map_location='cuda',weights_only=False);model.load_state_dict(checkpoint['model'])
    assert checkpoint['epoch']==choice['epoch']
    ids,records,embeddings=load_inputs(family,('test',));rows=[];started=time.time();threshold=choice['threshold']
    temporary=output/'predictions.jsonl.tmp'
    with temporary.open('w') as stream,torch.inference_mode():
        for sid in ids['test']:
            record=records[sid];logits=model(*model_inputs(record,embeddings[sid],torch.device('cuda')))
            pred=set(decode_multi(logits.cpu().numpy(),record['sequence'],[threshold])[threshold])
            row=prediction_record(sid,family,record['sequence'],record['original'],pred,threshold)
            rows.append(row);stream.write(json.dumps(row)+'\n')
    temporary.replace(output/'predictions.jsonl')
    write_json(output/'summary.json',dict(completed=True,family=family,mode=mode,seed=seed,n=len(rows),
        rna_macro=average(rows),selection=choice,selection_lock_sha256=sha(OUT/'selection_lock.json'),
        predictions_sha256=sha(output/'predictions.jsonl'),seconds=time.time()-started))
    print('ARCHIVEII_FROZEN_TEST_COMPLETE',family,mode,seed,average(rows),flush=True)

class OfficialRiNALMo(nn.Module):
    """Same module names, head and forward pass as the official Lightning wrapper."""
    def __init__(self):
        super().__init__()
        from rinalmo.config import model_config
        from rinalmo.model.model import RiNALMo
        from rinalmo.model.downstream import SecStructPredictionHead
        self.lm=RiNALMo(model_config('giga'))
        self.pred_head=SecStructPredictionHead(self.lm.config['model']['transformer'].embed_dim,num_blocks=2)
    def forward(self,tokens):return self.pred_head(self.lm(tokens)['representation'][...,1:-1,:]).squeeze(-1)

def evaluate_baseline(family):
    from scripts.check_archiveii_frozen_sources import verify
    source_verification=verify()
    set_seed(17);manifest=load_manifest();directory=OUT/'baseline'/family;directory.mkdir(parents=True,exist_ok=True)
    weight=ROOT/'weights/archiveii_frozen'/f'rinalmo_giga_ss_archiveII-{family}_ft.pt'
    provenance=json.loads(Path(str(weight)+'.provenance.json').read_text());assert sha(weight)==provenance['sha256']
    if (directory/'summary.json').exists():
        summary=json.loads((directory/'summary.json').read_text())
        assert summary['checkpoint']['sha256']==provenance['sha256']
        assert summary['predictions_sha256']==sha(directory/'predictions.jsonl');return
    saved=torch.load(weight,map_location='cpu',weights_only=False);threshold=float(saved.pop('threshold'))
    model=OfficialRiNALMo();model.load_state_dict(saved,strict=True);del saved
    model=model.cuda().eval().requires_grad_(False)
    from rinalmo.data.alphabet import Alphabet
    alphabet=Alphabet(**model.lm.config['alphabet']);rows=[];started=time.time()
    with (directory/'predictions.jsonl.tmp').open('w') as stream,torch.inference_mode():
        for index,sid in enumerate(manifest['folds'][family]['test'],1):
            record=manifest['records'][sid];sequence=record['sequence']
            tokens=torch.tensor([alphabet.encode(sequence)],device='cuda',dtype=torch.int64)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(tokens);probs=torch.sigmoid(logits)[0].cpu().numpy()
            pred=pair_set(official.prob_mat_to_sec_struct(probs.copy(),sequence,threshold))
            truth={tuple(p) for p in record['pairs']}
            row=prediction_record(sid,family,normalized_sequence(sequence),truth,pred,threshold)
            rows.append(row);stream.write(json.dumps(row)+'\n')
            if index%200==0:print('OFFICIAL_BASELINE',family,index,flush=True)
    (directory/'predictions.jsonl.tmp').replace(directory/'predictions.jsonl')
    write_json(directory/'summary.json',dict(completed=True,family=family,mode='RiNALMo_official',n=len(rows),
        rna_macro=average(rows),threshold=threshold,checkpoint=provenance,predictions_sha256=sha(directory/'predictions.jsonl'),
        seconds=time.time()-started,data_manifest_sha256=sha(OUT/'data_manifest.json.gz'),evaluation_source_sha256=sha(Path(__file__)),
        official_source_verification=source_verification,
        description='Public official fine-tuned checkpoint, native head/threshold/FP16 inference; not a repeated-seed baseline.'))
    print('OFFICIAL_BASELINE_COMPLETE',family,average(rows),flush=True)

def verify_file(path,family,manifest):
    expected=set(manifest['folds'][family]['test']);seen=set();rows=[]
    for line in path.read_text().splitlines():
        row=json.loads(line);sid=row['rna_id'];assert sid in expected and sid not in seen;seen.add(sid)
        record=manifest['records'][sid];assert row['sequence']==normalized_sequence(record['sequence']) and row['family']==family
        truth={tuple(p) for p in record['pairs']};pred={tuple(p) for p in row['predicted_pairs']}
        assert {tuple(p) for p in row['true_pairs']}==truth
        assert len(pred)==len(row['predicted_pairs'])
        occupied=set();length=len(row['sequence'])
        target=np.zeros((length,length),dtype=np.int8);prediction=np.zeros_like(target)
        for a,b in truth:target[a,b]=target[b,a]=1
        for a,b in pred:
            assert 0<=a<b<length and b-a>=4 and a not in occupied and b not in occupied
            assert row['sequence'][a]+row['sequence'][b] in {'AU','UA','CG','GC','GU','UG'}
            occupied.update((a,b));prediction[a,b]=prediction[b,a]=1
        scores=(official.ss_precision(target,prediction),official.ss_recall(target,prediction),official.ss_f1(target,prediction),
            official.ss_precision(target,prediction,False),official.ss_recall(target,prediction,False),official.ss_f1(target,prediction,False))
        assert all(abs(row[key]-score)<1e-12 for key,score in zip(FIELDS,scores)),sid
        rows.append(row)
    assert seen==expected
    return rows

def finalize():
    manifest=load_manifest();lock_path=OUT/'selection_lock.json';lock=json.loads(lock_path.read_text());assert len(lock['choices'])==81
    verified=[];total=0
    for family in FAMILIES:
        paths=[OUT/'baseline'/family]+[OUT/'tests'/family/mode/f'seed{seed}' for mode in MODES for seed in SEEDS]
        for directory in paths:
            summary=json.loads((directory/'summary.json').read_text());assert summary['completed']
            assert summary['predictions_sha256']==sha(directory/'predictions.jsonl')
            if summary['mode']!='RiNALMo_official':
                assert summary['selection_lock_sha256']==sha(lock_path)
                assert summary['selection']==lock['choices'][f"{family}/{summary['mode']}/seed{summary['seed']}"]
            rows=verify_file(directory/'predictions.jsonl',family,manifest)
            assert len(rows)==summary['n']
            threshold=summary['selection']['threshold'] if summary['mode']!='RiNALMo_official' else summary['threshold']
            assert all(row['threshold']==threshold for row in rows)
            assert all(abs(average(rows)[key]-summary['rna_macro'][key])<1e-12 for key in FIELDS)
            total+=len(rows);verified.append(dict(path=directory.relative_to(ROOT).as_posix(),n=len(rows),predictions_sha256=summary['predictions_sha256']))
            print('INDEPENDENTLY_VERIFIED',directory.relative_to(OUT),len(rows),flush=True)
    assert len(verified)==90 and total==38650
    write_json(OUT/'artifact_validation.json',dict(passed=True,independently_verified_files=90,predictions=total,
        evaluator='Unmodified official matrix ss_precision/ss_recall/ss_f1 for both flexible and exact pairing; complete original CT truth.',
        selection_lock_sha256=sha(lock_path),files=verified))
    print('ARCHIVEII_FROZEN_ARTIFACTS_VALIDATED',total,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');parser.add_argument('--finalize',action='store_true')
    parser.add_argument('--family',choices=FAMILIES);parser.add_argument('--mode',choices=MODES)
    parser.add_argument('--seed',type=int,choices=SEEDS);parser.add_argument('--baseline',action='store_true');args=parser.parse_args()
    if args.seal:seal()
    elif args.finalize:finalize()
    elif args.baseline:evaluate_baseline(args.family)
    else:evaluate_e(args.family,args.mode,args.seed)
