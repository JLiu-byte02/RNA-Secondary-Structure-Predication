"""Seal all five validation-only horizons before any new held-out evaluation."""
import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.long_protocol import MODES,SEEDS,BUDGETS,select_epoch
from scripts.full_protocol import shared_threshold
from scripts.cache_thermo import file_hash,write_json
from scripts.run_long_epochs import code_identity
from scripts.run_thermo_fast import digest,input_hash
from scripts.eval_full_tr0 import metrics_file
OUT=ROOT/'outputs/long_epochs'

def read_rows(directory):
    rows=[json.loads(line) for line in (directory/'epochs.jsonl').read_text().splitlines() if line.strip()]
    if [r['epoch'] for r in rows]!=list(range(1,51)):raise ValueError('Require 50 complete consecutive epochs')
    if any(not math.isfinite(r['score']) or not math.isfinite(r['mean_pair_loss']) or not math.isfinite(r['validation']['mean_pair_bce']) for r in rows):
        raise ValueError('Nonfinite trajectory diagnostics')
    return rows

def evaluation_identity():
    return dict(runtime=code_identity(),evaluation={p:file_hash(ROOT/p) for p in
        ('scripts/eval_long_epochs.py','scripts/eval_full_tr0.py')})

def lock():
    import torch
    from scripts._common import load_split
    from scripts.run_thermo_fast import state_hash
    split=load_split(ROOT);choices={};heads={};configs=set();inputs=set();codes=set()
    curves={};summaries={}
    for mode in MODES:
        for seed in SEEDS:
            directory=OUT/mode/f'seed{seed}';summary=json.loads((directory/'summary.json').read_text())
            rows=read_rows(directory);key=f'{mode}/seed{seed}';summaries[key]=summary
            if summary['status']!='completed' or summary['epochs_completed']!=50 or summary['early_stopping']:
                raise ValueError('Incomplete fixed trajectory: '+key)
            if summary['train_count']!=len(split['TR0']) or summary['validation_count']!=len(split['VL0']):raise ValueError('Cohort counts changed')
            if summary['train_ids_hash']!=digest(split['TR0']) or summary['validation_ids_hash']!=digest(split['VL0']):raise ValueError('Cohort identity changed')
            if not summary['encoder_frozen'] or not summary['gradients_verified'] or summary['test_scores_used']:raise ValueError('Invalid training contract')
            expected=175553 if mode=='B0' else 186306
            if summary['parameter_count']!=expected:raise ValueError('Baseline architecture changed')
            heads.setdefault(seed,set()).add(summary['head_initialization_hash'])
            configs.add(summary['config_hash']);inputs.add(summary['input_provenance_hash']);codes.add(summary['code_hash'])
            for budget in BUDGETS:
                selected=select_epoch(rows,budget);path=directory/f"epoch{selected['epoch']}.pt"
                state=torch.load(path,map_location='cpu',weights_only=False)
                if state['epoch']!=selected['epoch'] or state['mode']!=mode or state['seed']!=seed:raise ValueError('Selected checkpoint metadata changed')
                identifier=f'{key}/budget{budget}'
                choices[identifier]=dict(mode=mode,seed=seed,budget=budget,epoch=selected['epoch'],
                    checkpoint_path=str(path.relative_to(ROOT)),checkpoint_sha256=file_hash(path),
                    parameter_state_sha256=state_hash(state['model']),epochs_sha256=file_hash(directory/'epochs.jsonl'),
                    validation_curve=selected['validation']['threshold_metrics'])
                curves[(mode,seed,budget)]={float(t):m['rna_macro_exact_f1'] for t,m in selected['validation']['threshold_metrics'].items()}
    if len(configs)!=1 or len(inputs)!=1 or len(codes)!=1 or any(len(v)!=1 for v in heads.values()):raise ValueError('Paired experiment identities differ')
    if next(iter(codes))!=code_identity()['sha256']:raise ValueError('Training runtime source changed')
    for mode in MODES:
        for budget in BUDGETS:
            threshold=shared_threshold([curves[(mode,s,budget)] for s in SEEDS])
            for seed in SEEDS:choices[f'{mode}/seed{seed}/budget{budget}']['threshold']=threshold
    path=OUT/'selection_lock.json'
    existing=json.loads(path.read_text()) if path.exists() else {}
    record=dict(protocol='LONG-50-v1',locked_unix=existing.get('locked_unix',time.time()),selection_split='VL0',
        test_scores_used=False,budgets=list(BUDGETS),seeds=list(SEEDS),choices=choices,
        input_provenance_hash=next(iter(inputs)),config_hash=next(iter(configs)),evaluation_identity=evaluation_identity())
    if existing and record!=existing:raise ValueError('Refuse to change sealed selections')
    write_json(path,record);print('ALL_60_BUDGET_SELECTIONS_LOCKED',flush=True)

def test_run(mode,seed):
    import torch
    from scripts._common import load_manifest,load_split,set_seed
    from scripts.long_models import make_model
    from scripts.run_thermo_fast import build_records,evaluate
    set_seed(seed);device=torch.device('cuda')
    sealed=json.loads((OUT/'selection_lock.json').read_text());lock_sha=file_hash(OUT/'selection_lock.json')
    if sealed['evaluation_identity']!=evaluation_identity():raise ValueError('Sealed evaluation source changed')
    audit=json.loads((ROOT/'outputs/full_tr0/split_audit.json').read_text())
    cache=json.loads((ROOT/'outputs/full_tr0/thermo_cache_manifest.json').read_text())
    if input_hash(ROOT,audit,cache)!=sealed['input_provenance_hash']:raise ValueError('Evaluation inputs changed')
    manifest,split=load_manifest(ROOT),load_split(ROOT)
    identifiers=split['VL0']+split['TS0']+split['bprna_new']
    records=build_records(ROOT,identifiers,manifest,cache['samples']);embeddings={}
    for name in ('VL0','TS0','bprna_new'):
        path=ROOT/'embeddings/full'/f'{name}.pt'
        if file_hash(path)!=audit['embedding_sha256'][name]:raise ValueError('Test embedding changed')
        embeddings.update(torch.load(path,map_location='cpu',weights_only=False))
    model=make_model(mode).to(device);directory=OUT/mode/f'seed{seed}'
    reuse={};completed=[]
    for budget in BUDGETS:
        selection=sealed['choices'][f'{mode}/seed{seed}/budget{budget}']
        checkpoint=ROOT/selection['checkpoint_path'];threshold=selection['threshold']
        if file_hash(checkpoint)!=selection['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
        target=directory/'tests'/f'budget{budget}';target.mkdir(parents=True,exist_ok=True)
        summary_path=target/'summary.json'
        if summary_path.exists():
            summary=json.loads(summary_path.read_text())
            if summary['selection_lock_sha256']!=lock_sha:raise ValueError('Completed test selection changed')
            for name,metrics in summary['metrics'].items():
                path=target/f'{name}_predictions.jsonl'
                if file_hash(path)!=metrics['predictions_sha256']:raise ValueError('Completed prediction changed')
                reuse[(selection['checkpoint_sha256'],threshold,name)]=(path,metrics)
            completed.append(summary);continue
        state=torch.load(checkpoint,map_location=device,weights_only=False);model.load_state_dict(state['model'])
        metrics={};runtime={}
        for name in ('VL0','TS0','bprna_new'):
            path=target/f'{name}_predictions.jsonl';key=(selection['checkpoint_sha256'],threshold,name)
            if key in reuse:
                origin,values=reuse[key];shutil.copy2(origin,path);metrics[name]=dict(values)
                runtime[name]=dict(fresh_inference=False,seconds=0.,source=str(origin.relative_to(ROOT)))
            else:
                torch.cuda.synchronize();started=time.perf_counter()
                evaluate(model,split[name],embeddings,records,[threshold],device,path)
                torch.cuda.synchronize();seconds=time.perf_counter()-started
                metrics[name]=metrics_file(path,split[name],manifest,threshold)
                runtime[name]=dict(fresh_inference=True,seconds=seconds,source=None)
                reuse[key]=(path,metrics[name])
            print('BUDGET_TEST',mode,seed,budget,name,metrics[name]['rna_macro_exact_f1'],flush=True)
        summary=dict(status='completed',mode=mode,seed=seed,budget=budget,epoch=selection['epoch'],threshold=threshold,
            checkpoint_sha256=selection['checkpoint_sha256'],selection_lock_sha256=lock_sha,metrics=metrics,runtime=runtime)
        write_json(summary_path,summary);completed.append(summary)
    write_json(directory/'test_summary.json',dict(status='completed',mode=mode,seed=seed,budgets=list(BUDGETS),
        selection_lock_sha256=lock_sha,completed_budget_count=len(completed)))

def finalize():
    from scripts._common import load_manifest,load_split
    manifest,split=load_manifest(ROOT),load_split(ROOT)
    sealed=json.loads((OUT/'selection_lock.json').read_text());lock_sha=file_hash(OUT/'selection_lock.json')
    if sealed['evaluation_identity']!=evaluation_identity():raise ValueError('Evaluation code changed')
    rows=[];checked={};conceptual_predictions=fresh=0
    for mode in MODES:
        for seed in SEEDS:
            directory=OUT/mode/f'seed{seed}';trajectory=read_rows(directory)
            if file_hash(directory/'epochs.jsonl')!=sealed['choices'][f'{mode}/seed{seed}/budget10']['epochs_sha256']:
                raise ValueError('Trajectory changed after lock')
            for budget in BUDGETS:
                selection=sealed['choices'][f'{mode}/seed{seed}/budget{budget}']
                if select_epoch(trajectory,budget)['epoch']!=selection['epoch']:raise ValueError('Budget selection mismatch')
                if file_hash(ROOT/selection['checkpoint_path'])!=selection['checkpoint_sha256']:raise ValueError('Checkpoint changed')
                target=directory/'tests'/f'budget{budget}';summary=json.loads((target/'summary.json').read_text())
                if summary['selection_lock_sha256']!=lock_sha or summary['threshold']!=selection['threshold']:raise ValueError('Test lock mismatch')
                for name in ('VL0','TS0','bprna_new'):
                    path=target/f'{name}_predictions.jsonl';sha=file_hash(path);key=(sha,name,selection['threshold'])
                    if key not in checked:checked[key]=metrics_file(path,split[name],manifest,selection['threshold'])
                    metrics=checked[key]
                    if metrics!=summary['metrics'][name]:raise ValueError('Independent metric recomputation differs')
                    if name=='VL0':
                        expected=selection['validation_curve'][str(selection['threshold'])]['rna_macro_exact_f1']
                        if abs(metrics['rna_macro_exact_f1']-expected)>1e-12:raise ValueError('VL0 predictions differ from sealed curve')
                    rows.append(dict(mode=mode,seed=seed,budget=budget,split=name,selected_epoch=selection['epoch'],threshold=selection['threshold'],**metrics))
                    conceptual_predictions+=metrics['n'];fresh+=int(summary['runtime'][name]['fresh_inference'])
    with (OUT/'budget_results.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(OUT/'artifact_validation.json',dict(passed=True,trajectories=12,epochs_per_trajectory=50,
        budget_cohort_evaluations=len(rows),conceptual_predictions=conceptual_predictions,
        unique_prediction_files_independently_recomputed=len(checked),
        unique_predictions_independently_recomputed=sum(m['n'] for m in checked.values()),
        fresh_forward_cohort_evaluations=fresh,selection_lock_sha256=lock_sha,test_scores_used_for_selection=False))
    print('LONG_BUDGET_ARTIFACTS_VALIDATED',len(rows),'unique prediction files',len(checked),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--lock',action='store_true');parser.add_argument('--finalize',action='store_true')
    parser.add_argument('--mode',choices=MODES);parser.add_argument('--seed',type=int,choices=SEEDS);args=parser.parse_args()
    if args.lock:lock()
    elif args.finalize:finalize()
    elif args.mode and args.seed:test_run(args.mode,args.seed)
    else:parser.error('Choose --lock, --finalize, or --mode/--seed')
