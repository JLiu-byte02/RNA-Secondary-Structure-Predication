"""Lock thresholds from all seeds' VL0 curves, then evaluate fixed test cohorts."""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.full_protocol import SEEDS, shared_threshold
from scripts.cache_thermo import file_hash, write_json
OUT=ROOT/'outputs/full_tr0'
MODES=('E0_no_thermo','E1_bpp','E2_bpp_stack')


def metrics_file(path, ids, manifest, threshold):
    from src.metrics import pair_prf, rna_macro
    from src.decoder import candidate_pairs
    from src.utils import pair_set_0b, pairs_from_str
    metrics=[];tp=nr=npred=0
    with path.open() as handle:
        for index,line in enumerate(handle):
            row=json.loads(line)
            if index>=len(ids) or row['rna_id']!=ids[index] or row['threshold']!=threshold:
                raise ValueError('Prediction cohort / threshold mismatch: '+str(path))
            expected=manifest[row['rna_id']]
            truth=pair_set_0b(pairs_from_str(expected['original_pairs']))
            pred=set(map(tuple,row['predicted_pairs']))
            if set(map(tuple,row['reference_pairs']))!=truth or row['sequence']!=expected['sequence']:
                raise ValueError('Full truth / sequence mismatch')
            if not pred.issubset(set(candidate_pairs(row['sequence']))) or len({i for pair in pred for i in pair})!=2*len(pred):
                raise ValueError('Invalid decoded geometry')
            prf=pair_prf(truth,pred)
            if abs(prf[2]-row['exact_f1'])>1e-12: raise ValueError('Stored F1 mismatch')
            metrics.append(prf);tp+=len(truth&pred);nr+=len(truth);npred+=len(pred)
    if len(metrics)!=len(ids):raise ValueError('Incomplete predictions: '+str(path))
    p,r,f=rna_macro(metrics)
    return dict(n=len(metrics),precision=p,recall=r,rna_macro_exact_f1=f,
                micro_precision=tp/npred if npred else 0.,micro_recall=tp/nr if nr else 0.,
                micro_exact_f1=2*tp/(nr+npred) if nr+npred else 1.,
                true_positive_pairs=tp,reference_pairs=nr,predicted_pairs=npred,
                predictions_sha256=file_hash(path))


def lock_selection():
    from scripts._common import load_split
    from scripts.run_thermo_fast import digest
    from scripts.full_protocol import validate_config
    runs={};per_seed={};configs=set();inputs=set()
    split=load_split(ROOT)
    for mode in MODES:
        curves=[]
        for seed in SEEDS:
            directory=OUT/mode/f'seed{seed}'
            summary=json.loads((directory/'summary.json').read_text())
            config=json.loads((directory/'config.json').read_text());validate_config(config)
            if summary['status']!='completed' or summary['train_count']!=len(split['TR0']) or summary['dev_count']!=len(split['VL0']):
                raise ValueError('Training incomplete')
            if summary['train_ids_hash']!=digest(split['TR0']) or summary['dev_ids_hash']!=digest(split['VL0']) or summary['test_data_evaluated']:
                raise ValueError('Cohort / test separation mismatch')
            if not summary['gradients_verified'] or not summary['encoder_frozen']:raise ValueError('Invalid gradient contract')
            configs.add(summary['config_hash']);inputs.add(summary['input_provenance_hash'])
            per_seed.setdefault(seed,set()).add(summary['initialization_hash'])
            curve={float(k):v for k,v in json.loads((directory/'validation_curve.json').read_text()).items()}
            if sorted(curve)!=config['thresholds']:raise ValueError('Incomplete validation curve')
            curves.append(curve)
            runs[f'{mode}/seed{seed}']=dict(checkpoint_sha256=file_hash(directory/'best.pt'),
                                           summary_sha256=file_hash(directory/'summary.json'),
                                           best_epoch=summary['best_epoch'])
        threshold=shared_threshold(curves)
        for seed in SEEDS:runs[f'{mode}/seed{seed}']['threshold']=threshold
    if len(configs)!=1 or len(inputs)!=1 or any(len(v)!=1 for v in per_seed.values()):
        raise ValueError('Paired experiments differ in initialization, configuration or inputs')
    record=dict(protocol='FULL-TR0-v1',selection_split='VL0',test_scores_used=False,
                seeds=SEEDS,runs=runs,config_hash=next(iter(configs)),input_provenance_hash=next(iter(inputs)))
    path=OUT/'selection_lock.json'
    if path.exists() and json.loads(path.read_text())!=record:raise ValueError('Refuse to alter a locked selection')
    write_json(path,record)
    print('SELECTION_LOCKED',json.dumps({m:runs[m+'/seed17']['threshold'] for m in MODES}),flush=True)


def test_run(mode,seed):
    import torch
    from scripts._common import load_manifest, load_split
    from scripts.run_thermo_fast import ThermoFusion, build_records, evaluate
    lock=json.loads((OUT/'selection_lock.json').read_text())
    directory=OUT/mode/f'seed{seed}'
    selection=lock['runs'][f'{mode}/seed{seed}']; threshold=selection['threshold']
    if file_hash(directory/'best.pt')!=selection['checkpoint_sha256']:raise ValueError('Checkpoint changed after selection')
    summary_path=directory/'test_summary.json'
    if summary_path.exists():
        old=json.loads(summary_path.read_text())
        if old['selection_lock_sha256']!=file_hash(OUT/'selection_lock.json'):raise ValueError('Test selection changed')
        for name,record in old['metrics'].items():
            if file_hash(directory/f'locked_{name}_predictions.jsonl')!=record['predictions_sha256']:raise ValueError('Test predictions changed')
        print('TEST_ALREADY_COMPLETED',mode,seed,flush=True);return
    model=ThermoFusion(mode).cuda()
    checkpoint=torch.load(directory/'best.pt',map_location='cuda',weights_only=False)
    if checkpoint['epoch']!=selection['best_epoch']:raise ValueError('Epoch changed')
    model.load_state_dict(checkpoint['model'])
    manifest,split=load_manifest(ROOT),load_split(ROOT)
    features=json.loads((OUT/'thermo_cache_manifest.json').read_text())['samples']
    metrics={}
    for name in ('VL0','TS0','bprna_new'):
        ids=split[name]
        records=build_records(ROOT,ids,manifest,features)
        embeddings=torch.load(ROOT/'embeddings/full'/f'{name}.pt',map_location='cpu',weights_only=False)
        path=directory/f'locked_{name}_predictions.jsonl'
        evaluate(model,ids,embeddings,records,[threshold],torch.device('cuda'),path)
        metrics[name]=metrics_file(path,ids,manifest,threshold)
        del records,embeddings
        print('TEST_COHORT_COMPLETE',mode,seed,name,json.dumps(metrics[name]),flush=True)
    write_json(summary_path,dict(status='completed',mode=mode,seed=seed,threshold=threshold,
               checkpoint_sha256=selection['checkpoint_sha256'],selection_lock_sha256=file_hash(OUT/'selection_lock.json'),metrics=metrics))


def finalize():
    from scripts._common import load_manifest,load_split
    manifest,split=load_manifest(ROOT),load_split(ROOT)
    lock=json.loads((OUT/'selection_lock.json').read_text())
    rows=[];reviewed=0
    for mode in MODES:
        for seed in SEEDS:
            directory=OUT/mode/f'seed{seed}'
            selection=lock['runs'][f'{mode}/seed{seed}']
            summary=json.loads((directory/'test_summary.json').read_text())
            training=json.loads((directory/'summary.json').read_text())
            if file_hash(directory/'best.pt')!=selection['checkpoint_sha256'] or summary['selection_lock_sha256']!=file_hash(OUT/'selection_lock.json'):
                raise ValueError('Final selection provenance mismatch')
            log=[json.loads(line) for line in (directory/'train_log.jsonl').read_text().splitlines() if line]
            winner=max(log,key=lambda x:x['metrics']['rna_macro_exact_f1'])
            if winner['epoch']!=training['best_epoch'] or winner['threshold']!=training['threshold']:
                raise ValueError('Selected checkpoint differs from earliest best VL0 epoch')
            for name in ('VL0','TS0','bprna_new'):
                record=metrics_file(directory/f'locked_{name}_predictions.jsonl',split[name],manifest,selection['threshold'])
                if record!=summary['metrics'][name]:raise ValueError('Metrics changed during independent recomputation')
                rows.append(dict(mode=mode,seed=seed,split=name,threshold=selection['threshold'],best_epoch=selection['best_epoch'],**record))
                reviewed+=record['n']
    with (OUT/'full_results.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,list(rows[0]));writer.writeheader();writer.writerows(rows)
    aggregate=[]
    for name in ('VL0','TS0','bprna_new'):
        for mode in MODES:
            selected=[r for r in rows if r['mode']==mode and r['split']==name]
            aggregate.append(dict(mode=mode,split=name,n=selected[0]['n'],
                f1_mean=statistics.mean(r['rna_macro_exact_f1'] for r in selected),
                f1_sample_sd=statistics.stdev(r['rna_macro_exact_f1'] for r in selected),
                precision_mean=statistics.mean(r['precision'] for r in selected),
                recall_mean=statistics.mean(r['recall'] for r in selected)))
    write_json(OUT/'aggregate_results.json',aggregate)
    write_json(OUT/'artifact_validation.json',dict(passed=True,runs=9,cohort_evaluations=27,
               independently_recomputed_predictions=reviewed,selection_lock_sha256=file_hash(OUT/'selection_lock.json'),
               complete_TR0=True,selection_split='VL0',test_scores_used_for_selection=False))
    print('FULL_ARTIFACTS_VALIDATED',json.dumps(aggregate),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--lock',action='store_true')
    parser.add_argument('--finalize',action='store_true')
    parser.add_argument('--mode',choices=MODES)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args()
    if args.lock:lock_selection()
    elif args.finalize:finalize()
    elif args.mode and args.seed:test_run(args.mode,args.seed)
    else:parser.error('Choose --lock, --finalize, or --mode / --seed')
