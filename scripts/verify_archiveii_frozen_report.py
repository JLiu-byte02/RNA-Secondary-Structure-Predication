"""Read-only independent verification of the delivered real ArchiveII report."""
import argparse
import csv
from datetime import datetime
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.archiveii_frozen_protocol import FAMILIES,MODES,SEEDS,PRETRAINED_SHA,shared_threshold

def verify(report,strict_manifest=False):
    report=report.resolve()
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    def j(name):return json.loads((report/name).read_text(encoding='utf8'))
    def c(name):
        with (report/name).open(encoding='utf8',newline='') as stream:return list(csv.DictReader(stream))
    def close(a,b):assert math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-12),(a,b)
    manifest=j('delivered_files.json')
    for name,value in manifest.items():
        path=report/name;assert path.resolve().is_relative_to(report)
        assert path.stat().st_size==value['bytes'] and sha(path)==value['sha256'],name
    if strict_manifest:
        actual={p.relative_to(report).as_posix() for p in report.rglob('*') if p.is_file()}
        assert actual==set(manifest)|{'delivered_files.json'}
    identity=j('experiment_identity.json');sources=identity['training_source']['files']
    for name,expected in sources.items():assert sha(ROOT/name)==expected,name
    assert hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()==identity['training_source']['sha256']
    assert sha(report/'archiveii_frozen.yaml')==identity['config_sha256']
    assert sha(report/'embedding_provenance.json')==identity['inputs']['files']['embedding_provenance.json']
    assert sha(report/'preparation_complete.json')==identity['preparation_sha256']
    embedding=j('embedding_provenance.json');assert embedding['frozen'] and not embedding['supervised_weights_loaded']
    assert embedding['weight_sha256']==PRETRAINED_SHA
    assert embedding['n']==3865 and embedding['embeddings_sha256']==identity['inputs']['files']['embeddings']
    audit=j('split_audit.json');assert audit['passed'] and audit['total_test_RNAs']==3865
    assert all(n==0 for n in audit['identical_sequence_groups_involving_test'].values())
    with gzip.open(report/'fold_membership.csv.gz','rt',encoding='utf8',newline='') as stream:membership=list(csv.DictReader(stream))
    assert len(membership)==9*3865
    ids_by_role={};canonical={}
    for family in FAMILIES:
        rows=[row for row in membership if row['fold_test_family']==family];seen=set()
        assert len(rows)==3865
        for row in rows:
            sid=row['rna_id'];assert sid not in seen;seen.add(sid)
            assert row['role'] in ('train','valid','test')
            assert (row['rna_family']==family)==(row['role']=='test')
            content=(row['rna_family'],row['length'],row['ct_sha256'])
            if sid in canonical:assert canonical[sid]==content
            else:canonical[sid]=content
        assert set(row['rna_family'] for row in rows if row['role']!='test')==set(FAMILIES)-{family}
        for role in ('train','valid','test'):
            ids_by_role[family,role]={row['rna_id'] for row in rows if row['role']==role}
            assert len(ids_by_role[family,role])==audit['fold_counts'][family][role]
    assert len(canonical)==3865
    source=j('source_dataset.json');assert source['fold_membership_rows']==len(membership)
    assert source['archive_sha256']==audit['archive_sha256']
    assert not source['new_length_filter'] and not source['new_deduplication']
    empty_candidates=set(source['empty_candidate_RNAs'])
    runs=c('run_results.csv');families=c('family_results.csv');overall=c('overall_results.csv')
    paired=c('paired_E_vs_E0.csv');epochs=c('epoch_curves.csv')
    expected={(f,m,str(s)) for f in FAMILIES for m in MODES for s in SEEDS}|{(f,'RiNALMo_official','') for f in FAMILIES}
    assert len(runs)==90 and {(r['family'],r['mode'],r['seed']) for r in runs}==expected
    assert sum(int(r['n']) for r in runs)==38650
    expected_epochs={(f,m,str(s),str(e)) for f in FAMILIES for m in MODES for s in SEEDS for e in range(1,51)}
    assert len(epochs)==4050 and {(r['family'],r['mode'],r['seed'],r['epoch']) for r in epochs}==expected_epochs
    for row in epochs:
        f=row['family'];train=ids_by_role[f,'train'];valid=ids_by_role[f,'valid']
        assert int(row['n_training_RNAs'])==len(train-empty_candidates)
        assert int(row['n_expected_training_RNAs'])==len(train) and int(row['n_validated_RNAs'])==len(valid)
        for field in ('train_bce_online','validation_bce','validation_exact_f1','candidate_accuracy','candidate_balanced_accuracy','training_seconds'):
            assert math.isfinite(float(row[field])) and float(row[field])>=0
    for row in runs:
        assert int(row['n'])==audit['test_counts'][row['family']]
        assert float(row['f1'])+1e-12>=float(row['exact_f1'])
    metrics=('precision','recall','f1','exact_precision','exact_recall','exact_f1')
    assert len(families)==36
    all_modes=('RiNALMo_official',)+MODES
    assert {(r['family'],r['mode']) for r in families}=={(f,m) for f in FAMILIES for m in all_modes}
    for group in families:
        values=[r for r in runs if r['family']==group['family'] and r['mode']==group['mode']]
        assert len(values)==int(group['n_models'])
        for metric in metrics:
            scores=[float(r[metric]) for r in values];close(statistics.mean(scores),group[metric+'_mean'])
            if len(scores)==3:close(statistics.stdev(scores),group[metric+'_sample_sd'])
            else:assert group[metric+'_sample_sd']==''
    assert len(overall)==8
    assert {(r['mode'],r['weighting']) for r in overall}=={(m,w) for m in all_modes for w in ('equal_family','RNA_count_weighted')}
    for group in overall:
        seeds=('',) if group['mode']=='RiNALMo_official' else tuple(map(str,SEEDS))
        for metric in metrics:
            scores=[]
            for seed in seeds:
                values=[r for r in runs if r['mode']==group['mode'] and r['seed']==seed];assert len(values)==9
                weights=[1 if group['weighting']=='equal_family' else int(r['n']) for r in values]
                scores.append(sum(float(r[metric])*w for r,w in zip(values,weights))/sum(weights))
            close(statistics.mean(scores),group[metric+'_mean'])
            if len(scores)==3:close(statistics.stdev(scores),group[metric+'_sample_sd'])
            else:assert group[metric+'_sample_sd']==''
    assert len(paired)==20
    assert {(r['family'],r['contrast']) for r in paired}=={(f,m+' - E0_no_thermo') for f in tuple(FAMILIES)+('ALL_9_equal_family',) for m in MODES[1:]}
    for group in paired:
        mode=group['contrast'].split(' - ')[0]
        for metric in ('f1','exact_f1'):
            values=[]
            for seed in SEEDS:
                def score(method):
                    return statistics.mean(float(r[metric]) for r in runs if r['mode']==method and r['seed']==str(seed)
                        and (r['family']==group['family'] or group['family']=='ALL_9_equal_family'))
                difference=score(mode)-score(MODES[0]);values.append(difference);close(difference,group[f'{metric}_seed{seed}'])
            close(statistics.mean(values),group[metric+'_mean']);close(statistics.stdev(values),group[metric+'_sample_sd'])
            assert sum(v>0 for v in values)==int(group[metric+'_positive_seeds'])
    lock=j('selection_lock.json');summaries=j('trajectory_summaries.json')
    assert lock['passed'] and not lock['test_scores_used'] and len(lock['choices'])==len(summaries)==81
    assert lock['experiment_identity_sha256']==sha(report/'experiment_identity.json')
    for family in FAMILIES:
        for mode in MODES:
            group=[r for r in summaries if r['family']==family and r['mode']==mode];assert len(group)==3
            curves=[]
            for summary in group:
                seed=summary['seed'];choice=lock['choices'][f'{family}/{mode}/seed{seed}']
                track=sorted([r for r in epochs if r['family']==family and r['mode']==mode and r['seed']==str(seed)],key=lambda r:int(r['epoch']))
                peak=max(track,key=lambda r:float(r['validation_exact_f1']))
                assert int(peak['epoch'])==choice['epoch']==summary['best_epoch']
                assert summary['epochs_completed']==50 and summary['encoder_frozen'] and summary['gradients_verified']
                assert summary['code_hash']==identity['training_source']['sha256'] and summary['input_hash']==identity['inputs']['sha256']
                assert summary['parameter_count']==186306
                assert summary['train_count']==audit['fold_counts'][family]['train']
                assert summary['validation_count']==audit['fold_counts'][family]['valid']
                assert summary['best_checkpoint_sha256']==choice['checkpoint_sha256']
                close(summary['training_seconds'],sum(float(r['training_seconds']) for r in track))
                curves.append({float(t):v['rna_macro_exact_f1'] for t,v in summary['best_validation']['threshold_metrics'].items()})
            threshold=shared_threshold(curves)
            for summary in group:
                choice=lock['choices'][f"{family}/{mode}/seed{summary['seed']}"]
                close(choice['threshold'],threshold)
                close(choice['validation_exact_f1'],summary['best_validation']['threshold_metrics'][str(threshold)]['rna_macro_exact_f1'])
        for seed in SEEDS:
            selected=[r for r in summaries if r['family']==family and r['seed']==seed]
            assert len({r['initialization_hash'] for r in selected})==len({r['head_initialization_hash'] for r in selected})==1
    validation=j('artifact_validation.json');assert validation['passed'] and validation['predictions']==38650
    assert validation['independently_verified_files']==90 and validation['selection_lock_sha256']==sha(report/'selection_lock.json')
    jobs=j('slurm_jobs.json');assert len(jobs)==102
    assert all(row['status']=='COMPLETED' and row['exit_code']=='0:0' for row in jobs.values())
    accounting={fields[0]:fields for line in (report/'slurm_all_attempts.txt').read_text().splitlines()
        if len(fields:=line.split('|'))>=7 and fields[0].isdigit()}
    assert all(row['id'] in accounting for row in jobs.values())
    for row in jobs.values():assert accounting[row['id']][1]=='COMPLETED' and accounting[row['id']][2]=='0:0'
    parse=lambda value:datetime.strptime(value,'%Y-%m-%dT%H:%M:%S')
    last_train=max(parse(accounting[row['id']][5]) for name,row in jobs.items() if name.startswith('train_'))
    first_test=min(parse(accounting[row['id']][4]) for name,row in jobs.items() if name.startswith('test_'))
    assert last_train<=first_test,'An E test started before all E training finished'
    previous=j('previous_preparation_jobs.json')
    assert all(row['id'] in accounting and accounting[row['id']][1].startswith(row['status']) for row in previous)
    trends_verified=0
    if (report/'training_budget_trends.csv').exists():
        trends=c('training_budget_trends.csv');assert len(trends)==15
        assert {(r['mode'],int(r['epoch_budget'])) for r in trends}=={(m,b) for m in MODES for b in (10,20,30,40,50)}
        for row in trends:
            budget=int(row['epoch_budget']);assert int(row['n_trajectories'])==27
            group=[r for r in epochs if r['mode']==row['mode'] and int(r['epoch'])==budget];assert len(group)==27
            for field in ('train_bce_online','validation_bce','validation_exact_f1','candidate_accuracy','candidate_balanced_accuracy'):
                close(statistics.mean(float(r[field]) for r in group),row[field+'_mean'])
            tracks=[[r for r in epochs if r['mode']==row['mode'] and r['family']==f and r['seed']==str(s) and int(r['epoch'])<=budget]
                    for f in FAMILIES for s in SEEDS]
            assert all(len(t)==budget for t in tracks)
            close(statistics.mean(max(float(r['validation_exact_f1']) for r in t) for t in tracks),row['best_validation_exact_f1_within_budget_mean'])
            close(statistics.mean(sum(float(r['training_seconds'])+float(r['validation_seconds']) for r in t)/60 for t in tracks),row['cumulative_train_and_validation_minutes_mean'])
        trends_verified=len(trends)
    return dict(passed=True,manifest_files_verified=len(manifest),fold_membership_rows_verified=34785,
        complete_training_epochs=4050,complete_RNA_coverage_each_epoch=True,model_family_seed_results=90,
        family_aggregates_recomputed=36,overall_aggregates_recomputed=8,paired_contrasts_recomputed=20,
        validation_only_choices_verified=81,logical_predictions=38650,successful_final_slurm_jobs=102,
        E_tests_started_after_all_training=True,numerical_source_unchanged=True,
        historical_preparation_attempts_verified=len(previous),
        training_budget_trends_recomputed=trends_verified,
        strict_manifest=strict_manifest,limitation='Raw checkpoints and prediction streams remain on the server; their official-function recomputation is recorded in artifact_validation.json.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,default=ROOT/'reports/archiveii_frozen_2026-10-22')
    parser.add_argument('--strict-manifest',action='store_true');args=parser.parse_args()
    print(json.dumps(verify(args.report,args.strict_manifest)))
