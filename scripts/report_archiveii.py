"""Audit real LOFO predictions and report equal-family and pooled-RNA results."""
import argparse
import concurrent.futures
import csv
import gzip
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'third_party/rinalmo_upstream'))
from scripts.archiveii_data import OUT,FAMILIES,EXPECTED,load_manifest,sha,write_json
from scripts.archiveii_metrics import sparse_prf
MODES=('E0_no_thermo','E1_bpp','E2_bpp_stack');SEEDS=(17,23,41)
FIELDS=('precision','recall','f1','exact_precision','exact_recall','exact_f1')
REPORT=ROOT/'reports/archiveii_ood_2026-10-22'

def csv_write(path,rows):
    assert rows;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def directories():
    for family in FAMILIES:
        for seed in SEEDS:
            for mode in MODES:yield family,mode,seed,OUT/'runs'/family/mode/f'seed{seed}'
        yield family,'RiNALMo_official',None,OUT/'baseline'/family

def partial():
    rows=[]
    for family,mode,seed,directory in directories():
        path=directory/'summary.json'
        if path.exists():
            s=json.loads(path.read_text());assert s['completed']
            rows.append(dict(family=family,mode=mode,seed=seed,n=s['n'],**s['rna_macro']))
    if rows:csv_write(OUT/'progress_completed_runs.csv',rows)
    complete=[f for f in FAMILIES if sum(r['family']==f and r['mode']!='RiNALMo_official' for r in rows)==9]
    write_json(OUT/'progress_summary.json',dict(final=False,completed_E_runs=sum(r['mode']!='RiNALMo_official' for r in rows),
        completed_baselines=sum(r['mode']=='RiNALMo_official' for r in rows),completed_families=complete))
    print('ACTUAL_PROGRESS',len(complete),'families;',len(rows),'completed runs',flush=True)

def validate():
    sys.path.insert(0,str(ROOT/'setup'))
    from check_archiveii_environment import verify_official_sources
    import numpy as np
    from rinalmo.utils.sec_struct import ss_precision,ss_recall,ss_f1
    manifest=load_manifest();identity=json.loads((OUT/'experiment_identity.json').read_text())
    assert verify_official_sources()==identity['official_sources']
    assert sha(OUT/'data_manifest.json.gz')==identity['data_manifest_sha256']
    assert sha(OUT/'physical_manifest.json')==identity['physical_manifest_sha256']
    for name,digest in identity['training_source']['files'].items():assert sha(ROOT/name)==digest,name
    runs=[];curves=[];checksums=[];initializations={seed:set() for seed in SEEDS};count=0;official_spotchecks=0
    for family,mode,seed,directory in directories():
        summary=json.loads((directory/'summary.json').read_text());assert summary['completed']
        pred_path=directory/'predictions.jsonl';assert sha(pred_path)==summary['predictions_sha256']
        predictions=[json.loads(line) for line in pred_path.read_text().splitlines()]
        assert len(predictions)==EXPECTED[family]
        assert {r['rna_id'] for r in predictions}==set(manifest['folds'][family]['test'])
        values={key:[] for key in FIELDS}
        for index,pred in enumerate(predictions):
            truth=manifest['records'][pred['rna_id']];length=truth['length']
            assert truth['family']==family and pred['sequence']==truth['sequence']
            target={tuple(p) for p in pred['true_pairs']};pairs={tuple(p) for p in pred['predicted_pairs']}
            assert target=={tuple(p) for p in truth['pairs']}
            assert len(pairs)==len(pred['predicted_pairs'])
            nts=[n for pair in pairs for n in pair];assert len(nts)==len(set(nts))
            seq=truth['sequence'].replace('T','U')
            assert all(0<=a<b<length and b-a>=4 and seq[a]+seq[b] in {'AU','UA','CG','GC','GU','UG'} for a,b in pairs)
            flex=sparse_prf(target,pairs,length,True);exact=sparse_prf(target,pairs,length,False)
            computed=dict(zip(FIELDS,flex+exact))
            for name,value in computed.items():
                assert math.isclose(value,float(pred[name]),abs_tol=1e-12,rel_tol=0),(family,mode,seed,pred['rna_id'],name)
                values[name].append(value)
            assert math.isclose(float(pred['threshold']),float(summary['threshold']),abs_tol=1e-12,rel_tol=0)
            if index in (0,len(predictions)-1):
                t=np.zeros((length,length));p=np.zeros_like(t)
                for a,b in target:t[a,b]=t[b,a]=1
                for a,b in pairs:p[a,b]=p[b,a]=1
                official=(ss_precision(t,p),ss_recall(t,p),ss_f1(t,p),ss_precision(t,p,False),ss_recall(t,p,False),ss_f1(t,p,False))
                assert np.allclose(official,flex+exact,rtol=0,atol=1e-15);official_spotchecks+=1
        for name in FIELDS:assert math.isclose(statistics.mean(values[name]),summary['rna_macro'][name],rel_tol=0,abs_tol=1e-12)
        for name in ('precision','recall','f1'):
            assert abs(summary['metrics']['test/'+name]-summary['rna_macro'][name])<1e-6,'Lightning float32 logging discrepancy exceeds tolerance'
        if mode!='RiNALMo_official':
            assert summary['epochs_completed']==15 and summary['code_hash']==identity['training_source']['sha256']
            initializations[seed].add(summary['head_initialization_hash'])
            lock=json.loads((directory/'selection_lock.json').read_text())
            assert sha(directory/'selection_lock.json')==summary['selection_lock_sha256']
            assert lock['epoch']==15 and not lock['test_scores_used'] and lock['threshold']==summary['threshold']
            assert lock['data_manifest_sha256']==identity['data_manifest_sha256']
            epochs=[json.loads(line) for line in (directory/'epochs.jsonl').read_text().splitlines()]
            assert [r['epoch'] for r in epochs]==list(range(1,16))
            for epoch in epochs:
                phase=(epoch['epoch']-1)//3
                names=epoch['trainable_lm_parameter_names']
                blocks=sorted({int(n.split('.')[2]) for n in names if n.startswith('transformer.blocks.')})
                assert blocks==list(range(33-phase*3,33))
                scores=epoch['threshold_f1'];threshold=max((i/100 for i in range(1,30)),key=lambda t:scores[str(t)])
                assert threshold==epoch['threshold']
                curves.append(dict(family=family,mode=mode,seed=seed,**{k:v for k,v in epoch.items() if k not in ('threshold_f1','trainable_lm_parameter_names')},trainable_blocks=','.join(map(str,blocks))))
            assert epochs[-1]['threshold']==summary['threshold']
            checksums.append((directory/'final.pt',summary['checkpoint_sha256']))
        else:
            checksums.append((ROOT/summary['checkpoint']['path'],summary['checkpoint']['sha256']))
        count+=len(predictions)
        runs.append(dict(family=family,mode=mode,seed=seed,n=len(predictions),threshold=summary['threshold'],
            **{name:statistics.mean(values[name]) for name in FIELDS},predictions_sha256=summary['predictions_sha256']))
        print('PREDICTIONS_RECOMPUTED',family,mode,seed,len(predictions),flush=True)
    assert count==38650 and len(runs)==90 and len(curves)==1215
    assert all(len(values)==1 for values in initializations.values())
    def verify_weight(item):
        path,digest=item;assert sha(path)==digest,str(path);return str(path.relative_to(ROOT))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for index,_ in enumerate(pool.map(verify_weight,checksums),1):
            if index%10==0:print('CHECKPOINTS_REHASHED',index,'/',len(checksums),flush=True)
    csv_write(OUT/'verified_runs.csv',runs);csv_write(OUT/'epoch_curves.csv',curves)
    write_json(OUT/'artifact_validation.json',dict(passed=True,completed_E_runs=81,completed_official_baselines=9,
        independently_recomputed_predictions=count,official_function_spotchecks=official_spotchecks,
        prediction_metrics_verifier='Independent sparse implementation, parity-tested at 2030 threshold cases and actual 512nt logits; official functions used by every test worker.',
        rehashed_checkpoints=len(checksums),epoch_rows=len(curves),head_initialization_identical_within_seed=True,
        data_manifest_sha256=identity['data_manifest_sha256'],source_sha256=identity['training_source']['sha256'],
        verified_runs_sha256=sha(OUT/'verified_runs.csv'),epoch_curves_sha256=sha(OUT/'epoch_curves.csv')))

def report():
    validation=json.loads((OUT/'artifact_validation.json').read_text());assert validation['passed']
    assert sha(OUT/'verified_runs.csv')==validation['verified_runs_sha256']
    state=json.loads((OUT/'cluster_status.json').read_text());assert all(r['status']=='COMPLETED' and r['exit_code']=='0:0' for r in state['jobs'].values())
    runs=list(csv.DictReader((OUT/'verified_runs.csv').open()));groups=[]
    for family in FAMILIES:
        for mode in (*MODES,'RiNALMo_official'):
            rows=[r for r in runs if r['family']==family and r['mode']==mode]
            assert len(rows)==(1 if mode=='RiNALMo_official' else 3)
            group=dict(family=family,mode=mode,n_RNAs=EXPECTED[family],n_seeds=len(rows))
            for field in FIELDS:
                values=[float(r[field]) for r in rows]
                group[field+'_mean']=statistics.mean(values);group[field+'_sample_sd']=statistics.stdev(values) if len(values)>1 else ''
            groups.append(group)
    overall=[]
    for mode in (*MODES,'RiNALMo_official'):
        seeds=('',) if mode=='RiNALMo_official' else tuple(map(str,SEEDS))
        for weighting in ('family_macro_9_equal_families','rna_macro_3865_RNAs'):
            row=dict(mode=mode,aggregation=weighting,n_families=9,n_RNAs=3865,n_seeds=len(seeds))
            for field in FIELDS:
                values=[]
                for seed in seeds:
                    selected=[r for r in runs if r['mode']==mode and r['seed']==seed];assert len(selected)==9
                    values.append(statistics.mean(float(r[field]) for r in selected) if weighting.startswith('family_')
                        else sum(int(r['n'])*float(r[field]) for r in selected)/3865)
                row[field+'_mean']=statistics.mean(values);row[field+'_sample_sd']=statistics.stdev(values) if len(values)>1 else ''
            overall.append(row)
    REPORT.mkdir(parents=True,exist_ok=True)
    csv_write(REPORT/'family_results.csv',groups);csv_write(REPORT/'overall_results.csv',overall)
    for name in ('verified_runs.csv','epoch_curves.csv','artifact_validation.json','experiment_identity.json','preparation_complete.json',
        'environment_verification.json','data_manifest.json.gz','split_audit.json','physical_manifest.json','baseline_download_manifest.json'):
        shutil.copyfile(OUT/name,REPORT/name)
    write_json(REPORT/'slurm_jobs.json',state['jobs'])
    jobids=[r['id'] for r in state['jobs'].values()]
    accounting=subprocess.check_output(['sacct','-X','-P','-j',','.join(jobids),'-o','JobIDRaw,JobName,State,ExitCode,ElapsedRaw,Start,End,NodeList'],text=True)
    (REPORT/'slurm_accounting.txt').write_text(accounting)
    def formatted(row,field='f1'):
        sd=row[field+'_sample_sd'];return f"{row[field+'_mean']:.6f}"+(f" ± {sd:.6f}" if sd!='' else '')
    family_table=[]
    for family in FAMILIES:
        by={r['mode']:r for r in groups if r['family']==family}
        family_table.append('| '+family+' | '+str(EXPECTED[family])+' | '+' | '.join(formatted(by[m]) for m in ('RiNALMo_official',*MODES))+' |')
    overall_table=[]
    for weighting in ('family_macro_9_equal_families','rna_macro_3865_RNAs'):
        by={r['mode']:r for r in overall if r['aggregation']==weighting}
        overall_table.append('| '+weighting+' | '+' | '.join(formatted(by[m]) for m in ('RiNALMo_official',*MODES))+' |')
    comparisons=[]
    for mode in MODES:
        e=next(r for r in overall if r['mode']==mode and r['aggregation'].startswith('family_'))
        baseline=next(r for r in overall if r['mode']=='RiNALMo_official' and r['aggregation'].startswith('family_'))
        wins=sum(next(r for r in groups if r['mode']==mode and r['family']==f)['f1_mean']>
            next(r for r in groups if r['mode']=='RiNALMo_official' and r['family']==f)['f1_mean'] for f in FAMILIES)
        comparisons.append(f"- {mode}：九家族等权平均F1相对官方检查点差值 {e['f1_mean']-baseline['f1_mean']:+.6f}；家族均值较高 {wins}/9。")
    text='''# ArchiveII 跨RNA家族预测：九折结果

E0/E1/E2各完成9家族×3种子的15轮在线微调，另重评估9个官方RiNALMo检查点。训练、验证和测试使用官方固定ArchiveII fam-fold文件，完整测试3865条RNA，不新增长度过滤或重新划分。

主指标为官方允许1nt配对偏移的逐RNA macro F1。E结果为种子17/23/41均值±样本SD；官方列是每家族一个发布检查点的实测分数。模型和阈值均按官方训练结束规则固定后才测试；主表不使用此前TR0实验的exact F1口径。

| 测试家族 | RNA数 | 官方RiNALMo重评估 | E0 | E1 | E2 |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(family_table)+'''

## 九折汇总

| 汇总方式 | 官方RiNALMo | E0 | E1 | E2 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(overall_table)+'''

九家族等权均值为主汇总；RNA加权均值单独报告，避免大样本家族主导“九折平均”。E的汇总SD先在每个种子内跨九家族汇总，再计算三个种子的样本SD，不把家族差异混入重复训练波动。

'''+ '\n'.join(comparisons)+'''

## 公平性、过程和局限

沿用官方逐步解冻日程、15轮、Adam初始LR1e-4、LinearLR总7000步、batch1、16-mixed及完整严格上三角BCE。阈值在各折验证集上从0.01至0.29逐0.01选择，每种子独立选择，使用第15轮参数及其阈值。E系列保持项目自身的PairHead+CorrectionCNN结构及P/Q通道定义；官方基线保持原生预测头。对齐的是数据和训练/评估过程，E头的架构差异是比较对象。

物理特征只由序列计算，不使用CT结构标签。官方数据中训练/验证的重复序列保持原状并在split_audit.json披露；九折测试与其训练/验证的完全相同序列交集均为0。Family-OOD指监督训练家族留出，不声称通用语言模型预训练从未包含这些家族。

本次E系列三种子重复和官方单检查点的重复数量不同，样本SD不是置信区间，不据此宣称统计显著性。官方论文与CLI默认学习率存在差异，显式采用论文Methods的1e-4；所有选用值、源码版本和无法逐位重建作者随机轨迹的限制见[实验协议](../../docs/ARCHIVEII_OOD_PROTOCOL.md)。

附加exact P/R/F1、主P/R/F1及每种子结果在family_results.csv、overall_results.csv和verified_runs.csv。epoch_curves.csv保留全部1215条轮记录、验证阈值、损失和实际解冻层；81条E轨迹和9个基线共38650条预测通过独立重算，全部90个模型文件重新核验SHA256。数据清单、源码指纹、官方权重公开MD5与本地SHA256、最终阈值锁定及Slurm退出证据一起保留。原始模型、各折训练日志及逐RNA预测在服务器outputs/archiveii_ood/。

来源：[官方论文](https://doi.org/10.1038/s41467-025-60872-5)、[固定官方源码](https://github.com/lbcb-sci/RiNALMo/tree/2c2c5c14a5ae609d8c560a5d9ca32e51e0288955)、[官方权重](https://zenodo.org/records/15043668)。
'''
    (REPORT/'README_results.md').write_text(text,encoding='utf8')
    write_json(REPORT/'delivered_files.json',{p.relative_to(REPORT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(REPORT.rglob('*')) if p.is_file() and p.name!='delivered_files.json'})
    with tarfile.open(ROOT/'.tmp/archiveii_report_2026-10-22.tar.gz','w:gz') as f:f.add(REPORT,arcname=REPORT.relative_to(ROOT).as_posix())
    print('ARCHIVEII_FINAL_REPORT_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');p.add_argument('--validate',action='store_true');p.add_argument('--report',action='store_true');a=p.parse_args()
    if a.partial:partial()
    if a.validate:validate()
    if a.report:report()
