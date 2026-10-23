"""Publish only the complete, independently verified nine-family frozen-E study."""
import csv
import gzip
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.archiveii_frozen_data import OUT,load_manifest,sha,write_json
from scripts.archiveii_frozen_protocol import FAMILIES,MODES,SEEDS
REPORT=ROOT/'reports/archiveii_frozen_2026-10-22'
FIELDS=('precision','recall','f1','exact_precision','exact_recall','exact_f1')

def csv_file(name,rows):
    with (REPORT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    validation=json.loads((OUT/'artifact_validation.json').read_text());assert validation['passed']
    assert validation['predictions']==38650 and validation['independently_verified_files']==90
    state=json.loads((OUT/'cluster_status.json').read_text())
    assert all(row['status']=='COMPLETED' and row['exit_code']=='0:0' for row in state['jobs'].values())
    manifest=load_manifest();REPORT.mkdir(parents=True,exist_ok=True)
    summaries=[];epochs=[];runs=[]
    for family in FAMILIES:
        baseline=json.loads((OUT/'baseline'/family/'summary.json').read_text())
        assert baseline['completed'];runs.append(dict(family=family,mode='RiNALMo_official',seed='',n=baseline['n'],
            **baseline['rna_macro'],epoch='',threshold=baseline['threshold'],checkpoint_sha256=baseline['checkpoint']['sha256']))
        for mode in MODES:
            for seed in SEEDS:
                directory=OUT/'runs'/family/mode/f'seed{seed}'
                summary=json.loads((directory/'summary.json').read_text());summaries.append(summary)
                result=json.loads((OUT/'tests'/family/mode/f'seed{seed}'/'summary.json').read_text())
                assert result['completed'];choice=result['selection']
                runs.append(dict(family=family,mode=mode,seed=seed,n=result['n'],**result['rna_macro'],
                    epoch=choice['epoch'],threshold=choice['threshold'],checkpoint_sha256=choice['checkpoint_sha256']))
                rows=[json.loads(line) for line in (directory/'epochs.jsonl').read_text().splitlines()]
                assert [row['epoch'] for row in rows]==list(range(1,51))
                total=0
                for row in rows:
                    total+=row['training_seconds'];v=row['validation']
                    epochs.append(dict(family=family,mode=mode,seed=seed,epoch=row['epoch'],
                        n_training_RNAs=row['n_steps'],n_expected_training_RNAs=summary['train_count'],n_validated_RNAs=v['n'],
                        train_bce_online=row['mean_pair_loss'],validation_bce=v['mean_pair_bce'],
                        validation_exact_f1=row['score'],threshold_at_epoch=row['threshold'],
                        candidate_accuracy=v['candidate_accuracy'],candidate_balanced_accuracy=v['candidate_balanced_accuracy'],
                        training_seconds=row['training_seconds'],validation_seconds=v['seconds'],cumulative_training_minutes=total/60))
    assert len(runs)==90 and len(epochs)==4050 and sum(row['n'] for row in runs)==38650
    family_rows=[];all_modes=('RiNALMo_official',)+MODES
    for family in FAMILIES:
        for mode in all_modes:
            group=[row for row in runs if row['family']==family and row['mode']==mode]
            assert len(group)==(1 if mode=='RiNALMo_official' else 3)
            row=dict(family=family,mode=mode,n=group[0]['n'],n_models=len(group))
            for key in FIELDS:
                values=[r[key] for r in group];row[key+'_mean']=statistics.mean(values)
                row[key+'_sample_sd']=statistics.stdev(values) if len(values)>1 else ''
            family_rows.append(row)
    overall=[]
    for mode in all_modes:
        for weighting in ('equal_family','RNA_count_weighted'):
            seeds=('',) if mode=='RiNALMo_official' else SEEDS
            per_seed=[]
            for seed in seeds:
                group=[row for row in runs if row['mode']==mode and row['seed']==seed];assert len(group)==9
                weights=[1 if weighting=='equal_family' else row['n'] for row in group]
                per_seed.append({key:sum(row[key]*weight for row,weight in zip(group,weights))/sum(weights) for key in FIELDS})
            row=dict(mode=mode,weighting=weighting,n_families=9,n_RNAs=3865,n_models_per_family=len(seeds))
            for key in FIELDS:
                values=[r[key] for r in per_seed];row[key+'_mean']=statistics.mean(values)
                row[key+'_sample_sd']=statistics.stdev(values) if len(values)>1 else ''
            overall.append(row)
    paired=[]
    for family in tuple(FAMILIES)+('ALL_9_equal_family',):
        for mode in MODES[1:]:
            row=dict(family=family,contrast=mode+' - E0_no_thermo')
            for metric in ('f1','exact_f1'):
                values=[]
                for seed in SEEDS:
                    def score(method):
                        selected=[r[metric] for r in runs if r['seed']==seed and r['mode']==method and
                            (r['family']==family or family=='ALL_9_equal_family')]
                        return statistics.mean(selected)
                    difference=score(mode)-score(MODES[0]);values.append(difference);row[f'{metric}_seed{seed}']=difference
                row[metric+'_mean']=statistics.mean(values);row[metric+'_sample_sd']=statistics.stdev(values)
                row[metric+'_positive_seeds']=sum(value>0 for value in values)
            paired.append(row)
    csv_file('run_results.csv',runs);csv_file('family_results.csv',family_rows)
    csv_file('overall_results.csv',overall);csv_file('paired_E_vs_E0.csv',paired);csv_file('epoch_curves.csv',epochs)
    write_json(REPORT/'trajectory_summaries.json',summaries)
    for name in ('artifact_validation.json','selection_lock.json','experiment_identity.json','split_audit.json',
        'embedding_provenance.json','official_source_verification.json','preparation_complete.json'):
        shutil.copy2(OUT/name,REPORT/name)
    previous_preparation=json.loads((OUT/'previous_preparation_jobs.json').read_text()) if (OUT/'previous_preparation_jobs.json').exists() else []
    write_json(REPORT/'previous_preparation_jobs.json',previous_preparation)
    if (OUT/'historical_preparation_correction.json').exists():shutil.copy2(OUT/'historical_preparation_correction.json',REPORT/'historical_preparation_correction.json')
    if (OUT/'download_transport_change.json').exists():shutil.copy2(OUT/'download_transport_change.json',REPORT/'download_transport_change.json')
    shutil.copy2(ROOT/'configs/archiveii_frozen.yaml',REPORT/'archiveii_frozen.yaml')
    write_json(REPORT/'slurm_jobs.json',state['jobs']);write_json(REPORT/'previous_slurm_jobs.json',state.get('previous_jobs',[]))
    preparation=previous_preparation
    ids=sorted({row['id'] for row in state['jobs'].values()}|{row['id'] for row in state.get('previous_jobs',[])}|{row['id'] for row in preparation},key=int)
    accounting=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(ids),'-o','JobIDRaw,State,ExitCode,ElapsedRaw,Start,End,NodeList'],text=True)
    (REPORT/'slurm_all_attempts.txt').write_text(accounting)
    allocated=sum(int(fields[3]) for line in accounting.splitlines() if len(fields:=line.split('|'))>=4 and fields[0] in ids)
    membership=[]
    for family in FAMILIES:
        for role in ('train','valid','test'):
            for sid in manifest['folds'][family][role]:
                r=manifest['records'][sid]
                membership.append(dict(fold_test_family=family,role=role,rna_id=sid,rna_family=r['family'],length=r['length'],ct_sha256=r['ct_sha256']))
    path=REPORT/'fold_membership.csv'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,list(membership[0]));writer.writeheader();writer.writerows(membership)
    with (REPORT/'fold_membership.csv.gz').open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as handle:handle.write(path.read_bytes())
    path.unlink()
    write_json(REPORT/'source_dataset.json',dict(url=manifest['source'],archive_sha256=manifest['archive_sha256'],
        data_manifest_sha256=sha(OUT/'data_manifest.json.gz'),physical_manifest_sha256=sha(OUT/'physical_manifest.json'),
        physical_unique_sequences=len(json.loads((OUT/'physical_manifest.json').read_text())['samples']),
        empty_truth_RNAs=[sid for sid,row in manifest['records'].items() if not row['pairs']],
        empty_candidate_RNAs=[sid for sid,row in manifest['records'].items() if not any(
            row['sequence'][a]+row['sequence'][b] in {'AU','UA','CG','GC','GU','UG'}
            for a in range(row['length']) for b in range(a+4,row['length']))],
        new_length_filter=False,new_deduplication=False,fold_membership_rows=len(membership)))
    lookup={(row['family'],row['mode']):row for row in family_rows}
    def value(row,metric):
        average=row[metric+'_mean'];sd=row[metric+'_sample_sd']
        return f'{average:.6f}' if sd=='' else f'{average:.6f} ± {sd:.6f}'
    def table(metric):
        lines=['| 测试家族 | n | 官方RiNALMo | E0 | E1 | E2 |','|---|---:|---:|---:|---:|---:|']
        for family in FAMILIES:
            values=[value(lookup[family,mode],metric) for mode in all_modes]
            lines.append('| '+family+' | '+str(lookup[family,MODES[0]]['n'])+' | '+' | '.join(values)+' |')
        return '\n'.join(lines)
    overall_table=['| 方法 | 九家族等权F1 | 按RNA数加权F1 | 九家族等权exact F1 |','|---|---:|---:|---:|']
    for mode in all_modes:
        equal=next(row for row in overall if row['mode']==mode and row['weighting']=='equal_family')
        weighted=next(row for row in overall if row['mode']==mode and row['weighting']=='RNA_count_weighted')
        overall_table.append(f"| {mode} | {value(equal,'f1')} | {value(weighted,'f1')} | {value(equal,'exact_f1')} |")
    paired_table=['| 家族 | 对比 | 配对F1差值均值 ± SD | 正差值种子数 |','|---|---|---:|---:|']
    for row in paired:paired_table.append(f"| {row['family']} | {row['contrast']} | {row['f1_mean']:+.6f} ± {row['f1_sample_sd']:.6f} | {row['f1_positive_seeds']}/3 |")
    preparation_note=('首次表征准备已算完表示，但记录安装来源时遇到namespace package没有__file__的问题。改用包搜索路径后回归测试通过，重新运行并验收准备。作业9218失败；依赖烟雾检查9219实际启动后因缺少embedding_provenance.json失败，原先“已取消”的中间记录已依据Slurm最终状态和错误日志更正，见historical_preparation_correction.json。后续准备9220及烟雾检查9221成功，旧产物单独保留。'
        if any(row['id']=='9218' for row in preparation) else '本次准备作业及任何重试均在Slurm记录中核对，失败尝试与正式成功作业分开保留。')
    transport_note=('同一官方权重的1MiB读取探测中，记录页入口6.2秒、API入口28.7秒，字节哈希一致。因此恢复下载时改用官方记录页文件入口，保留既有分块、相同Range及最终MD5校验；该探测不用于推断整批权重或训练的加速比例。详情见download_transport_change.json。'
        if (OUT/'download_transport_change.json').exists() else '')
    text=f'''# ArchiveII Family-OOD：冻结E模型九折结果

全部81条E训练轨迹均已完成50轮，使用种子17、23、41及相同官方ArchiveII家族划分。全部81个仅按验证集得到的epoch和阈值选择统一锁定后执行E测试。九个官方微调检查点在相同测试CT上重评。独立复核90份预测文件、38650条逐RNA预测，原始真值完整保留。

## 各家族性能：官方允许1nt偏移的F1

每条RNA独立计算再在家族内平均。E为三种子均值±样本SD；官方RiNALMo为一个公开微调模型的实测重评估分数，不具有本轮三种子SD。

{table('f1')}

## 完全精确配对的F1

{table('exact_f1')}

## 九轮汇总

九家族等权均值是主汇总。按RNA数量加权的均值会更受5S/SRP等大族影响，分别报告。E的汇总SD由每个种子的九家族汇总分数计算，不能解释为家族间SD或置信区间。

{chr(10).join(overall_table)}

## E部分的配对比较

相同家族、种子的E1/E2减E0。额外的物理信息是这组配对对比的受控变化。三个种子不用于宣称统计显著性。

{chr(10).join(paired_table)}

## 数据和对比边界

本轮严格采用用户确认的“冻结编码器、50轮、仅替换数据”方案，保留原E的AdamW、候选BCE、BF16计算、0.1–0.9阈值网格及验证exact F1选模规则。官方RiNALMo基线具有微调后的编码器、原生预测头及其发布阈值；它与E使用同一测试数据和指标，但训练方式与架构不同，不能把两者的全部差异归因于纠错模块。

官方数据包括3865条RNA、3354条唯一序列；保留官方train/valid/test成员，无新增去重、长度过滤或裁剪。16S/23S是该基准提供的domain。官方train/valid中的重复序列组保留并在split_audit.json披露，测试角色没有相同序列组。五条空真值RNA保留：官方报告指标的空/空分数为0，原E验证选模函数的空/空分数为1，两个用途不混称同一数值。Family-OOD指监督阶段的家族留出，不声称通用语言模型预训练排除了测试家族。

数据与协议来源、样本数及重现命令见 [实验协议](../../docs/ARCHIVEII_FROZEN_PROTOCOL.md)。每折每条RNA的角色及CT哈希见fold_membership.csv.gz；逐种子结果、4050条逐轮曲线、配对差值、选择锁定及源代码/输入指纹随报告保存。

## 计算成本与运行问题

81条轨迹保留的完整训练段累计 {sum(row['training_seconds'] for row in summaries)/3600:.2f} GPU小时，验证段累计 {sum(row['validation_seconds'] for row in summaries)/3600:.2f} 小时。全部列入审计的准备/训练/测试/验证尝试累计单GPU分配时间为 {allocated/3600:.2f} 小时；包含历史失败尝试，不含登录节点CPU预处理、网络下载与排队。这些是累计资源时间，不能当成用户等待的墙钟时间，也不据不同节点负载比较算法效率。

{preparation_note}官方权重采用分块恢复下载，公开MD5与本地SHA256校验通过后才进入基线评估。{transport_note}

## 图表和产物

![逐家族官方口径F1](figures/family_flexible_f1.png)

![P/Q相对E0的同种子配对差值](figures/paired_flexible_vs_E0.png)

![逐家族exact F1](figures/family_exact_f1.png)

图表由真实family_results.csv/run_results.csv生成，E显示三种子的均值、样本SD和各种子点，官方检查点单独标记。SVG/PDF/PNG及实际QA记录位于figures/；自动检查不替代最终人工审阅。模型和逐RNA预测位于服务器outputs/archiveii_frozen/。
'''
    (REPORT/'README_results.md').write_text(text,encoding='utf8')
    from scripts.summarize_archiveii_frozen import append_analysis
    append_analysis(REPORT)
    write_json(REPORT/'delivered_files.json',{p.relative_to(REPORT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(REPORT.rglob('*')) if p.is_file() and p.name!='delivered_files.json'})
    archive=ROOT/'.tmp/archiveii_frozen_reports_2026-10-22.tar.gz'
    with tarfile.open(archive,'w:gz') as handle:handle.add(REPORT,arcname=REPORT.relative_to(ROOT).as_posix())
    print(json.dumps(dict(report=REPORT.relative_to(ROOT).as_posix(),runs=len(runs),epochs=len(epochs),archive_bytes=archive.stat().st_size)))

if __name__=='__main__':main()
