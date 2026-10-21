"""Collect all observed epochs and locked budget results without plotting dependencies."""
import csv
import json
import math
import shutil
import statistics
import subprocess
import sys
import tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.long_protocol import MODES,SEEDS,BUDGETS
from scripts.eval_long_epochs import read_rows
from scripts.run_long_epochs import code_identity
from scripts.cache_thermo import file_hash,write_json
OUT=ROOT/'outputs/long_epochs'
REPORT=ROOT/'reports/long_epochs_2026-10-21'

def write_csv(name,rows):
    with (REPORT/name).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,list(rows[0]));writer.writeheader();writer.writerows(rows)

def mean_sd(values):return statistics.mean(values),statistics.stdev(values)

def main():
    validation=json.loads((OUT/'artifact_validation.json').read_text())
    if not validation['passed']:raise RuntimeError('Final budget validation has not passed')
    identity=json.loads((OUT/'experiment_identity.json').read_text())
    if identity['training_source']!=code_identity():raise RuntimeError('Training source changed')
    if identity['config_sha256']!=file_hash(ROOT/'configs/long_epochs.yaml'):raise RuntimeError('Training configuration changed')
    REPORT.mkdir(parents=True,exist_ok=True)
    cluster=json.loads((OUT/'cluster_status.json').read_text())
    if len(cluster['jobs'])!=25 or any(j['status']!='COMPLETED' or j['exit_code']!='0:0' for j in cluster['jobs'].values()):
        raise RuntimeError('Require successful preparation, twelve training and twelve evaluation jobs')
    write_json(REPORT/'slurm_jobs.json',cluster['jobs'])
    accounting=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(j['id'] for j in cluster['jobs'].values()),
        '-o','JobID,JobName,State,ExitCode,Elapsed,NodeList'],text=True)
    (REPORT/'slurm_accounting.txt').write_text(accounting)
    tests=list(csv.DictReader((OUT/'budget_results.csv').open()))
    epochs=[];convergence=[];summaries=[]
    for mode in MODES:
        for seed in SEEDS:
            directory=OUT/mode/f'seed{seed}';rows=read_rows(directory)
            summary=json.loads((directory/'summary.json').read_text());summaries.append(summary)
            cumulative=0.;cumulative_validation=0.;best=-1.
            for row in rows:
                cumulative+=row['training_seconds'];cumulative_validation+=row['validation']['seconds'];best=max(best,row['score'])
                diagnostic=row['validation'];metrics=diagnostic['threshold_metrics'][str(row['threshold'])]
                epochs.append(dict(mode=mode,seed=seed,epoch=row['epoch'],origin=row['origin'],
                    train_bce_online=row['mean_pair_loss'],validation_bce=diagnostic['mean_pair_bce'],
                    validation_f1=row['score'],validation_precision=metrics['precision'],validation_recall=metrics['recall'],
                    best_so_far_f1=best,candidate_accuracy=diagnostic['candidate_accuracy'],
                    candidate_balanced_accuracy=diagnostic['candidate_balanced_accuracy'],
                    training_seconds=row['training_seconds'],validation_seconds=diagnostic['seconds'],
                    cumulative_training_minutes=cumulative/60,cumulative_validation_minutes=cumulative_validation/60))
            peak=max(rows,key=lambda r:r['score']);target=.95*peak['score']
            first95=next(r['epoch'] for r in rows if r['score']>=target)
            first_close=next(r['epoch'] for r in rows if peak['score']-r['score']<=.001)
            convergence.append(dict(mode=mode,seed=seed,parameter_count=summary['parameter_count'],
                best_epoch=peak['epoch'],best_validation_f1=peak['score'],last_validation_f1=rows[-1]['score'],
                peak_to_last_f1_drop=peak['score']-rows[-1]['score'],last_validation_bce=rows[-1]['validation']['mean_pair_bce'],
                minimum_validation_bce=min(r['validation']['mean_pair_bce'] for r in rows),
                train_bce_epoch10=rows[9]['mean_pair_loss'],train_bce_epoch30=rows[29]['mean_pair_loss'],train_bce_epoch50=rows[49]['mean_pair_loss'],
                epoch_to_95_percent_peak=first95,training_minutes_to_95_percent_peak=sum(r['training_seconds'] for r in rows[:first95])/60,
                epoch_within_001_peak=first_close,training_minutes_within_001_peak=sum(r['training_seconds'] for r in rows[:first_close])/60,
                training_minutes_50=cumulative/60,validation_minutes_50=cumulative_validation/60,
                imported_prefix_epochs=summary['imported_prefix_epochs'],new_training_minutes=summary['new_training_seconds']/60,
                new_stage_wall_minutes=summary['new_stage_wall_seconds']/60,peak_cuda_GiB=summary['peak_cuda_bytes']/1024**3))
    write_csv('epoch_curves.csv',epochs);write_csv('convergence_by_seed.csv',convergence)
    aggregate=[]
    for mode in MODES:
        for budget in BUDGETS:
            last=[r for r in epochs if r['mode']==mode and r['epoch']==budget]
            result=dict(mode=mode,budget=budget)
            for field in ('train_bce_online','validation_bce','validation_f1','candidate_accuracy','candidate_balanced_accuracy','cumulative_training_minutes'):
                result[field+'_mean'],result[field+'_sd']=mean_sd([r[field] for r in last])
            for split in ('VL0','TS0','bprna_new'):
                selected=[r for r in tests if r['mode']==mode and int(r['budget'])==budget and r['split']==split]
                if len(selected)!=3:raise ValueError('Incomplete three-seed budget results')
                for field in ('precision','recall','rna_macro_exact_f1'):
                    result[f'{split}_{field}_mean'],result[f'{split}_{field}_sd']=mean_sd([float(r[field]) for r in selected])
                if split=='VL0':result['selected_epochs']=','.join(str(r['selected_epoch']) for r in selected)
            aggregate.append(result)
    write_csv('budget_aggregate.csv',aggregate)
    paired=[]
    for budget in BUDGETS:
        for split in ('VL0','TS0','bprna_new'):
            for seed in SEEDS:
                scores={r['mode']:float(r['rna_macro_exact_f1']) for r in tests if int(r['budget'])==budget and r['split']==split and int(r['seed'])==seed}
                paired.append(dict(budget=budget,split=split,seed=seed,E0_minus_B0=scores['E0_no_thermo']-scores['B0'],
                    E1_minus_B0=scores['E1_bpp']-scores['B0'],E1_minus_E0=scores['E1_bpp']-scores['E0_no_thermo'],
                    E2_minus_E1=scores['E2_bpp_stack']-scores['E1_bpp']))
    write_csv('paired_deltas.csv',paired)
    for name in ('artifact_validation.json','selection_lock.json','experiment_identity.json','preparation_complete.json','budget_results.csv'):
        shutil.copy2(OUT/name,REPORT/name)
    write_json(REPORT/'trajectory_summaries.json',summaries)
    shutil.copy2(ROOT/'configs/long_epochs.yaml',REPORT/'long_epochs.yaml')
    full_previous=ROOT/'reports/full_tr0_2026-10-21/aggregate_results.csv'
    if full_previous.exists():shutil.copy2(full_previous,REPORT/'historical_10_epoch_with_early_stopping.csv')
    final=[r for r in aggregate if r['budget']==50]
    fmt=lambda r,name:f"{r[name+'_mean']:.6f} ± {r[name+'_sd']:.6f}"
    final_table='\n'.join(f"| {r['mode']} | {fmt(r,'VL0_rna_macro_exact_f1')} | {fmt(r,'TS0_rna_macro_exact_f1')} | {fmt(r,'bprna_new_rna_macro_exact_f1')} |" for r in final)
    budget_table='\n'.join(f"| {r['mode']} | {r['budget']} | {r['selected_epochs']} | {fmt(r,'VL0_rna_macro_exact_f1')} | {fmt(r,'TS0_rna_macro_exact_f1')} | {fmt(r,'bprna_new_rna_macro_exact_f1')} |" for r in aggregate)
    cost_table='\n'.join(f"| {r['mode']} | {r['budget']} | {fmt(r,'train_bce_online')} | {fmt(r,'validation_bce')} | {fmt(r,'candidate_balanced_accuracy')} | {r['cumulative_training_minutes_mean']:.1f} ± {r['cumulative_training_minutes_sd']:.1f} |" for r in aggregate)
    interpretation=[]
    for mode in MODES:
        groups={r['budget']:r for r in aggregate if r['mode']==mode}
        c=[r for r in convergence if r['mode']==mode]
        interpretation.append(f"- {mode}：10→30轮，VL0 最后轮 F1 变化 {groups[30]['validation_f1_mean']-groups[10]['validation_f1_mean']:+.4f}；30→50轮变化 {groups[50]['validation_f1_mean']-groups[30]['validation_f1_mean']:+.4f}。累计训练时间从 {groups[10]['cumulative_training_minutes_mean']:.1f} 分钟增加到 {groups[30]['cumulative_training_minutes_mean']:.1f} / {groups[50]['cumulative_training_minutes_mean']:.1f} 分钟。50轮平均峰值回落 {statistics.mean(r['peak_to_last_f1_drop'] for r in c):.4f}；最佳 epoch 分别为 {', '.join(str(r['best_epoch']) for r in c)}。")
        interpretation.append(f"  在预算内选择的模型上，30→50轮的 TS0 / bpRNA-new 平均 F1 变化分别为 {groups[50]['TS0_rna_macro_exact_f1_mean']-groups[30]['TS0_rna_macro_exact_f1_mean']:+.4f} / {groups[50]['bprna_new_rna_macro_exact_f1_mean']-groups[30]['bprna_new_rna_macro_exact_f1_mean']:+.4f}，额外训练 {groups[50]['cumulative_training_minutes_mean']-groups[30]['cumulative_training_minutes_mean']:.1f} 分钟（每种子均值）。这些是预定预算的描述性比较，不据测试分数挑选预算。")
    text=f'''# B0 / E 系列 50 轮训练研究

12条轨迹全部完成50轮。训练为完整 TR0 10,682条，VL0 1,286条用于验证和选择，TS0 1,288条及 bpRNA-new 5,388条用于固定预算测试；种子17、23、41。每个预算内先选各自 VL0 最佳 epoch，再选择方法共享阈值，全部60个选择统一锁定后才提交测试。测试结果不用于决定预算。

## 50轮预算下的最终效果

下表为预算内所选模型的完整真值 RNA-macro exact F1，均值 ± 三种子样本 SD。

| 方法 | VL0（选择集） | TS0 | bpRNA-new |
|---|---:|---:|---:|
{final_table}

## 增加轮数的收益与代价

以下“最后轮 F1”使用该轮的 VL0 阈值网格最大 F1；与后文预算内选出的最佳模型分开报告。

{chr(10).join(interpretation)}

训练损失下降但 VL0 BCE 上升、VL0 F1 从峰值回落是后期过拟合的观察指标。峰值回落不等于显著性结论；应结合全部三个种子曲线和 validation_bce 判断。最佳值曲线的单调性来自选择规则，不能据此声称轮末模型持续改善。在线训练 BCE 使用每步更新中的参数，VL0 BCE 使用轮末参数，二者不能直接相减为精确泛化间隙。

收敛速度见 convergence_by_seed.csv：预先规定达到最大 VL0 F1 的95%，以及距最大值不超过0.001的首次 epoch 与累计训练时间。所有逐轮数据保留在 epoch_curves.csv，包含 accuracy / balanced accuracy 与时长；候选 accuracy 易被大量负配对抬高，不能代替结构 F1。

## B0 的过程与公平性

B0 是原始175,553参数 PairHead，不含纠错 CNN，不读取 P/Q/距离作为模型证据；E系列为186,306参数。同一随机种子的四个 PairHead 初始化逐张量相同，B0和E使用相同训练集合、候选损失、优化器、BF16、梯度累积、解码、完整真值、验证与测试规则。B0从头训练50轮，E从已验证的最后 epoch 与 AdamW 状态继续至50轮；继承前缀的每轮VL0曲线重新推理并精确重现，旧输出保持原样。E2/seed23的旧轨迹在9轮早停，本次从第10轮继续。

全部方法关闭提前停止，以完整观测50轮轨迹；学习率固定1e-4，没有新增调度器。历史10轮上限+patience3的表格单独保存在 historical_10_epoch_with_early_stopping.csv，不能与本研究的固定10轮前缀视为完全相同停止策略，也不混入此前家族留出 B0 的开发分数。

时间为实际测得的训练段时间，包含继承轨迹的已发生训练成本；本阶段新增成本、诊断成本和实际作业时间另列 trajectory_summaries.json。各作业独占一张 Slurm GPU，集群排队、节点负载和前缀复用会影响墙钟时间，不把排队时间或不同节点负载解释为方法速度差异。种子SD不等于置信区间。

## 固定预算性能比较

| 方法 | 预算轮数 | 所选epoch（17/23/41） | VL0 | TS0 | bpRNA-new |
|---|---:|---|---:|---:|---:|
{budget_table}

## 轮末损失和训练成本

| 方法 | 轮数 | 在线训练BCE | VL0 BCE | 候选balanced accuracy | 累计训练分钟 |
|---|---:|---:|---:|---:|---:|
{cost_table}

## 产物与复核

固定预算共 {validation['budget_cohort_evaluations']} 个方法/种子/预算/集合组合，对应 {validation['conceptual_predictions']:,} 条逻辑预测。相同checkpoint/阈值/集合的重复组合复用完全相同、校验SHA256的预测文件；独立复算 {validation['unique_prediction_files_independently_recomputed']} 份不同预测文件、{validation['unique_predictions_independently_recomputed']:,} 条独立记录。完整选择锁定、源代码指纹、逐种子结果、配对差值和检查记录随报告保存。

服务器模型及逐RNA预测在 outputs/long_epochs/，表征复用 embeddings/full/。图表、脚本及源数据与报告统一管理。历史测试集曾出现在项目方法决策材料中，本研究保持预先固定的预算与选择规则，不声称整个项目历史从未查看这些测试集。
'''
    (REPORT/'README_results.md').write_text(text,encoding='utf8')
    write_json(REPORT/'delivered_files.json',{p.name:dict(bytes=p.stat().st_size,sha256=file_hash(p)) for p in REPORT.iterdir() if p.is_file() and p.name!='delivered_files.json'})
    archive=ROOT/'.tmp/long_epoch_reports_2026-10-21.tar.gz'
    with tarfile.open(archive,'w:gz') as handle:handle.add(REPORT,arcname=str(REPORT.relative_to(ROOT)))
    print(json.dumps(dict(report=str(REPORT),archive=str(archive),bytes=archive.stat().st_size,epochs=len(epochs),budget_rows=len(aggregate))))

if __name__=='__main__':main()
