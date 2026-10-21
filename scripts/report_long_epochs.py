"""Collect all observed epochs and locked budget results without plotting dependencies."""
import csv
import hashlib
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
    previous_jobs=cluster.get('previous_jobs',[])
    write_json(REPORT/'previous_slurm_jobs.json',previous_jobs)
    all_ids=sorted({j['id'] for j in list(cluster['jobs'].values())+previous_jobs},key=int)
    all_accounting=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(all_ids),
        '-o','JobID,JobName,State,ExitCode,ElapsedRaw,Start,End,NodeList'],text=True)
    (REPORT/'slurm_all_attempts.txt').write_text(all_accounting)
    allocation_seconds=sum(int(line.split('|')[4]) for line in all_accounting.splitlines() if line.strip())
    migration_note='本次没有训练资源迁移。'
    migration_path=OUT/'resource_migration.json'
    if migration_path.exists():
        raw=json.loads(migration_path.read_text());proof=raw.get('resume_verification',{})
        if not proof.get('passed') or raw['numerical_runtime_changed']:raise RuntimeError('Migration continuation has not been verified')
        checkpoint=raw['checkpoint']
        rows=read_rows(OUT/raw['mode']/f"seed{raw['seed']}")
        prefix=rows[:checkpoint['epoch']]
        prefix_sha=hashlib.sha256(json.dumps(prefix,sort_keys=True).encode()).hexdigest()
        if prefix_sha!=checkpoint['retained_prefix_sha256'] or checkpoint['code_hash']!=identity['training_source']['sha256']:
            raise RuntimeError('Migration changed the retained trajectory or runtime')
        write_json(REPORT/'migration_retained_prefix.json',prefix)
        # Raw operator evidence remains on the server; published audit avoids private working paths.
        keys=('status','mode','seed','original_job_id','replacement_job_id','node','original_gpu_index','replacement_gpu_index',
            'reason','unrelated_process_memory_observed_MiB','unrelated_process_signalled','submitted_unix','switch_started_unix',
            'original_cancel_requested_unix','checkpoint','original_job_accounting_after_cancel','numerical_runtime_changed',
            'cost_note','resume_verification')
        migration={k:raw[k] for k in keys if k in raw}
        write_json(REPORT/'resource_migration.json',migration)
        migration_note=(f"E2/seed41 原作业 {raw['original_job_id']} 的分配 GPU 上出现另一个不属于该作业的计算进程，"
            f"占用约 {raw['unrelated_process_memory_observed_MiB']/1024:.1f} GiB 显存。通过 Slurm 先申请空闲 GPU 后，"
            f"主动取消原分配，以作业 {raw['replacement_job_id']} 从第 {checkpoint['epoch']+1} 轮继续；"
            f"保留前 {checkpoint['epoch']} 轮的逐项记录、参数、AdamW 和随机数状态。"
            f"AdamW 步数从 {checkpoint['optimizer_step_counts'][0]:,} 连续增加到 {proof['first_resumed_optimizer_step_counts'][0]:,}。"
            f"数值代码、输入和超参数均未改变，原有前缀的完整内容哈希一致。迁移前 {proof['pre_migration_complete_new_epochs']} 个完整新轮"
            f"平均 {proof['pre_migration_mean_epoch_seconds']:.1f} 秒，迁移后最初 {len(proof['post_migration_probe_epochs'])} 个完整轮"
            f"平均 {proof['post_migration_mean_epoch_seconds']:.1f} 秒；这是消除资源竞争的观察，不代表算法本身加速。")
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
    convergence_table='\n'.join(
        f"| {mode} | {', '.join(str(r['best_epoch']) for r in convergence if r['mode']==mode)} | "
        f"{statistics.mean(r['epoch_to_95_percent_peak'] for r in convergence if r['mode']==mode):.1f} | "
        f"{statistics.mean(r['epoch_within_001_peak'] for r in convergence if r['mode']==mode):.1f} | "
        f"{statistics.mean(r['training_minutes_50'] for r in convergence if r['mode']==mode):.1f} ± "
        f"{statistics.stdev(r['training_minutes_50'] for r in convergence if r['mode']==mode):.1f} | "
        f"{statistics.mean(r['new_training_minutes'] for r in convergence if r['mode']==mode):.1f} |"
        for mode in MODES)
    comparisons=[]
    for mode,field in (('E0_no_thermo','E0_minus_B0'),('E1_bpp','E1_minus_B0'),('E2_bpp_stack',None)):
        for split in ('TS0','bprna_new'):
            differences=[r[field] if field else r['E1_minus_B0']+r['E2_minus_E1']
                for r in paired if r['budget']==50 and r['split']==split]
            comparisons.append(f"| {mode} − B0 | {split} | {statistics.mean(differences):+.6f} ± {statistics.stdev(differences):.6f} | "
                f"{sum(d>0 for d in differences)}/3 |")
    comparison_table='\n'.join(comparisons)
    interpretation=[]
    for mode in MODES:
        groups={r['budget']:r for r in aggregate if r['mode']==mode}
        c=[r for r in convergence if r['mode']==mode]
        interpretation.append(f"- {mode}：10→30轮，VL0 最后轮 F1 变化 {groups[30]['validation_f1_mean']-groups[10]['validation_f1_mean']:+.4f}；30→50轮变化 {groups[50]['validation_f1_mean']-groups[30]['validation_f1_mean']:+.4f}。累计训练时间从 {groups[10]['cumulative_training_minutes_mean']:.1f} 分钟增加到 {groups[30]['cumulative_training_minutes_mean']:.1f} / {groups[50]['cumulative_training_minutes_mean']:.1f} 分钟。50轮平均峰值回落 {statistics.mean(r['peak_to_last_f1_drop'] for r in c):.4f}；最佳 epoch 分别为 {', '.join(str(r['best_epoch']) for r in c)}。")
        interpretation.append(f"  在预算内选择的模型上，30→50轮的 TS0 / bpRNA-new 平均 F1 变化分别为 {groups[50]['TS0_rna_macro_exact_f1_mean']-groups[30]['TS0_rna_macro_exact_f1_mean']:+.4f} / {groups[50]['bprna_new_rna_macro_exact_f1_mean']-groups[30]['bprna_new_rna_macro_exact_f1_mean']:+.4f}，额外训练 {groups[50]['cumulative_training_minutes_mean']-groups[30]['cumulative_training_minutes_mean']:.1f} 分钟（每种子均值）。这些是预定预算的描述性比较，不据测试分数挑选预算。")
        interpretation.append(f"  在线训练 BCE（10/30/50轮）为 {groups[10]['train_bce_online_mean']:.5f} / {groups[30]['train_bce_online_mean']:.5f} / {groups[50]['train_bce_online_mean']:.5f}；同期 VL0 BCE 为 {groups[10]['validation_bce_mean']:.5f} / {groups[30]['validation_bce_mean']:.5f} / {groups[50]['validation_bce_mean']:.5f}。结合逐种子曲线判断后期变化，不能仅凭训练损失给出泛化结论。")
        joint=0
        for seed in SEEDS:
            track=sorted([r for r in epochs if r['mode']==mode and r['seed']==seed],key=lambda r:r['epoch'])
            joint+=int(track[-1]['train_bce_online']<track[9]['train_bce_online']
                and track[-1]['validation_bce']>track[9]['validation_bce']
                and max(r['validation_f1'] for r in track)-track[-1]['validation_f1']>.001)
        interpretation.append(f"  {joint}/3 个种子同时出现：50轮训练 BCE 低于10轮、验证 BCE 高于10轮、轮末 F1 比本轨迹峰值低超过0.001。"+
            ("这为后期过拟合提供联合观察证据，仍是描述性判断。" if joint else "未观察到上述三个信号同时出现，不能仅由训练轮数推断过拟合。")+
            f" 10→50轮累计训练段时间增加到 {groups[50]['cumulative_training_minutes_mean']/groups[10]['cumulative_training_minutes_mean']:.2f} 倍，预算内所选 VL0 平均 F1 变化 {groups[50]['VL0_rna_macro_exact_f1_mean']-groups[10]['VL0_rna_macro_exact_f1_mean']:+.4f}。")
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

收敛速度见 convergence_by_seed.csv：预先规定达到最大 VL0 F1 的95%，以及距最大值不超过0.001的首次 epoch 与累计训练时间。这些指标需要完整轨迹的峰值，是事后收敛描述，不是可直接在线执行的停止规则。所有逐轮数据保留在 epoch_curves.csv，包含候选 micro accuracy / balanced accuracy 与时长；候选 accuracy 易被大量负配对抬高，不能代替结构 F1。

| 方法 | 最佳epoch（17/23/41） | 首次达到峰值95%的平均epoch | 首次距峰值≤0.001的平均epoch | 50轮训练分钟/种子 | 本阶段新增训练分钟/种子 |
|---|---|---:|---:|---:|---:|
{convergence_table}

12条完整轨迹累计训练段成本 {sum(r['training_minutes_50'] for r in convergence)/60:.2f} GPU小时；其中本阶段新增训练段 {sum(r['new_training_minutes'] for r in convergence)/60:.2f} GPU小时。各训练作业申请一张GPU，上述数值为完整训练轮段的分配时间，验证、前缀诊断、最终测试及排队另计；并行作业的累计GPU小时不能当成用户等待的墙钟时间。

## B0 的过程与公平性

B0 是原始175,553参数 PairHead，保留原有卷积残差块，不含额外的 CorrectionCNN，不读取 P/Q/距离作为模型证据；E系列为186,306参数。同一随机种子的四个 PairHead 初始化逐张量相同，B0和E使用相同训练集合、候选损失、优化器、BF16、梯度累积、解码、完整真值、验证与测试规则。B0从头训练50轮，E从已验证的最后 epoch 与 AdamW 状态继续至50轮；继承前缀的每轮VL0曲线重新推理并精确重现，旧输出保持原样。E2/seed23的旧轨迹在9轮早停，本次从第10轮继续。

全部方法关闭提前停止，以完整观测50轮轨迹；学习率固定1e-4，没有新增调度器。历史10轮上限+patience3的表格单独保存在 historical_10_epoch_with_early_stopping.csv，不能与本研究的固定10轮前缀视为完全相同停止策略，也不混入此前家族留出 B0 的开发分数。

以下按相同种子计算50轮预算所选模型的 F1 差值，正数表示 E 优于 B0。配对差值的 SD 是三个种子差值的样本 SD，不从两组边际 SD 推算；三个种子不构成统计显著性的证据。

| 比较 | 集合 | 配对F1差值均值 ± SD | 正差值种子数 |
|---|---|---:|---:|
{comparison_table}

时间为实际测得的完整轮训练段时间，包含继承轨迹的已发生训练成本；本阶段新增成本和诊断成本另列 trajectory_summaries.json。若从检查点重启，summary 的 new_stage_wall_seconds 只描述最后一次进程运行。各作业采用单GPU的 Slurm 分配；集群排队、节点负载、实际GPU竞争和前缀复用会影响墙钟时间，不把这些差异解释为方法计算效率的差异。种子SD不等于置信区间。

## 资源竞争与检查点续训记录

{migration_note}

25个最终准备/训练/测试作业均为 COMPLETED、ExitCode 0:0；含被替代尝试共提交 {len(all_ids)} 个不同作业ID。previous_slurm_jobs.json 与 slurm_all_attempts.txt 保留主动取消及所有最终作业的状态、开始/结束时间和分配秒数。由于所有这些作业均申请一张GPU，本阶段所有尝试累计 GPU 分配时长为 {allocation_seconds/3600:.2f} 小时（含准备、验证、测试、重新加载及取消前的部分轮；不含排队，也不含此前10轮阶段）。

逐轮时间图仅统计完成且保留的 epoch 训练段，未完成轮的部分工作、迁移时短暂重叠的资源预留和重新加载没有混入这些曲线，均包含在全部尝试的 Slurm 分配时长中。RiNALMo 表征预处理是此前完成的共同步骤，不包含在本阶段的训练轮数成本比较中。

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

## 图表与源数据

图表由 `python scripts/plot_long_epochs.py` 在上述真实结果验证后生成。细线为各种子，粗线为均值，阴影或误差线为三种子样本SD；不平滑、不剔除数据。可编辑SVG/PDF、300dpi PNG、数据哈希和实际字体/碰撞检查见 [figures/figure_manifest.json](figures/figure_manifest.json)。每张单图使用自己的清晰轴范围，跨图数值比较须读取刻度。

![轮末验证F1](figures/validation_f1_by_epoch.png)

![在线训练损失](figures/training_loss_by_epoch.png)

![轮末验证损失](figures/validation_loss_by_epoch.png)

![TS0预算比较](figures/TS0_f1_vs_budget.png)

![bpRNA-new预算比较](figures/bprna_new_f1_vs_budget.png)

![B0配对比较：TS0](figures/TS0_paired_vs_B0.png)

![B0配对比较：bpRNA-new](figures/bprna_new_paired_vs_B0.png)

其余两张图为 [验证选择包络](figures/validation_best_so_far.png) 和 [验证收益与训练耗时](figures/validation_vs_time.png)。选择包络单调来自规则，不能视为轮末性能持续改善。
'''
    (REPORT/'README_results.md').write_text(text,encoding='utf8')
    write_json(REPORT/'delivered_files.json',{p.relative_to(REPORT).as_posix():dict(bytes=p.stat().st_size,sha256=file_hash(p)) for p in sorted(REPORT.rglob('*')) if p.is_file() and p.name!='delivered_files.json'})
    archive=ROOT/'.tmp/long_epoch_reports_2026-10-21.tar.gz'
    with tarfile.open(archive,'w:gz') as handle:handle.add(REPORT,arcname=str(REPORT.relative_to(ROOT)))
    print(json.dumps(dict(report=str(REPORT),archive=str(archive),bytes=archive.stat().st_size,epochs=len(epochs),budget_rows=len(aggregate))))

if __name__=='__main__':main()
