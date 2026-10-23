"""Describe actual frozen-E learning curves without further test-set selection."""
import csv
import json
from pathlib import Path
import statistics as st

MODES=('E0_no_thermo','E1_bpp','E2_bpp_stack')

def append_analysis(report):
    report=Path(report)
    def read(name):
        with (report/name).open(encoding='utf8',newline='') as stream:return list(csv.DictReader(stream))
    epochs=read('epoch_curves.csv');runs=read('run_results.csv')
    assert len(epochs)==4050 and len(runs)==90
    summaries=json.loads((report/'trajectory_summaries.json').read_text())
    output=[]
    for mode in MODES:
        tracks={}
        for row in epochs:
            if row['mode']==mode:tracks.setdefault((row['family'],row['seed']),[]).append(row)
        assert len(tracks)==27
        for track in tracks.values():
            track.sort(key=lambda r:int(r['epoch']));assert [int(r['epoch']) for r in track]==list(range(1,51))
        for budget in (10,20,30,40,50):
            current=[track[budget-1] for track in tracks.values()]
            result=dict(mode=mode,epoch_budget=budget,n_trajectories=27)
            for key in ('train_bce_online','validation_bce','validation_exact_f1','candidate_accuracy','candidate_balanced_accuracy'):
                result[key+'_mean']=st.mean(float(r[key]) for r in current)
            result['best_validation_exact_f1_within_budget_mean']=st.mean(max(float(r['validation_exact_f1']) for r in track[:budget]) for track in tracks.values())
            result['cumulative_train_and_validation_minutes_mean']=st.mean(sum(float(r['training_seconds'])+float(r['validation_seconds']) for r in track[:budget])/60 for track in tracks.values())
            output.append(result)
    with (report/'training_budget_trends.csv').open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,list(output[0]));writer.writeheader();writer.writerows(output)
    lookup={(r['mode'],r['epoch_budget']):r for r in output}
    families=read('family_results.csv');overall=read('overall_results.csv')
    def family_value(family,mode):return float(next(r for r in families if r['family']==family and r['mode']==mode)['f1_mean'])
    def aggregate(mode,metric='f1_mean',weight='equal_family'):return float(next(r for r in overall if r['mode']==mode and r['weighting']==weight)[metric])
    gains=[100*(aggregate(mode)-aggregate(MODES[0])) for mode in MODES[1:]]
    all_families=sorted({r['family'] for r in families})
    positive=[sum(family_value(f,mode)>family_value(f,MODES[0]) for f in all_families) for mode in MODES[1:]]
    above=[f for f in all_families if all(family_value(f,m)>family_value(f,'RiNALMo_official') for m in MODES[1:])]
    lines=['## 结果解读与训练轮数代价','',
        f'E1、E2 的九家族等权 flexible F1 相对 E0 分别变化 {gains[0]:+.2f}、{gains[1]:+.2f} 个百分点。E1/E2 的家族均值分别在 {positive[0]}/9、{positive[1]}/9 个家族高于 E0；tmRNA 上 E2 相对 E0 变化 {100*(family_value("tmRNA",MODES[2])-family_value("tmRNA",MODES[0])):+.2f} 个百分点。端粒酶家族只有 35 条 RNA，E2 相对 E0 的变化为 {100*(family_value("telomerase",MODES[2])-family_value("telomerase",MODES[0])):+.2f} 个百分点，应结合小家族规模理解总体宏平均。', '',
        f'E2 的等权 flexible F1 为 {aggregate(MODES[2]):.6f}，官方基线为 {aggregate("RiNALMo_official"):.6f}，均值差 {100*(aggregate(MODES[2])-aggregate("RiNALMo_official")):+.2f} 个百分点；E2 三种子 SD 为 {aggregate(MODES[2],"f1_sample_sd"):.6f}，且官方只有一个固定模型，不能据此宣称稳定超越。按 RNA 数量加权时官方为 {aggregate("RiNALMo_official",weight="RNA_count_weighted"):.6f}，E2 为 {aggregate(MODES[2],weight="RNA_count_weighted"):.6f}；exact F1 等权均值分别为 {aggregate("RiNALMo_official","exact_f1_mean"):.6f} 和 {aggregate(MODES[2],"exact_f1_mean"):.6f}。E1/E2 均值均高于官方的家族为：{", ".join(above)}。', '',
        '下表描述同一批 81 条固定 50 轮轨迹的中途状态，每个方法在九家族和三个种子间等权平均。轮末验证 F1 与预算内验证峰值均为原 E 的 exact 验证口径，不是测试分数；没有为这些中间预算追加测试或重新选择最终方案。BCE 为候选配对损失，训练值是在线更新均值，不能直接解释成全矩阵准确率。', '',
        '| 方法 | 轮数 | 训练 BCE | 验证 BCE | 轮末验证 F1 | 预算内验证峰值 | 累计训练+验证分钟/轨迹 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for mode in MODES:
        for budget in (10,30,50):
            r=lookup[mode,budget]
            lines.append(f"| {mode.split('_')[0]} | {budget} | {r['train_bce_online_mean']:.6f} | {r['validation_bce_mean']:.6f} | {r['validation_exact_f1_mean']:.6f} | {r['best_validation_exact_f1_within_budget_mean']:.6f} | {r['cumulative_train_and_validation_minutes_mean']:.2f} |")
    lines += ['', '| 方法 | 最佳验证轮数中位数 | 最佳轮数 >30 的轨迹 /27 | 30→50 平均额外分钟/轨迹 | 30→50 验证峰值均值增量 |', '|---|---:|---:|---:|---:|']
    for mode in MODES:
        group=[r for r in summaries if r['mode']==mode];assert len(group)==27
        a,b=lookup[mode,30],lookup[mode,50]
        lines.append(f"| {mode.split('_')[0]} | {st.median(r['best_epoch'] for r in group):g} | {sum(r['best_epoch']>30 for r in group)}/27 | {b['cumulative_train_and_validation_minutes_mean']-a['cumulative_train_and_validation_minutes_mean']:.2f} | {b['best_validation_exact_f1_within_budget_mean']-a['best_validation_exact_f1_within_budget_mean']:+.6f} |")
    lines += ['', '训练损失继续降低而验证损失上升、轮末 F1 低于历史峰值的组合提示过拟合风险；预算内验证峰值随预算单调不降是选择规则本身的性质，不能当成测试泛化持续改善的证据。累计耗时来自实际逐轮记录，不含排队、下载与预处理；各轨迹节点负载不同，不用于比较算法吞吐。完整 10/20/30/40/50 轮损失、accuracy、balanced accuracy 和时间见 training_budget_trends.csv。','']
    target=report/'README_results.md';text=target.read_text(encoding='utf8').split('## 结果解读与训练轮数代价')[0].rstrip()
    target.write_text(text+'\n\n'+'\n'.join(lines),encoding='utf8')
    return dict(training_budget_rows=len(output),learning_curves=len(epochs))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True)
    print(json.dumps(append_analysis(parser.parse_args().report)))
