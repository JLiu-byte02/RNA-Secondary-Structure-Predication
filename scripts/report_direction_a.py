"""Build an independently rechecked pilot report, never synthetic result rows."""
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.direction_a_protocol import ROOT,OUT,config,file_sha,write_json,digest
from scripts.direction_a_evaluate import official_verify,aggregate,metric_row
from scripts.direction_a_control import promotion,resource_status
from scripts.archiveii_frozen_data import load_manifest

METHODS=('base','p_only','mfe','mea','fixed_mix','residual','global_gate','local_gate','safe_local_gate')
REPORT=ROOT/'reports/direction_a_pilot_2026-10-23'
OFFICIAL_REFERENCE_SHA256='75c971655e9a12cbbe18d3f75b8377303d0acc1795fab0f0253721ec14b750d0'

def verify_reference(path):
    actual=file_sha(path);assert actual==OFFICIAL_REFERENCE_SHA256,'Official verifier source changed'
    return actual

def verify_records(rows,workers=4):
    """Bounded CPU audit; each original metric function remains unchanged."""
    chunks=[rows[i:i+32] for i in range(0,len(rows),32)]
    if not chunks:return dict(passed=True,n=0)
    with ProcessPoolExecutor(max_workers=min(workers,len(chunks))) as pool:
        results=list(pool.map(official_verify,chunks))
    assert all(r['passed'] for r in results) and sum(r['n'] for r in results)==len(rows)
    return dict(passed=True,n=len(rows))

def write_csv(path,rows):
    assert rows;fields=list(rows[0])
    with path.open('w',newline='',encoding='utf8') as h:
        writer=csv.DictWriter(h,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def build():
    cfg=config();summaries={};sources={};metric_rows=[];diagnostics=[];curves=[];all_rows=0
    manifest=load_manifest();roles=json.loads((OUT/'pilot_roles.json').read_text())
    official_path='third_party/rinalmo_frozen_reference/sec_struct.py'
    # The literal digest was independently extracted from Git917dbaf on the login node.
    # CPU compute nodes need no Git installation or network access for this verification.
    official_sha=verify_reference(ROOT/official_path)
    for case in cfg['pilot_development_families']:
        directory=OUT/'pilot'/case/'correction_seed17';p=directory/'development_summary.json';summary=json.loads(p.read_text())
        assert summary['completed'] and not summary['smoke'] and summary['case']==case
        assert file_sha(directory/'development_predictions.json')==summary['predictions_sha256']
        assert file_sha(directory/'choices_sealed.json')==summary['choices_sha256']
        sealed=json.loads((directory/'choices_sealed.json').read_text())
        predictions=json.loads((directory/'development_predictions.json').read_text());expected=set(roles['cases'][case]['development'])
        assert set(predictions)==set(METHODS)
        for method in METHODS:
            rows=predictions[method]['selected'];assert len(rows)==len(expected) and {r['sample_id'] for r in rows}==expected
            verify_records(rows);all_rows+=len(rows)
            if 'common_base_threshold' in predictions[method]:
                common=predictions[method]['common_base_threshold'];assert len(common)==len(expected) and {r['sample_id'] for r in common}==expected
                verify_records(common);all_rows+=len(common)
                actual_common=aggregate(common)
                for level in ('equal_family','rna_macro'):
                    for key,value in actual_common[level].items():assert abs(value-summary['methods'][method]['common_base_threshold'][level][key])<1e-12
            recalculated=[]
            base_rows={r['sample_id']:r for r in predictions['base']['selected']}
            for row in rows:
                sid=row['sample_id'];raw=manifest['records'][sid]
                assert raw['family']==case and row['family']==case and len(raw['sequence'])==row['length']
                assert set(map(tuple,row['reference']))==set(map(tuple,raw['pairs']))
                base=set(map(tuple,base_rows[sid]['prediction']));pred=set(map(tuple,row['prediction']));truth=set(map(tuple,row['reference']))
                assert len(pred)==len({i for a,b in pred for i in (a,b)})/2
                updated=metric_row(sid,dict(family=case,sequence=raw['sequence'],reference=truth),pred,base)
                for key in ('exact_f1','flexible_f1','helpful','harmful','harmful_rate'):assert abs(updated[key]-row[key])<1e-12
                recalculated.append(updated)
                added=pred-base;removed=base-pred
                diagnostics.append(dict(case=case,method=method,sample_id=sid,sequence_group=hashlib.sha256(raw['sequence'].encode()).hexdigest(),
                    exact_f1=row['exact_f1'],delta_exact_f1=row['exact_f1']-base_rows[sid]['exact_f1'],
                    correct_added=len(added&truth),incorrect_added=len(added-truth),correct_removed=len(removed&truth),incorrect_removed=len(removed-truth),
                    gate_mean=row.get('gate_mean',''),gate_std=row.get('gate_std','')))
            actual=aggregate(recalculated);selected=summary['methods'][method]['selected']
            for level in ('equal_family','rna_macro'):
                for key,value in actual[level].items():assert abs(value-selected[level][key])<1e-12
            values=selected['equal_family'];choice=summary['methods'][method]['choice']
            metric_rows.append(dict(case=case,method=method,n=len(rows),exact_f1=values['exact_f1'],flexible_f1=values['flexible_f1'],
                 precision=values['exact_precision'],recall=values['exact_recall'],harmful_edit_rate=values['harmful_rate'],
                 helpful_edits_per_rna=values['helpful'],harmful_edits_per_rna=values['harmful'],
                 threshold=choice.get('threshold',''),alpha=choice.get('alpha',''),
                 selected_epoch=sealed['learned'].get(method,{}).get('selected_epoch',''),
                 calibration_exact_f1=choice.get('equal_family_exact_f1',''),
                 common_threshold_exact_f1=summary['methods'][method].get('common_base_threshold',selected)['equal_family']['exact_f1']))
        for method in ('residual','global_gate','local_gate','safe_local_gate'):
            for line in (directory/method/'epochs.jsonl').read_text().splitlines():
                row=json.loads(line);curves.append(dict(case=case,method=method,epoch=row['epoch'],fit_loss=row['mean_loss'],fit_bce=row['mean_bce'],
                    calibration_bce=row['calibration_bce'],calibration_exact_f1=row['calibration']['equal_family_exact_f1'],
                    calibration_gate_mean=row['calibration_gate_mean'],training_seconds=row['training_seconds']))
        summaries[case]=summary;sources[case]=dict(summary_sha256=file_sha(p),predictions_sha256=summary['predictions_sha256'],choices_sha256=summary['choices_sha256'])
    decision=promotion(summaries,cfg);resources=resource_status(json.loads((OUT/'jobs.json').read_text()))
    REPORT.mkdir(parents=True,exist_ok=True)
    write_csv(REPORT/'family_metrics.csv',metric_rows);write_csv(REPORT/'paired_rna_diagnostics.csv',diagnostics);write_csv(REPORT/'learning_curves.csv',curves)
    write_json(REPORT/'numeric_source.json',dict(sources=sources,metrics=metric_rows,promotion=decision,resources=resources,
        official_reference_sha256=official_sha,independently_verified_prediction_rows=all_rows,
        role_manifest_sha256=file_sha(OUT/'pilot_roles.json'),config_sha256=file_sha(ROOT/'configs/direction_a.json'),
        statement='Pilot development, one seed17, two predeclared pseudo-OOD families; not nine-fold confirmation'))
    text=['# 方向 A：局部热力学纠错开发实验','',
        '本报告为一个种子（17）、两个预先指定的伪 OOD 开发家族的筛选结果，不能当作九折确认或论文最终结论。外层5S家族未参与本轮拟合和评分。',
        '', '每条 RNA 先计算完整 CT 真值上的配对指标，再取家族内平均。表中没有多种子误差条；flex F1允许官方1nt偏移。MFE/MEA为非交叉物理参考，MEA gamma固定为1；其余方法使用相同候选约束和贪心解码器。','',
        '| 方法 | SRP exact F1 | tmRNA exact F1 | 两族等权 exact F1 | 两族等权 flex F1 | 有害编辑比例 |','|---|---:|---:|---:|---:|---:|']
    for method in METHODS:
        rows=[r for r in metric_rows if r['method']==method];assert len(rows)==2
        text.append('| '+method+' | '+' | '.join(f'{v:.6f}' for v in [rows[0]['exact_f1'],rows[1]['exact_f1'],np.mean([r['exact_f1'] for r in rows]),np.mean([r['flexible_f1'] for r in rows]),np.mean([r['harmful_edit_rate'] for r in rows])])+' |')
    text+=['','## 预设晋级判断','',f"判定：{'达到' if decision['promote'] else '未达到'}扩展九折实验的门槛；候选：{decision['winner'] or '无'}。",
        '门槛为相对校准集选择的固定融合：两开发家族平均 exact F1增益至少0.005、任一家族退步不超过0.01、有害编辑比例不高于残差纠错器。这是计算预算决策，不是显著性检验。',
        '',f"独立官方函数复核 {all_rows} 条预测，原始 CT 真值和每条 RNA 的成员关系均核对通过。",
        '', '## 成本与限制','',
        f"本报告生成时，pilot已分配GPU时间为{resources['phases']['pilot']['charged_seconds']/3600:.3f}小时，仍在运行作业的剩余预留为{resources['phases']['pilot']['reserved_remaining_seconds']/3600:.3f}小时；上限8小时。smoke另计{resources['phases']['smoke']['charged_seconds']/3600:.3f}小时。所有失败作业保留在资源记录中。",
        '已有通用表征和P缓存被重复利用，此处增量成本不包含历史编码器预训练、权重下载及特征生成。OOF基础模型与完整基础模型的训练家族数量不同，可能产生分数分布偏移。元校准标签可出现在预测另一家族块的基础模型拟合中，不构成独立端到端风险校准。',
        '本轮开发整体使用了除5S之外八个家族的标签。今后全九折可作为方法开发后的统一基准对照，不能称为九个独立盲确认；只有5S仍完整保留于本阶段方法开发之外。更强论文论断需新外部数据或外层家族专属的嵌套选择。',
        '作业9345因完整基础模型9341尚未结束而提前启动，在读取缺失summary时失败；该次消耗计入预算，后续重试必须在依赖完成后执行。',
        '若本轮未达晋级门槛，停止自动全量扩展，先根据固定融合、局部门控空间变化和有害编辑统计定位原因；最多在原预算内尝试一次明确记录的改动。即使晋级，仍需九折多种子、外部数据和更强头/第二编码器证据后才能评估投稿准备度。',
        '', '## 可复核文件','',
        '- family_metrics.csv：所有方法和两家族指标、选择阈值、固定混合系数、共同阈值敏感性。',
        '- paired_rna_diagnostics.csv：逐RNA配对差值、四类编辑及门控均值/标准差。',
        '- learning_curves.csv：全部纠错训练轮次的拟合损失、校准损失/F1及门控趋势。',
        '- numeric_source.json：来源哈希、独立复核数量、资源记录和晋级判定。','']
    (REPORT/'README_results.md').write_text('\n'.join(text),encoding='utf8')
    print(json.dumps(dict(report=str(REPORT.relative_to(ROOT)),verified_prediction_rows=all_rows,promotion=decision)))

if __name__=='__main__':build()
