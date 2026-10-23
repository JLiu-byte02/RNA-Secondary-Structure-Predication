"""CPU-only independent audit and decision for the single structured-risk revision."""
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.direction_a_protocol import ROOT,OUT,config,file_sha,write_json,digest
from scripts.direction_a_risk import revision_decision,choose_policy
from scripts.direction_a_evaluate import aggregate
from scripts.archiveii_frozen_data import load_manifest
from scripts.report_direction_a import verify_reference,verify_records
from scripts.direction_a_control import resource_status

REPORT=ROOT/'reports/direction_a_revision1_2026-10-23'

def run():
    original_audit=ROOT/'reports/direction_a_pilot_2026-10-23/numeric_source.json'
    original=json.loads(original_audit.read_text());assert original['independently_verified_prediction_rows']>0
    manifest=load_manifest();roles=json.loads((OUT/'pilot_roles.json').read_text());sources={};metrics=[];cases={};verified=0
    reference=verify_reference(ROOT/'third_party/rinalmo_frozen_reference/sec_struct.py')
    for case in config()['pilot_development_families']:
        directory=OUT/'revision1'/case;path=directory/'development_summary.json';row=json.loads(path.read_text());assert row['completed'] and row['case']==case
        assert row['choices_sha256']==file_sha(directory/'choices_sealed.json') and row['predictions_sha256']==file_sha(directory/'development_predictions.json')
        sealed=json.loads((directory/'choices_sealed.json').read_text());seal_hash=sealed.pop('sha256');assert digest(sealed)==seal_hash
        assert sealed['identity']==row['identity'] and row['identity']['roles_sha256']==roles['sha256']
        for name,sha in row['identity']['source']['revision_files'].items():assert file_sha(ROOT/name)==sha
        for name,sha in row['identity']['source']['v1']['files'].items():assert file_sha(ROOT/name)==sha
        assert row['identity']['source']['official_reference_sha256']==reference
        for method,choice in sealed['choices'].items():
            assert choose_policy(sealed['profiles'][method],sealed['budget'])==choice==row['methods'][method]['choice']
        predictions=json.loads((directory/'development_predictions.json').read_text());expected=set(roles['cases'][case]['development'])
        old_path=OUT/'pilot'/case/'correction_seed17/development_summary.json';old=json.loads(old_path.read_text())
        assert original['sources'][case]['summary_sha256']==file_sha(old_path),'Original audit is stale'
        assert row['identity']['v1_choices_sha256']==old['choices_sha256']
        for method,views in predictions.items():
            for view,values in views.items():
                assert len(values)==len(expected) and {r['sample_id'] for r in values}==expected
                for r in values:
                    raw=manifest['records'][r['sample_id']];assert raw['family']==case and len(raw['sequence'])==r['length']
                    assert set(map(tuple,r['reference']))==set(map(tuple,raw['pairs']))
                verify_records(values);verified+=len(values);actual=aggregate(values)
                for level in ('equal_family','rna_macro'):
                    for key,value in actual[level].items():assert abs(value-row['methods'][method][view][level][key])<1e-12
            stats=row['methods'][method]['selected']['equal_family'];choice=row['methods'][method]['choice']
            metrics.append(dict(case=case,method=method,n=row['n'],exact_f1=stats['exact_f1'],flexible_f1=stats['flexible_f1'],harmful_fraction=stats['harmful_rate'],
                mean_harmful_edits=stats['harmful'],mean_helpful_edits=stats['helpful'],beta=choice['beta'],threshold=choice['threshold'],
                common_threshold_exact_f1=row['methods'][method]['common_base_threshold']['equal_family']['exact_f1']))
        get=lambda m:row['methods'][m]['selected']['equal_family']
        old_get=lambda m:old['methods'][m]['selected']['equal_family']
        cases[case]=dict(candidate_f1=get('risk_local_gate')['exact_f1'],risk_baseline_f1=get('risk_fixed_mix')['exact_f1'],
            original_baseline_f1=old_get('fixed_mix')['exact_f1'],candidate_harm=get('risk_local_gate')['harmful_rate'],residual_harm=old_get('residual')['harmful_rate'])
        sources[case]=dict(summary_sha256=file_sha(path),predictions_sha256=row['predictions_sha256'],choices_sha256=row['choices_sha256'])
    revision_config=json.loads((ROOT/'configs/direction_a_revision1.json').read_text());decision=revision_decision(cases,revision_config)
    resources=resource_status(json.loads((OUT/'jobs.json').read_text()));REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'family_metrics.csv').open('w',newline='',encoding='utf8') as h:
        writer=csv.DictWriter(h,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    numeric=dict(sources=sources,metrics=metrics,promotion=decision,resources=resources,official_reference_sha256=reference,
        original_independent_audit_sha256=file_sha(original_audit),independently_verified_prediction_rows=verified,
        verification_source={n:file_sha(ROOT/n) for n in ['scripts/report_direction_a_risk.py','scripts/report_direction_a.py','scripts/direction_a_risk.py']})
    write_json(REPORT/'numeric_source.json',numeric);write_json(OUT/'revision1/promotion.json',numeric)
    lines=['# 方向A：解码后结构风险校准的有界改进','',
        '本轮为查看v1开发结果后的一次明确记录的改进，使用同一开发集，不能当作盲测试。基础模型、神经纠错器权重、数据角色和随机种子均复用；新增模型训练次数为0。','',
        '所有策略都在相同校准集上搜索五档修改强度和原有阈值网格，并用真实解码结构上的有害编辑比例约束选择。约束预算来自旧固定融合与残差纠错器校准风险中较低者。两案例、三策略均锁定后才执行改进版开发评分。','',
        '| 方法 | SRP exact F1 | tmRNA exact F1 | 两族平均 | 平均有害编辑比例 |','|---|---:|---:|---:|---:|']
    for method in ['risk_fixed_mix','risk_global_gate','risk_local_gate']:
        values=[r for r in metrics if r['method']==method];assert len(values)==2
        lines.append(f"| {method} | {values[0]['exact_f1']:.6f} | {values[1]['exact_f1']:.6f} | {sum(r['exact_f1'] for r in values)/2:.6f} | {sum(r['harmful_fraction'] for r in values)/2:.6f} |")
    lines+=['',f"晋级判断：{'达到' if decision['promote'] else '未达到'}预设门槛。相对同规则固定融合的两族平均exact F1差为{decision['mean_exact_gain']:+.6f}；相对原固定融合为{decision['mean_gain_over_original_fixed_mix']:+.6f}。",
        f"官方原函数独立复核{verified}条改进预测记录（含共同阈值视图），完整CT真值与成员关系均核对。",'',
        '单个开发种子、两个家族不足以证明总体泛化。此规则只约束校准集观测到的经验风险，没有任意Family-OOD分布下的安全保证。仅5S在本阶段方法开发中完整保留。',
        '若改进仍不达标，停止自动九折扩展，保留全部正负结果，后续研究应检验针对结构编辑的效用/风险学习和独立外部证据，而不是增加未经证实的全量训练。','']
    (REPORT/'README_results.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps(dict(report=str(REPORT.relative_to(ROOT)),verified=verified,promotion=decision)))

if __name__=='__main__':run()
