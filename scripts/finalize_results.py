"""Validate real run artifacts and render Chinese report."""
import csv,hashlib,json,statistics,sys
from collections import defaultdict
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.metrics import pair_prf,family_macro
from src.decoder import candidate_pairs
from src.utils import pairs_from_str,pair_set_0b
import torch
O=R/'outputs/thermo_fast'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main():
 assert Path.cwd().resolve()==R
 audit=read(O/'split_audit.json');cache=read(O/'thermo_cache_manifest.json');decision=read(O/'decision.json')
 assert audit['passed'] and read(O/'provenance/preflight.json')['passed']
 manifest={x['sample_id']:x for x in csv.DictReader((R/'data/processed/manifest.tsv').open(),delimiter='\t')}
 split=read(R/'data/processed/split_ids.json');rows=list(csv.DictReader((O/'development_summary.csv').open()))
 checks=[];groups=defaultdict(list);families=defaultdict(list);pairtot=defaultdict(lambda:defaultdict(int))
 for row in rows:
  mode,seed=row['mode'],int(row['seed']);d=O/mode/f'seed{seed}';summary=read(d/'summary.json')
  assert summary['status']=='completed' and summary['train_count']==3299
  assert summary['dev_count']==1068 and summary['VL0_count']==1286 and summary['gradients_verified']
  assert summary['encoder_frozen'] and summary['supervised_initialization']=='none_identical_random'
  logs=[json.loads(x) for x in (d/'train_log.jsonl').read_text().splitlines() if x.strip()]
  assert [x['epoch'] for x in logs]==list(range(1,summary['epochs_completed']+1))
  selected=max(logs,key=lambda x:x['metrics']['family_macro_exact_f1'])
  assert selected['epoch']==summary['best_epoch'] and selected['threshold']==summary['threshold']
  best=torch.load(d/'best.pt',map_location='cpu',weights_only=False)
  epoch=torch.load(d/f"epoch{summary['best_epoch']}.pt",map_location='cpu',weights_only=False)
  assert best['epoch']==summary['best_epoch'] and all(torch.equal(v,epoch['model'][k]) for k,v in best['model'].items())
  del best,epoch
  eligible_scores=[]
  for name,key in [('family-dev','family_dev'),('VL0','VL0')]:
   pred=[json.loads(x) for x in (d/f'{name}_predictions.jsonl').read_text().splitlines() if x.strip()]
   assert [p['rna_id'] for p in pred]==cache['shared_successful_cohort'][name]
   scores=[];fams=[]
   for p in pred:
    m=manifest[p['rna_id']];truth=pair_set_0b(pairs_from_str(m['original_pairs']));eligible=pair_set_0b(pairs_from_str(m['eligible_pairs']))
    actual={tuple(v) for v in p['predicted_pairs']};reported={tuple(v) for v in p['reference_pairs']}
    assert truth==reported=={tuple(v) for v in p['full_reference_pairs']}
    assert eligible=={tuple(v) for v in p['eligible_reference_pairs']}
    assert p['sequence']==m['sequence'] and (m.get('family_id') or 'unknown')==p['family_id']
    assert actual<=set(candidate_pairs(m['sequence']))
    endpoints=[v for pair in actual for v in pair];assert len(endpoints)==len(set(endpoints))
    f1=pair_prf(truth,actual)[2];assert abs(f1-p['exact_f1'])<1e-12
    scores.append(f1);fams.append((p['family_id'],f1))
    if name=='family-dev':eligible_scores.append(pair_prf(eligible,actual)[2])
   mean=sum(scores)/len(scores)
   if name=='family-dev':
    assert abs(mean-float(row['rna_macro_exact_f1']))<1e-10
    assert abs(family_macro(fams)-float(row['family_macro_exact_f1']))<1e-10
   else:assert abs(mean-summary['VL0_exact_f1'])<1e-10
  record=dict(mode=mode,seed=seed,passed=True,epochs=summary['epochs_completed'],best_epoch=summary['best_epoch'],threshold=summary['threshold'],checkpoint_sha256=sha(d/'best.pt'),prediction_sha256=sha(d/'family-dev_predictions.jsonl'),initialization_hash=summary['initialization_hash'],config_hash=summary['config_hash'],eligible_RNA_macro=sum(eligible_scores)/len(eligible_scores))
  checks.append(record);groups[mode].append(row)
 assert len({x['config_hash'] for x in checks})==1
 for seed in {x['seed'] for x in checks}:assert len({x['initialization_hash'] for x in checks if x['seed']==seed})==1
 required=[('E0_no_thermo',17),('E1_bpp',17)]
 if decision['run_E2']:required=[(m,s) for m in ('E0_no_thermo','E1_bpp','E2_bpp_stack') for s in (17,23,41)]
 elif '23' in decision['supporting_metrics']:required += [('E0_no_thermo',23),('E1_bpp',23)]
 if decision['Q_retention']=='retain_candidate':required += [('E3_bpp_product',s) for s in (17,23,41)]
 assert set(required)<={(x['mode'],x['seed']) for x in checks}
 census=defaultdict(int);success=defaultdict(int)
 for sid in split['family_dev']:census[manifest[sid].get('family_id') or 'unknown']+=1
 for sid in cache['shared_successful_cohort']['family-dev']:success[manifest[sid].get('family_id') or 'unknown']+=1
 family_path=O/'family_delta.csv';family_rows=list(csv.DictReader(family_path.open()))
 for x in family_rows:
  f=x['family_id'];x.update(fixed_family_RNAs=census[f],thermo_successful_RNAs=success[f],feature_coverage=success[f]/census[f]);families[x['mode']].append(float(x['delta_exact_f1']))
 if family_rows:
  with family_path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(family_rows[0]));w.writeheader();w.writerows(family_rows)
 corrections=list(csv.DictReader((O/'pair_stem_corrections.csv').open()))
 for x in corrections:
  for k in ('full_original_true_added','full_original_false_deleted','full_original_false_added','full_original_true_deleted','reference_stems','complete_stems_recovered','E0_complete_stems_recovered','reference_stem_pair_count','recovered_stem_pair_count'):
   pairtot[x['mode']][k]+=int(x[k])
 status_rows=[]
 for mode in ('E0_no_thermo','E1_bpp','E2_bpp_stack','E3_bpp_product'):
  for seed in (17,23,41):
   found=next((x for x in checks if x['mode']==mode and x['seed']==seed),None)
   state='completed_validated' if found else ('skipped_by_rule' if (mode,seed) not in required else 'not_run')
   status_rows.append(dict(mode=mode,seed=seed,status=state))
 with (O/'experiment_status.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['mode','seed','status']);w.writeheader();w.writerows(status_rows)
 save(O/'artifact_validation.json',dict(passed=True,runs=checks,required=required,metrics_recomputed_from_all_predictions=True,checkpoint_equals_selected_epoch=True,test_data_evaluated=False))
 save(O/'structure_analysis_summary.json',dict(pair_stem_totals=pairtot,decision=decision))
 report=['# RNA ThermoFusion FAST-v1 实际结果','', f'工作根目录：`{R}`。以下仅纳入本目录实际完成并通过核验的运行。', '', '主指标使用**完整原始真值**的逐家族等权 exact F1，包含 B0 无法解码的配对。未使用 TS0 或 bpRNA-new 预测指标。', '', '|方法|seed|家族宏 F1|RNA宏 F1|VL0 RNA宏 F1（辅助）|最佳/实际epoch|阈值|', '|---|---:|---:|---:|---:|---:|---:|']
 for x in rows:report.append(f"|{x['mode']}|{x['seed']}|{float(x['family_macro_exact_f1']):.6f}|{float(x['rna_macro_exact_f1']):.6f}|{float(x['VL0_exact_f1']):.6f}|{x['best_epoch']}/{x['epochs_completed']}|{x['threshold']}|")
 report+=['','|方法|跨种子家族宏均值±样本SD|RNA宏均值±样本SD|','|---|---:|---:|']
 for mode,values in groups.items():
  a=[float(x['family_macro_exact_f1']) for x in values];b=[float(x['rna_macro_exact_f1']) for x in values]
  report.append(f'|{mode}|{statistics.mean(a):.6f} ± {statistics.stdev(a) if len(a)>1 else 0:.6f}|{statistics.mean(b):.6f} ± {statistics.stdev(b) if len(b)>1 else 0:.6f}|')
 report+=['','继续/停止判定：`'+decision['decision']+'`；Q保留判定：`'+decision['Q_retention']+'`。逐种子证据见 decision.json。0.015 是预定计算预算门槛，并非统计显著性。','', '|方|补回真配对|删除错配对|新增错配对|误删真配对|完整stem恢复/参考|','|---|---:|---:|---:|---:|---:|']
 for mode,v in pairtot.items():report.append(f"|{mode}|{v['full_original_true_added']}|{v['full_original_false_deleted']}|{v['full_original_false_added']}|{v['full_original_true_deleted']}|{v['complete_stems_recovered']}/{v['reference_stems']}|")
 report+=['','配对变化按同种子E0逐RNA比较，表中为各种子计数之和。Stem复用 maximal_positive_stems(min_stem=3)，参考为完整真值中的连续最大片段；完整恢复要求每个配对均正确，部分恢复为stem内配对召回。', '', '固定数量：fit-train3299、family-dev1068/292家族、VL01286、TS01288、bpRNA-new5388。审计见split_audit.json与provenance/preflight.json，包含去重、家族隔离、CLS/EOS、实际预训练权重SHA256和embedding来源。VL0全部1286条家族unknown，保留并报告RNA宏，不声称VL0家族宏。', '', '特征覆盖率：'+json.dumps(cache['coverage'],ensure_ascii=False)+'。逐家族覆盖率见family_delta.csv；失败原因见thermo_cache_manifest.json。', '', '热力学：ViennaRNA2.6.4，37°C/dangles2/noLP=False/uniq_ML=True/Turner2004，cutoff1e-8；一次PF取得P/Q。Python无stack_prob包装，使用同版本共享库vrna_stack_prob；1-based和i=0哨兵经独立短序列全结构枚举核验。Q是真实联合概率，product只作为明确标记的条件对照。']
 report += [
 '',
 '统一训练：原B0 head与候选BCE，CNN末层全零，head与CNN联合训练，encoder冻结。',
 '沿用旧AdamW lr1e-4/wd1e-4/accumulation4/batch1/clip1/BF16/max10/patience3。',
 'Thresholds0.1..0.9: select only by family-dev full-truth family macro F1; lower threshold and earlier epoch break ties.',
 '每条RNA完整输入，无padding。同种子各模式同初始化hash、顺序、预算。',
 '运行成本见development_summary.csv和thermo_cache_manifest.json，wall time含冷启动。',
 'All experiment statuses are explicitly recorded in experiment_status.csv.',

 'Smoke只验证程序。旧历史B0不作本轮公平比较基线。',
 '限制：同一family-dev用于选模型，开发结果不能替代独立最终测试。种子SD不是置信区间。',
 '监督家族留出不代表RiNALMo预训练没有见过相关RNA。',
 '本次不扩展全量TR0重训、最终测试或九张扰动BPP。',
 '',
 '恢复：在Astra运行 `. ./env.sh`，再运行 `.venv/bin/python -u scripts/run_thermo_fast.py --pipeline`。',
 '源码/输入哈希见implementation_manifest.json和各组code_identity.json、input_provenance.json。',
 '日志、PID、退出状态见logs/和pipeline_status.json。进度见PROGRESS.md。',
 ]
 (O/'README_results.md').write_text('\n'.join(report)+'\n')
 print(json.dumps(dict(validation='passed',formal_runs=len(checks),required=len(required),decision=decision['decision'],Q=decision['Q_retention'])),flush=True)
if __name__=='__main__':main()
