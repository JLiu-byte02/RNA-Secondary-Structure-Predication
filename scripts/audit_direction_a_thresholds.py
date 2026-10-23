"""Post-hoc baseline-only search-boundary sensitivity; never replaces main results."""
import csv,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.direction_a_protocol import ROOT,OUT,config,file_sha,write_json,source_identity
from scripts.direction_a_data import prepare_roles,load_records
from scripts.run_direction_a_correction import load_bases
from scripts.direction_a_evaluate import probabilities,calibrate,metric_row,aggregate
from scripts.archiveii_frozen_metrics import greedy_all_thresholds
from scripts.report_direction_a import verify_records,verify_reference

def run():
    cfg=config();roles=prepare_roles();thresholds=sorted(set([.0005,.001,.002,.005,.01,.02,.03,.04]+cfg['thresholds']))
    seals={};source=dict(code=file_sha(Path(__file__)),base_code=source_identity('base'),roles_sha256=roles['sha256'],
        official_reference_sha256=verify_reference(ROOT/'third_party/rinalmo_frozen_reference/sec_struct.py'))
    for case in ('srp','tmRNA'):
        ids=roles['cases'][case]['calibration'];records=load_records(ids);tensors,parents=load_bases(case,17,records)
        base={s:v.numpy() for s,v in tensors.items()};physical={s:np.log(np.clip(r['physical'][0].numpy(),1e-4,1-1e-4)/(1-np.clip(r['physical'][0].numpy(),1e-4,1-1e-4))) for s,r in records.items()}
        base_choice=calibrate(records,ids,{s:probabilities(v) for s,v in base.items()},thresholds);grid=[]
        for alpha in cfg['mixture_alphas']:
            values={s:probabilities((1-alpha)*base[s]+alpha*physical[s]) for s in ids}
            grid.append(dict(alpha=alpha,**calibrate(records,ids,values,thresholds)))
        seals[case]=dict(parents=parents,base=base_choice,fixed_mix=max(grid,key=lambda r:r['equal_family_exact_f1']),grid=grid)
    directory=OUT/'threshold_sensitivity';write_json(directory/'choices_sealed.json',dict(source=source,thresholds=thresholds,cases=seals,
        scope='Baseline-only post-hoc sensitivity, all cases selected before new scoring. Neural methods were not retuned; not a matched primary comparison.'))
    summary=[];predictions={};verified=0
    for case in ('srp','tmRNA'):
        ids=roles['cases'][case]['development'];records=load_records(ids);tensors,parents=load_bases(case,17,records,development=True);assert parents==seals[case]['parents']
        old=json.loads((OUT/'pilot'/case/'correction_seed17/development_summary.json').read_text());base={s:v.numpy() for s,v in tensors.items()}
        base_predictions={s:greedy_all_thresholds(probabilities(base[s]),records[s]['sequence'],[old['methods']['base']['choice']['threshold']])[old['methods']['base']['choice']['threshold']] for s in ids}
        for method in ('base','fixed_mix'):
            choice=seals[case][method];rows=[]
            for sid in ids:
                r=records[sid];value=base[sid]
                if method=='fixed_mix':
                    p=np.clip(r['physical'][0].numpy(),1e-4,1-1e-4);value=(1-choice['alpha'])*value+choice['alpha']*np.log(p/(1-p))
                pred=greedy_all_thresholds(probabilities(value),r['sequence'],[choice['threshold']])[choice['threshold']]
                rows.append(metric_row(sid,r,pred,base_predictions[sid]))
            verify_records(rows);verified+=len(rows);actual=aggregate(rows)['equal_family'];predictions[case+'_'+method]=rows
            summary.append(dict(case=case,method=method,original_threshold=old['methods'][method]['choice']['threshold'],new_threshold=choice['threshold'],alpha=choice.get('alpha',''),
                original_exact_f1=old['methods'][method]['selected']['equal_family']['exact_f1'],new_exact_f1=actual['exact_f1'],new_flexible_f1=actual['flexible_f1'],harmful_fraction=actual['harmful_rate']))
    report=ROOT/'reports/direction_a_revision1_2026-10-23'
    with (report/'threshold_sensitivity.csv').open('w',newline='',encoding='utf8') as h:
        writer=csv.DictWriter(h,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    write_json(directory/'predictions.json',predictions)
    write_json(report/'threshold_sensitivity_source.json',dict(source=source,rows=summary,verified_prediction_rows=verified,
        choices_sha256=file_sha(directory/'choices_sealed.json'),predictions_sha256=file_sha(directory/'predictions.json'),
        interpretation='Expanded-grid baseline sensitivity only. Does not replace original selections, promotion or matched neural comparisons.'))
    print(json.dumps(dict(rows=summary,verified=verified)))

if __name__=='__main__':run()
