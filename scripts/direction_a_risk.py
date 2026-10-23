"""Empirical decoded-edit calibration. No arbitrary-distribution risk guarantee."""
from collections import defaultdict
from statistics import mean
import math

def family_metrics(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['family']].append(row)
    assert groups
    return {key:mean(mean(r[key] for r in values) for values in groups.values()) for key in ('f1','harm')}

def choose_policy(profiles,budget):
    assert math.isfinite(budget) and budget>=0
    assert all(all(math.isfinite(r[k]) for k in ('f1','harm','beta','threshold')) for r in profiles)
    assert all(0<=r['beta']<=1 and 0<r['threshold']<1 and 0<=r['f1']<=1 and 0<=r['harm']<=1 for r in profiles),'Unbounded or invalid policy'
    feasible=[r for r in profiles if r['harm']<=budget+1e-12];assert feasible,'Identity fallback must be feasible'
    return max(feasible,key=lambda r:(r['f1'],-r['harm'],-r['beta'],-r['threshold']))

def revision_decision(rows,cfg):
    gains={case:r['candidate_f1']-r['risk_baseline_f1'] for case,r in rows.items()}
    original=mean(r['candidate_f1']-r['original_baseline_f1'] for r in rows.values())
    harm=mean(r['candidate_harm'] for r in rows.values());residual=mean(r['residual_harm'] for r in rows.values())
    return dict(promote=mean(gains.values())>=cfg['promotion_mean_exact_gain'] and min(gains.values())>=-cfg['promotion_max_family_loss'] and harm<=residual and original>=0,
                per_case_exact_gain=gains,mean_exact_gain=mean(gains.values()),mean_gain_over_original_fixed_mix=original,
                mean_harmful_rate=harm,residual_mean_harmful_rate=residual,interpretation='Development compute gate, not a significance test or OOD risk guarantee')
