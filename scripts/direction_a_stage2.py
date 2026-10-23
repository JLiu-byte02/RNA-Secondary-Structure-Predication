"""Pure, fail-closed choices for matched calibration and family-chain roles."""
import math
from scripts.direction_a_protocol import digest,group_calibration

def thresholds(legacy):return sorted(set([.0005,.001,.002,.005,.01,.02,.03,.04]+list(legacy)))

def choose(profiles,budget=None):
    if not profiles:raise ValueError('Empty calibration profiles')
    for row in profiles:
        if not all(math.isfinite(row[k]) for k in ('f1','harm','beta','threshold')):raise ValueError('Nonfinite profile')
        if not (0<=row['f1']<=1 and 0<=row['harm']<=1 and 0<=row['beta']<=1 and 0<row['threshold']<1):raise ValueError('Invalid profile')
    feasible=[r for r in profiles if budget is None or r['harm']<=budget+1e-12]
    if not feasible:raise ValueError('No feasible calibration choice')
    return max(feasible,key=lambda r:(r['f1'],-r['harm'],-r['beta'],-r['threshold']))

def utility_roles(records,old,development):
    families=sorted({records[s]['family'] for s in old['block_a']},key=lambda f:(digest(f),f))
    assert len(families)==4
    proposal_families=set(families[:2]);utility_families=set(families[2:])
    utility=[s for s in old['block_a'] if records[s]['family'] in utility_families]
    fit,cal=group_calibration(records,utility,.2)
    role=dict(base_train=sorted(old['block_b']),proposal_fit=sorted(s for s in old['block_a'] if records[s]['family'] in proposal_families),
              utility_fit=fit,utility_cal=cal,development=sorted(old['development']))
    groups=[set(role[n]) for n in ('base_train','proposal_fit','utility_fit','utility_cal','development')]
    assert not any(a&b for i,a in enumerate(groups) for b in groups[i+1:])
    for i,a in enumerate(groups):
        for b in groups[i+1:]:assert not({records[s]['sequence'] for s in a}&{records[s]['sequence'] for s in b}),'Cross-role identical sequence'
    family_sets=[{records[s]['family'] for s in role[n]} for n in ('base_train','proposal_fit','utility_fit','development')]
    assert list(map(len,family_sets))==[3,2,2,1]
    assert not any(a&b for i,a in enumerate(family_sets) for b in family_sets[i+1:])
    assert family_sets[-1]=={development} and not any('5s' in f for f in family_sets)
    return role
