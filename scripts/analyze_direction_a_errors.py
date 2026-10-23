"""Descriptive error stratification after locked policies; no exclusions or tuning."""
from collections import defaultdict
import csv,json
from pathlib import Path
from statistics import mean,median
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.direction_a_protocol import ROOT,OUT,file_sha,write_json

def crossing_pairs(pairs):
    pairs=sorted(map(tuple,pairs));result=set()
    for index,(a,b) in enumerate(pairs):
        for c,d in pairs[index+1:]:
            if a<c<b<d:result.update(((a,b),(c,d)))
    return result

def run():
    from scripts.archiveii_frozen_data import load_manifest,OUT as HISTORICAL
    from scripts.direction_a_edit_packets import oracle
    manifest=load_manifest();roles=json.loads((OUT/'pilot_roles.json').read_text());cohorts={};metrics=[];sources={};oracles=[]
    report=ROOT/'reports/direction_a_revision1_2026-10-23';audit=json.loads((report/'numeric_source.json').read_text())
    assert audit['independently_verified_prediction_rows']>0
    for case in ('srp','tmRNA'):
        ids=roles['cases'][case]['development'];raw={s:manifest['records'][s] for s in ids};pk={s:crossing_pairs(raw[s]['pairs']) for s in ids}
        lengths=[len(r['sequence']) for r in raw.values()];sequences=defaultdict(set);eligible=total=0
        for sid,r in raw.items():
            truth=frozenset(map(tuple,r['pairs']));sequences[r['sequence']].add(truth);total+=len(truth)
            eligible+=sum(b-a>=4 and r['sequence'][a]+r['sequence'][b] in {'AU','UA','CG','GC','GU','UG'} for a,b in truth)
        cohorts[case]=dict(n=len(ids),unique_sequences=len(sequences),mean_length=mean(lengths),median_length=median(lengths),max_length=max(lengths),
            reference_crossing_rna_n=sum(bool(v) for v in pk.values()),fraction_reference_pairs_in_crossings=sum(map(len,pk.values()))/total,
            candidate_truth_coverage=eligible/total,contradictory_sequence_groups=sum(len(v)>1 for v in sequences.values()))
        old_path=OUT/'pilot'/case/'correction_seed17/development_predictions.json';new_path=OUT/'revision1'/case/'development_predictions.json'
        original=json.loads(old_path.read_text());revision=json.loads(new_path.read_text());base={r['sample_id']:r['exact_f1'] for r in original['base']['selected']}
        base_pairs={r['sample_id']:set(map(tuple,r['prediction'])) for r in original['base']['selected']}
        sources[case]=dict(original_predictions_sha256=file_sha(old_path),revision_predictions_sha256=file_sha(new_path))
        for name,views in {**original,**revision}.items():
            rows=views['selected'];assert len(rows)==len(ids) and {r['sample_id'] for r in rows}==set(ids)
            grouped=defaultdict(list)
            for row in rows:grouped[raw[row['sample_id']]['sequence']].append(row['exact_f1'])
            deltas=[r['exact_f1']-base[r['sample_id']] for r in rows];harm=sum(r['harmful'] for r in rows);helpful=sum(r['helpful'] for r in rows)
            crossing=[r for r in rows if pk[r['sample_id']]];noncrossing=[r for r in rows if not pk[r['sample_id']]]
            pk_total=sum(len(pk[r['sample_id']]) for r in rows);pk_hits=sum(len(pk[r['sample_id']]&set(map(tuple,r['prediction']))) for r in rows)
            metrics.append(dict(case=case,method=name,n=len(rows),sequence_group_macro_f1=mean(mean(v) for v in grouped.values()),mean_delta_f1=mean(deltas),median_delta_f1=median(deltas),
                fraction_improved=mean(d>1e-12 for d in deltas),fraction_worse=mean(d< -1e-12 for d in deltas),
                edit_coverage=mean(r['harmful']+r['helpful']>0 for r in rows),mean_harmful_fraction=mean(r['harmful_rate'] for r in rows),
                pooled_harmful_fraction=harm/(harm+helpful) if harm+helpful else 0.,mean_harmful_edits=harm/len(rows),mean_helpful_edits=helpful/len(rows),
                crossing_rna_n=len(crossing),crossing_rna_f1=mean(r['exact_f1'] for r in crossing) if crossing else None,
                noncrossing_rna_n=len(noncrossing),noncrossing_rna_f1=mean(r['exact_f1'] for r in noncrossing) if noncrossing else None,
                crossing_involved_pair_recall=pk_hits/pk_total if pk_total else None))
        for name,views in [('local_gate',original['local_gate']),('risk_local_gate',revision['risk_local_gate'])]:
            for merged in (False,True):
                results=[]
                for row in views['selected']:
                    sid=row['sample_id'];result=oracle(base_pairs[sid],row['prediction'],raw[sid]['pairs'],merged)
                    assert result['f1']>=max(base[sid],row['exact_f1'])-1e-12
                    results.append(result)
                oracles.append(dict(case=case,proposal=name,stem_merged=merged,n=len(results),
                    base_f1=mean(base.values()),proposal_f1=mean(r['exact_f1'] for r in views['selected']),
                    whole_rna_choice_oracle_f1=mean(max(base[r['sample_id']],r['exact_f1']) for r in views['selected']),
                    oracle_f1=mean(r['f1'] for r in results),mean_packets=mean(r['packets'] for r in results),
                    mean_accepted_packets=mean(r['accepted'] for r in results),uses_ground_truth=True))
    with (report/'error_diagnostics.csv').open('w',newline='',encoding='utf8') as h:
        writer=csv.DictWriter(h,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    write_json(report/'error_mechanism_source.json',dict(cohorts=cohorts,metrics=metrics,sources=sources,data_manifest_sha256=file_sha(HISTORICAL/'data_manifest.json.gz'),
        code_sha256=file_sha(Path(__file__)),excluded_rnas=0,interpretation='Post-evaluation descriptive strata, no causal attribution, statistical significance or method selection'))
    with (report/'oracle_headroom.csv').open('w',newline='',encoding='utf8') as h:
        writer=csv.DictWriter(h,fieldnames=list(oracles[0]));writer.writeheader();writer.writerows(oracles)
    write_json(report/'oracle_diagnostic_source.json',dict(rows=oracles,sources=sources,code_sha256=file_sha(ROOT/'scripts/direction_a_edit_packets.py'),
        interpretation='Truth-using optimization bound over validity-preserving edits of a fixed proposal. Not model performance, not deployable, not used for primary metric or promotion.'))
    print(json.dumps(dict(cohorts=cohorts,metric_rows=len(metrics),oracle_diagnostics=oracles)))

if __name__=='__main__':run()
