"""Fail-closed allocation admission and prospectively declared pilot decision."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.direction_a_protocol import OUT,config,write_json,file_sha

TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}

def terminal(state):return state.split()[0].rstrip('+') in TERMINAL

def budget_decision(jobs,accounting,new_wall_seconds,cap_hours,max_inflight,other_active=0):
    charged=reserved=0;active=other_active
    assert new_wall_seconds>=0 and cap_hours>0
    for job in jobs:
        if job['job_id'] not in accounting:raise ValueError('Missing accounting: '+job['job_id'])
        row=accounting[job['job_id']];elapsed=row['elapsed'];assert elapsed>=0
        charged+=elapsed*job['gpus']
        if not terminal(row['state']):
            active+=1;reserved+=max(0,job['wall_seconds']-elapsed)*job['gpus']
    return dict(allowed=charged+reserved+new_wall_seconds<=cap_hours*3600 and (active<max_inflight or new_wall_seconds==0),
                charged_seconds=charged,reserved_remaining_seconds=reserved,active=active,cap_seconds=cap_hours*3600,
                new_wall_seconds=new_wall_seconds)

def accounting(jobs):
    if not jobs:return {}
    ids=','.join(r['job_id'] for r in jobs);assert all(r['job_id'].isdigit() for r in jobs)
    text=subprocess.check_output(['sacct','-j',ids,'-X','-n','-P','-o','JobID,State,ElapsedRaw'],text=True)
    result={}
    for line in text.splitlines():
        fields=line.split('|')
        if len(fields)>=3 and fields[0] in ids.split(','):result[fields[0]]=dict(state=fields[1],elapsed=int(fields[2]))
    return result

def resource_status(ledger,rows=None):
    all_jobs=[r for values in ledger.values() for r in values]
    if rows is None:rows=accounting(all_jobs)
    cfg=config();report={}
    for phase in ('smoke','pilot','core'):
        other=sum(not terminal(rows[r['job_id']]['state']) for r in all_jobs if r not in ledger[phase])
        report[phase]=budget_decision(ledger[phase],rows,0,cfg[phase+'_gpu_hour_cap'],cfg['max_inflight'],other)
    return dict(phases=report,accounting=rows)

def submit(args):
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'jobs.json';cfg=config()
    with (OUT/'.submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=json.loads(path.read_text());status=resource_status(ledger)
        assert not any(r.get('tag')==args.tag for values in ledger.values() for r in values),'Duplicate job tag'
        other=sum(not terminal(status['accounting'][r['job_id']]['state']) for phase,values in ledger.items() if phase!=args.phase for r in values)
        decision=budget_decision(ledger[args.phase],status['accounting'],args.minutes*60,cfg[args.phase+'_gpu_hour_cap'],cfg['max_inflight'],other)
        assert decision['allowed'],json.dumps(decision)
        if args.phase=='core':
            promoted=json.loads((OUT/'pilot/promotion.json').read_text());assert promoted['promote'],'No measured promotion'
        command=['sbatch','--parsable',f'--time=00:{args.minutes:02d}:00',args.script,*args.arguments]
        if args.minutes>=60:command[2]=f'--time={args.minutes//60:02d}:{args.minutes%60:02d}:00'
        receipt=subprocess.check_output(command,text=True).strip();jid=receipt.split(';')[0];assert jid.isdigit()
        ledger[args.phase].append(dict(job_id=jid,phase=args.phase,tag=args.tag,wall_seconds=args.minutes*60,gpus=1,command=command))
        write_json(path,ledger)
        # Slurm's successful submission receipt precedes sacct database visibility.
        # Reserve its full walltime immediately; later admissions still require actual accounting.
        known=dict(status['accounting']);known[jid]=dict(state='SUBMITTED',elapsed=0)
        write_json(OUT/'resource_status.json',resource_status(ledger,known))
        print(json.dumps(dict(submitted=jid,tag=args.tag,admission=decision)))

def promotion(summaries,cfg):
    candidates={}
    for method in ('local_gate','safe_local_gate'):
        gain={case:row['methods'][method]['selected']['equal_family']['exact_f1']-row['methods']['fixed_mix']['selected']['equal_family']['exact_f1'] for case,row in summaries.items()}
        harm=sum(row['methods'][method]['selected']['equal_family']['harmful_rate'] for row in summaries.values())/len(summaries)
        residual_harm=sum(row['methods']['residual']['selected']['equal_family']['harmful_rate'] for row in summaries.values())/len(summaries)
        mean=sum(gain.values())/len(gain)
        candidates[method]=dict(mean_exact_gain=mean,per_case_exact_gain=gain,mean_harmful_rate=harm,residual_mean_harmful_rate=residual_harm,
                               promote=mean>=cfg['promotion_mean_exact_gain'] and min(gain.values())>=-cfg['promotion_max_family_loss'] and harm<=residual_harm)
    eligible=[m for m,row in candidates.items() if row['promote']]
    winner=max(eligible,key=lambda m:candidates[m]['mean_exact_gain']) if eligible else None
    return dict(promote=bool(eligible),winner=winner,candidates=candidates,interpretation='Compute-admission decision, not a significance test')

def decide():
    cfg=config();summaries={};sources={}
    for case in cfg['pilot_development_families']:
        path=OUT/'pilot'/case/'correction_seed17/development_summary.json';row=json.loads(path.read_text())
        assert row['completed'] and not row['smoke'] and row['case']==case
        assert all(v['official_verification']['passed'] for v in row['methods'].values())
        summaries[case]=row;sources[case]=file_sha(path)
    result=dict(source_summaries=sources,**promotion(summaries,cfg));write_json(OUT/'pilot/promotion.json',result);print(json.dumps(result))

if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('status');sub.add_parser('decide');s=sub.add_parser('submit')
    s.add_argument('--phase',choices=('smoke','pilot','core'),required=True);s.add_argument('--tag',required=True)
    s.add_argument('--minutes',type=int,required=True);s.add_argument('script');s.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if args.action=='submit':submit(args)
    elif args.action=='decide':decide()
    else:
        result=resource_status(json.loads((OUT/'jobs.json').read_text()));write_json(OUT/'resource_status.json',result);print(json.dumps(result))
