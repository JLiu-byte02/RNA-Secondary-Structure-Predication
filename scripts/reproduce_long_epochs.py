"""Resume-safe Slurm supervisor for the fixed B0/E 50-epoch study."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.long_protocol import MODES,SEEDS
from scripts.run_long_epochs import code_identity
OUT=ROOT/'outputs/long_epochs';STATE=OUT/'cluster_status.json'
TASKS=[('B0',s) for s in SEEDS]+[(m,s) for s in SEEDS for m in MODES if m!='B0']
FAILED={'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'}

def save(state):
    path=STATE.with_suffix('.tmp');path.write_text(json.dumps(state,indent=2));path.replace(STATE)

def main(args):
    os.chdir(ROOT);(ROOT/'logs').mkdir(exist_ok=True)
    state=json.loads(STATE.read_text()) if STATE.exists() else dict(jobs={},started_unix=time.time())
    state.update(status='running',pid=os.getpid());state.pop('error',None)
    if args.prepare_job and 'prepare' not in state['jobs']:state['jobs']['prepare']=dict(id=args.prepare_job,status='submitted')
    save(state)
    def submit(name,script,*values):
        if name in state['jobs'] and state['jobs'][name].get('status') in FAILED:
            state.setdefault('previous_jobs',[]).append(dict(name=name,**state['jobs'].pop(name)))
        if name not in state['jobs']:
            job=subprocess.check_output(['sbatch','--parsable',script,*map(str,values)],text=True).strip().split(';')[0]
            if not job.isdigit():raise ValueError('Unexpected Slurm job: '+job)
            state['jobs'][name]=dict(id=job,status='submitted');save(state);print('SUBMITTED',name,job,flush=True)
    def wait(names):
        state['stage']='waiting:'+','.join(names);save(state);pending=set(names)
        while pending:
            for name in sorted(pending):
                record=state['jobs'][name]
                result=subprocess.check_output(['sacct','-X','-n','-P','-j',record['id'],'-o','State,ExitCode'],text=True).strip()
                if not result:continue
                status,code=result.splitlines()[0].split('|')[:2];status=status.strip().split()[0]
                record.update(status=status,exit_code=code)
                if status=='COMPLETED' and code=='0:0':pending.remove(name);print('COMPLETED',name,record['id'],flush=True)
                elif status in FAILED or (status=='COMPLETED' and code!='0:0'):raise RuntimeError('Job failed: '+name+' '+str(record))
            state['updated_unix']=time.time();save(state)
            if pending:time.sleep(30)
    def command(*values):subprocess.run([sys.executable,*values],check=True)
    try:
        submit('prepare','slurm/prepare_long.sbatch');wait(['prepare'])
        if not json.loads((OUT/'preparation_complete.json').read_text())['passed']:raise RuntimeError('Smoke tests failed')
        identity=OUT/'experiment_identity.json'
        if not identity.exists():
            identity.write_text(json.dumps(dict(training_source=code_identity(),
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                config_sha256=__import__('hashlib').sha256((ROOT/'configs/long_epochs.yaml').read_bytes()).hexdigest()),indent=2))
        names=[]
        for mode,seed in TASKS:
            name=f'train_{mode}_seed{seed}';submit(name,'slurm/train_long.sbatch',mode,seed);names.append(name)
        wait(names)
        state['stage']='lock_all_validation_only_budget_selections';save(state)
        command('scripts/eval_long_epochs.py','--lock')
        names=[]
        for mode,seed in TASKS:
            name=f'test_{mode}_seed{seed}';submit(name,'slurm/test_long.sbatch',mode,seed);names.append(name)
        wait(names)
        state['stage']='independent_final_validation';save(state)
        command('scripts/eval_long_epochs.py','--finalize')
        state['stage']='collect_tables_and_report';save(state)
        command('scripts/report_long_epochs.py')
        state.update(status='completed',stage='validated',elapsed_seconds=time.time()-state['started_unix']);save(state)
        print('LONG_EPOCH_STUDY_COMPLETED',flush=True)
    except Exception:
        state.update(status='failed',error=traceback.format_exc());save(state);raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-job')
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.cluster.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main(parser.parse_args())
