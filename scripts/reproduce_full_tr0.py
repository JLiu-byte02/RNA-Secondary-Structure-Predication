"""Slurm supervisor for the complete nine-run TR0 / VL0 / held-out test stage."""
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/full_tr0'
STATE=OUT/'cluster_status.json'
MODES=('E0_no_thermo','E1_bpp','E2_bpp_stack')
SEEDS=(17,23,41)
FAILED={'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'}

def save(state):
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(STATE)

def main():
    os.chdir(ROOT)
    state=json.loads(STATE.read_text()) if STATE.exists() else dict(jobs={},started_unix=time.time())
    state.update(status='running',pid=os.getpid());state.pop('error',None);save(state)
    def submit(name,script,*args):
        if name in state['jobs'] and state['jobs'][name].get('status') in FAILED:
            state.setdefault('previous_jobs',[]).append(dict(name=name,**state['jobs'].pop(name)))
        if name not in state['jobs']:
            job=subprocess.check_output(['sbatch','--parsable',script,*map(str,args)],text=True).strip().split(';')[0]
            if not job.isdigit():raise ValueError('Unexpected Slurm response: '+job)
            state['jobs'][name]=dict(id=job,status='submitted');save(state)
            print('SUBMITTED',name,job,flush=True)
    def wait(names):
        state['stage']='waiting:'+','.join(names);save(state)
        pending=set(names)
        while pending:
            for name in sorted(pending):
                info=state['jobs'][name]
                text=subprocess.check_output(['sacct','-X','-n','-P','-j',info['id'],'-o','State,ExitCode'],text=True).strip()
                if not text:continue
                status,code=text.splitlines()[0].split('|')[:2];status=status.strip().split()[0]
                info.update(status=status,exit_code=code)
                if status=='COMPLETED' and code=='0:0':
                    pending.remove(name);print('COMPLETED',name,info['id'],flush=True)
                elif status in FAILED:raise RuntimeError('Slurm failure: '+name+' '+str(info))
            state['updated_unix']=time.time();save(state)
            if pending:time.sleep(30)
    try:
        submit('prepare','slurm/prepare_full.sbatch');wait(['prepare'])
        names=[]
        for seed in SEEDS:
            for mode in MODES:
                name=f'train_{mode}_seed{seed}';submit(name,'slurm/train_full.sbatch',mode,seed);names.append(name)
        wait(names)
        state['stage']='validation_selection_lock';save(state)
        subprocess.run([sys.executable,'scripts/eval_full_tr0.py','--lock'],check=True)
        names=[]
        for seed in SEEDS:
            for mode in MODES:
                name=f'test_{mode}_seed{seed}';submit(name,'slurm/test_full.sbatch',mode,seed);names.append(name)
        wait(names)
        state['stage']='independent_final_validation';save(state)
        subprocess.run([sys.executable,'scripts/eval_full_tr0.py','--finalize'],check=True)
        state.update(status='completed',stage='validated',elapsed_seconds=time.time()-state['started_unix'])
        save(state);print('FULL_TR0_COMPLETED',flush=True)
    except Exception:
        state.update(status='failed',error=traceback.format_exc());save(state);raise

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.cluster.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
