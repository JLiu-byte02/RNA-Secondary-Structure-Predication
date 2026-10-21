"""Run the unchanged FAST-v1 gate on Slurm and validate all required outputs."""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/thermo_fast'
STATE = OUT/'cluster_status.json'


def save(state):
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state,indent=2))
    temporary.replace(STATE)


def command(args):
    print('COMMAND', ' '.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),cwd=ROOT,check=True)


def main():
    os.chdir(ROOT)
    OUT.mkdir(parents=True,exist_ok=True)
    (ROOT/'logs').mkdir(exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else dict(jobs={})
    state.update(status='running',pid=os.getpid(),started_unix=state.get('started_unix',time.time()))
    state.pop('error',None)
    save(state)

    def submit(name, script, *args):
        failed_states={'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'}
        if name in state['jobs'] and state['jobs'][name].get('status') in failed_states:
            state.setdefault('previous_jobs',[]).append(dict(name=name,**state['jobs'].pop(name)))
        if name not in state['jobs']:
            result=subprocess.check_output(['sbatch','--parsable',str(ROOT/script),*map(str,args)],text=True).strip()
            job=result.split(';')[0]
            if not job.isdigit():
                raise RuntimeError('Unexpected sbatch output: '+result)
            state['jobs'][name]=dict(id=job,status='submitted')
            save(state)
            print('SUBMITTED',name,job,flush=True)

    def wait(names):
        state['stage']='waiting:'+','.join(names)
        save(state)
        pending=set(names)
        while pending:
            for name in sorted(pending):
                info=state['jobs'][name]
                status=subprocess.check_output(['sacct','-X','-n','-P','-j',info['id'],'-o','State,ExitCode'],text=True).strip()
                if not status:
                    continue
                status,exit_code=status.splitlines()[0].split('|')[:2]
                status=status.strip().split()[0]
                info.update(status=status,exit_code=exit_code)
                if status=='COMPLETED' and exit_code=='0:0':
                    pending.remove(name)
                    print('COMPLETED',name,info['id'],flush=True)
                elif status in {'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'}:
                    raise RuntimeError('Slurm job failed: '+name+' '+json.dumps(info))
            state['updated_unix']=time.time()
            save(state)
            if pending:
                time.sleep(30)

    try:
        command([sys.executable,'scripts/unpack_data.py'])
        state['stage']='download_weights';save(state)
        command([sys.executable,'-u','scripts/download_weights.py'])
        submit('prepare','slurm/prepare.sbatch')
        wait(['prepare'])
        first=[]
        for mode in ('E0_no_thermo','E1_bpp'):
            name=mode+'_seed17';submit(name,'slurm/train.sbatch',mode,17);first.append(name)
        wait(first)
        def evaluate(seeds):
            command([sys.executable,'scripts/eval_thermo_fast.py','--seeds',','.join(map(str,seeds))])
            return json.loads((OUT/'decision.json').read_text())
        decision=evaluate([17])
        stack=decision['run_E2']
        seeds=[17,23,41] if stack else sorted({17,*decision.get('seeds_to_run',[23])})
        remaining=[]
        for seed in seeds:
            for mode in ('E0_no_thermo','E1_bpp','E2_bpp_stack') if stack else ('E0_no_thermo','E1_bpp'):
                name=mode+f'_seed{seed}'
                if name not in first:
                    submit(name,'slurm/train.sbatch',mode,seed);remaining.append(name)
        wait(remaining)
        decision=evaluate(seeds)
        if decision.get('Q_retention')=='retain_candidate':
            extra=[]
            for seed in (17,23,41):
                name=f'E3_bpp_product_seed{seed}';submit(name,'slurm/train.sbatch','E3_bpp_product',seed);extra.append(name)
            wait(extra)
            command([sys.executable,'scripts/eval_thermo_fast.py','--seeds','17,23,41','--compare','E0_no_thermo,E1_bpp,E2_bpp_stack,E3_bpp_product'])
        state['stage']='final_validation';save(state)
        command([sys.executable,'scripts/finalize_results.py'])
        state.update(status='completed',stage='validated',seeds=seeds,E2_executed=stack,
                     elapsed_seconds=time.time()-state['started_unix'],test_data_evaluated=False)
        save(state)
        print('CLUSTER_REPRODUCTION_COMPLETED',flush=True)
    except Exception as exc:
        state.update(status='failed',error=str(exc));save(state);raise


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.cluster.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
