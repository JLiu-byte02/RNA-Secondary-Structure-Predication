"""Run the predeclared frozen-E nine-family study under Slurm, with audited retries."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.archiveii_frozen_data import OUT,sha,write_json
from scripts.archiveii_frozen_protocol import FAMILIES,MODES,SEEDS,code_identity
from scripts.run_archiveii_frozen import input_identity
STATE=OUT/'cluster_status.json'
FAILED={'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'}

def main(args):
    os.chdir(ROOT);(ROOT/'logs').mkdir(exist_ok=True)
    state=json.loads(STATE.read_text()) if STATE.exists() else dict(jobs={},started_unix=time.time(),previous_jobs=[])
    state.update(status='running',pid=os.getpid());state.pop('error',None)
    if args.embed_job:
        state['jobs'].setdefault('embed',dict(id=args.embed_job,status='submitted'))
    if args.smoke_job:state['jobs'].setdefault('smoke',dict(id=args.smoke_job,status='submitted'))
    def save():state['updated_unix']=time.time();write_json(STATE,state)
    def submit(name,script,*values):
        if name in state['jobs'] and state['jobs'][name]['status'] in FAILED:
            state['previous_jobs'].append(dict(name=name,**state['jobs'].pop(name)))
        if name not in state['jobs']:
            job=subprocess.check_output(['sbatch','--parsable',script,*map(str,values)],text=True).strip().split(';')[0]
            assert job.isdigit();state['jobs'][name]=dict(id=job,status='submitted');save();print('SUBMITTED',name,job,flush=True)
    def poll():
        active={row['id']:name for name,row in state['jobs'].items() if row['status'] not in {'COMPLETED'}|FAILED}
        if not active:return
        lines=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(active),'-o','JobIDRaw,State,ExitCode'],text=True).splitlines()
        error=None
        for line in lines:
            fields=line.split('|')
            if len(fields)<3 or fields[0] not in active:continue
            job,status,exit_code=fields[:3];name=active[job];status=status.strip().split()[0]
            state['jobs'][name].update(status=status,exit_code=exit_code)
            if status=='COMPLETED' and exit_code=='0:0':print('COMPLETED',name,job,flush=True)
            elif status in FAILED or (status=='COMPLETED' and exit_code!='0:0'):
                state['jobs'][name]['status']=status if status in FAILED else 'FAILED';error=name+' '+str(state['jobs'][name])
        state['completed_train_runs']=sum(row['status']=='COMPLETED' for name,row in state['jobs'].items() if name.startswith('train_'))
        state['completed_baselines']=sum(row['status']=='COMPLETED' for name,row in state['jobs'].items() if name.startswith('baseline_'))
        save()
        if error:raise RuntimeError('Inspect failed job before retrying: '+error)
    def inflight():return sum(row['status'] not in {'COMPLETED'}|FAILED for row in state['jobs'].values())
    def wait(names):
        while True:
            poll()
            if all(state['jobs'][name]['status']=='COMPLETED' for name in names):return
            time.sleep(30)
    def command(*values):subprocess.run([sys.executable,*values],check=True)
    def ensure_download():
        if (OUT/'baseline_download_manifest.json').exists():return
        pid=state.get('download_pid')
        if pid:
            path=Path(f'/proc/{pid}/cmdline')
            if path.exists() and b'download_archiveii_frozen_baselines.py' in path.read_bytes():return
        if state.get('download_attempts',0)>=4:raise RuntimeError('Official checkpoint download exhausted four attempts; inspect its log')
        with (ROOT/'logs/archiveii_frozen_download.log').open('ab') as stream:
            process=subprocess.Popen([sys.executable,'-u','scripts/download_archiveii_frozen_baselines.py'],
                stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True)
        state['download_pid']=process.pid;state['download_attempts']=state.get('download_attempts',0)+1;save()
    def baseline_ready():
        for family in FAMILIES:
            name='baseline_'+family
            if name in state['jobs'] and state['jobs'][name]['status'] not in FAILED:continue
            if inflight()>=args.max_inflight:return
            path=ROOT/'weights/archiveii_frozen'/f'rinalmo_giga_ss_archiveII-{family}_ft.pt.provenance.json'
            if path.exists():submit(name,'slurm/baseline_archiveii_frozen.sbatch',family)
    try:
        save();state['stage']='preparation';save()
        if 'embed' not in state['jobs']:submit('embed','slurm/embed_archiveii_frozen.sbatch')
        wait(['embed'])
        if 'smoke' not in state['jobs']:submit('smoke','slurm/smoke_archiveii_frozen.sbatch')
        wait(['smoke'])
        prep=json.loads((OUT/'preparation_complete.json').read_text())
        assert prep['passed'] and prep['training_source']==code_identity() and prep['inputs']==input_identity()
        command('scripts/check_archiveii_frozen_sources.py')
        path=OUT/'experiment_identity.json'
        if not path.exists():
            assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Publish and synchronize source before formal training'
            write_json(path,dict(protocol='ARCHIVEII-FROZEN-E-LOFO-v2',training_source=code_identity(),inputs=input_identity(),
                config_sha256=sha(ROOT/'configs/archiveii_frozen.yaml'),preparation_sha256=sha(OUT/'preparation_complete.json'),
                official_source_verification_sha256=sha(OUT/'official_source_verification.json'),
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                data_policy='Exact official ArchiveII family-fold CT membership; no new filtering or deduplication',
                encoder_frozen=True,seeds=list(SEEDS),families=list(FAMILIES),test_policy='All 81 choices sealed before any E test'))
        else:
            identity=json.loads(path.read_text());assert identity['training_source']==code_identity() and identity['inputs']==input_identity()
        if not state.get('download_pid') and (OUT/'download_pid.json').exists():
            state['download_pid']=json.loads((OUT/'download_pid.json').read_text())['pid']
        tasks=[(f,m,s,f'train_{f}_{m}_seed{s}') for f in FAMILIES for s in SEEDS for m in MODES]
        state['stage']='training';state['queue_policy']=dict(family_order=list(FAMILIES),max_inflight=args.max_inflight,independent_jobs_may_overlap=True);save()
        while True:
            ensure_download();baseline_ready()
            for family,mode,seed,name in tasks:
                if name in state['jobs'] and state['jobs'][name]['status'] not in FAILED:continue
                if inflight()>=args.max_inflight:break
                submit(name,'slurm/train_archiveii_frozen.sbatch',family,mode,seed)
            poll();state.setdefault('families_training_completed',[])
            for family in FAMILIES:
                names=[name for f,_,_,name in tasks if f==family]
                if family not in state['families_training_completed'] and all(state['jobs'].get(name,{}).get('status')=='COMPLETED' for name in names):
                    state['families_training_completed'].append(family);save();print('FAMILY_TRAINING_COMPLETED',family,flush=True)
            if all(state['jobs'].get(name,{}).get('status')=='COMPLETED' for *_,name in tasks):break
            time.sleep(30)
        command('scripts/eval_archiveii_frozen.py','--seal');state['stage']='testing';save()
        while True:
            ensure_download();baseline_ready()
            for family in FAMILIES:
                name='test_'+family
                if name in state['jobs'] and state['jobs'][name]['status'] not in FAILED:continue
                if inflight()>=args.max_inflight:break
                submit(name,'slurm/test_archiveii_frozen.sbatch',family)
            poll()
            names=['test_'+f for f in FAMILIES]+['baseline_'+f for f in FAMILIES]
            if all(state['jobs'].get(name,{}).get('status')=='COMPLETED' for name in names):break
            time.sleep(30)
        submit('validate','slurm/validate_archiveii_frozen.sbatch');state['stage']='validation';save();wait(['validate'])
        command('scripts/report_archiveii_frozen.py')
        state.update(status='completed',stage='validated',elapsed_seconds=time.time()-state['started_unix']);save()
        print('ARCHIVEII_FROZEN_NINE_FAMILY_STUDY_COMPLETED',flush=True)
    except Exception:
        state.update(status='failed',error=traceback.format_exc());save();raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--embed-job');parser.add_argument('--smoke-job')
    parser.add_argument('--max-inflight',type=int,default=8);args=parser.parse_args()
    assert 1<=args.max_inflight<=12
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.cluster.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main(args)
