"""Resume-safe nine-family LOFO supervisor; no test-based method decisions."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'setup'))
from scripts.archiveii_data import OUT,FAMILIES,sha,write_json
from scripts.run_archiveii import MODES,SEEDS,code_identity
from check_archiveii_environment import verify_official_sources
STATE=OUT/'cluster_status.json'
FAILED={'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'}

def main(args):
    os.chdir(ROOT);(ROOT/'logs').mkdir(exist_ok=True)
    state=json.loads(STATE.read_text()) if STATE.exists() else dict(jobs={},started_unix=time.time())
    state.update(status='running',pid=os.getpid());state.pop('error',None)
    if args.prepare_job:
        if 'prepare' in state['jobs']:assert state['jobs']['prepare']['id']==args.prepare_job
        else:state['jobs']['prepare']=dict(id=args.prepare_job,status='submitted')
    if args.capacity_job:
        state['jobs'].setdefault('prepare_capacity',dict(id=args.capacity_job,status='submitted'))
    if args.download_pid:state['download_pid']=int(args.download_pid)
    def save():state['updated_unix']=time.time();write_json(STATE,state)
    def submit(name,script,*values):
        if name in state['jobs'] and state['jobs'][name].get('status') in FAILED:
            state.setdefault('previous_jobs',[]).append(dict(name=name,**state['jobs'].pop(name)))
        if name not in state['jobs']:
            job=subprocess.check_output(['sbatch','--parsable',script,*map(str,values)],text=True).strip().split(';')[0]
            assert job.isdigit();state['jobs'][name]=dict(id=job,status='submitted');save();print('SUBMITTED',name,job,flush=True)
    def poll():
        active={r['id']:name for name,r in state['jobs'].items() if r.get('status') not in ({'COMPLETED'}|FAILED)}
        if not active:return
        lines=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(active),'-o','JobIDRaw,State,ExitCode'],text=True).splitlines()
        for line in lines:
            fields=line.split('|')
            if len(fields)<3 or fields[0] not in active:continue
            job,status,exit_code=fields[:3];name=active[job];status=status.strip().split()[0]
            state['jobs'][name].update(status=status,exit_code=exit_code)
            if status=='COMPLETED' and exit_code=='0:0':print('COMPLETED',name,job,flush=True)
            elif status in FAILED or (status=='COMPLETED' and exit_code!='0:0'):raise RuntimeError('Failed job: '+name+' '+str(state['jobs'][name]))
        state['completed_train_runs']=sum(r.get('status')=='COMPLETED' for k,r in state['jobs'].items() if k.startswith('train_'))
        state['completed_baselines']=sum(r.get('status')=='COMPLETED' for k,r in state['jobs'].items() if k.startswith('baseline_'))
        save()
    def baseline_ready():
        for family in FAMILIES:
            provenance=ROOT/'weights/archiveii'/f'rinalmo_giga_ss_archiveII-{family}_ft.pt.provenance.json'
            name='baseline_'+family
            if name in state['jobs'] and state['jobs'][name].get('status') not in FAILED:continue
            if inflight()>=9:return
            if provenance.exists():submit(name,'slurm/test_archiveii_baseline.sbatch',family)
    def inflight():
        return sum(r.get('status') not in ({'COMPLETED'}|FAILED) for r in state['jobs'].values())
    def wait(names,with_baselines=False):
        state['stage']='waiting:'+','.join(names);save()
        while True:
            if with_baselines:baseline_ready()
            poll()
            if all(state['jobs'][name].get('status')=='COMPLETED' for name in names):return
            time.sleep(30)
    def command(*values):subprocess.run([sys.executable,*values],check=True)
    def ensure_download():
        if (OUT/'baseline_download_manifest.json').exists():return
        pid=state.get('download_pid');cmdline=Path(f'/proc/{pid}/cmdline') if pid else None
        if cmdline and cmdline.exists() and b'scripts/download_archiveii_baselines.py' in cmdline.read_bytes():return
        if state.get('download_attempts',0)>=3:raise RuntimeError('Official checkpoint downloader exhausted three attempts; inspect archiveii-download.log')
        with (ROOT/'logs/archiveii-download.log').open('ab') as stream:
            p=subprocess.Popen([sys.executable,'-u','scripts/download_archiveii_baselines.py'],stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True)
        state['download_pid']=p.pid;state['download_attempts']=state.get('download_attempts',0)+1;save()
    try:
        submit('prepare','slurm/prepare_archiveii.sbatch');wait(['prepare']+(['prepare_capacity'] if 'prepare_capacity' in state['jobs'] else []))
        preparation=json.loads((OUT/'preparation_complete.json').read_text());assert preparation['passed']
        assert preparation['lightning_15_phase_fixture_passed']
        assert preparation['training_source']==code_identity(),'Source changed since smoke checks'
        frozen=OUT/'experiment_identity.json'
        if not frozen.exists():
            assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Publish and synchronize the reviewed source before formal training'
            write_json(frozen,dict(training_source=code_identity(),official_sources=verify_official_sources(),
                data_manifest_sha256=sha(OUT/'data_manifest.json.gz'),physical_manifest_sha256=sha(OUT/'physical_manifest.json'),
                config_sha256=sha(ROOT/'configs/archiveii_ood.yaml'),pretrained_sha256=sha(ROOT/'weights/rinalmo_giga_pretrained.pt'),
                preparation_sha256=sha(OUT/'preparation_complete.json'),environment_sha256=sha(OUT/'environment_verification.json'),
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),seeds=list(SEEDS),families=list(FAMILIES)))
        else:assert json.loads(frozen.read_text())['training_source']==code_identity()
        ensure_download()
        # Ordered rolling queue: independent folds may overlap, with at most nine
        # E/baseline jobs submitted or running. Never mix model state across folds.
        tasks=[(f,m,s,f'train_{f}_{m}_seed{s}') for f in FAMILIES for s in SEEDS for m in MODES]
        state['queue_policy']=dict(family_submission_order=list(FAMILIES),max_inflight_jobs=9,independent_folds_may_overlap=True)
        while True:
            ensure_download();baseline_ready()
            for family,mode,seed,name in tasks:
                if name in state['jobs'] and state['jobs'][name].get('status') not in FAILED:continue
                if inflight()>=9:break
                submit(name,'slurm/train_archiveii.sbatch',family,mode,seed)
            poll();state.setdefault('completed_families',[])
            for family in FAMILIES:
                names=[name for f,_,_,name in tasks if f==family]
                if family not in state['completed_families'] and all(state['jobs'].get(n,{}).get('status')=='COMPLETED' for n in names):
                    state['completed_families'].append(family);save();command('scripts/report_archiveii.py','--partial');print('FAMILY_COMPLETED',family,flush=True)
            baseline_names=['baseline_'+f for f in FAMILIES]
            if all(state['jobs'].get(name,{}).get('status')=='COMPLETED' for *_,name in tasks) and all(state['jobs'].get(name,{}).get('status')=='COMPLETED' for name in baseline_names):break
            state['stage']='rolling_family_queue';save();time.sleep(30)
        submit('validate','slurm/validate_archiveii.sbatch');wait(['validate'])
        command('scripts/report_archiveii.py','--report')
        state.update(status='completed',stage='validated',elapsed_seconds=time.time()-state['started_unix']);save()
        print('ARCHIVEII_NINE_FAMILY_STUDY_COMPLETED',flush=True)
    except Exception:
        state.update(status='failed',error=traceback.format_exc());save();raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-job');p.add_argument('--capacity-job');p.add_argument('--download-pid');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.cluster.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main(a)
