"""Bounded A/B/C family-chain mechanism experiment, separate from v1 promotion."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.direction_a_protocol import ROOT,OUT,config,digest,file_sha,write_json,source_identity
from scripts.direction_a_data import prepare_roles,load_records
from scripts.direction_a_stage2 import utility_roles,thresholds
from scripts.direction_a_runtime import execution_spec
from scripts.archiveii_frozen_data import load_manifest
from scripts.direction_a_models import LocalCorrection,physical_entropy
from scripts.run_direction_a_correction import train_one_epoch,infer
from scripts.direction_a_evaluate import probabilities,calibrate,metric_row,aggregate,score,edits
from scripts.archiveii_frozen_metrics import greedy_all_thresholds
from scripts.direction_a_edit_packets import edit_packets
from scripts.direction_a_utility import features,target,predict_changes,apply_packets
from scripts.long_models import B0
from scripts.run_thermo_fast import model_inputs,save_checkpoint,state_hash
from scripts._common import set_seed
from scripts.report_direction_a import verify_records,verify_reference,write_csv

STAGE=ROOT/'outputs/direction_a_stage2/utility'
RUNTIME=execution_spec(os.environ.get('RNA_DIRECTION_A_DEVICE','cuda'))
KINDS=('pair_mlp','atomic_mlp','stem_mlp','stem_no_physics','stem_no_topology','stem_linear')
LAMBDAS=(0.,.25,.5,1.,2.,4.)
COSTS=(0.,.1,.25)
RULE_CUTS=(-.5,-.25,-.1,0.,.1,.25,.5,1.,1e6)

def source():
    names=['scripts/run_direction_a_utility.py','scripts/direction_a_utility.py','scripts/direction_a_edit_packets.py','scripts/direction_a_stage2.py','scripts/report_direction_a.py','scripts/direction_a_runtime.py']
    return dict(v1=source_identity('correction'),base_source=source_identity('base'),files={n:file_sha(ROOT/n) for n in names},official=verify_reference(ROOT/'third_party/rinalmo_frozen_reference/sec_struct.py'))

def identity(case):
    manifest=prepare_roles();raw=load_manifest()['records'];role=utility_roles(raw,manifest['cases'][case],case)
    directory=OUT/'pilot'/case/'base_b_seed17';summary=json.loads((directory/'summary.json').read_text())
    assert summary['completed'] and summary['epochs']==50 and summary['train_ids']==role['base_train']
    assert summary['identity']['code']==source_identity('base') and summary['identity']['inputs']==manifest['historical_inputs']
    assert file_sha(directory/'final.pt')==summary['checkpoint_sha256']
    return dict(source=source(),runtime=RUNTIME,case=case,seed=17,role=role,roles_sha256=manifest['sha256'],historical_inputs=manifest['historical_inputs'],
                base_summary_sha256=file_sha(directory/'summary.json'),base_checkpoint_sha256=summary['checkpoint_sha256'],
                proposal_epochs=15,utility_epochs=150,utility_hidden=32,utility_lr=.003)

def frozen_base(case,ids,records,device):
    directory=OUT/'pilot'/case/'base_b_seed17';row=json.loads((directory/'summary.json').read_text());values={}
    if RUNTIME['device']=='cpu':
        assert file_sha(directory/'final.pt')==row['checkpoint_sha256']
        fresh,embeddings=load_records(ids,True);model=B0();model.load_state_dict(torch.load(directory/'final.pt',map_location='cpu',weights_only=False)['model']);model.eval()
        cache=STAGE/'formal'/case/'base_cpu_fit';cache.mkdir(parents=True,exist_ok=True);artifacts={}
        with torch.no_grad():
            for i,sid in enumerate(ids):
                value=model(*model_inputs(fresh[sid],embeddings[sid],torch.device('cpu'))).float();assert torch.isfinite(value).all()
                values[sid]=value;path=cache/(digest(sid)+'.npz');np.savez_compressed(path,base=value.numpy());artifacts[sid]=file_sha(path)
                if (i+1)%100==0:print('UTILITY_BASE_CPU',case,i+1,len(ids),flush=True)
        write_json(cache/'manifest.json',dict(checkpoint_sha256=row['checkpoint_sha256'],runtime=RUNTIME,files=artifacts))
        return values
    for sid in ids:
        item=row['artifacts'][sid];path=ROOT/item['path'];assert file_sha(path)==item['sha256']
        with np.load(path,allow_pickle=False) as data:values[sid]=torch.from_numpy(data['base'].copy())
    return values

def ablate(x,kind):
    x=np.asarray(x,dtype=np.float32).copy()
    if kind=='stem_no_physics':x[:,[10,11,13,14]]=0
    if kind=='stem_no_topology':x[:,[3,15,16]]=0
    return x

def examples(ids,records,base,proposal,base_t,proposal_t):
    result={}
    for sid in ids:
        r=records[sid];s=base[sid].numpy();p=r['physical'][0].numpy();entropy=physical_entropy(r['physical'][0]).numpy()
        bp=greedy_all_thresholds(probabilities(s),r['sequence'],[base_t])[base_t]
        pp=greedy_all_thresholds(probabilities(proposal[sid]),r['sequence'],[proposal_t])[proposal_t]
        groups={k:edit_packets(bp,pp,merge_stems=(k=='stem')) for k in ('atomic','stem')}
        groups['pair']=[dict(added=[edge],removed=[]) for edge in sorted(pp-bp)]+[dict(added=[],removed=[edge]) for edge in sorted(bp-pp)]
        detail={}
        for kind,packets in groups.items():
            x=np.asarray([features(packet,bp,s,p,entropy) for packet in packets],dtype=np.float32).reshape(-1,18)
            detail[kind]=dict(packets=packets,x=x)
        result[sid]=dict(base=bp,proposal=pp,groups=detail)
    return result

def learning_rows(ids,records,ex,kind):
    group='pair' if kind=='pair_mlp' else 'atomic' if kind=='atomic_mlp' else 'stem'
    counts={f:sum(records[s]['family']==f for s in ids) for f in {records[s]['family'] for s in ids}}
    xs=[];ys=[];weights=[]
    for sid in ids:
        data=ex[sid]['groups'][group];n=len(data['packets'])
        for x,packet in zip(data['x'],data['packets']):
            xs.append(x);ys.append(target(packet,records[sid]['reference']));weights.append(1/(counts[records[sid]['family']]*max(1,n)))
    if not xs:raise ValueError('No editable packets in utility data; stop rather than inventing results')
    x=ablate(xs,kind);y=np.asarray(ys,dtype=np.float32);w=np.asarray(weights,dtype=np.float32);w/=w.sum()
    return x,y,w

def fit_predictor(kind,fit,cal,records,ex,directory):
    set_seed(17);x,y,w=learning_rows(fit,records,ex,kind);cx,cy,cw=learning_rows(cal,records,ex,kind)
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1
    x=(x-mean)/scale;cx=(cx-mean)/scale
    if kind=='stem_linear':
        a=np.column_stack((x,np.ones(len(x))));b=np.column_stack((cx,np.ones(len(cx))));pen=np.eye(19)*.01;pen[-1,-1]=0
        coef=np.linalg.solve(a.T@(w[:,None]*a)+pen,a.T@(w*y));prediction=b@coef
        model=dict(kind=kind,mean=mean,scale=scale,coef=coef,parameter_count=19,selected_epoch=0,
                   calibration_mse=float(np.sum(cw*(prediction-cy)**2)),training_examples=len(y))
    else:
        net=nn.Sequential(nn.Linear(18,32),nn.GELU(),nn.Linear(32,1),nn.Tanh());opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=1e-4)
        tx,ty,tw=torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(w);vx=torch.from_numpy(cx)
        best=None;curve=[]
        for epoch in range(1,151):
            net.train();opt.zero_grad(set_to_none=True);loss=((net(tx).flatten()-ty).square()*tw).sum();loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in net.parameters())
            opt.step();net.eval()
            with torch.no_grad():mse=float(np.sum(cw*(net(vx).flatten().numpy()-cy)**2))
            curve.append(dict(epoch=epoch,train_mse=float(loss.detach()),calibration_mse=mse))
            if best is None or mse<best['calibration_mse']:best=dict(state={k:v.detach().clone() for k,v in net.state_dict().items()},calibration_mse=mse,selected_epoch=epoch)
        model=dict(kind=kind,mean=mean,scale=scale,**best,parameter_count=sum(p.numel() for p in net.parameters()),training_examples=len(y))
        write_json(directory/(kind+'_curve.json'),curve)
    save_checkpoint(directory/(kind+'.pt'),model);return model

def predict(model,x):
    if len(x)==0:return np.zeros(0)
    x=(ablate(x,model['kind'])-model['mean'])/model['scale']
    if model['kind']=='stem_linear':return np.column_stack((x,np.ones(len(x))))@model['coef']
    net=nn.Sequential(nn.Linear(18,32),nn.GELU(),nn.Linear(32,1),nn.Tanh());net.load_state_dict(model['state']);net.eval()
    with torch.no_grad():return net(torch.from_numpy(x)).flatten().numpy()

def estimates(model,ex):
    if model['kind']=='pair_mlp':
        pairs=ex['groups']['pair'];raw=predict(model,pairs['x']);mapping={tuple((p['added'] or p['removed'])[0]):predict_changes(p,r)[0] for p,r in zip(pairs['packets'],raw)}
        packets=ex['groups']['stem']['packets'];tp=[sum(mapping[tuple(edge)] for edge in p['added']+p['removed']) for p in packets]
        return packets,[(max(-len(p['removed']),min(len(p['added']),dt)),len(p['added'])-len(p['removed'])-max(-len(p['removed']),min(len(p['added']),dt))) for p,dt in zip(packets,tp)]
    group='atomic' if model['kind']=='atomic_mlp' else 'stem';data=ex['groups'][group]
    return data['packets'],[predict_changes(p,r) for p,r in zip(data['packets'],predict(model,data['x']))]

def candidate_prediction(method,choice,ex,estimated=None):
    if method=='identity':return ex['base']
    if method=='all_edits':return ex['proposal']
    if method in ('margin_rule','physics_rule'):
        data=ex['groups']['stem'];index=12 if method=='margin_rule' else 13
        return apply_packets(ex['base'],data['packets'],[float(x[index])>choice['cut'] for x in data['x']])
    packets,changes=estimated
    return apply_packets(ex['base'],packets,[dt-choice['lambda']*df-choice['cost']*(len(p['added'])+len(p['removed']))>0 for p,(dt,df) in zip(packets,changes)])

def profile(ids,records,ex,predictions):
    families={}
    for sid,pred in zip(ids,predictions):
        r=records[sid];families.setdefault(r['family'],[]).append((score(r['reference'],pred,len(r['sequence']))['f1'],edits(ex[sid]['base'],pred,r['reference'])['harmful_rate']))
    return dict(f1=float(np.mean([np.mean([r[0] for r in v]) for v in families.values()])),harm=float(np.mean([np.mean([r[1] for r in v]) for v in families.values()])))

def choose_profiles(profiles):return max(profiles,key=lambda r:(r['f1'],-r['harm'],r.get('cost',0),r.get('cut',0),-r.get('lambda',0)))

def fit(case,smoke=False):
    torch.set_num_threads(RUNTIME['threads']);set_seed(17);ident=identity(case);role=ident['role'];directory=STAGE/('smoke' if smoke else 'formal')/case;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'choices_sealed.json').exists():raise FileExistsError('Existing experiment; do not retrain')
    ids=sorted(set(role['proposal_fit']+role['utility_fit']+role['utility_cal']))
    if smoke:
        role={k:v[:4] for k,v in role.items()};ids=sorted(set(role['proposal_fit']+role['utility_fit']+role['utility_cal']))
    records=load_records(ids);base=frozen_base(case,ids,records,torch.device('cpu'));device=torch.device(RUNTIME['device'])
    if device.type=='cuda':assert torch.cuda.is_available()
    set_seed(17)
    model=LocalCorrection('local_gate').to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001);initial=state_hash(model.state_dict());curve=[]
    for epoch in range(1,(2 if smoke else 15)+1):
        curve.append(dict(epoch=epoch,**train_one_epoch(model,opt,role['proposal_fit'],records,base,17,epoch,config(),device,.05)))
        print('UTILITY_PROPOSAL_EPOCH',case,epoch,curve[-1]['n_steps'],flush=True)
    save_checkpoint(directory/'proposal.pt',dict(identity=ident,smoke=smoke,model=model.state_dict(),initialization_hash=initial,curve=curve))
    outputs,_=infer(model,ids,records,base,device);ts=thresholds(config()['thresholds']);bp={s:base[s].numpy() for s in ids}
    bchoice=calibrate(records,role['proposal_fit'],{s:probabilities(bp[s]) for s in role['proposal_fit']},ts)
    pchoice=calibrate(records,role['proposal_fit'],{s:probabilities(outputs[s]) for s in role['proposal_fit']},ts)
    ex=examples(ids,records,base,outputs,bchoice['threshold'],pchoice['threshold'])
    if smoke:
        packet_counts={part:sum(len(ex[s]['groups']['stem']['packets']) for s in role[part]) for part in ('proposal_fit','utility_fit','utility_cal')}
        write_json(directory/'smoke.json',dict(passed=True,proposal_epochs=2,gradients_verified=all(r['gradients_verified'] for r in curve),
            packet_counts=packet_counts,utility_training='Not attempted in tiny smoke; an empty proposal is a valid inference case, not a scientific performance result.',source=source()))
        print('UTILITY_SMOKE_COMPLETE',case,json.dumps(packet_counts),flush=True);return
    models={kind:fit_predictor(kind,role['utility_fit'],role['utility_cal'],records,ex,directory) for kind in KINDS};profiles={};choices={}
    cal=role['utility_cal'];estimated={kind:{s:estimates(model,ex[s]) for s in cal} for kind,model in models.items()}
    for method in ('identity','all_edits','margin_rule','physics_rule')+KINDS:
        options=[{}] if method in ('identity','all_edits') else [dict(cut=c) for c in RULE_CUTS] if method.endswith('_rule') else [dict(lambda_=l,cost=c) for l in LAMBDAS for c in COSTS]
        options=[{('lambda' if k=='lambda_' else k):v for k,v in o.items()} for o in options]
        profiles[method]=[dict(choice=o,**profile(cal,records,ex,[candidate_prediction(method,o,ex[s],estimated.get(method,{}).get(s)) for s in cal])) for o in options]
        choices[method]=max(profiles[method],key=lambda r:(r['f1'],-r['harm'],r['choice'].get('cost',0),r['choice'].get('cut',0),-r['choice'].get('lambda',0)))
    # Strong ordinary score controls get the same expanded thresholds and calibration RNA.
    for method in ('calibrated_base','fixed_mix'):
        grid=[]
        for alpha in ((0.,) if method=='calibrated_base' else (0.,.25,.5,.75,1.)):
            by_t={t:[] for t in ts}
            for sid in cal:
                p=np.clip(records[sid]['physical'][0].numpy(),1e-4,1-1e-4);v=(1-alpha)*bp[sid]+alpha*np.log(p/(1-p))
                for t,pred in greedy_all_thresholds(probabilities(v),records[sid]['sequence'],ts).items():by_t[t].append(pred)
            grid.extend(dict(choice=dict(alpha=alpha,threshold=t),**profile(cal,records,ex,preds)) for t,preds in by_t.items())
        choices[method]=max(grid,key=lambda r:(r['f1'],-r['harm'],-r['choice']['alpha'],-r['choice']['threshold']));profiles[method]=grid
    assert ident==identity(case)
    sealed=dict(identity=ident,smoke=smoke,base_threshold=bchoice['threshold'],proposal_threshold=pchoice['threshold'],choices=choices,profiles=profiles,
        proposal_sha256=file_sha(directory/'proposal.pt'),model_files={k:file_sha(directory/(k+'.pt')) for k in KINDS},
        main_candidate='stem_mlp',training_rows={k:models[k]['training_examples'] for k in KINDS},parameters={k:models[k]['parameter_count'] for k in KINDS},sealed_unix=time.time(),job_id=os.environ.get('SLURM_JOB_ID'))
    sealed['sha256']=digest(sealed);write_json(directory/'choices_sealed.json',sealed);print('UTILITY_FIT_SEALED',case,smoke,flush=True)

def seal(case):
    p=STAGE/'formal'/case;row=json.loads((p/'choices_sealed.json').read_text());claimed=row.pop('sha256')
    assert digest(row)==claimed and row['identity']==identity(case) and not row['smoke'];assert file_sha(p/'proposal.pt')==row['proposal_sha256']
    assert all(file_sha(p/(k+'.pt'))==h for k,h in row['model_files'].items());row['sha256']=claimed;return row

def evaluate(case):
    for other in ('srp','tmRNA'):seal(other)
    torch.set_num_threads(RUNTIME['threads']);set_seed(17);sealed=seal(case);ids=sealed['identity']['role']['development'];records,embeddings=load_records(ids,True)
    directory=STAGE/'formal'/case;device=torch.device(RUNTIME['device']);b=B0().to(device)
    path=OUT/'pilot'/case/'base_b_seed17/final.pt';assert file_sha(path)==sealed['identity']['base_checkpoint_sha256']
    b.load_state_dict(torch.load(path,map_location=device,weights_only=False)['model']);b.eval();base={}
    with torch.no_grad():
        for sid in ids:base[sid]=b(*model_inputs(records[sid],embeddings[sid],device)).float().cpu()
    model=LocalCorrection('local_gate').to(device);model.load_state_dict(torch.load(directory/'proposal.pt',map_location=device,weights_only=False)['model'])
    outputs,_=infer(model,ids,records,base,device);ex=examples(ids,records,base,outputs,sealed['base_threshold'],sealed['proposal_threshold'])
    models={k:torch.load(directory/(k+'.pt'),map_location='cpu',weights_only=False) for k in KINDS};predictions={m:[] for m in sealed['choices']}
    for sid in ids:
        for method,entry in sealed['choices'].items():
            choice=entry['choice']
            if method in ('calibrated_base','fixed_mix'):
                p=np.clip(records[sid]['physical'][0].numpy(),1e-4,1-1e-4);v=(1-choice['alpha'])*base[sid].numpy()+choice['alpha']*np.log(p/(1-p))
                pred=greedy_all_thresholds(probabilities(v),records[sid]['sequence'],[choice['threshold']])[choice['threshold']]
            else:pred=candidate_prediction(method,choice,ex[sid],estimates(models[method],ex[sid]) if method in KINDS else None)
            predictions[method].append(metric_row(sid,records[sid],pred,ex[sid]['base']))
    assert sealed==seal(case);write_json(directory/'predictions.json',predictions)
    write_json(directory/'summary.json',dict(completed=True,case=case,n=len(ids),source=source(),choices_sha256=file_sha(directory/'choices_sealed.json'),predictions_sha256=file_sha(directory/'predictions.json'),
        methods={m:aggregate(v) for m,v in predictions.items()},official_verification='pending',job_id=os.environ.get('SLURM_JOB_ID')))
    print('UTILITY_DEVELOPMENT_COMPLETE',case,len(ids),flush=True)

def report():
    rows=[];verified=0;sources={};per_case={}
    for case in ('srp','tmRNA'):
        selected=seal(case);p=STAGE/'formal'/case;summary=json.loads((p/'summary.json').read_text());predictions=json.loads((p/'predictions.json').read_text())
        assert summary['source']==source() and summary['completed'] and file_sha(p/'predictions.json')==summary['predictions_sha256'] and file_sha(p/'choices_sealed.json')==summary['choices_sha256']
        ids=selected['identity']['role']['development'];records=load_records(ids);base={r['sample_id']:r['prediction'] for r in predictions['identity']}
        for method,values in predictions.items():
            assert len(values)==len(ids) and {r['sample_id'] for r in values}==set(ids);verify_records(values);verified+=len(values)
            for r in values:
                rebuilt=metric_row(r['sample_id'],records[r['sample_id']],set(map(tuple,r['prediction'])),set(map(tuple,base[r['sample_id']])))
                for key in ('exact_f1','flexible_f1','helpful','harmful','harmful_rate'):assert abs(r[key]-rebuilt[key])<1e-10
            actual=aggregate(values);assert actual==summary['methods'][method]
            rows.append(dict(case=case,method=method,n=len(ids),**actual['equal_family']))
        controls=('identity','all_edits','margin_rule','physics_rule','calibrated_base','fixed_mix','stem_linear')
        control=max(controls,key=lambda m:(selected['choices'][m]['f1'],-selected['choices'][m]['harm']))
        per_case[case]=dict(control=control,candidate=summary['methods']['stem_mlp']['equal_family'],baseline=summary['methods'][control]['equal_family'])
        sources[case]=dict(summary_sha256=file_sha(p/'summary.json'),choices_sha256=file_sha(p/'choices_sealed.json'),predictions_sha256=file_sha(p/'predictions.json'),roles=selected['identity']['role'],parameters=selected['parameters'])
    gains={case:v['candidate']['exact_f1']-v['baseline']['exact_f1'] for case,v in per_case.items()};harm=np.mean([v['candidate']['harmful_rate']-v['baseline']['harmful_rate'] for v in per_case.values()])
    decision=dict(promote=bool(np.mean(list(gains.values()))>=.005 and min(gains.values())>=-.01 and harm<=0),mean_gain=float(np.mean(list(gains.values()))),per_case_gain=gains,mean_harm_difference=float(harm),per_case=per_case,
        scope='Development compute gate for the predeclared stem_mlp, not significance, an OOD guarantee or a core admission override.')
    target_dir=ROOT/'reports/direction_a_stage2_2026-10-23';target_dir.mkdir(exist_ok=True);write_csv(target_dir/'utility_metrics.csv',rows)
    write_json(target_dir/'utility_verification.json',dict(passed=True,official_predictions_verified=verified,source=source(),artifacts=sources,decision=decision))
    write_json(STAGE/'promotion.json',decision);print('UTILITY_REPORT_COMPLETE',verified,json.dumps(decision),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('fit','smoke','evaluate','report'));p.add_argument('--case',choices=('srp','tmRNA'));a=p.parse_args()
    report() if a.action=='report' else fit(a.case,a.action=='smoke') if a.action in ('fit','smoke') else evaluate(a.case)
