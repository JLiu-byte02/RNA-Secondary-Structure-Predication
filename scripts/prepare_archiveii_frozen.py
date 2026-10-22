"""Generate fresh generic frozen representations once, shared by all nine folds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts._common import set_seed
from scripts.archiveii_frozen_data import OUT,load_manifest,normalized_sequence,sha,write_json
from scripts.archiveii_frozen_protocol import PRETRAINED_SHA
from rinalmo.config import model_config
from rinalmo.data.alphabet import Alphabet
from rinalmo.model.model import RiNALMo

def rinalmo_source_files():
    import rinalmo
    package=Path(next(iter(rinalmo.__path__))).resolve()
    return {p.relative_to(package).as_posix():sha(p) for p in sorted(package.rglob('*.py'))}

def main():
    started=time.time();set_seed(17)
    manifest=load_manifest();assert manifest['passed']
    weight=ROOT/'weights/rinalmo_giga_pretrained.pt';assert sha(weight)==PRETRAINED_SHA
    implementation=sha(Path(__file__));manifest_sha=sha(OUT/'data_manifest.json.gz')
    directory=ROOT/'embeddings/archiveii_frozen';directory.mkdir(parents=True,exist_ok=True)
    chunks=directory/'chunks';chunks.mkdir(exist_ok=True)
    sequences=sorted({normalized_sequence(row['sequence']) for row in manifest['records'].values()})
    if any(set(s)-set('ACGU') for s in sequences):raise ValueError('Unrecognized sequence alphabet: do not silently filter RNAs')
    cached={}
    for path in sorted(chunks.glob('*.pt')):
        value=torch.load(path,map_location='cpu',weights_only=False)
        assert value['weight_sha256']==PRETRAINED_SHA and value['implementation_sha256']==implementation
        assert value['data_manifest_sha256']==manifest_sha
        for key,embedding in value['embeddings'].items():
            assert key not in cached;cached[key]=embedding
    config=model_config('giga');model=RiNALMo(config)
    model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=False))
    model=model.cuda().eval().requires_grad_(False);alphabet=Alphabet(**config['alphabet'])
    pending=[s for s in sequences if hashlib.sha256(s.encode()).hexdigest() not in cached]
    batch={}
    for index,sequence in enumerate(pending,1):
        key=hashlib.sha256(sequence.encode()).hexdigest()
        tokens=torch.tensor([alphabet.encode(sequence)],dtype=torch.int64,device='cuda')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            embedding=model(tokens)['representation'][0,1:-1].half().cpu()
        assert embedding.shape==(len(sequence),1280) and torch.isfinite(embedding).all()
        cached[key]=batch[key]=embedding
        if index%100==0 or index==len(pending):
            path=chunks/f'{len(cached):06d}.pt';temporary=path.with_suffix('.tmp')
            torch.save(dict(weight_sha256=PRETRAINED_SHA,implementation_sha256=implementation,
                data_manifest_sha256=manifest_sha,embeddings=batch),temporary);temporary.replace(path);batch={}
            print('ARCHIVEII_EMBED',len(cached),'/',len(sequences),'seconds',round(time.time()-started,1),flush=True)
    assert set(cached)=={hashlib.sha256(s.encode()).hexdigest() for s in sequences}
    path=directory/'all.pt';temporary=path.with_suffix('.tmp')
    torch.save({sid:cached[hashlib.sha256(normalized_sequence(row['sequence']).encode()).hexdigest()]
        for sid,row in manifest['records'].items()},temporary);temporary.replace(path)
    source_files=rinalmo_source_files()
    write_json(OUT/'embedding_provenance.json',dict(passed=True,n=3865,unique_sequences=len(sequences),
        frozen=True,supervised_weights_loaded=False,weight_sha256=PRETRAINED_SHA,
        implementation_sha256=implementation,data_manifest_sha256=manifest_sha,
        embeddings_sha256=sha(path),precision='BF16 inference / FP16 CPU storage',special_tokens_removed='CLS_EOS',
        rinalmo_source_files=source_files,python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),slurm_job_id=os.environ.get('SLURM_JOB_ID'),seconds=time.time()-started))
    print('ARCHIVEII_FROZEN_EMBEDDINGS_COMPLETE',flush=True)

if __name__=='__main__':main()
