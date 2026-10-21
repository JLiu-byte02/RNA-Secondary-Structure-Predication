"""Read-only source provenance checks; write evidence only below Astra."""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXPECTED='cd93c3f21eb3e767373c9491192686b5846247bd1110693e453c1dd0f321c0db'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def resolve_weight(root):
    return root/'weights/rinalmo_giga_pretrained.pt'


def validate_source(root, reference):
    errors, records = [], []
    reused = {'_common.py', 'pair_head.py', 'decoder.py', 'losses.py', 'metrics.py', 'utils.py', 'stem_candidates.py', '__init__.py'}
    for relative, expected in sorted(reference.items()):
        path = root/relative
        if path.name not in reused or path.suffix != '.py':
            continue
        actual = sha(path) if path.is_file() else None
        records.append(dict(path=relative, sha256=actual, original_sha256=expected, unchanged=actual==expected))
        if actual != expected:
            errors.append('Reused B0 module changed or missing: '+relative)
    return errors, records

def main():
    if Path.cwd().resolve()!=ROOT:
        raise RuntimeError('Run preflight from the project root: '+str(ROOT))
    output=ROOT/'outputs/thermo_fast/provenance'
    output.mkdir(parents=True,exist_ok=True)
    manifest={r['sample_id']:r for r in csv.DictReader((ROOT/'data/processed/manifest.tsv').open(),delimiter='\t')}
    split=json.loads((ROOT/'data/processed/split_ids.json').read_text())
    expected={'fit_train':3299,'family_dev':1068,'VL0':1286,'TS0':1288,'bprna_new':5388}
    errors=[]
    for name,n in expected.items():
        if len(split.get(name,[]))!=n: errors.append(f'{name}: {len(split.get(name,[]))} != {n}')
    reference=json.loads((ROOT/'provenance/source_snapshot.json').read_text())
    source_errors, source=validate_source(ROOT,reference)
    errors.extend(source_errors)
    indices={}
    for name in ('fit_train','family_dev','VL0'):
        index={r['sample_id']:r for r in csv.DictReader((ROOT/f'embeddings/{name}_index.tsv').open(),delimiter='\t')}
        for sid in split[name]:
            r,m=index.get(sid,{}),manifest[sid]
            checks=[r.get('sequence_hash')==m['sequence_hash'], r.get('weight_sha256')==EXPECTED,
                    r.get('length')==m['length'],r.get('embedding_length')==m['length'],
                    r.get('dimension')=='1280',r.get('layer')=='last',r.get('special_tokens_removed')=='CLS_EOS']
            if not all(checks): errors.append(f'Embedding index mismatch: {sid}')
        indices[name]={'n':len(split[name]),'index_sha256':sha(ROOT/f'embeddings/{name}_index.tsv')}
    weight=resolve_weight(ROOT)
    actual=sha(weight)
    if actual!=EXPECTED: errors.append('Generic pretrained weight hash mismatch')
    overlaps={}
    for a,b in [('fit_train','family_dev'),('fit_train','VL0'),('family_dev','VL0')]:
        ha={manifest[x]['sequence_hash'] for x in split[a]}
        hb={manifest[x]['sequence_hash'] for x in split[b]}
        overlaps[a+'__'+b]=len(ha&hb)
        if ha&hb: errors.append(f'Sequence overlap: {a}/{b}')
    # Pseudoknots are allowed by B0 greedy decoder but absent from Vienna's ensemble.
    labels={}
    for name in ('fit_train','family_dev','VL0'):
        totals={'original':0,'unreachable':0,'noncanonical':0,'short_loop':0,'crossing_pairs':0,'crossing_RNAs':0,'unknown_RNAs':0,'eligible_extra_exclusions':0}
        for sid in split[name]:
            r=manifest[sid];seq=r['sequence']
            pairs={(int(t.split('-')[0])-1,int(t.split('-')[1])-1) for t in r['original_pairs'].split(';') if t}
            eligible={(int(t.split('-')[0])-1,int(t.split('-')[1])-1) for t in r['eligible_pairs'].split(';') if t}
            allowed={(i,j) for i,j in pairs if j-i>=4 and seq[i]+seq[j] in {'AU','UA','CG','GC','GU','UG'}}
            crossings=set()
            for i,j in pairs:
                for k,l in pairs:
                    if i<k<j<l: crossings.update(((i,j),(k,l)))
            totals['original']+=len(pairs);totals['unreachable']+=len(pairs-allowed)
            totals['noncanonical']+=sum(seq[i]+seq[j] not in {'AU','UA','CG','GC','GU','UG'} for i,j in pairs)
            totals['short_loop']+=sum(j-i<4 for i,j in pairs)
            totals['crossing_pairs']+=len(crossings);totals['crossing_RNAs']+=bool(crossings)
            totals['eligible_extra_exclusions']+=len(allowed-eligible)
            totals['unknown_RNAs']+=r.get('family_id','') in ('','unknown','NA')
        labels[name]=totals
    record=dict(passed=not errors,errors=errors,counts={k:len(v) for k,v in split.items()},
                old_modules=source,embedding_indices=indices,pretrained_weight=dict(path=str(weight),sha256=actual,
                architecture='generic RiNALMo-Giga pretrained; never loads any supervised checkpoint'),
                sequence_overlaps=overlaps,label_details=labels,
                pseudoknot_policy='B0 decoder allows crossings; Vienna ensemble excludes them; original truth retained for all metrics',
                unknown_policy='unknown retained, excluded only from known-family macro and included in RNA macro',
                primary_truth='full_original_pairs',source_snapshot='provenance/source_snapshot.json',new_root=str(ROOT),
                argv=sys.argv,cwd=os.getcwd(),environment={k:v for k,v in os.environ.items() if k in ['TMPDIR','PIP_CACHE_DIR','TORCH_HOME','HF_HOME','CUDA_CACHE_PATH','PYTHONPYCACHEPREFIX']})
    (output/'preflight.json').write_text(json.dumps(record,indent=2))
    if errors: raise RuntimeError('; '.join(errors))
    print(json.dumps({'preflight':'passed','counts':record['counts'],'label_details':labels}),flush=True)

if __name__=='__main__': main()
