"""Audit all fixed full-stage cohorts and generate P/Q with the original PF code."""
import json
import multiprocessing
import sys
import time
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts._common import load_manifest, load_split
from scripts.full_protocol import audit_cohorts, SPLITS
from scripts.cache_thermo import cache_one, file_hash, sequence_hash, PARAMS, pairs_from_text, valid_pair, write_json


def main():
    import RNA
    if RNA.__version__ != '2.6.4':
        raise RuntimeError('Keep the original ViennaRNA 2.6.4')
    started=time.time()
    out=ROOT/'outputs/full_tr0'
    out.mkdir(parents=True,exist_ok=True)
    manifest,splits=load_manifest(ROOT),load_split(ROOT)
    audit=audit_cohorts(manifest,splits)
    hashes={}
    for name in SPLITS:
        path=ROOT/'embeddings/full'/f'{name}.pt'
        hashes[name]=file_hash(path)
        for sid in splits[name]:
            row=manifest[sid];seq=row['sequence']
            original=pairs_from_text(row['original_pairs'],len(seq))
            eligible=pairs_from_text(row['eligible_pairs'],len(seq))
            if eligible != {pair for pair in original if valid_pair(seq,*pair)}:
                raise ValueError('Eligible labels differ from the original candidate policy: '+sid)
    audit.update(manifest_sha256=file_hash(ROOT/'data/processed/manifest.tsv'),
                 split_ids_sha256=file_hash(ROOT/'data/processed/split_ids.json'),embedding_sha256=hashes)
    write_json(out/'split_audit.json',audit)
    grouped=defaultdict(list)
    for name in SPLITS:
        for sid in splits[name]:grouped[manifest[sid]['sequence']].append(sid)
    previous=json.loads((ROOT/'outputs/thermo_fast/thermo_cache_manifest.json').read_text())
    reusable={v['sequence_hash']:ROOT/v['cache_path'] for v in previous['samples'].values() if v['status']=='success'}
    tasks=[(seq,ids,reusable.get(sequence_hash(seq),out/'thermo_cache'/f'{sequence_hash(seq)}.npz'),
            1e-8,['bpp','stack'],RNA.__version__) for seq,ids in grouped.items()]
    results=[]
    with multiprocessing.get_context('spawn').Pool(8) as pool:
        for result in pool.imap_unordered(cache_one,tasks,chunksize=1):
            results.append(result)
            if result['status']!='success': print('CACHE_ERROR',json.dumps(result),flush=True)
            if len(results)%100==0: print('FULL_CACHE',len(results),'/',len(tasks),flush=True)
    mapping={}
    assignment={sid:name for name in SPLITS for sid in splits[name]}
    for result in results:
        for sid in result['sample_ids']:
            name=assignment[sid]
            mapping[sid]=dict(result,cache_path=str(Path(result['cache_path']).relative_to(ROOT)),split=name)
    failures=[r for r in results if r['status']!='success']
    record=dict(viennarna_version=RNA.__version__,params=PARAMS,stack_cutoff=1e-8,
                features=['bpp','stack'],samples=mapping,failures=failures,
                counts={n:len(splits[n]) for n in SPLITS},seconds=time.time()-started,
                successful_unique_sequences=len(results)-len(failures),failed_unique_sequences=len(failures))
    write_json(out/'thermo_cache_manifest.json',record)
    if failures:raise RuntimeError('Full stage requires 100% of the fixed cohorts; resolve failed caches')
    print('FULL_CACHE_COMPLETE',json.dumps(dict(counts=record['counts'],seconds=record['seconds'])),flush=True)

if __name__=='__main__':main()
