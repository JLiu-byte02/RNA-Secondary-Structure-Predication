"""Keep the official ArchiveII family-fold membership and CT bytes unchanged."""
import argparse
from collections import Counter, defaultdict
import concurrent.futures
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
FAMILIES=('5s','16s','23s','tRNA','srp','tmRNA','RNaseP','grp1','telomerase')
EXPECTED={'5s':1283,'16s':67,'23s':15,'tRNA':557,'srp':918,'tmRNA':462,'RNaseP':454,'grp1':74,'telomerase':35}
OUT=ROOT/'outputs/archiveii_ood'
DATA=ROOT/'data/archiveII'
ARCHIVE_SHA='0ff7209016ba1775288572794e850d5857937c391d67c9ccacfe9ae77c0188e5'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8');tmp.replace(path)

def read_ct(path):
    rows=[r.split() for r in path.read_text().splitlines() if r.strip() and not r.lstrip().startswith('#')]
    length=int(rows[0][0]);assert len(rows)==length+1,path
    sequence=''.join(r[1] for r in rows[1:])
    partners={int(r[0])-1:int(r[-2])-1 for r in rows[1:] if int(r[-2])>0}
    assert all(0<=i<length and 0<=j<length and i!=j and partners.get(j)==i for i,j in partners.items()),path
    return sequence,sorted([i,j] for i,j in partners.items() if i<j)

def normalized_sequence(sequence):return sequence.upper().replace('T','U')
def feature_path(sequence):return ROOT/'.cache/archiveii'/f'{hashlib.sha256(normalized_sequence(sequence).encode()).hexdigest()}.npz'

def prepare():
    archive=ROOT/'data/archiveii_source/ct-splits.tar.gz'
    assert sha(archive)==ARCHIVE_SHA,'Official archive checksum differs'
    DATA.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            parts=Path(member.name).parts
            if len(parts)<5 or parts[:2]!=('ct','fam-fold') or member.isdir():continue
            assert member.isfile() and '..' not in parts and parts[2] in FAMILIES
            assert parts[3] in ('train','valid','test') and parts[-1].endswith('.ct')
            destination=DATA/Path(*parts[2:]);destination.parent.mkdir(parents=True,exist_ok=True)
            content=handle.extractfile(member).read()
            if destination.exists():assert destination.read_bytes()==content,destination
            else:destination.write_bytes(content)
    records={};folds={};sequence_ids=defaultdict(list)
    # The exact same per-RNA content must occur in all folds, not only the same ID.
    for family in FAMILIES:
        for path in (DATA/family/'test').glob('**/*.ct'):
            identifier=path.stem;assert identifier not in records
            sequence,pairs=read_ct(path)
            records[identifier]=dict(id=identifier,family=family,sequence=sequence,pairs=pairs,length=len(sequence),
                ct_sha256=sha(path),ct_path=path.relative_to(ROOT).as_posix())
            sequence_ids[normalized_sequence(sequence)].append(identifier)
        assert sum(r['family']==family for r in records.values())==EXPECTED[family],family
    assert len(records)==3865
    for family in FAMILIES:
        fold={};roles={}
        for role in ('train','valid','test'):
            ids=[]
            # Same unsorted Path.glob collection rule as the official Dataset.
            for path in (DATA/family/role).glob('**/*.ct'):
                identifier=path.stem;record=records[identifier]
                assert sha(path)==record['ct_sha256'],path
                assert identifier not in roles,'ID leakage across roles'
                assert (record['family']==family)==(role=='test'),'Held-out family leakage'
                roles[identifier]=role;ids.append(identifier)
            assert ids and len(ids)==len(set(ids));fold[role]=ids
        assert set(roles)==set(records)
        fold['counts']={role:len(fold[role]) for role in ('train','valid','test')}
        fold['membership_sha256']=hashlib.sha256(json.dumps({k:sorted(fold[k]) for k in ('train','valid','test')},sort_keys=True).encode()).hexdigest()
        overlaps=[]
        for seq,ids in sequence_ids.items():
            if len({roles[i] for i in ids})>1:overlaps.append(ids)
        fold['cross_role_identical_sequences']=overlaps
        fold['identical_sequence_groups_involving_test']=[ids for ids in overlaps if 'test' in {roles[i] for i in ids}]
        folds[family]=fold
    duplicates=[ids for ids in sequence_ids.values() if len(ids)>1]
    manifest=dict(source='https://github.com/marcellszi/dl-rna/releases/download/Data/ct-splits.tar.gz',
        archive_sha256=ARCHIVE_SHA,official_commit='2c2c5c14a5ae609d8c560a5d9ca32e51e0288955',
        records=records,folds=folds,unique_sequences=len(sequence_ids),identical_sequence_groups=duplicates,
        new_length_filter=False,new_deduplication=False,passed=True)
    path=OUT/'data_manifest.json.gz';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as f:f.write(json.dumps(manifest,sort_keys=True).encode())
    write_json(OUT/'split_audit.json',dict(passed=True,archive_sha256=ARCHIVE_SHA,manifest_sha256=sha(path),
        total_test_RNAs=3865,unique_sequences=len(sequence_ids),test_counts=EXPECTED,
        fold_counts={f:v['counts'] for f,v in folds.items()},identical_sequence_groups=len(duplicates),
        cross_role_identical_sequence_groups={f:len(v['cross_role_identical_sequences']) for f,v in folds.items()},
        identical_sequence_groups_involving_test={f:len(v['identical_sequence_groups_involving_test']) for f,v in folds.items()},
        lengths_by_family={f:dict(min=min(r['length'] for r in records.values() if r['family']==f),
            max=max(r['length'] for r in records.values() if r['family']==f)) for f in FAMILIES},
        note='Membership and CT bytes unchanged; duplicate sequence groups disclosed without changing the official benchmark.'))
    print(json.dumps(dict(passed=True,records=len(records),fold_counts={f:v['counts'] for f,v in folds.items()})),flush=True)

def load_manifest():
    with gzip.open(OUT/'data_manifest.json.gz','rt',encoding='utf8') as f:return json.load(f)

def cache(workers):
    import RNA
    from scripts.cache_thermo import cache_one,PARAMS
    manifest=load_manifest();sequences={}
    for r in manifest['records'].values():sequences.setdefault(normalized_sequence(r['sequence']),[]).append(r['id'])
    tasks=[(s,ids,feature_path(s),1e-8,['bpp','stack'],str(RNA.__version__)) for s,ids in sequences.items()]
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(cache_one,tasks,chunksize=1):
            row['cache_path']=str(Path(row['cache_path']).relative_to(ROOT));results.append(row)
            if row['status']!='success':raise RuntimeError(str(row))
            if len(results)%100==0:print('PHYSICAL_CACHE',len(results),'/',len(tasks),flush=True)
    write_json(OUT/'physical_manifest.json',dict(passed=True,samples=results,params=PARAMS,stack_cutoff=1e-8,
        viennarna_version=str(RNA.__version__),source='Only each sequence; no CT labels consumed by cache_one.',
        data_manifest_sha256=sha(OUT/'data_manifest.json.gz')))
    print('ARCHIVEII_PHYSICAL_CACHE_COMPLETE',len(results),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--cache',action='store_true');p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    if a.prepare:prepare()
    if a.cache:cache(a.workers)
