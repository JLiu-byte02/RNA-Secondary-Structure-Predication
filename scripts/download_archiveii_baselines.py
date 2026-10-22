"""Download only public official checkpoints; verify Zenodo MD5 and local SHA256."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.archiveii_data import FAMILIES,sha,write_json
DEST=ROOT/'weights/archiveii';DEST.mkdir(parents=True,exist_ok=True)

def download(entry):
    path=DEST/entry['key'];expected=entry['checksum'].removeprefix('md5:')
    def valid(p):
        if not p.is_file() or p.stat().st_size!=int(entry['size']):return False
        h=hashlib.md5()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        return h.hexdigest()==expected
    if not valid(path):
        size=int(entry['size']);chunk_size=16*1024*1024
        parts=DEST/'download_parts'/path.name;parts.mkdir(parents=True,exist_ok=True)
        ranges=[(i,start,min(size,start+chunk_size)-1) for i,start in enumerate(range(0,size,chunk_size))]
        # Retain fully received prefix chunks from the earlier single-stream attempt.
        old=path.with_suffix('.part')
        if old.exists():
            with old.open('rb') as source:
                for i,start,end in ranges:
                    if end>=old.stat().st_size:break
                    target=parts/f'{i:05d}.bin'
                    if target.exists():continue
                    source.seek(start);content=source.read(end-start+1)
                    assert len(content)==end-start+1;target.write_bytes(content)
        def get_part(item):
            i,start,end=item;target=parts/f'{i:05d}.bin';temporary=target.with_suffix('.part');length=end-start+1
            if target.exists() and target.stat().st_size==length:return target
            for attempt in range(8):
                try:
                    offset=temporary.stat().st_size if temporary.exists() else 0
                    if offset<length:
                        left=start+offset
                        request=urllib.request.Request(entry['links']['self'],headers={'User-Agent':'RNA-ThermoFusion-repro/1.0','Range':f'bytes={left}-{end}'})
                        with urllib.request.urlopen(request,timeout=90) as response:
                            assert response.status==206 and response.headers.get('Content-Range')==f'bytes {left}-{end}/{size}','Unverified range response'
                            with temporary.open('ab') as f:
                                for content in iter(lambda:response.read(1024*1024),b''):f.write(content)
                    if temporary.stat().st_size!=length:raise RuntimeError('Incomplete range')
                    temporary.replace(target);return target
                except Exception as e:
                    print('RANGE_RETRY',path.name,i,attempt+1,type(e).__name__,str(e),flush=True)
                    if attempt==7:raise
                    time.sleep(min(30,2**attempt))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for count,_ in enumerate(pool.map(get_part,ranges),1):
                if count%20==0:print('CHECKPOINT_PARTS',path.name,count,'/',len(ranges),flush=True)
        part=path.with_suffix('.assembled')
        with part.open('wb') as destination:
            for i,start,end in ranges:
                with (parts/f'{i:05d}.bin').open('rb') as source:
                    for content in iter(lambda:source.read(1024*1024),b''):destination.write(content)
        if not valid(part):raise RuntimeError('Assembled checkpoint MD5 mismatch: '+path.name)
        part.replace(path)
    row=dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,md5=expected,sha256=sha(path),url=entry['links']['self'])
    write_json(DEST/(path.name+'.provenance.json'),row)
    print('VERIFIED_OFFICIAL_CHECKPOINT',path.name,flush=True);return row

def main():
    record_path=DEST/'zenodo_record.json'
    if not record_path.exists():
        with urllib.request.urlopen('https://zenodo.org/api/records/15043668',timeout=90) as response:record_path.write_bytes(response.read())
    record=json.loads(record_path.read_text());entries={e['key']:e for e in record['files']}
    selected=[entries[f'rinalmo_giga_ss_archiveII-{family}_ft.pt'] for family in FAMILIES]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(download,selected))
    write_json(ROOT/'outputs/archiveii_ood/baseline_download_manifest.json',dict(passed=True,zenodo_record='https://zenodo.org/records/15043668',checkpoints=rows))

if __name__=='__main__':main()
