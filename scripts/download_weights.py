"""Resumable byte-range download of the official generic pretrained weight."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://zenodo.org/api/records/15043668/files/rinalmo_giga_pretrained.pt/content'
EXPECTED = 'cd93c3f21eb3e767373c9491192686b5846247bd1110693e453c1dd0f321c0db'
SIZE = 2603787622


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--source', choices=['zenodo','google-drive'], default=os.environ.get('RNA_WEIGHT_SOURCE','zenodo'))
    args = parser.parse_args()
    directory = ROOT/'weights'
    directory.mkdir(exist_ok=True)
    destination = directory/'rinalmo_giga_pretrained.pt'
    if destination.exists() and destination.stat().st_size == SIZE and digest(destination) == EXPECTED:
        print('WEIGHT_ALREADY_VERIFIED', flush=True)
        return
    url, cookies = URL, {}
    source_url = URL
    if args.source == 'google-drive':
        from bs4 import BeautifulSoup
        mapping = json.loads((ROOT/'vendor/RiNALMo/rinalmo/resources/model2gdisk.json').read_text())
        source_url = 'https://drive.google.com/uc?export=download&id='+mapping['giga-v1']
        session = requests.Session()
        response = session.get(source_url,timeout=(15,30))
        response.raise_for_status()
        form = BeautifulSoup(response.text,'html.parser').find('form')
        if form is None:
            raise RuntimeError('Official Google Drive download confirmation unavailable')
        params = {item['name']:item.get('value','') for item in form.find_all('input') if item.get('name')}
        url = requests.Request('GET',form['action'],params=params).prepare().url
        cookies = session.cookies.get_dict()
    parts = directory/'download_parts'
    parts.mkdir(exist_ok=True)
    block_size = 8*1024*1024
    started = time.time()

    def fetch(index):
        start, end = index*block_size, min((index+1)*block_size, SIZE)-1
        path = parts/f'{index:05d}.part'
        if path.exists() and path.stat().st_size == end-start+1:
            return path
        temporary = path.with_suffix('.tmp')
        for attempt in range(10):
            try:
                received = temporary.stat().st_size if temporary.exists() else 0
                if received == end-start+1:
                    os.replace(temporary,path)
                    return path
                if received > end-start+1:
                    raise RuntimeError('Download part exceeds expected byte range')
                current_start = start+received
                with requests.get(url, headers={'Range':f'bytes={current_start}-{end}'}, cookies=cookies,
                                  stream=True, timeout=(30,90)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {current_start}-{end}/{SIZE}':
                        raise RuntimeError('Unexpected server byte-range response')
                    with temporary.open('ab') as handle:
                        for data in response.iter_content(256*1024):
                            handle.write(data)
                if temporary.stat().st_size != end-start+1:
                    raise RuntimeError('Incomplete download part')
                os.replace(temporary,path)
                return path
            except Exception:
                if attempt == 9:
                    raise
                time.sleep(min(30, 2**attempt))

    count = (SIZE+block_size-1)//block_size
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for path in pool.map(fetch, range(count)):
            completed += path.stat().st_size
            elapsed = time.time()-started
            print(json.dumps(dict(downloaded_bytes=completed,total_bytes=SIZE,seconds=round(elapsed,1),
                                  average_MBps=round(completed/max(1,elapsed)/1e6,3))), flush=True)
    temporary = destination.with_suffix('.assembling')
    with temporary.open('wb') as output:
        for index in range(count):
            with (parts/f'{index:05d}.part').open('rb') as handle:
                for block in iter(lambda:handle.read(1024*1024),b''):
                    output.write(block)
    actual = digest(temporary)
    if actual != EXPECTED:
        raise RuntimeError('Pretrained weight checksum mismatch: '+actual)
    os.replace(temporary,destination)
    (directory/'download_validation.json').write_text(json.dumps(dict(url=source_url,sha256=actual,bytes=SIZE,
                                                                    seconds=time.time()-started),indent=2))
    print('WEIGHT_DOWNLOAD_AND_SHA256_COMPLETE',actual,flush=True)


if __name__ == '__main__':
    main()
