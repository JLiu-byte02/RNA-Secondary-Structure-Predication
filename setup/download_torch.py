import concurrent.futures
import hashlib
import html
import os
from pathlib import Path
import re
import time
import requests
from urllib.parse import urljoin, urlsplit, parse_qs
root = Path(__file__).resolve().parents[1]
name = 'torch-2.1.2+cu118-cp311-cp311-linux_x86_64.whl'
dest = root / 'setup/wheels' / name
partial = dest.with_suffix(dest.suffix + '.partial')
index = 'https://download.pytorch.org/whl/cu118/torch/'
response = requests.get(index, timeout=30)
response.raise_for_status()
links = re.findall(r'href="([^"]+)"', response.text)
link = next(html.unescape(x) for x in links if 'torch-2.1.2%2Bcu118-cp311-cp311-linux_x86_64.whl#sha256=' in x)
link = urljoin(index, link)
expected = parse_qs(urlsplit(link).fragment)['sha256'][0]
url = link.split('#')[0]
probe = requests.get(url, headers={'Range':'bytes=0-0'}, stream=True, timeout=(15,30))
assert probe.status_code == 206, probe.status_code
total = int(probe.headers['Content-Range'].split('/')[-1])
probe.close()
offset = partial.stat().st_size if partial.exists() else 0
assert 0 <= offset <= total
chunk_size = 32 * 1024 * 1024
chunks = root / '.cache/torch_download'
chunks.mkdir(exist_ok=True)
work = [(start, min(start+chunk_size,total)-1) for start in range(offset,total,chunk_size)]
def fetch(bounds):
    start, end = bounds
    path = chunks / f'{start}-{end}.part'
    for attempt in range(4):
        try:
            with requests.get(url, headers={'Range':f'bytes={start}-{end}'}, stream=True, timeout=(15,60)) as r:
                assert r.status_code == 206, r.status_code
                assert r.headers['Content-Range'] == f'bytes {start}-{end}/{total}', r.headers
                with path.open('wb') as out:
                    for block in r.iter_content(1024*1024):
                        out.write(block)
            assert path.stat().st_size == end-start+1
            return path
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt+1)
print(f'RESUME {offset}/{total}; workers=4; expected_sha256={expected}',flush=True)
started = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(fetch,w) for w in work]
    done = offset
    for future in concurrent.futures.as_completed(futures):
        done += future.result().stat().st_size
        print(f'DOWNLOAD {done}/{total} new_MBps={(done-offset)/1e6/max(.01,time.monotonic()-started):.2f}',flush=True)
with partial.open('ab') as out:
    for bounds in work:
        path = chunks / f'{bounds[0]}-{bounds[1]}.part'
        with path.open('rb') as inp:
            while block := inp.read(1024*1024):
                out.write(block)
assert partial.stat().st_size == total
h = hashlib.sha256()
with partial.open('rb') as inp:
    while block := inp.read(8*1024*1024):
        h.update(block)
assert h.hexdigest() == expected, (h.hexdigest(),expected)
os.replace(partial, dest)
(root/'setup/torch_wheel_sha256.txt').write_text(f'{expected}  {name}\n')
print('HASH_VERIFIED '+str(dest),flush=True)
