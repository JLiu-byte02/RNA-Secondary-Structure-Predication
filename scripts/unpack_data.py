"""Restore the fixed manifest from its versioned compressed copy."""
import gzip
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    path=ROOT/'data/processed/manifest.tsv'
    expected=json.loads((ROOT/'provenance/source_snapshot.json').read_text())['data/processed/manifest.tsv']
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Existing fixed manifest differs from the recorded source')
        return
    with gzip.open(path.with_suffix('.tsv.gz'),'rb') as handle:
        content=handle.read()
    if hashlib.sha256(content).hexdigest()!=expected:
        raise RuntimeError('Compressed manifest checksum mismatch')
    temporary=path.with_suffix('.tmp')
    temporary.write_bytes(content)
    os.replace(temporary,path)


if __name__=='__main__':
    main()
