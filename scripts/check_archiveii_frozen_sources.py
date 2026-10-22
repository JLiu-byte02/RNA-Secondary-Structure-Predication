"""Check installed numerical RiNALMo modules against the pinned official Git blobs."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.archiveii_frozen_data import OUT,write_json,sha

def verify():
    import rinalmo
    package=Path(next(iter(rinalmo.__path__))).resolve()
    reference=json.loads((ROOT/'setup/archiveii_frozen_official_modules.json').read_text())
    rows={}
    for name,expected in reference['modules'].items():
        relative=Path(name).relative_to('rinalmo')
        path=ROOT/'third_party/rinalmo_frozen_reference/sec_struct.py' if name.endswith('/utils/sec_struct.py') else package/relative
        content=path.read_bytes();actual=hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()
        assert actual==expected,'Installed numerical source differs from official commit: '+name
        rows[name]=dict(git_blob_sha1=actual,sha256=sha(path))
    embedding=json.loads((OUT/'embedding_provenance.json').read_text())
    for name,row in rows.items():
        if name.endswith('/utils/sec_struct.py'):continue
        assert embedding['rinalmo_source_files'][Path(name).relative_to('rinalmo').as_posix()]==row['sha256']
    return dict(passed=True,official_commit=reference['commit'],files=rows,
        embedding_provenance_sha256=sha(OUT/'embedding_provenance.json'))

if __name__=='__main__':
    record=verify();write_json(OUT/'official_source_verification.json',record)
    print(json.dumps(dict(passed=True,official_numerical_modules=len(record['files']))))
