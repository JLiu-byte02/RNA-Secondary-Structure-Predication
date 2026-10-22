"""Verify pinned official package bytes and the isolated fine-tuning overlay."""
import hashlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def blob_hash(path):
    value=path.read_bytes();return hashlib.sha1(f'blob {len(value)}\0'.encode()+value).hexdigest()

def verify_official_sources():
    sys.path.insert(0,str(ROOT/'third_party/rinalmo_upstream'))
    from rinalmo.model.model import RiNALMo
    manifest=json.loads((ROOT/'setup/archiveii_official_source_manifest.json').read_text())
    package=Path(inspect.getfile(RiNALMo)).resolve().parents[1]
    assert package==(ROOT/'third_party/rinalmo_upstream/rinalmo').resolve()
    for name,module in list(sys.modules.items()):
        if name.startswith('rinalmo.') and getattr(module,'__file__',None):
            assert Path(module.__file__).resolve().is_relative_to(package),('Mixed official sources',name)
    actual={}
    for row in manifest['package_files']:
        relative=Path(row['path']).relative_to('rinalmo');path=package/relative
        assert path.is_file() and blob_hash(path)==row['git_blob_sha1'],row['path']
        actual[row['path']]=row['git_blob_sha1']
    for local,upstream in manifest['reference_files'].items():
        assert blob_hash(ROOT/local)==manifest['upstream_blobs'][upstream],local
    return dict(passed=True,official_commit=manifest['commit'],package_files_verified=len(actual),
        reference_files_verified=len(manifest['reference_files']),
        package_source_sha256=hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest())

def main():
    import torch
    from scripts.archiveii_data import OUT,write_json
    required={'torch':'2.1.2+cu118','numpy':'1.26.4','flash-attn':'2.3.2','lightning':'2.2.0.post0',
        'pytorch-lightning':'2.1.3','torchmetrics':'1.2.1','ViennaRNA':'2.6.4','scikit-learn':'1.3.1'}
    actual={n:metadata.version(n) for n in required};assert actual==required,(actual,required)
    sources=verify_official_sources()
    subprocess.run([sys.executable,'-m','pip','check'],check=True)
    packages={name:metadata.version(name) for name in sorted({d.metadata['Name'] for d in metadata.distributions()})}
    write_json(OUT/'environment_verification.json',dict(passed=True,python=platform.python_version(),
        platform=platform.platform(),glibc=platform.libc_ver(),cuda=torch.version.cuda,
        prefix_isolated=Path(sys.prefix).resolve()==(ROOT/'.venv_archiveii').resolve(),
        required_packages=actual,all_packages=packages,official_sources=sources))
    assert Path(sys.prefix).resolve()==(ROOT/'.venv_archiveii').resolve()
    print('ARCHIVEII_ENVIRONMENT_AND_OFFICIAL_SOURCE_VERIFIED',sources['package_files_verified'],flush=True)

if __name__=='__main__':main()
