"""Generate frozen generic RiNALMo-Giga embeddings with resumable chunks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._common import load_manifest, load_split, set_seed
from scripts.preflight import EXPECTED, sha
from rinalmo.config import model_config
from rinalmo.data.alphabet import Alphabet
from rinalmo.model.model import RiNALMo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', nargs='+', choices=['fit_train', 'family_dev', 'VL0'],
                        default=['fit_train', 'family_dev', 'VL0'])
    args = parser.parse_args()
    weight = ROOT/'weights/rinalmo_giga_pretrained.pt'
    actual = sha(weight)
    if actual != EXPECTED:
        raise RuntimeError('Public pretrained weight SHA256 differs from the original experiment')
    set_seed(17)
    manifest, splits = load_manifest(ROOT), load_split(ROOT)
    cfg = model_config('giga')
    model = RiNALMo(cfg)
    model.load_state_dict(torch.load(weight, map_location='cpu', weights_only=False))
    model = model.cuda().eval().requires_grad_(False)
    alphabet = Alphabet(**cfg['alphabet'])
    started = time.time()
    provenance = ROOT/'outputs/thermo_fast/provenance'
    provenance.mkdir(parents=True, exist_ok=True)
    directory = ROOT/'embeddings'
    directory.mkdir(exist_ok=True)
    implementation_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    results = {}
    for name in args.splits:
        chunk_directory = directory/'chunks'/name
        chunk_directory.mkdir(parents=True, exist_ok=True)
        cache = {}
        for chunk in sorted(chunk_directory.glob('*.pt')):
            payload = torch.load(chunk, map_location='cpu', weights_only=False)
            if payload['weight_sha256'] != actual or payload['implementation_sha256'] != implementation_hash:
                raise RuntimeError('Chunk provenance mismatch: '+str(chunk))
            cache.update(payload['embeddings'])
        missing = [sid for sid in splits[name] if sid not in cache]
        batch = {}
        for index, sid in enumerate(missing, 1):
            sequence = manifest[sid]['sequence']
            tokens = torch.tensor([alphabet.encode(sequence)], dtype=torch.long, device='cuda')
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                representation = model(tokens)['representation'][0, 1:-1].half().cpu()
            if representation.shape != (len(sequence), 1280) or not torch.isfinite(representation).all():
                raise RuntimeError('Invalid embedding: '+sid)
            cache[sid] = batch[sid] = representation
            if index % 100 == 0 or index == len(missing):
                destination = chunk_directory/f'{len(cache):06d}.pt'
                temporary = destination.with_suffix('.tmp')
                torch.save(dict(weight_sha256=actual, implementation_sha256=implementation_hash,
                                embeddings=batch), temporary)
                os.replace(temporary, destination)
                batch = {}
                print('EMBED', name, len(cache), '/', len(splits[name]),
                      'elapsed_seconds', round(time.time()-started, 1), flush=True)
        destination = directory/f'{name}.pt'
        temporary = destination.with_suffix('.tmp')
        torch.save({sid:cache[sid] for sid in splits[name]}, temporary)
        os.replace(temporary, destination)
        rows = [dict(sample_id=sid, split=name, sequence_hash=manifest[sid]['sequence_hash'],
                     length=len(manifest[sid]['sequence']), embedding_length=cache[sid].shape[0],
                     dimension=1280, weight_sha256=actual, layer='last', special_tokens_removed='CLS_EOS',
                     dtype='float16', path=str(destination)) for sid in splits[name]]
        with (directory/f'{name}_index.tsv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, list(rows[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)
        results[name] = dict(n=len(rows), sha256=sha(destination), bytes=destination.stat().st_size)
        del cache
    download_record = json.loads((ROOT/'weights/download_validation.json').read_text())
    record = dict(weight_sha256=actual, weight_source=download_record['url'],
                  frozen=True, supervised_weights_loaded=False, precision='BF16 inference, FP16 storage',
                  implementation_sha256=implementation_hash, splits=results, seconds=time.time()-started,
                  gpu=torch.cuda.get_device_name(0), peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                  torch=torch.__version__, cuda=torch.version.cuda, slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    (provenance/('embedding_completion_'+'_'.join(args.splits)+'.json')).write_text(json.dumps(record, indent=2))
    print('EMBEDDINGS_COMPLETE', json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
