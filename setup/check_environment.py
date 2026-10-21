import argparse
import ctypes
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
packages = {'torch':'torch', 'numpy':'numpy', 'scipy':'scipy', 'pandas':'pandas', 'matplotlib':'matplotlib', 'seaborn':'seaborn', 'scikit-learn':'sklearn', 'biopython':'Bio', 'PyYAML':'yaml', 'tqdm':'tqdm', 'ViennaRNA':'RNA', 'einops':'einops', 'ml-collections':'ml_collections', 'gdown':'gdown', 'requests':'requests', 'packaging':'packaging', 'pyarrow':'pyarrow', 'tensorboard':'tensorboard', 'psutil':'psutil', 'pytest':'pytest', 'flash-attn':'flash_attn', 'rinalmo':'rinalmo'}
versions = {}
for distribution, module in packages.items():
    importlib.import_module(module)
    versions[distribution] = importlib.metadata.version(distribution)
from rinalmo.config import model_config
from rinalmo.model.model import RiNALMo
from rinalmo.data.alphabet import Alphabet
import numpy as np
import RNA
import torch
torch.set_num_threads(2)
assert sys.prefix == str(root / '.venv'), sys.prefix
assert RNA.__version__ == '2.6.4'
assert np.__version__ == '1.26.4'
assert torch.__version__ == '2.1.2+cu118'
assert torch.version.cuda == '11.8'
layer = torch.nn.Linear(8, 2)
output = layer(torch.ones(2, 8))
output.square().mean().backward()
assert torch.isfinite(layer.weight.grad).all()
assert np.array_equal(torch.from_numpy(np.arange(5)).numpy(), np.arange(5))
from torch.utils.tensorboard import SummaryWriter
import tempfile
with tempfile.TemporaryDirectory(dir=root/'.cache') as logdir:
    with SummaryWriter(log_dir=logdir) as writer:
        writer.add_scalar('environment_check/value', 1.0, 0)
config = model_config('giga')
alphabet = Alphabet(**config['alphabet'])
assert len(alphabet.encode('ACGU')) == 6
sequence = 'GGGAAACCC'
md = RNA.md()
md.uniq_ML = 1
fc = RNA.fold_compound(sequence, md)
structure, mfe = fc.mfe()
fc.exp_params_rescale(mfe)
fc.pf()
bpp = np.asarray(fc.bpp())
assert bpp.shape == (10, 10) and np.isfinite(bpp).all()
class Ep(ctypes.Structure):
    _fields_ = [('i', ctypes.c_int), ('j', ctypes.c_int), ('p', ctypes.c_float), ('type', ctypes.c_int)]
lib = ctypes.CDLL(RNA._RNA.__file__)
compute = lib.vrna_stack_prob
compute.argtypes = [ctypes.c_void_p, ctypes.c_double]
compute.restype = ctypes.POINTER(Ep)
release = ctypes.CDLL(None).free
release.argtypes = [ctypes.c_void_p]
release.restype = None
entries = compute(ctypes.c_void_p(int(fc.this)), 1e-8)
assert entries
stacks = []
try:
    for position in range(len(sequence) ** 2):
        entry = entries[position]
        if entry.i == 0:
            break
        i, j, p = int(entry.i), int(entry.j), float(entry.p)
        assert 0 <= p <= min(bpp[i, j], bpp[i+1, j-1]) + 1e-5
        stacks.append([i, j, p])
    else:
        raise AssertionError('Stack probability sentinel missing')
finally:
    release(ctypes.cast(entries, ctypes.c_void_p))
assert stacks
report = dict(host=socket.gethostname(), python=sys.version, executable=sys.executable, libc=platform.libc_ver(), packages=versions, cuda_runtime=torch.version.cuda, cpu_backward='passed', numpy_torch_bridge='passed', rinalmo_import='passed', vienna_bpp='passed', vienna_stack_C_API='passed', stack_count=len(stacks), gpu_validation='pending_allocation')
if args.gpu:
    assert os.environ.get('SLURM_JOB_ID'), 'GPU check requires a Slurm allocation'
    assert torch.cuda.is_available(), 'CUDA is unavailable in the allocated job'
    device = torch.cuda.get_device_name(0)
    assert torch.cuda.is_bf16_supported()
    with torch.no_grad():
        x = torch.randn(32, 32, device='cuda', dtype=torch.bfloat16)
        assert torch.isfinite(x @ x).all()
        from flash_attn import flash_attn_qkvpacked_func
        qkv = torch.randn(1, 16, 3, 2, 32, device='cuda', dtype=torch.bfloat16)
        result = flash_attn_qkvpacked_func(qkv, dropout_p=0.0, causal=False)
        assert result.shape == (1, 16, 2, 32) and torch.isfinite(result).all()
        from flash_attn.layers.rotary import RotaryEmbedding
        rotated = RotaryEmbedding(dim=32).cuda()(qkv.clone())
        assert rotated.shape == qkv.shape and torch.isfinite(rotated).all()
        torch.cuda.synchronize()
    report.update(gpu_validation='passed', gpu=device, capability=torch.cuda.get_device_capability(0), flash_attention_cuda='passed', triton_rotary_cuda='passed', slurm_job_id=os.environ['SLURM_JOB_ID'])
output_path = root / 'setup' / ('gpu_validation.json' if args.gpu else 'environment_validation.json')
output_path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
