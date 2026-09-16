import hashlib, json, os, random
import numpy as np
import torch
import torch.distributed as dist


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def hash_jsonable(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def init_distributed():
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    if world > 1:
        torch.cuda.set_device(local)
        dist.init_process_group(backend='nccl')
    return rank, local, world


def barrier():
    if dist.is_available() and dist.is_initialized(): dist.barrier()


def all_reduce_long(values, device):
    t = torch.tensor(values, dtype=torch.long, device=device)
    if dist.is_available() and dist.is_initialized(): dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return [int(x) for x in t.tolist()]


def all_reduce_float(value, device):
    t = torch.tensor([float(value)], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized(): dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier(); dist.destroy_process_group()
