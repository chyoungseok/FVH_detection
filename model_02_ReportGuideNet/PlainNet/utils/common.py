import os
import random
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_dist_avail_and_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(backend: str = "nccl") -> int:
    if is_dist_avail_and_initialized():
        return get_rank()
    dist.init_process_group(backend=backend, timeout=torch.distributed.constants.default_pg_timeout)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    return get_rank()


def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location="cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0
        self.cnt = 0
    def update(self, val, n=1):
        self.sum += float(val) * n
        self.cnt += n
    @property
    def avg(self):
        return self.sum / max(1, self.cnt)
