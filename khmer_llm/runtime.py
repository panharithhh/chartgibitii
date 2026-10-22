"""Hardware detection, mixed precision, DDP helpers and small run-wide utilities."""
import gc
import os
import re
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist

from .config import Config

_START_TIME = time.time()
_TORCH_VERSION = tuple(int(v) for v in re.findall(r"\d+", torch.__version__)[:2])
# A dict (not a bool) so modules that imported it see the value set by setup_runtime().
RUNTIME_FLAGS = {"sdpa_gqa": False}   # native GQA in SDPA (torch>=2.5, sm>=80)


def ensure_package(module: str, pkg: Optional[str] = None):
    """Imports an optional dependency, pip-installing it first if missing (Kaggle / Colab)."""
    try:
        __import__(module)
    except ImportError:
        print(f"[*] Installing {pkg or module}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg or module])


@dataclass
class Runtime:
    device: str
    device_type: str
    amp_dtype: Optional[torch.dtype]   # None -> fp32
    use_scaler: bool
    ns_dtype: torch.dtype              # Newton-Schulz precision
    ddp: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    gpu_name: str = "cpu"

    @property
    def master(self) -> bool:
        return self.rank == 0


def setup_runtime(allow_ddp: bool = True) -> Runtime:
    ddp = allow_ddp and int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank, local_rank, world = 0, 0, 1
    if ddp:
        dist.init_process_group("nccl", timeout=timedelta(hours=4))
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    device_type = device.split(":")[0]
    if device_type == "cuda":
        major, _ = torch.cuda.get_device_capability()
        ampere_plus = major >= 8
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        RUNTIME_FLAGS["sdpa_gqa"] = ampere_plus and _TORCH_VERSION >= (2, 5)
        return Runtime(
            device=device, device_type=device_type,
            amp_dtype=torch.bfloat16 if ampere_plus else torch.float16,
            use_scaler=not ampere_plus,
            ns_dtype=torch.bfloat16 if ampere_plus else torch.float32,
            ddp=ddp, rank=rank, local_rank=local_rank, world_size=world,
            gpu_name=torch.cuda.get_device_name(),
        )
    return Runtime(device=device, device_type=device_type, amp_dtype=None, use_scaler=False,
                   ns_dtype=torch.float32, gpu_name=device)


def autocast_ctx(rt: Runtime):
    if rt.amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=rt.amp_dtype)


def make_scaler(rt: Runtime):
    try:
        return torch.amp.GradScaler("cuda", enabled=rt.use_scaler)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=rt.use_scaler)


def barrier(rt: Runtime):
    if rt.ddp:
        dist.barrier()


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def hours_left(cfg: Config) -> float:
    """Hours left of cfg.max_train_hours for the whole run (all stages)."""
    return cfg.max_train_hours - (time.time() - _START_TIME) / 3600


def time_left(cfg: Config) -> bool:
    return hours_left(cfg) > 0


def link_previous_outputs(cfg: Config):
    """Links files from a previous Kaggle run (added as notebook input) into work_dir."""
    if not cfg.import_dir or not os.path.isdir(cfg.import_dir):
        return
    for name in sorted(os.listdir(cfg.import_dir)):
        src, dst = os.path.join(cfg.import_dir, name), cfg.path(name)
        if name.endswith(".tmp"):   # half-written leftovers of a killed session
            continue
        if os.path.isfile(src) and not os.path.lexists(dst):
            os.symlink(os.path.abspath(src), dst)
            print(f"[↪] linked {name} from {cfg.import_dir}")
