"""Training checkpoints (resume) and half-precision inference weights."""
import os
from dataclasses import asdict
from typing import List, Optional

import torch

from ..config import Config
from ..runtime import Runtime


def atomic_torch_save(obj, path: str):
    """Write-then-rename: never leaves a half-written file, and replaces (instead of writing
    through) a symlink that link_previous_outputs created to a read-only /kaggle/input file."""
    tmp = path + ".tmp"
    if os.path.lexists(tmp):   # leftover from a killed run (possibly a link)
        os.remove(tmp)
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(path, raw_model, muon, adam, scaler, step: int, cfg: Config):
    obj = dict(model=raw_model.state_dict(), muon=muon.state_dict(), adam=adam.state_dict(),
               scaler=scaler.state_dict() if scaler.is_enabled() else None, step=step, config=asdict(cfg))
    atomic_torch_save(obj, path)


def load_checkpoint(path, raw_model, muon, adam, scaler, rt: Runtime) -> int:
    ckpt = torch.load(path, map_location=rt.device, weights_only=False)
    raw_model.load_state_dict(ckpt["model"])
    muon.load_state_dict(ckpt["muon"])
    adam.load_state_dict(ckpt["adam"])
    if ckpt.get("scaler") and scaler.is_enabled():
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt["step"])


def save_inference_weights(path: str, raw_model, cfg: Config, tokenizer_file: str, rt: Runtime):
    half = rt.amp_dtype or torch.float32
    keep_fp32 = ("expert_bias",)   # routing biases: rounding them would change top-k expert choice
    state = {k: (v.to(half) if v.is_floating_point() and not k.endswith(keep_fp32) else v)
             for k, v in raw_model.state_dict().items()}
    atomic_torch_save(dict(model=state, config=asdict(cfg), tokenizer=tokenizer_file), path)


def load_model_weights(raw_model, path: str):
    """Loads weights from a training checkpoint or an inference-weights file."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw_model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)


def first_existing(paths: List[Optional[str]]) -> Optional[str]:
    return next((p for p in paths if p and os.path.exists(p)), None)
