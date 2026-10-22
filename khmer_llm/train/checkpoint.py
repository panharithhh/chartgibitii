"""Training checkpoints (resume) and half-precision inference weights."""
import os
from dataclasses import asdict
from typing import List, Optional

import torch

from ..config import Config
from ..runtime import Runtime


def save_checkpoint(path, raw_model, muon, adam, scaler, step: int, cfg: Config):
    obj = dict(model=raw_model.state_dict(), muon=muon.state_dict(), adam=adam.state_dict(),
               scaler=scaler.state_dict() if scaler.is_enabled() else None, step=step, config=asdict(cfg))
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


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
    torch.save(dict(model={k: (v.to(half) if v.is_floating_point() else v)
                           for k, v in raw_model.state_dict().items()},
                    config=asdict(cfg), tokenizer=tokenizer_file), path)


def load_model_weights(raw_model, path: str):
    """Loads weights from a training checkpoint or an inference-weights file."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw_model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)


def first_existing(paths: List[Optional[str]]) -> Optional[str]:
    return next((p for p in paths if p and os.path.exists(p)), None)
