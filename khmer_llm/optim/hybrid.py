"""Hybrid Muon + AdamW setup and the shared learning-rate schedule."""
import math

import torch
import torch.nn as nn

from ..config import Config
from ..runtime import Runtime
from .muon import Muon


def split_param_groups(model: nn.Module):
    """Muon: hidden 2D/3D matrices in attention + FFN/experts. AdamW: embeddings, head, router, norms."""
    muon, adam_decay, adam_no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("layers.") and p.ndim >= 2 and ".router." not in name:
            muon.append((name, p))
        elif p.ndim >= 2:
            adam_decay.append((name, p))
        else:
            adam_no_decay.append((name, p))
    return muon, adam_decay, adam_no_decay


def build_optimizers(model: nn.Module, cfg: Config, rt: Runtime):
    muon_p, decay_p, no_decay_p = split_param_groups(model)
    muon = Muon([p for _, p in muon_p], lr=cfg.lr, momentum=cfg.muon_momentum, nesterov=True,
                weight_decay=cfg.weight_decay, ns_steps=cfg.ns_steps, ns_dtype=rt.ns_dtype)
    adam = torch.optim.AdamW(
        [{"params": [p for _, p in decay_p], "weight_decay": cfg.weight_decay},
         {"params": [p for _, p in no_decay_p], "weight_decay": 0.0}],
        lr=cfg.lr, betas=cfg.adam_betas, eps=1e-8, fused=rt.device_type == "cuda",
    )
    for opt in (muon, adam):
        for g in opt.param_groups:
            g["base_lr"] = cfg.lr
    return muon, adam


def lr_mult(step: int, cfg: Config) -> float:
    if step < cfg.warmup_iters:
        return (step + 1) / cfg.warmup_iters
    if cfg.lr_schedule == "wsd":
        decay_start = max(cfg.warmup_iters, int(cfg.max_iters * (1 - cfg.wsd_decay_frac)))
        if step < decay_start:
            return 1.0
        progress = min(1.0, (step - decay_start) / max(1, cfg.max_iters - decay_start))
        return 1.0 - (1.0 - cfg.min_lr_ratio) * progress
    progress = min(1.0, (step - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters))
    return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
