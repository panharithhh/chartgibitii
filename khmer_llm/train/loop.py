"""Shared training loop for pretraining and SFT: accumulation, clipping, MoE balancing, eval, resume."""
import math
import os
import time
from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

from ..config import Config
from ..model.layers import MoE
from ..model.lm import KimiMoELM
from ..optim.hybrid import build_optimizers, lr_mult
from ..runtime import Runtime, autocast_ctx, hours_left, make_scaler, time_left
from .checkpoint import load_checkpoint, save_checkpoint


def update_router_bias(moes: List[MoE], cfg: Config, rt: Runtime) -> Optional[torch.Tensor]:
    """All-reduce expert loads, nudge biases toward balance, return load fractions [layers, E]."""
    if not moes:
        return None
    counts = torch.stack([m.load_counts for m in moes])
    if rt.ddp:
        dist.all_reduce(counts)
    for m, c in zip(moes, counts):
        m.update_bias(cfg.bias_update_rate, c)
        m.load_counts.zero_()
    return counts / counts.sum(dim=-1, keepdim=True).clamp(min=1)


def train_step(model, raw_model, muon, adam, scaler, loader, batch, step: int, cfg: Config, rt: Runtime):
    mult = lr_mult(step, cfg)
    for opt in (muon, adam):
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * mult

    X, Y = batch
    loss_sum = torch.zeros((), device=rt.device)
    for micro in range(cfg.grad_accum_steps):
        last = micro == cfg.grad_accum_steps - 1
        with model.no_sync() if rt.ddp and not last else nullcontext():
            with autocast_ctx(rt):
                _, loss, _ = model(X, Y)
                loss = loss / cfg.grad_accum_steps
            X, Y = loader.get()                  # prefetch next batch while the GPU is busy
            scaler.scale(loss).backward()
        loss_sum += loss.detach()

    scaler.unscale_(muon)
    scaler.unscale_(adam)
    grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
    # An inf/nan anywhere makes the shared clip zero every grad, so skip BOTH optimizers together
    # (GradScaler alone would still step the optimizer whose own grads looked finite).
    if torch.isfinite(grad_norm):
        scaler.step(muon)
        scaler.step(adam)
    scaler.update()   # lowers the fp16 loss scale when unscale_ found infs
    muon.zero_grad(set_to_none=True)
    adam.zero_grad(set_to_none=True)
    load = update_router_bias(raw_model.moe_layers(), cfg, rt)
    return (X, Y), loss_sum, grad_norm, load, mult


@torch.no_grad()
def estimate_loss(raw_model, loaders: Dict[str, object], cfg: Config, rt: Runtime) -> Dict[str, float]:
    raw_model.eval()
    out = {}
    for name, loader in loaders.items():
        total = 0.0
        for _ in range(cfg.eval_iters):
            X, Y = loader.get()
            with autocast_ctx(rt):
                _, loss, _ = raw_model(X, Y)
            total += loss.item()
        out[name] = total / cfg.eval_iters
    raw_model.train()
    return out


def any_rank(flag: bool, rt: Runtime) -> bool:
    """True if flag is True on any rank (keeps DDP ranks in lockstep on stop decisions)."""
    if not rt.ddp:
        return flag
    t = torch.tensor(float(flag), device=rt.device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item() > 0


def load_summary(load: Optional[torch.Tensor]) -> str:
    if load is None:
        return ""
    return f" | expert load min {load.min().item() * 100:.1f}% max {load.max().item() * 100:.1f}%"


def wrap_model(raw_model: KimiMoELM, cfg: Config, rt: Runtime):
    model = raw_model
    if cfg.compile and rt.amp_dtype == torch.bfloat16:
        model = torch.compile(model)
    if rt.ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # broadcast_buffers=False: per-rank expert load counters must not be overwritten by rank 0
        model = DDP(model, device_ids=[rt.local_rank], broadcast_buffers=False)
    return model


def run_training(cfg: Config, rt: Runtime, model, raw_model: KimiMoELM, train_loader,
                 eval_loaders: Dict[str, object], ckpt_path: str, sample_fn, title: str) -> int:
    """Runs (or resumes) training up to cfg.max_iters. Returns the final step."""
    muon, adam = build_optimizers(raw_model, cfg, rt)
    scaler = make_scaler(rt)
    start_step = 0
    if cfg.resume and os.path.exists(ckpt_path):
        start_step = load_checkpoint(ckpt_path, raw_model, muon, adam, scaler, rt)
        if rt.master:
            print(f"[↻] Resumed {title} from {ckpt_path} at step {start_step}")
    if start_step >= cfg.max_iters:
        if rt.master:
            print(f"[✓] {title} already complete ({start_step}/{cfg.max_iters} steps).")
        return start_step
    if any_rank(not time_left(cfg), rt):
        if rt.master:
            print(f"[⏱] No time left for {title}; skipping.")
        return start_step

    tokens_per_step = cfg.batch_size * cfg.context_length * cfg.grad_accum_steps * rt.world_size
    if rt.master:
        print("\n" + "=" * 78)
        print(f"🔥 {title} | {tokens_per_step:,} tokens/step | {cfg.max_iters:,} steps "
              f"| ~{tokens_per_step * cfg.max_iters / 1e6:,.0f}M tokens | peak lr {cfg.lr:.1e}")
        print("=" * 78)

    model.train()
    batch = train_loader.get()
    t_log = time.time()
    loss_acc, n_acc, step, load = torch.zeros((), device=rt.device), 0, start_step, None
    saved_step, eta_shown = start_step, False

    while step < cfg.max_iters:
        batch, loss_sum, grad_norm, load, mult = train_step(
            model, raw_model, muon, adam, scaler, train_loader, batch, step, cfg, rt)
        step += 1
        loss_acc += loss_sum
        n_acc += 1

        if step % cfg.log_interval == 0 or step == cfg.max_iters:
            dt = time.time() - t_log
            stop = any_rank(not time_left(cfg), rt)
            if rt.master:
                print(f"step {step:5d}/{cfg.max_iters} | loss {(loss_acc / n_acc).item():.4f} "
                      f"| lr {cfg.lr * mult:.2e} | gnorm {grad_norm.item():.2f} "
                      f"| {n_acc * tokens_per_step / dt:,.0f} tok/s"
                      f"{' | scale %.0f' % scaler.get_scale() if scaler.is_enabled() else ''}"
                      f"{load_summary(load)}")
            if rt.master and not eta_shown and step - start_step >= 2 * cfg.log_interval:
                eta_shown = True   # after warm-up of the timer: how long the rest takes vs the time budget
                tok_s = n_acc * tokens_per_step / max(dt, 1e-9)
                need, left = (cfg.max_iters - step) * tokens_per_step / tok_s / 3600, hours_left(cfg)
                print(f"  [eta] ~{need:.1f} h for the remaining {cfg.max_iters - step:,} steps at {tok_s:,.0f} tok/s"
                      + (f"; only {left:.1f} h of budget left -> it will save and stop, continue next session "
                         f"with Config(import_dir=...)" if need > left else ""))
            loss_acc.zero_()
            n_acc, t_log = 0, time.time()
            if stop:
                if rt.master:
                    print(f"[⏱] Time budget of {cfg.max_train_hours}h reached; stopping {title}.")
                break

        if rt.master and (step % cfg.eval_interval == 0 or step == cfg.max_iters):
            losses = estimate_loss(raw_model, eval_loaders, cfg, rt)
            print("  [eval] " + " | ".join(f"{k} {v:.4f}" for k, v in losses.items())
                  + f" | ppl {math.exp(min(list(losses.values())[-1], 20)):.1f}")
            if load is not None:
                print("  [experts] max load per MoE layer: "
                      + " ".join(f"{v * 100:.0f}%" for v in load.max(dim=-1).values.tolist())
                      + f" (ideal {100 / cfg.n_routed_experts:.1f}%)")
            print(f"  [sample] {sample_fn()}")

        if rt.master and step % cfg.ckpt_interval == 0:
            save_checkpoint(ckpt_path, raw_model, muon, adam, scaler, step, cfg)
            saved_step = step
            print(f"  [ckpt] saved {ckpt_path} @ step {step}")

    if rt.master and step > saved_step:
        save_checkpoint(ckpt_path, raw_model, muon, adam, scaler, step, cfg)
    if rt.master and step > start_step:
        print(f"[✓] {title} stopped at step {step}. Checkpoint: {ckpt_path}")
    return step
