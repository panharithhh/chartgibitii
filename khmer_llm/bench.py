"""Tokenizer benchmark: the same model, data and number of steps with each tokenizer, ranked by
bits per byte (BPB) on held-out text. Loss per token is NOT comparable across tokenizers (a tokenizer
with longer tokens has fewer, harder tokens), BPB is: total NLL / ln 2 / UTF-8 bytes of the text.
"""
import json
import math
import os
import time
import zlib
from dataclasses import replace
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .chat import generate
from .config import Config
from .data.bpe import load_tokenizer
from .data.corpus import TokenLoader, harvest_khmer_corpus, iter_lines, prepare_token_bins
from .model.lm import build_model
from .runtime import Runtime, autocast_ctx, barrier, free_memory
from .text import COENG, KHMER_SIGN_RE
from .train.checkpoint import first_existing
from .train.loop import run_training, wrap_model

BENCH_PROMPT = "ប្រទេសកម្ពុជាគឺជា"


def tokenizer_stats(tok, lines: List[str]) -> Dict[str, float]:
    """Compression and Khmer-cluster integrity of a tokenizer on held-out lines."""
    pieces: Dict[int, str] = {}
    n_tok = n_chars = n_bytes = broken = partial = 0
    used = set()
    for line in lines:
        ids = tok.encode(line)
        n_tok += len(ids)
        n_chars += len(line)
        n_bytes += len(line.encode("utf-8"))
        used.update(ids)
        for i in ids:
            if i not in pieces:
                pieces[i] = tok.decode([i])
            piece = pieces[i]
            core = piece.strip()
            if "�" in piece or not piece:
                partial += 1        # a raw byte piece: not even a whole character
            elif core and (KHMER_SIGN_RE.match(core) or core.endswith(COENG)):
                broken += 1         # starts with a vowel sign / ends with coeng: a Khmer cluster was split
    n = max(1, n_tok)
    return dict(chars_per_token=n_chars / n, bytes_per_token=n_bytes / n, tokens_used=len(used),
                byte_piece_rate=partial / n, broken_cluster_rate=broken / n)


@torch.no_grad()
def evaluate_bpb(model, val_path: str, val_bytes: int, cfg: Config, rt: Runtime, batch_size: int = 8) -> Dict[str, float]:
    """Exact NLL over every token of the val bin (non-overlapping windows) -> bits per byte."""
    data = torch.from_numpy(np.memmap(val_path, dtype=np.uint16, mode="r").astype(np.int64))
    T, n = cfg.context_length, len(data)
    full = (n - 1) // T
    total_nll, total_tok = 0.0, 0
    model.eval()
    windows = [(s * T, T) for s in range(full)]
    if (n - 1) % T:
        windows.append((full * T, (n - 1) % T))
    for i in range(0, len(windows), batch_size):
        group = windows[i:i + batch_size]
        for length in sorted({w[1] for w in group}):   # the last window may be shorter
            starts = [s for s, l in group if l == length]
            x = torch.stack([data[s:s + length] for s in starts]).to(rt.device)
            y = torch.stack([data[s + 1:s + 1 + length] for s in starts]).to(rt.device)
            with autocast_ctx(rt):
                logits = model(x)[0]
            total_nll += F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
                                         reduction="sum").item()
            total_tok += y.numel()
    model.train()
    return dict(val_loss_per_token=total_nll / max(1, total_tok), bpb=total_nll / math.log(2) / max(1, val_bytes),
                val_tokens=total_tok)


def _held_out_lines(cfg: Config, source: str, limit: int = 3000) -> List[str]:
    out = []
    for line in iter_lines(source):
        if zlib.crc32(line.encode("utf-8")) % cfg.val_every_n_lines == 0:   # same split as the val bins
            out.append(line)
            if len(out) >= limit:
                break
    return out


def run_tokenizer_benchmark(cfg: Config, rt: Runtime) -> dict:
    kinds = [k.strip() for k in cfg.bench_tokenizers.split(",") if k.strip()]
    source = first_existing([cfg.tokenizer_sample_path, cfg.corpus_path])
    if source is None:
        if rt.master:
            harvest_khmer_corpus(cfg)
        barrier(rt)
        source = first_existing([cfg.tokenizer_sample_path, cfg.corpus_path])
    held_out = _held_out_lines(cfg, source) if rt.master else []
    if rt.master:
        print("\n" + "=" * 78)
        print(f"📏 TOKENIZER BENCHMARK: {', '.join(kinds)} | {cfg.bench_steps} steps each | text: {os.path.basename(source)}")
        print("=" * 78)

    results = {}
    for kind in kinds:
        shared = replace(cfg, tokenizer_type=kind)   # tokenizer lives in work_dir (shared with the main pipeline)
        vcfg = replace(shared, work_dir=os.path.join(cfg.work_dir, f"bench_{kind}"), import_dir=None,
                       max_iters=cfg.bench_steps, warmup_iters=min(cfg.warmup_iters, max(1, cfg.bench_steps // 10)),
                       eval_interval=max(1, cfg.bench_steps // 4), ckpt_interval=max(1, cfg.bench_steps // 2))
        if rt.master:
            os.makedirs(vcfg.work_dir, exist_ok=True)
            t0 = time.time()
            tok = load_tokenizer(shared, source)   # trains on `source` if missing
            tok_seconds = time.time() - t0
            prepare_token_bins(vcfg, tok, source=source)
        barrier(rt)
        tok = load_tokenizer(shared)

        T, B = vcfg.context_length, vcfg.batch_size
        train_loader = TokenLoader(vcfg.train_bin, B, T, rt.device, vcfg.seed + rt.rank)
        val_loader = TokenLoader(vcfg.val_bin, B, T, rt.device, vcfg.seed + 20_000)
        raw_model = build_model(vcfg, rt)
        model = wrap_model(raw_model, vcfg, rt)
        sample_fn = lambda: generate(raw_model, tok, BENCH_PROMPT, rt, max_new_tokens=40)
        t0 = time.time()
        step = run_training(vcfg, rt, model, raw_model, train_loader, {"val": val_loader}, vcfg.ckpt_path,
                            sample_fn, f"BENCH [{kind}]")
        train_seconds = time.time() - t0

        if rt.master:
            with open(vcfg.train_bin + ".json") as f:
                meta = json.load(f)
            stats = tokenizer_stats(tok, held_out)
            ev = evaluate_bpb(raw_model, vcfg.val_bin, meta["val_bytes"], vcfg, rt)
            tokens_seen = step * B * T * vcfg.grad_accum_steps * rt.world_size
            results[kind] = dict(**stats, **ev, steps=step, tokens_seen=tokens_seen,
                                 chars_seen=tokens_seen * stats["chars_per_token"], train_minutes=train_seconds / 60,
                                 tokenizer_minutes=tok_seconds / 60, corpus_tokens=meta["train_tokens"],
                                 sample=generate(raw_model, tok, BENCH_PROMPT, rt, max_new_tokens=40, temperature=0.7))
            print(f"[✓] {kind}: BPB {ev['bpb']:.4f} | {stats['chars_per_token']:.2f} chars/token")
        del model, raw_model
        free_memory()
        barrier(rt)

    if rt.master and results:
        print("\n" + "=" * 78)
        print("📏 RESULTS (same model, data and steps; lower BPB = better)")
        print("=" * 78)
        print(f"{'tokenizer':<10} {'val BPB':>8} {'chars/tok':>9} {'broken':>7} {'byte pcs':>8} "
              f"{'loss/tok':>8} {'text seen':>10} {'train':>7}")
        for kind, r in sorted(results.items(), key=lambda kv: kv[1]["bpb"]):
            print(f"{kind:<10} {r['bpb']:>8.4f} {r['chars_per_token']:>9.2f} {r['broken_cluster_rate']:>6.1%} "
                  f"{r['byte_piece_rate']:>7.1%} {r['val_loss_per_token']:>8.3f} {r['chars_seen'] / 1e6:>8.0f}Mc "
                  f"{r['train_minutes']:>5.0f}m")
        best = min(results, key=lambda k: results[k]["bpb"])
        print(f"\n🏆 Best: {best}  (use it with Config(tokenizer_type='{best}'))")
        for kind, r in results.items():
            print(f"   [{kind}] {r['sample'][:150]}")
        with open(cfg.path("bench_results.json"), "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        print(f"Saved {cfg.path('bench_results.json')}")
    return results
