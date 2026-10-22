"""Pretraining data: stream + clean + dedup the Khmer corpus, tokenize to uint16 bins, batch."""
import json
import os
import time
import zlib
from typing import List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from ..config import Config
from ..text import text_to_lines
from .hf import maybe_hf_login, open_hf_stream, row_text
from .sources import HARVEST_SOURCES
from .tokenizer import KhmerTokenizer


def harvest_khmer_corpus(cfg: Config) -> str:
    out_path, done_flag = cfg.corpus_path, cfg.corpus_path + ".done"
    if os.path.exists(out_path) and os.path.exists(done_flag):
        print(f"[✓] Corpus found at {out_path} ({os.path.getsize(out_path) / 2**20:.1f} MB). Skipping harvest.")
        return out_path

    maybe_hf_login()
    print("\n[📥] Streaming Khmer corpora from HuggingFace...")
    seen, total = set(), 0
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for src in HARVEST_SOURCES:
            budget = cfg.max_harvest_lines - total
            if budget <= 0:
                break
            if src.get("cap"):
                budget = min(budget, src["cap"])
            n_src, t0 = 0, time.time()
            try:
                label, ds = open_hf_stream(src["loads"])
                print(f"  -> {src['name']} [{label}] (budget {budget:,} lines)")
                for row in ds:
                    for line in text_to_lines(row_text(row, src.get("fields"), src.get("sep", "\n"))):
                        h = hash(line)
                        if h in seen:
                            continue
                        seen.add(h)
                        f.write(line + "\n")
                        n_src += 1
                        if n_src % 200_000 == 0:
                            print(f"     ... {n_src:,} lines ({time.time() - t0:.0f}s)")
                        if n_src >= budget:
                            break
                    if n_src >= budget or time.time() - t0 > cfg.max_seconds_per_source:
                        break
            except Exception as e:   # gated / renamed / network error -> keep what we have
                print(f"  -> {src['name']}: [x] stopped: {type(e).__name__}: {str(e)[:300]}")
            total += n_src
            print(f"     [✓] +{n_src:,} lines | total {total:,} | {time.time() - t0:.0f}s")

    if total == 0:
        raise RuntimeError("No Khmer text harvested. Turn on Internet in the Kaggle notebook settings.")
    os.replace(tmp_path, out_path)
    with open(done_flag, "w") as fl:
        fl.write(str(total))
    print(f"[✓] Saved {total:,} deduplicated lines to {out_path} ({os.path.getsize(out_path) / 2**20:.1f} MB)")
    return out_path


def prepare_token_bins(cfg: Config, tok: KhmerTokenizer):
    meta_path = cfg.train_bin + ".json"
    if os.path.exists(cfg.train_bin) and os.path.exists(cfg.val_bin) and os.path.exists(meta_path):
        print(f"[✓] Token bins found: {json.load(open(meta_path))}")
        return
    assert tok.vocab_size < 2**16, "uint16 bins need vocab < 65536"
    print("\n[🔄] Tokenizing corpus -> uint16 memmap bins...")
    n_train = n_val = 0
    with open(cfg.corpus_path, encoding="utf-8") as fin, \
         open(cfg.train_bin + ".tmp", "wb") as ft, open(cfg.val_bin + ".tmp", "wb") as fv:

        def flush(lines: List[str]):
            nonlocal n_train, n_val
            tr, va = [], []
            for line, ids in zip(lines, tok.encode_batch(lines)):
                dst = va if zlib.crc32(line.encode("utf-8")) % cfg.val_every_n_lines == 0 else tr
                dst.extend(ids)
                dst.append(tok.eos_id)          # EOS separates documents
            np.asarray(tr, dtype=np.uint16).tofile(ft)
            np.asarray(va, dtype=np.uint16).tofile(fv)
            n_train += len(tr)
            n_val += len(va)

        batch = []
        for line in tqdm(fin, desc="Tokenizing", unit=" lines", mininterval=10):
            line = line.rstrip("\n")
            if line:
                batch.append(line)
            if len(batch) >= 50_000:
                flush(batch)
                batch = []
        if batch:
            flush(batch)

    os.replace(cfg.train_bin + ".tmp", cfg.train_bin)
    os.replace(cfg.val_bin + ".tmp", cfg.val_bin)
    meta = dict(train_tokens=n_train, val_tokens=n_val, vocab_size=tok.vocab_size)
    json.dump(meta, open(meta_path, "w"))
    print(f"[✓] Train {n_train:,} tokens | Val {n_val:,} tokens")


class TokenLoader:
    """Random-window batches from a uint16 token array (memmap or in-memory)."""

    def __init__(self, data, batch_size: int, seq_len: int, device: str, seed: int):
        self.data = np.memmap(data, dtype=np.uint16, mode="r") if isinstance(data, str) else data
        assert len(self.data) > seq_len + 1, f"not enough tokens ({len(self.data)}) for seq_len {seq_len}"
        self.B, self.T, self.device = batch_size, seq_len, device
        self.pin = device.startswith("cuda")
        self.rng = np.random.default_rng(seed)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ix = self.rng.integers(0, len(self.data) - self.T - 1, size=self.B)
        buf = torch.from_numpy(np.stack([self.data[i:i + self.T + 1] for i in ix]).astype(np.int64))
        if self.pin:
            buf = buf.pin_memory().to(self.device, non_blocking=True)
        else:
            buf = buf.to(self.device)
        return buf[:, :-1], buf[:, 1:]
