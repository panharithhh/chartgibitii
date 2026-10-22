"""Pretraining data: stream + clean + dedup the Khmer corpus (gzipped), tokenize to uint16 bins, batch."""
import gzip
import html
import itertools
import json
import os
import random
import time
import zlib
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from ..config import Config
from ..text import text_to_lines
from .hf import maybe_hf_login, open_hf_stream, row_text
from .sources import HARVEST_SOURCES
from .tokenizer import KhmerTokenizer


def open_text(path: str, mode: str = "rt", gz: Optional[bool] = None):
    """Opens .gz transparently (the corpus is stored gzipped: Khmer is 3 bytes/char in UTF-8).
    gz overrides the suffix check, e.g. for a temp file that will be renamed to *.gz."""
    if path.endswith(".gz") if gz is None else gz:
        return gzip.open(path, mode, encoding="utf-8", **({"compresslevel": 3} if "w" in mode else {}))
    return open(path, mode, encoding="utf-8")


def iter_lines(path: str) -> Iterator[str]:
    with open_text(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                yield line


def harvest_khmer_corpus(cfg: Config) -> str:
    """
    Streams every source in HARVEST_SOURCES into one gzipped, line-deduplicated corpus until
    cfg.max_corpus_chars (or the harvest time budget) is reached. Also keeps a uniform reservoir
    sample over ALL harvested lines for tokenizer training, so the vocabulary matches the mix.
    """
    out_path, done_flag = cfg.corpus_path, cfg.corpus_path + ".done"
    if os.path.exists(out_path) and os.path.exists(done_flag):
        print(f"[✓] Corpus found at {out_path} ({os.path.getsize(out_path) / 2**30:.2f} GB gz). Skipping harvest.")
        return out_path

    maybe_hf_login()
    print(f"\n[📥] Streaming Khmer corpora from HuggingFace (budget {cfg.max_corpus_chars / 1e9:.1f}B chars, "
          f"{cfg.max_harvest_hours:g} h)...")
    rng = random.Random(cfg.seed)
    reservoir, n_seen = [], 0
    seen, total_chars, total_lines, stats = set(), 0, 0, {}
    t_harvest = time.time()
    tmp_path = out_path + ".tmp"
    with open_text(tmp_path, "wt", gz=out_path.endswith(".gz")) as f:
        for src in HARVEST_SOURCES:
            if total_chars >= cfg.max_corpus_chars or time.time() - t_harvest > cfg.max_harvest_hours * 3600:
                print(f"  -> budget reached; skipping the remaining sources (from {src['name']})")
                break
            cap = src.get("cap_chars") or float("inf")   # per-source share, so one crawl cannot eat the budget
            n_src, c_src, t0 = 0, 0, time.time()
            try:
                label, ds = open_hf_stream(src["loads"], src.get("require"))
                print(f"  -> {src['name']} [{label}]")
                for row in ds:
                    raw = row_text(row, src.get("fields"), src.get("sep", "\n"))
                    if src.get("unescape"):   # MADLAD-400 stores newlines as a literal backslash-n
                        raw = html.unescape(raw.replace("\\n", "\n"))
                    for line in text_to_lines(raw):
                        h = hash(line)
                        if h in seen:
                            continue
                        seen.add(h)
                        f.write(line + "\n")
                        n_src += 1
                        c_src += len(line)
                        n_seen += 1   # reservoir sampling (Algorithm R) for the tokenizer sample
                        if len(reservoir) < cfg.tokenizer_sample_lines:
                            reservoir.append(line)
                        else:
                            j = rng.randrange(n_seen)
                            if j < cfg.tokenizer_sample_lines:
                                reservoir[j] = line
                        if n_src % 500_000 == 0:
                            print(f"     ... {n_src:,} lines, {c_src / 1e6:,.0f}M chars ({time.time() - t0:.0f}s)")
                    if (c_src >= cap or total_chars + c_src >= cfg.max_corpus_chars
                            or time.time() - t0 > cfg.max_seconds_per_source):
                        break
            except Exception as e:   # gated / renamed / network error -> keep what we have
                print(f"  -> {src['name']}: [x] stopped: {type(e).__name__}: {str(e)[:300]}")
            total_chars += c_src
            total_lines += n_src
            stats[src["name"]] = dict(lines=n_src, chars=c_src)
            print(f"     [✓] +{n_src:,} lines / {c_src / 1e6:,.0f}M chars | total {total_chars / 1e9:.2f}B chars "
                  f"| {time.time() - t0:.0f}s")

    if total_lines == 0:
        raise RuntimeError("No Khmer text harvested. Turn on Internet in the Kaggle notebook settings.")
    with open_text(cfg.tokenizer_sample_path + ".tmp", "wt", gz=cfg.tokenizer_sample_path.endswith(".gz")) as fs:
        fs.writelines(line + "\n" for line in reservoir)
    os.replace(cfg.tokenizer_sample_path + ".tmp", cfg.tokenizer_sample_path)
    os.replace(tmp_path, out_path)
    with open(done_flag + ".tmp", "w") as fl:
        json.dump(dict(lines=total_lines, chars=total_chars, sources=stats), fl, ensure_ascii=False, indent=1)
    os.replace(done_flag + ".tmp", done_flag)
    print(f"[✓] Corpus: {total_lines:,} deduplicated lines, {total_chars / 1e9:.2f}B chars "
          f"-> {out_path} ({os.path.getsize(out_path) / 2**30:.2f} GB gz)")
    return out_path


def _flatten_with_eos(seqs: List[List[int]], eos_id: int) -> np.ndarray:
    """[[a, b], [c]] -> [a, b, EOS, c, EOS] as uint16, without a per-token Python loop."""
    if not seqs:
        return np.zeros(0, dtype=np.uint16)
    lengths = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs))
    flat = np.fromiter(itertools.chain.from_iterable(seqs), dtype=np.uint16, count=int(lengths.sum()))
    return np.insert(flat, np.cumsum(lengths), eos_id)


def prepare_token_bins(cfg: Config, tok: KhmerTokenizer):
    meta_path = cfg.train_bin + ".json"
    if os.path.exists(cfg.train_bin) and os.path.exists(cfg.val_bin) and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("tokenizer") in (None, tok.fingerprint):   # None: bins from before fingerprints existed
            print(f"[✓] Token bins found: {meta}")
            return
        print("[!] Token bins were built with a different tokenizer; rebuilding.")
    assert tok.vocab_size < 2**16, "uint16 bins need vocab < 65536"
    print("\n[🔄] Tokenizing corpus -> uint16 memmap bins...")
    n_train = n_val = 0
    with open(cfg.train_bin + ".tmp", "wb") as ft, open(cfg.val_bin + ".tmp", "wb") as fv:

        def flush(lines: List[str]):
            nonlocal n_train, n_val
            tr, va = [], []
            for line, ids in zip(lines, tok.encode_batch(lines)):
                if ids:
                    (va if zlib.crc32(line.encode("utf-8")) % cfg.val_every_n_lines == 0 else tr).append(ids)
            tr_arr, va_arr = _flatten_with_eos(tr, tok.eos_id), _flatten_with_eos(va, tok.eos_id)   # EOS ends a doc
            tr_arr.tofile(ft)
            va_arr.tofile(fv)
            n_train += len(tr_arr)
            n_val += len(va_arr)

        batch = []
        for line in tqdm(iter_lines(cfg.corpus_path), desc="Tokenizing", unit=" lines", mininterval=10):
            batch.append(line)
            if len(batch) >= 100_000:
                flush(batch)
                batch = []
        if batch:
            flush(batch)

    os.replace(cfg.train_bin + ".tmp", cfg.train_bin)
    os.replace(cfg.val_bin + ".tmp", cfg.val_bin)
    meta = dict(train_tokens=n_train, val_tokens=n_val, vocab_size=tok.vocab_size, tokenizer=tok.fingerprint)
    with open(meta_path + ".tmp", "w") as f:
        json.dump(meta, f)
    os.replace(meta_path + ".tmp", meta_path)
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
