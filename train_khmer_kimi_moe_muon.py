"""
========================================================================================
KHMER LLM FROM SCRATCH — KIMI-STYLE SPARSE MoE TRANSFORMER + MUON (single file)
========================================================================================
Copy-paste into ONE Kaggle notebook cell (GPU T4 / P100 / A100 / H100, Internet ON) and run.
Optional 2x T4: save this file and run `!torchrun --nproc_per_node=2 train_khmer_kimi_moe_muon.py`

Pipeline:
1. Package installer + GPU autodetect (bf16 on sm>=80, fp16 + GradScaler on T4/P100)
2. Streaming multi-source Khmer harvester (dedup + cleaning, skips broken sources)
3. SentencePiece Unigram tokenizer (byte fallback) -> uint16 memmap token bins
4. Model:
   - Pre-norm decoder, RMSNorm, no linear biases
   - GQA (8 Q heads / 2 KV heads) + QK-Norm + RoPE (theta = 50k), SDPA / FlashAttention
   - Fine-grained MoE: 1 shared SwiGLU expert + 8 routed SwiGLU experts, top-2,
     softmax gating, renormalized weights, aux-loss-free bias balancing (DeepSeek-V3)
   - First layer dense SwiGLU (DeepSeek-style)
5. Hybrid optimizer: Muon (Newton-Schulz-5) for hidden 2D matrices, AdamW for the rest
6. Warmup + WSD/cosine LR, grad accumulation, clipping, checkpoint save/resume
7. KV-cache autoregressive sampler (temperature / top-k / top-p)
========================================================================================
"""

import os
import re
import sys
import json
import math
import time
import zlib
import tempfile
import subprocess
import unicodedata
from contextlib import nullcontext
from dataclasses import dataclass, asdict, replace
from datetime import timedelta
from typing import Optional, List, Tuple, Dict


# -----------------------------------------------------------------------------
# 1. Dependency auto-installation (Kaggle / Colab)
# -----------------------------------------------------------------------------
def _pip_install(pkg: str):
    print(f"[*] Installing {pkg}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for _mod in ["sentencepiece", "datasets", "tqdm", "numpy"]:
    try:
        __import__(_mod)
    except ImportError:
        _pip_install(_mod)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import sentencepiece as spm
from tqdm.auto import tqdm

_TORCH_VERSION = tuple(int(v) for v in re.findall(r"\d+", torch.__version__)[:2])


# -----------------------------------------------------------------------------
# 2. Configuration
# -----------------------------------------------------------------------------
@dataclass
class Config:
    # ---- model ----
    vocab_size: int = 16000            # 16000 or 32000
    context_length: int = 1024
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8                   # query heads (head_dim = d_model / n_heads = 64)
    n_kv_heads: int = 2                # GQA 4:1
    rope_theta: float = 50000.0
    norm_eps: float = 1e-6
    n_dense_layers: int = 1            # first N layers use a dense SwiGLU FFN
    n_shared_experts: int = 1
    n_routed_experts: int = 8
    top_k: int = 2
    d_ff_shared: int = 1536            # hidden size of the shared expert
    d_ff_expert: int = 384             # fine-grained routed expert = shared / 4
    bias_update_rate: float = 1e-3     # gamma for aux-loss-free load balancing
    tie_embeddings: bool = False

    # ---- optimization ----
    batch_size: int = 8                # micro-batch per GPU (fits a 16 GB T4)
    grad_accum_steps: int = 4          # 8 x 1024 x 4 = 32k tokens / step / GPU
    max_iters: int = 6000
    warmup_iters: int = 200
    lr: float = 2e-3                   # shared peak LR (Muon is RMS-matched to AdamW)
    min_lr_ratio: float = 0.1
    lr_schedule: str = "wsd"           # "wsd" (warmup-stable-decay) or "cosine"
    wsd_decay_frac: float = 0.2        # last 20% of steps decay linearly
    weight_decay: float = 0.1
    muon_momentum: float = 0.95
    ns_steps: int = 5
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    # ---- logging / eval / checkpoints ----
    log_interval: int = 20
    eval_interval: int = 500
    eval_iters: int = 40
    ckpt_interval: int = 500
    max_train_hours: float = 11.0      # Kaggle sessions stop at 12h -> save before that
    resume: bool = True
    compile: bool = False              # torch.compile (A100/H100 only; MoE loop graph-breaks)
    seed: int = 1337
    run_self_test: bool = True         # ~seconds; set KHMER_SELF_TEST_ONLY=1 to stop after it

    # ---- data ----
    work_dir: str = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    max_harvest_lines: int = 4_000_000
    max_seconds_per_source: int = 1800
    tokenizer_sample_lines: int = 2_000_000
    val_every_n_lines: int = 200       # ~0.5% of lines (hash-based) go to validation

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_ff_dense(self) -> int:
        # dense layer matches the active FFN width of an MoE layer
        return self.d_ff_shared * self.n_shared_experts + self.top_k * self.d_ff_expert

    @property
    def corpus_path(self) -> str:
        return os.path.join(self.work_dir, "khmer_corpus.txt")

    @property
    def tokenizer_prefix(self) -> str:
        return os.path.join(self.work_dir, f"khmer_sp_unigram_{self.vocab_size // 1000}k")

    @property
    def train_bin(self) -> str:
        return os.path.join(self.work_dir, f"train_{self.vocab_size}.bin")

    @property
    def val_bin(self) -> str:
        return os.path.join(self.work_dir, f"val_{self.vocab_size}.bin")

    @property
    def ckpt_path(self) -> str:
        return os.path.join(self.work_dir, "khmer_kimi_moe_latest.pt")

    @property
    def final_path(self) -> str:
        return os.path.join(self.work_dir, "khmer_kimi_moe_final.pt")


# -----------------------------------------------------------------------------
# 3. Hardware / runtime detection
# -----------------------------------------------------------------------------
_SDPA_ENABLE_GQA = False   # set by setup_runtime(); native GQA in SDPA (torch>=2.5, sm>=80)


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


def setup_runtime() -> Runtime:
    global _SDPA_ENABLE_GQA
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
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
        _SDPA_ENABLE_GQA = ampere_plus and _TORCH_VERSION >= (2, 5)
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


# -----------------------------------------------------------------------------
# 4. Multi-source Khmer dataset harvester (streaming + dedup + cleaning)
# -----------------------------------------------------------------------------
KHMER_RE = re.compile(r"[ក-៿᧠-᧿]")
ZERO_WIDTH_RE = re.compile(r"[​‌‍⁠﻿]")
CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
SPACE_RE = re.compile(r"[ \t 　]+")
SENT_END_RE = re.compile(r"(?<=[។៕?!])")
MAX_LINE_CHARS = 2000

# Order matters: curated sources first, large web crawls fill the rest of the budget.
HARVEST_SOURCES = [
    dict(name="Chuon Nath Dictionary 44k", path="seanghay/khmer-dictionary-44k",
         fields=["word", "definition"], sep=" ", cap=None),
    dict(name="Bactrian-X Khmer 67k", path="rinabuoy/Bactrian-X-km",
         fields=["instruction", "input", "output"], cap=None),
    dict(name="Alpaca Khmer", path="saillab/alpaca_khmer_taco",
         fields=["instruction", "input", "output"], cap=None),
    dict(name="CADT Khmer text pairs", path="rinabuoy/text-pairs-khmer-part-2",
         fields=None, cap=200_000),
    dict(name="FineWeb-2 khm_Khmr", path="HuggingFaceFW/fineweb-2", config="khm_Khmr",
         fields=["text"], cap=800_000),
    dict(name="khmer-lm-corpus 1.25M", path="phonsobon/khmer-lm-corpus",
         fields=["text"], cap=1_000_000),
    dict(name="khmer-pile 627k", path="sovannpanhaseng/khmer-pile",
         fields=["text"], cap=700_000),
    dict(name="khmer-text-corpus 4.9M", path="Panhapich/khmer-text-corpus",
         fields=["text"], cap=None),
]


def is_valid_khmer(s: str) -> bool:
    if len(s) < 10:
        return False
    kh = len(KHMER_RE.findall(s))
    return kh >= 5 and kh / len(s) >= 0.25


def text_to_lines(text: str) -> List[str]:
    """Clean a document and split it into paragraph-level lines (keeps '។' punctuation)."""
    text = unicodedata.normalize("NFC", str(text))
    text = CTRL_RE.sub(" ", ZERO_WIDTH_RE.sub("", text))
    out = []
    for para in text.split("\n"):
        para = SPACE_RE.sub(" ", para).strip()
        if not para:
            continue
        chunks = [para]
        if len(para) > MAX_LINE_CHARS:
            chunks, cur = [], ""
            for sent in SENT_END_RE.split(para):
                if cur and len(cur) + len(sent) > MAX_LINE_CHARS:
                    chunks.append(cur.strip())
                    cur = ""
                cur += sent
            if cur.strip():
                chunks.append(cur.strip())
        out.extend(c[:MAX_LINE_CHARS] for c in chunks if is_valid_khmer(c))
    return out


def _row_text(row: dict, fields: Optional[List[str]], sep: str) -> str:
    keys = [k for k in (fields or []) if k in row]
    if not keys:   # unknown schema -> use every string column
        keys = [k for k, v in row.items() if isinstance(v, str)]
    return sep.join(row[k] for k in keys if isinstance(row.get(k), str) and row[k].strip())


def _maybe_hf_login():
    """Picks up an HF_TOKEN Kaggle secret if present (needed only for gated datasets)."""
    if os.environ.get("HF_TOKEN"):
        return
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
    except Exception:
        pass


def harvest_khmer_corpus(cfg: Config) -> str:
    out_path, done_flag = cfg.corpus_path, cfg.corpus_path + ".done"
    if os.path.exists(out_path) and os.path.exists(done_flag):
        print(f"[✓] Corpus found at {out_path} ({os.path.getsize(out_path) / 2**20:.1f} MB). Skipping harvest.")
        return out_path

    from datasets import load_dataset
    _maybe_hf_login()
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
            print(f"  -> {src['name']} [{src['path']}] (budget {budget:,} lines)")
            n_src, t0 = 0, time.time()
            try:
                ds = load_dataset(src["path"], src.get("config"), split="train", streaming=True)
                for row in ds:
                    for line in text_to_lines(_row_text(row, src.get("fields"), src.get("sep", "\n"))):
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
                print(f"     [x] stopped: {type(e).__name__}: {str(e)[:200]}")
            total += n_src
            print(f"     [✓] +{n_src:,} lines | total {total:,} | {time.time() - t0:.0f}s")

    if total == 0:
        raise RuntimeError("No Khmer text harvested. Turn on Internet in the Kaggle notebook settings.")
    os.replace(tmp_path, out_path)
    with open(done_flag, "w") as fl:
        fl.write(str(total))
    print(f"[✓] Saved {total:,} deduplicated lines to {out_path} ({os.path.getsize(out_path) / 2**20:.1f} MB)")
    return out_path


# -----------------------------------------------------------------------------
# 5. SentencePiece tokenizer + uint16 memmap token bins
# -----------------------------------------------------------------------------
class KhmerTokenizer:
    def __init__(self, cfg: Config, corpus_path: Optional[str] = None):
        self.model_file = cfg.tokenizer_prefix + ".model"
        if not os.path.exists(self.model_file):
            assert corpus_path is not None, "tokenizer missing and no corpus to train on"
            print(f"\n[🔧] Training SentencePiece Unigram tokenizer ({cfg.vocab_size:,} vocab)...")
            spm.SentencePieceTrainer.train(
                input=corpus_path,
                model_prefix=cfg.tokenizer_prefix,
                vocab_size=cfg.vocab_size,
                model_type="unigram",
                character_coverage=0.9995,
                byte_fallback=True,
                split_digits=True,
                unk_id=0, bos_id=1, eos_id=2, pad_id=3,
                input_sentence_size=cfg.tokenizer_sample_lines,
                shuffle_input_sentence=True,
                max_sentence_length=16384,          # Khmer = 3 bytes/char in UTF-8
                num_threads=os.cpu_count() or 4,
            )
            print(f"[✓] Saved {self.model_file}")
        self.sp = spm.SentencePieceProcessor(model_file=self.model_file)
        self.vocab_size = self.sp.get_piece_size()
        self.bos_id, self.eos_id, self.pad_id = self.sp.bos_id(), self.sp.eos_id(), self.sp.pad_id()

    def encode(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        try:
            return self.sp.encode(texts, out_type=int, num_threads=os.cpu_count() or 4)
        except TypeError:   # old sentencepiece without num_threads
            return [self.sp.encode(t, out_type=int) for t in texts]

    def decode(self, ids: List[int]) -> str:
        return self.sp.decode(ids)


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


# -----------------------------------------------------------------------------
# 6. Model: GQA + QK-Norm + RoPE attention, shared + fine-grained routed MoE
# -----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """RMSNorm computed in fp32, returned in the input dtype."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


def precompute_rope(head_dim: int, max_len: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(max_len).float(), inv_freq)   # [max_len, head_dim/2]
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, D]; cos/sin: [T, D/2] (already sliced to the right positions)."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos, sin = cos.to(x.dtype)[None, None], sin.to(x.dtype)[None, None]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class GQAttention(nn.Module):
    """Grouped-query attention with per-head QK-Norm before RoPE and SDPA."""

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.n_heads, self.n_kv_heads, self.head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v) if use_cache else None   # cache stores only n_kv_heads (GQA saving)

        Tk = k.size(2)
        is_causal, attn_mask = Tk == T, None      # training / prefill -> causal; decode -> no mask
        if not is_causal and T > 1:               # chunked prefill on top of a cache
            attn_mask = torch.ones(T, Tk, dtype=torch.bool, device=x.device).tril(Tk - T)

        if _SDPA_ENABLE_GQA:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal,
                                                 enable_gqa=True)
        else:
            rep = self.n_heads // self.n_kv_heads
            if rep > 1:
                k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1)), new_kv


class SwiGLU(nn.Module):
    """down( SiLU(gate(x)) * up(x) )"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoE(nn.Module):
    """
    y = Shared(x) + sum_{i in top-k} w_i * Expert_i(x)
    - softmax router (fp32) -> top-k on (scores + bias) -> weights renormalized to sum 1
    - bias is selection-only and updated outside autograd (aux-loss-free balancing)
    - dropless dispatch: tokens sorted by expert, one matmul chain per expert
    """

    def __init__(self, cfg: Config):
        super().__init__()
        E, d, h = cfg.n_routed_experts, cfg.d_model, cfg.d_ff_expert
        self.n_experts, self.top_k = E, cfg.top_k
        self.shared = SwiGLU(d, cfg.d_ff_shared * cfg.n_shared_experts)
        self.router = nn.Linear(d, E, bias=False)
        # stacked expert weights [E, out, in]; Muon orthogonalizes each expert matrix separately
        self.w_gate = nn.Parameter(torch.empty(E, h, d))
        self.w_up = nn.Parameter(torch.empty(E, h, d))
        self.w_down = nn.Parameter(torch.empty(E, d, h))
        self.register_buffer("expert_bias", torch.zeros(E))
        self.register_buffer("load_counts", torch.zeros(E), persistent=False)

    def forward(self, x):
        B, T, D = x.shape
        xf = x.reshape(-1, D)
        N, k = xf.size(0), self.top_k

        with torch.autocast("cuda", enabled=False) if xf.is_cuda else nullcontext():
            scores = F.linear(xf.float(), self.router.weight.float()).softmax(dim=-1)   # [N, E]
        topk_idx = torch.topk(scores + self.expert_bias, k, dim=-1).indices             # bias: selection only
        topk_w = scores.gather(-1, topk_idx)
        topk_w = topk_w / topk_w.sum(-1, keepdim=True)

        flat_idx = topk_idx.reshape(-1)                                   # [N*k]
        counts = torch.bincount(flat_idx, minlength=self.n_experts)
        if self.training:
            self.load_counts += counts.detach().float()
        order = torch.argsort(flat_idx, stable=True)
        x_sorted = xf[order // k]

        outs, start = [], 0
        for e, c in enumerate(counts.tolist()):
            if c == 0:
                continue
            xe = x_sorted[start:start + c]
            start += c
            he = F.silu(F.linear(xe, self.w_gate[e])) * F.linear(xe, self.w_up[e])
            outs.append(F.linear(he, self.w_down[e]))
        y_sorted = torch.cat(outs, dim=0)
        y_flat = torch.empty_like(y_sorted)
        y_flat[order] = y_sorted                                          # back to (token, slot) order
        routed = (y_flat.view(N, k, D).float() * topk_w.unsqueeze(-1)).sum(dim=1)
        return (self.shared(xf) + routed).view(B, T, D)

    @torch.no_grad()
    def update_bias(self, gamma: float, counts: torch.Tensor):
        load = counts / counts.sum().clamp(min=1)
        self.expert_bias += gamma * torch.sign(load.mean() - load)


class Block(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GQAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff_dense) if layer_idx < cfg.n_dense_layers else MoE(cfg)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        a, new_kv = self.attn(self.attn_norm(x), cos, sin, past_kv, use_cache)
        x = x + a
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv


class KimiMoELM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        cos, sin = precompute_rope(cfg.head_dim, cfg.context_length, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._init_weights()

    def _init_weights(self):
        std = 0.02
        out_std = std / math.sqrt(2 * self.cfg.n_layers)   # residual-output projections
        for name, p in self.named_parameters():
            if p.ndim < 2:
                continue   # norm weights stay at 1
            if name.endswith(("o_proj.weight", "w_down.weight", ".w_down")):
                nn.init.normal_(p, 0.0, out_std)
            else:
                nn.init.normal_(p, 0.0, std)

    def moe_layers(self) -> List[MoE]:
        return [m for m in self.modules() if isinstance(m, MoE)]

    def forward(self, idx, targets=None, kv_cache: Optional[list] = None, last_only: bool = False):
        """kv_cache: None (no caching) or a list with one (k, v)/None entry per layer."""
        B, T = idx.shape
        use_cache = kv_cache is not None
        past = kv_cache[0][0].size(2) if use_cache and kv_cache[0] is not None else 0
        assert past + T <= self.cfg.context_length, "sequence longer than context_length"
        cos, sin = self.rope_cos[past:past + T], self.rope_sin[past:past + T]

        x = self.tok_emb(idx)
        new_cache = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            x, kv = layer(x, cos, sin, kv_cache[i] if use_cache else None, use_cache)
            if use_cache:
                new_cache.append(kv)
        x = self.norm(x)
        if targets is None and last_only:
            x = x[:, -1:]
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, new_cache

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int = 50,
                 top_p: float = 0.95, eos_id: Optional[int] = None, amp=None):
        """KV-cache sampling. When the window fills, re-prefill with the most recent half."""
        was_training = self.training
        self.eval()
        L = self.cfg.context_length
        idx = idx[:, -L:]
        with (amp if amp is not None else nullcontext()):
            logits, _, cache = self(idx, kv_cache=[None] * len(self.layers), last_only=True)
            for _ in range(max_new_tokens):
                nxt = sample_next_token(logits[:, -1].float(), temperature, top_k, top_p)
                idx = torch.cat([idx, nxt], dim=1)
                if eos_id is not None and bool((nxt == eos_id).all()):
                    break
                if cache[0][0].size(2) >= L:
                    logits, _, cache = self(idx[:, -(L // 2):], kv_cache=[None] * len(self.layers),
                                            last_only=True)
                else:
                    logits, _, cache = self(nxt, kv_cache=cache, last_only=True)
        self.train(was_training)
        return idx


def sample_next_token(logits, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        drop = probs.cumsum(dim=-1) - probs > top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return torch.multinomial(logits.softmax(dim=-1), num_samples=1)


def generate(model: KimiMoELM, tok: KhmerTokenizer, prompt: str, rt: Runtime, max_new_tokens: int = 80,
             temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95) -> str:
    ids = [tok.eos_id] + tok.encode(prompt)   # EOS = document start, as in training data
    idx = torch.tensor([ids], dtype=torch.long, device=rt.device)
    out = model.generate(idx, max_new_tokens, temperature, top_k, top_p, eos_id=tok.eos_id,
                         amp=autocast_ctx(rt))
    return tok.decode(out[0, 1:].tolist())


def count_params(model: KimiMoELM) -> Dict[str, int]:
    cfg = model.cfg
    total = sum(p.numel() for p in model.parameters())
    n_moe = cfg.n_layers - cfg.n_dense_layers
    inactive = n_moe * (cfg.n_routed_experts - cfg.top_k) * 3 * cfg.d_model * cfg.d_ff_expert
    emb = model.tok_emb.weight.numel() + (0 if cfg.tie_embeddings else model.lm_head.weight.numel())
    return dict(total=total, active=total - inactive, embedding=emb)


# -----------------------------------------------------------------------------
# 7. Muon (Newton-Schulz-5) + AdamW hybrid optimizer
# -----------------------------------------------------------------------------
def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7,
                   dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Approximately orthogonalizes G (2D or batched 3D): X_{k+1} = aX + (bA + cA^2)X, A = XX^T."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = (X / (X.norm(dim=(-2, -1), keepdim=True) + eps)).to(dtype)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Momentum (Nesterov) -> Newton-Schulz orthogonalization -> decoupled weight decay.
    Update is scaled by 0.2 * sqrt(max(m, n)) so its RMS matches AdamW (Moonlight recipe),
    which lets Muon and AdamW share one learning rate.
    """

    def __init__(self, params, lr=2e-3, momentum=0.95, nesterov=True, weight_decay=0.1,
                 ns_steps=5, ns_dtype=torch.bfloat16):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay, ns_steps=ns_steps))
        self.ns_dtype = ns_dtype

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(p.grad)
                g = p.grad.add(buf, alpha=mu) if group["nesterov"] else buf
                update = newton_schulz5(g, group["ns_steps"], dtype=self.ns_dtype)
                scale = 0.2 * math.sqrt(max(p.size(-2), p.size(-1)))
                if wd:
                    p.mul_(1.0 - lr * wd)
                p.add_(update.to(p.dtype), alpha=-lr * scale)
        return loss


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


# -----------------------------------------------------------------------------
# 8. Training step, eval, checkpoints
# -----------------------------------------------------------------------------
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
    scaler.step(muon)
    scaler.step(adam)
    scaler.update()
    muon.zero_grad(set_to_none=True)
    adam.zero_grad(set_to_none=True)
    load = update_router_bias(raw_model.moe_layers(), cfg, rt)
    return (X, Y), loss_sum, grad_norm, load, mult


@torch.no_grad()
def estimate_loss(raw_model, loaders: Dict[str, TokenLoader], cfg: Config, rt: Runtime) -> Dict[str, float]:
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


def load_summary(load: Optional[torch.Tensor]) -> str:
    if load is None:
        return ""
    return f" | expert load min {load.min().item() * 100:.1f}% max {load.max().item() * 100:.1f}%"


def train(cfg: Config, rt: Runtime, model, raw_model: KimiMoELM, tok: KhmerTokenizer):
    T, B = cfg.context_length, cfg.batch_size
    train_loader = TokenLoader(cfg.train_bin, B, T, rt.device, cfg.seed + rt.rank)
    train_eval_loader = TokenLoader(cfg.train_bin, B, T, rt.device, cfg.seed + 10_000)
    val_data = np.memmap(cfg.val_bin, dtype=np.uint16, mode="r")
    if len(val_data) <= T + 1:   # tiny corpus -> fall back to the tail of train
        val_data = np.memmap(cfg.train_bin, dtype=np.uint16, mode="r")[-(20 * T):]
    val_loader = TokenLoader(val_data, B, T, rt.device, cfg.seed + 20_000)

    muon, adam = build_optimizers(raw_model, cfg, rt)
    scaler = make_scaler(rt)
    start_step = 0
    if cfg.resume and os.path.exists(cfg.ckpt_path):
        start_step = load_checkpoint(cfg.ckpt_path, raw_model, muon, adam, scaler, rt)
        if rt.master:
            print(f"[↻] Resumed from {cfg.ckpt_path} at step {start_step}")

    tokens_per_step = B * T * cfg.grad_accum_steps * rt.world_size
    if rt.master:
        print("\n" + "=" * 78)
        print(f"🔥 TRAINING  | {tokens_per_step:,} tokens/step | {cfg.max_iters:,} steps "
              f"| ~{tokens_per_step * cfg.max_iters / 1e6:,.0f}M tokens")
        print("=" * 78)

    model.train()
    batch = train_loader.get()
    t_start = t_log = time.time()
    loss_acc, n_acc, step, load = torch.zeros((), device=rt.device), 0, start_step, None
    prompt = "ប្រទេសកម្ពុជាគឺជា"

    while step < cfg.max_iters:
        batch, loss_sum, grad_norm, load, mult = train_step(
            model, raw_model, muon, adam, scaler, train_loader, batch, step, cfg, rt)
        step += 1
        loss_acc += loss_sum
        n_acc += 1

        if step % cfg.log_interval == 0 or step == cfg.max_iters:
            dt = time.time() - t_log
            stop = torch.tensor(float((time.time() - t_start) / 3600 > cfg.max_train_hours), device=rt.device)
            if rt.ddp:
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if rt.master:
                print(f"step {step:5d}/{cfg.max_iters} | loss {(loss_acc / n_acc).item():.4f} "
                      f"| lr {cfg.lr * mult:.2e} | gnorm {grad_norm.item():.2f} "
                      f"| {n_acc * tokens_per_step / dt:,.0f} tok/s"
                      f"{' | scale %.0f' % scaler.get_scale() if scaler.is_enabled() else ''}"
                      f"{load_summary(load)}")
            loss_acc.zero_()
            n_acc, t_log = 0, time.time()
            if stop.item() > 0:
                if rt.master:
                    print(f"[⏱] Time budget of {cfg.max_train_hours}h reached; stopping.")
                break

        if rt.master and (step % cfg.eval_interval == 0 or step == cfg.max_iters):
            losses = estimate_loss(raw_model, {"train": train_eval_loader, "val": val_loader}, cfg, rt)
            print(f"  [eval] train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| val ppl {math.exp(min(losses['val'], 20)):.1f}")
            if load is not None:
                print("  [experts] max load per MoE layer: "
                      + " ".join(f"{v * 100:.0f}%" for v in load.max(dim=-1).values.tolist())
                      + f" (ideal {100 / cfg.n_routed_experts:.1f}%)")
            print(f"  [sample] {generate(raw_model, tok, prompt, rt, max_new_tokens=48)}")

        if rt.master and step % cfg.ckpt_interval == 0:
            save_checkpoint(cfg.ckpt_path, raw_model, muon, adam, scaler, step, cfg)
            print(f"  [ckpt] saved {cfg.ckpt_path} @ step {step}")

    if rt.master:
        save_checkpoint(cfg.ckpt_path, raw_model, muon, adam, scaler, step, cfg)
        half = rt.amp_dtype or torch.float32
        torch.save(dict(model={k: (v.to(half) if v.is_floating_point() else v)
                               for k, v in raw_model.state_dict().items()},
                        config=asdict(cfg), tokenizer=tok.model_file), cfg.final_path)
        print(f"\n[✓] Done at step {step}. Resume ckpt: {cfg.ckpt_path} | inference weights: {cfg.final_path}")


# -----------------------------------------------------------------------------
# 9. Self-test (tiny config, synthetic data)
# -----------------------------------------------------------------------------
def self_test(rt: Runtime):
    print("\n[🧪] Self-test (tiny model, synthetic data)...")
    t0 = time.time()
    rt = replace(rt, ddp=False)
    cfg = Config(vocab_size=128, context_length=64, d_model=128, n_layers=3, n_heads=8, n_kv_heads=2,
                 n_dense_layers=1, n_routed_experts=8, top_k=2, d_ff_shared=128, d_ff_expert=32,
                 batch_size=8, grad_accum_steps=2, max_iters=20, warmup_iters=2, lr=3e-3)
    torch.manual_seed(0)

    # 1) Newton-Schulz: output has singular values near 1, for wide / tall / batched inputs
    for shape in [(64, 96), (96, 64), (4, 32, 48)]:
        G = torch.randn(*shape, device=rt.device)
        X = newton_schulz5(G, dtype=rt.ns_dtype).float()
        assert X.shape == G.shape
        sv = torch.linalg.svdvals(X.cpu())
        assert 0.5 < sv.min() and sv.max() < 1.5, f"NS singular values out of range: {sv.min()}, {sv.max()}"

    # 2) shapes + loss
    model = KimiMoELM(cfg).to(rt.device)
    X = torch.randint(0, cfg.vocab_size, (2, cfg.context_length), device=rt.device)
    logits, loss, _ = model(X, X)
    assert logits.shape == (2, cfg.context_length, cfg.vocab_size) and torch.isfinite(loss)

    # 3) Muon/AdamW groups cover every parameter exactly once
    muon_p, decay_p, no_decay_p = split_param_groups(model)
    ids = [id(p) for _, p in muon_p + decay_p + no_decay_p]
    assert len(ids) == len(set(ids)) == len(list(model.parameters())), "param groups overlap or miss params"
    muon_names = {n for n, _ in muon_p}
    for n, p in model.named_parameters():
        expected_muon = any(s in n for s in ("q_proj", "k_proj", "v_proj", "o_proj", "w_gate", "w_up", "w_down"))
        assert (n in muon_names) == expected_muon, f"wrong optimizer for {n}"
    assert not any("router" in n or "emb" in n or "lm_head" in n or "norm" in n for n in muon_names)

    # 4) KV-cache decoding == full forward (fp32, no TF32)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    model.eval()
    with torch.no_grad():
        seq = torch.randint(0, cfg.vocab_size, (2, 24), device=rt.device)
        full = model(seq)[0]
        logits, _, cache = model(seq[:, :10], kv_cache=[None] * cfg.n_layers)
        max_err = (logits - full[:, :10]).abs().max().item()
        for t in range(10, 24):
            step_logits, _, cache = model(seq[:, t:t + 1], kv_cache=cache)
            max_err = max(max_err, (step_logits[:, 0] - full[:, t]).abs().max().item())
    torch.backends.cuda.matmul.allow_tf32 = tf32
    assert max_err < 1e-3, f"KV-cache mismatch: {max_err}"
    model.train()

    # 5) loss goes down with the real train_step (Muon + AdamW + scaler + bias balancing)
    motif = np.random.default_rng(0).integers(4, cfg.vocab_size, size=37)
    data = np.tile(motif, 600).astype(np.uint16)
    loader = TokenLoader(data, cfg.batch_size, cfg.context_length, rt.device, seed=0)
    muon, adam = build_optimizers(model, cfg, rt)
    scaler = make_scaler(rt)
    batch, losses = loader.get(), []
    for step in range(cfg.max_iters):
        batch, loss_sum, _, load, _ = train_step(model, model, muon, adam, scaler, loader, batch, step, cfg, rt)
        losses.append(loss_sum.item())
    assert losses[-1] < 0.7 * losses[0], f"loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"
    assert load is not None and torch.allclose(load.sum(-1), torch.ones_like(load.sum(-1)))
    assert all(m.expert_bias.abs().sum() > 0 for m in model.moe_layers()), "router bias not updated"

    # 6) checkpoint round-trip + generation
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.pt")
        save_checkpoint(path, model, muon, adam, scaler, 20, cfg)
        model2 = KimiMoELM(cfg).to(rt.device)
        m2, a2 = build_optimizers(model2, cfg, rt)
        assert load_checkpoint(path, model2, m2, a2, make_scaler(rt), rt) == 20
        for (k1, v1), (k2, v2) in zip(model.state_dict().items(), model2.state_dict().items()):
            assert k1 == k2 and torch.equal(v1, v2), f"checkpoint mismatch at {k1}"
    out = model.generate(torch.tensor([[5, 6, 7]], device=rt.device), 100, amp=autocast_ctx(rt))
    assert out.shape == (1, 103)   # 100 > context_length - 3 -> exercises window re-prefill

    print(f"[✓] Self-test passed in {time.time() - t0:.1f}s | loss {losses[0]:.3f} -> {losses[-1]:.3f} "
          f"| KV-cache err {max_err:.1e}")


# -----------------------------------------------------------------------------
# 10. Main
# -----------------------------------------------------------------------------
def main():
    cfg = Config()
    rt = setup_runtime()
    torch.manual_seed(cfg.seed + rt.rank)
    np.random.seed(cfg.seed + rt.rank)

    if rt.master:
        print("=" * 78)
        print("🚀 KHMER KIMI-STYLE MoE LLM (GQA + QK-Norm + RoPE | shared+routed MoE | Muon)")
        print(f"   Device: {rt.gpu_name} x{rt.world_size} | AMP: {rt.amp_dtype} | GradScaler: {rt.use_scaler} "
              f"| NS dtype: {rt.ns_dtype} | SDPA GQA: {_SDPA_ENABLE_GQA}")
        print("=" * 78)
        if cfg.run_self_test:
            self_test(rt)
    if os.environ.get("KHMER_SELF_TEST_ONLY") == "1":
        return

    os.makedirs(cfg.work_dir, exist_ok=True)
    if rt.master:
        harvest_khmer_corpus(cfg)
        tok = KhmerTokenizer(cfg, cfg.corpus_path)
        prepare_token_bins(cfg, tok)
    barrier(rt)
    if not rt.master:
        tok = KhmerTokenizer(cfg)
    assert tok.vocab_size == cfg.vocab_size, f"tokenizer vocab {tok.vocab_size} != config {cfg.vocab_size}"

    raw_model = KimiMoELM(cfg).to(rt.device)
    if rt.master:
        pc = count_params(raw_model)
        print(f"\n[🧠] Params: total {pc['total'] / 1e6:.1f}M | active/token {pc['active'] / 1e6:.1f}M "
              f"| non-embedding total {(pc['total'] - pc['embedding']) / 1e6:.1f}M "
              f"active {(pc['active'] - pc['embedding']) / 1e6:.1f}M")
        print(f"     {cfg.n_layers} layers ({cfg.n_dense_layers} dense) | d_model {cfg.d_model} "
              f"| {cfg.n_heads}Q/{cfg.n_kv_heads}KV heads | {cfg.n_shared_experts} shared + "
              f"{cfg.n_routed_experts} routed experts (top-{cfg.top_k}) | ctx {cfg.context_length}")
    model = raw_model
    if cfg.compile and rt.amp_dtype == torch.bfloat16:
        model = torch.compile(model)
    if rt.ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # broadcast_buffers=False: per-rank expert load counters must not be overwritten by rank 0
        model = DDP(model, device_ids=[rt.local_rank], broadcast_buffers=False)

    train(cfg, rt, model, raw_model, tok)

    if rt.master:
        print("\n" + "=" * 78)
        print("📝 FINAL GENERATION TEST (KHMER PROMPTS)")
        print("=" * 78)
        for p in ["សួស្តី! ខ្ញុំរស់នៅក្នុង", "ប្រទេសកម្ពុជាគឺជា", "រាជធានីភ្នំពេញ នៃព្រះរាជាណាចក្រកម្ពុជា"]:
            print(f"\nPrompt: {p}\nOutput: {generate(raw_model, tok, p, rt, max_new_tokens=80, temperature=0.75)}")
    if rt.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
