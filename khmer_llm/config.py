"""All hyperparameters and file paths for every stage, in one dataclass."""
import os
import typing
from dataclasses import dataclass, fields, replace
from typing import List, Optional, Tuple


@dataclass
class Config:
    # ---- pipeline ----
    stages: str = "pretrain,sft,rag"   # env KHMER_STAGES overrides; "data" = download + tokenizer only (CPU)
    launch_ui: bool = False            # Gradio web page for customers at the end (public share link)

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

    # ---- pretraining ----
    batch_size: int = 8                # micro-batch per GPU (fits a 16 GB T4)
    grad_accum_steps: int = 4          # 8 x 1024 x 4 = 32k tokens / step / GPU
    max_iters: int = 20000             # ~655M tokens on 1 GPU (2x with 2 GPUs); runs resume across sessions
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

    # ---- post-training (SFT) ----
    sft_epochs: float = 2.0
    sft_max_iters: Optional[int] = None    # overrides sft_epochs when set
    sft_warmup_iters: int = 30
    sft_lr: float = 3e-4
    sft_max_examples: int = 300_000
    sft_val_every_n: int = 50              # ~2% of examples (hash-based) held out
    sft_custom_jsonl: Optional[str] = None # your own Q&A: {"instruction", "response", "context"?} per line
    sft_custom_repeat: int = 3             # upsample your own examples
    pretrained_ckpt: Optional[str] = None  # base weights for SFT; default = this run's pretrain output

    # ---- RAG ----
    rag_docs_dir: Optional[str] = None     # folder of .txt / .md / .jsonl docs; None -> slice of the corpus
    rag_max_corpus_lines: int = 50_000     # used only when rag_docs_dir is None
    rag_chunk_chars: int = 500
    rag_chunk_overlap_chars: int = 100
    rag_top_k: int = 4
    rag_dense_model: Optional[str] = "intfloat/multilingual-e5-small"   # None -> BM25 only
    rag_max_context_tokens: int = 600
    rag_max_new_tokens: int = 200

    # ---- logging / eval / checkpoints ----
    log_interval: int = 20
    eval_interval: int = 500
    eval_iters: int = 40
    ckpt_interval: int = 500
    max_train_hours: float = 11.0      # whole run (all stages); Kaggle stops at 12h
    resume: bool = True
    compile: bool = False              # torch.compile (A100/H100 only; MoE loop graph-breaks)
    seed: int = 1337
    run_self_test: bool = True         # ~seconds; set KHMER_SELF_TEST_ONLY=1 to stop after it

    # ---- data ----
    work_dir: str = "/kaggle/working" if os.path.isdir("/kaggle/working") else "outputs"
    import_dir: Optional[str] = None   # previous run outputs, comma-separated, newest first; linked into work_dir
    max_corpus_chars: int = 6_000_000_000   # ~1.6-2B tokens; stored gzipped (~5 GB on disk)
    max_harvest_hours: float = 5.0         # stop downloading after this long (sources are tried in order)
    max_seconds_per_source: int = 5400
    tokenizer_sample_lines: int = 2_000_000   # uniform sample over all harvested lines
    val_every_n_lines: int = 200       # ~0.5% of lines (hash-based) go to validation

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_ff_dense(self) -> int:
        # dense layer matches the active FFN width of an MoE layer
        return self.d_ff_shared * self.n_shared_experts + self.top_k * self.d_ff_expert

    def path(self, name: str) -> str:
        return os.path.join(self.work_dir, name)

    @property
    def corpus_path(self) -> str:
        return self.path("khmer_corpus.txt.gz")

    @property
    def tokenizer_sample_path(self) -> str:
        return self.path("khmer_tokenizer_sample.txt.gz")

    @property
    def tokenizer_prefix(self) -> str:
        return self.path(f"khmer_sp_unigram_{self.vocab_size // 1000}k")

    @property
    def train_bin(self) -> str:
        return self.path(f"train_{self.vocab_size}.bin")

    @property
    def val_bin(self) -> str:
        return self.path(f"val_{self.vocab_size}.bin")

    @property
    def ckpt_path(self) -> str:
        return self.path("khmer_kimi_moe_latest.pt")

    @property
    def final_path(self) -> str:
        return self.path("khmer_kimi_moe_final.pt")

    @property
    def sft_jsonl_path(self) -> str:
        return self.path("khmer_sft.jsonl")

    @property
    def sft_packed_path(self) -> str:
        return self.path(f"sft_packed_{self.vocab_size}_{self.context_length}.npz")

    @property
    def sft_ckpt_path(self) -> str:
        return self.path("khmer_kimi_moe_sft_latest.pt")

    @property
    def sft_final_path(self) -> str:
        return self.path("khmer_kimi_moe_sft_final.pt")

    @property
    def rag_index_path(self) -> str:
        return self.path("khmer_rag_index.pt")


ARCH_FIELDS = ("vocab_size", "context_length", "d_model", "n_layers", "n_heads", "n_kv_heads", "rope_theta",
               "norm_eps", "n_dense_layers", "n_shared_experts", "n_routed_experts", "top_k", "d_ff_shared",
               "d_ff_expert", "tie_embeddings")


def with_architecture_of(cfg: Config, saved: dict) -> Config:
    """cfg with model-shape fields taken from a saved config (so loading never mismatches shapes)."""
    return replace(cfg, **{k: saved[k] for k in ARCH_FIELDS if k in saved})


def _coerce(raw: str, typ):
    if raw.strip().lower() in ("none", "null"):
        return None
    origin, args = typing.get_origin(typ), typing.get_args(typ)
    if origin is typing.Union:   # Optional[X]
        return _coerce(raw, next(a for a in args if a is not type(None)))
    if origin in (tuple, Tuple):
        return tuple(_coerce(v, args[0]) for v in raw.split(","))
    if typ is bool:
        return raw.strip().lower() in ("1", "true", "yes", "y", "on")
    if typ in (int, float):
        return typ(raw)
    return raw


def apply_overrides(cfg: Config, overrides: List[str]) -> Config:
    """Applies ["key=value", ...] (e.g. from `--set max_iters=100`) with type coercion."""
    known = {f.name: f.type for f in fields(cfg)}
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or key not in known:
            raise ValueError(f"bad override {item!r}; expected key=value with key in Config")
        setattr(cfg, key, _coerce(raw, known[key]))
    return cfg
