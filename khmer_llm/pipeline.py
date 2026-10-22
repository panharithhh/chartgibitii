"""Runs the stages in order: tokenizer -> pretrain -> SFT -> RAG (and optionally the web UI)."""
import os
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist

from .config import Config
from .data.corpus import harvest_khmer_corpus
from .data.tokenizer import KhmerTokenizer
from .runtime import RUNTIME_FLAGS, Runtime, barrier, link_previous_outputs, setup_runtime
from .selftest import self_test
from .stages.pretrain import run_pretrain
from .stages.rag import run_rag
from .stages.sft import run_sft

STAGES = ("pretrain", "sft", "rag")


def prepare_tokenizer(cfg: Config, rt: Runtime) -> KhmerTokenizer:
    """Loads the tokenizer, harvesting the corpus and training it first if it is missing."""
    if rt.master and not os.path.exists(cfg.tokenizer_prefix + ".model"):
        KhmerTokenizer(cfg, harvest_khmer_corpus(cfg))
    barrier(rt)
    tok = KhmerTokenizer(cfg)
    assert tok.vocab_size == cfg.vocab_size, f"tokenizer vocab {tok.vocab_size} != config {cfg.vocab_size}"
    return tok


def main(cfg: Optional[Config] = None, stages: Optional[str] = None) -> dict:
    cfg = cfg or Config()
    requested = stages or os.environ.get("KHMER_STAGES") or cfg.stages
    stage_list = [s.strip().lower() for s in requested.split(",") if s.strip()]
    unknown = set(stage_list) - set(STAGES)
    assert not unknown, f"unknown stages {unknown}; use {', '.join(STAGES)}"
    rt = setup_runtime()
    torch.manual_seed(cfg.seed + rt.rank)
    np.random.seed(cfg.seed + rt.rank)

    if rt.master:
        print("=" * 78)
        print("🚀 KHMER KIMI-STYLE MoE LLM (GQA + QK-Norm + RoPE | shared+routed MoE | Muon)")
        print(f"   Stages: {' → '.join(stage_list)} | work_dir: {os.path.abspath(cfg.work_dir)}")
        print(f"   Device: {rt.gpu_name} x{rt.world_size} | AMP: {rt.amp_dtype} | GradScaler: {rt.use_scaler} "
              f"| NS dtype: {rt.ns_dtype} | SDPA GQA: {RUNTIME_FLAGS['sdpa_gqa']}")
        print("=" * 78)
        if cfg.run_self_test:
            self_test(rt)
    if os.environ.get("KHMER_SELF_TEST_ONLY") == "1":
        return {}

    os.makedirs(cfg.work_dir, exist_ok=True)
    if rt.master:
        link_previous_outputs(cfg)
    barrier(rt)
    tok = prepare_tokenizer(cfg, rt)
    if rt.master and "pretrain" in stage_list:
        harvest_khmer_corpus(cfg)   # pretraining needs the corpus even when the tokenizer came from elsewhere
    barrier(rt)

    results = dict(config=cfg, tokenizer=tok)
    if "pretrain" in stage_list:
        run_pretrain(cfg, rt, tok)
    if "sft" in stage_list:
        run_sft(cfg, rt, tok)
    if rt.ddp:
        dist.destroy_process_group()
    if "rag" in stage_list and rt.master:
        if not os.path.exists(cfg.corpus_path) and not (cfg.rag_docs_dir and os.path.isdir(cfg.rag_docs_dir)):
            harvest_khmer_corpus(cfg)   # default knowledge base = harvested corpus
        results["rag"] = run_rag(cfg, rt, tok)
        if cfg.launch_ui:
            from .ui import launch_ui
            launch_ui(results["rag"])
    return results
