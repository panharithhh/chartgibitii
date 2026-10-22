"""Runs the stages in order: tokenizer -> pretrain -> SFT -> RAG (and optionally the web UI)."""
import os
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist

from .config import Config, apply_overrides
from .data.corpus import harvest_khmer_corpus, prepare_token_bins, token_bins_ready
from .data.tokenizer import KhmerTokenizer
from .runtime import RUNTIME_FLAGS, Runtime, barrier, link_previous_outputs, setup_runtime
from .selftest import self_test
from .stages.pretrain import run_pretrain
from .stages.rag import rag_fallback_path, run_rag
from .stages.sft import harvest_sft_examples, prepare_sft_data, run_sft
from .train.checkpoint import first_existing

STAGES = ("data", "pretrain", "sft", "rag")


def prepare_tokenizer(cfg: Config, rt: Runtime) -> KhmerTokenizer:
    """Loads the tokenizer, harvesting the corpus and training it first if it is missing."""
    if rt.master and not os.path.exists(cfg.tokenizer_prefix + ".model"):
        harvest_khmer_corpus(cfg)
        KhmerTokenizer(cfg, cfg.tokenizer_sample_path)
    barrier(rt)
    tok = KhmerTokenizer(cfg)
    assert tok.vocab_size == cfg.vocab_size, f"tokenizer vocab {tok.vocab_size} != config {cfg.vocab_size}"
    return tok


def check_prerequisites(cfg: Config, stage_list):
    """Fail fast (before hours of downloading) when a later stage is run without earlier outputs."""
    if "pretrain" in stage_list or "data" in stage_list:
        return
    hint = "include the 'pretrain' stage, or point work_dir / import_dir at a previous run's outputs"
    if not os.path.exists(cfg.tokenizer_prefix + ".model"):
        raise FileNotFoundError(f"No tokenizer at {cfg.tokenizer_prefix}.model: {hint}.")
    base = first_existing([cfg.pretrained_ckpt, cfg.ckpt_path, cfg.final_path])
    if "sft" in stage_list and base is None:
        raise FileNotFoundError(f"SFT needs pretrained weights in {cfg.work_dir}: {hint}.")
    if "rag" in stage_list and base is None and first_existing([cfg.sft_final_path, cfg.sft_ckpt_path]) is None:
        raise FileNotFoundError(f"RAG needs a trained model in {cfg.work_dir}: {hint}.")


def run_data(cfg: Config, rt: Runtime, tok: KhmerTokenizer):
    """Everything that needs no GPU: corpus + tokenizer (already done by prepare_tokenizer), token bins,
    SFT download + packing. Run it in a free CPU session, then import its output for GPU sessions."""
    if rt.master:
        prepare_token_bins(cfg, tok)
        harvest_sft_examples(cfg)
        prepare_sft_data(cfg, tok)
        print("[✓] Data stage done. In the GPU notebook add this run's output as input and use "
              "Config(import_dir='/kaggle/input/<this-run>').")
    barrier(rt)


def main(cfg: Optional[Config] = None, stages: Optional[str] = None) -> dict:
    cfg = cfg or Config()
    # KHMER_SET="key=value;key=value" overrides settings without editing code (e.g. for torchrun on Kaggle)
    apply_overrides(cfg, [kv for kv in os.environ.get("KHMER_SET", "").split(";") if kv.strip()])
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
    check_prerequisites(cfg, stage_list)
    tok = prepare_tokenizer(cfg, rt)
    if rt.master and ("pretrain" in stage_list or "data" in stage_list) and not token_bins_ready(cfg, tok):
        harvest_khmer_corpus(cfg)   # the raw corpus is only needed to (re)build the token bins
    barrier(rt)

    results = dict(config=cfg, tokenizer=tok)
    if "data" in stage_list:
        run_data(cfg, rt, tok)
    if "pretrain" in stage_list:
        run_pretrain(cfg, rt, tok)
    if "sft" in stage_list:
        run_sft(cfg, rt, tok)
    if rt.ddp:
        dist.destroy_process_group()
    if "rag" in stage_list and rt.master:
        if not rag_fallback_path(cfg) and not (cfg.rag_docs_dir and os.path.isdir(cfg.rag_docs_dir)):
            harvest_khmer_corpus(cfg)   # default knowledge base = harvested corpus
        results["rag"] = run_rag(cfg, rt, tok)
        if cfg.launch_ui:
            from .ui import launch_ui
            launch_ui(results["rag"])
    return results
