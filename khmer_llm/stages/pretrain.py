"""Stage 1 — pretraining: next-token prediction on the harvested Khmer corpus."""
import os

import numpy as np

from ..chat import generate
from ..config import Config
from ..data.corpus import TokenLoader, prepare_token_bins
from ..data.tokenizer import KhmerTokenizer
from ..model.lm import build_model, count_params
from ..runtime import Runtime, barrier, free_memory
from ..train.checkpoint import save_inference_weights
from ..train.loop import run_training, wrap_model

PRETRAIN_SAMPLE_PROMPTS = ["សួស្តី! ខ្ញុំរស់នៅក្នុង", "ប្រទេសកម្ពុជាគឺជា", "រាជធានីភ្នំពេញ នៃព្រះរាជាណាចក្រកម្ពុជា"]


def run_pretrain(cfg: Config, rt: Runtime, tok: KhmerTokenizer):
    if rt.master:
        prepare_token_bins(cfg, tok)
    barrier(rt)

    T, B = cfg.context_length, cfg.batch_size
    train_loader = TokenLoader(cfg.train_bin, B, T, rt.device, cfg.seed + rt.rank)
    train_eval_loader = TokenLoader(cfg.train_bin, B, T, rt.device, cfg.seed + 10_000)
    if os.path.getsize(cfg.val_bin) // 2 > T + 1:
        val_data = np.memmap(cfg.val_bin, dtype=np.uint16, mode="r")
    else:   # tiny corpus (np.memmap cannot even open an empty file) -> tail of train
        val_data = np.memmap(cfg.train_bin, dtype=np.uint16, mode="r")[-(20 * T):]
    val_loader = TokenLoader(val_data, B, T, rt.device, cfg.seed + 20_000)

    raw_model = build_model(cfg, rt)
    if rt.master:
        pc = count_params(raw_model)
        print(f"\n[🧠] Params: total {pc['total'] / 1e6:.1f}M | active/token {pc['active'] / 1e6:.1f}M "
              f"| non-embedding total {(pc['total'] - pc['embedding']) / 1e6:.1f}M "
              f"active {(pc['active'] - pc['embedding']) / 1e6:.1f}M")
        print(f"     {cfg.n_layers} layers ({cfg.n_dense_layers} dense) | d_model {cfg.d_model} "
              f"| {cfg.n_heads}Q/{cfg.n_kv_heads}KV heads | {cfg.n_shared_experts} shared + "
              f"{cfg.n_routed_experts} routed experts (top-{cfg.top_k}) | ctx {cfg.context_length}")
    model = wrap_model(raw_model, cfg, rt)
    sample_fn = lambda: generate(raw_model, tok, PRETRAIN_SAMPLE_PROMPTS[1], rt, max_new_tokens=48)
    step = run_training(cfg, rt, model, raw_model, train_loader,
                        {"train": train_eval_loader, "val": val_loader}, cfg.ckpt_path, sample_fn, "PRETRAIN")

    if rt.master and step == 0:
        print("[!] No pretraining steps ran (time budget?); base model weights not written.")
    elif rt.master:
        save_inference_weights(cfg.final_path, raw_model, cfg, tok.model_file, rt)
        print(f"[✓] Base model weights: {cfg.final_path} (step {step})")
        for p in PRETRAIN_SAMPLE_PROMPTS:
            print(f"  Prompt: {p}\n  Output: {generate(raw_model, tok, p, rt, max_new_tokens=60, temperature=0.75)}")
    del model, raw_model
    free_memory()
