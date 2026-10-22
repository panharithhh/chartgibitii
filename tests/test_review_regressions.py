"""Regression tests for issues found in code review."""
import json
import os

import numpy as np
import pytest
import torch

from khmer_llm.config import Config
from khmer_llm.model.lm import KimiMoELM, sample_next_token
from khmer_llm.optim.hybrid import build_optimizers
from khmer_llm.runtime import make_scaler
from khmer_llm.selftest import ToyTokenizer
from khmer_llm.train.checkpoint import atomic_torch_save, save_inference_weights
from khmer_llm.train.loop import train_step
from khmer_llm.data.corpus import TokenLoader


def test_repetition_penalty_spares_protected_eos():
    logits = torch.tensor([[2.0, 1.9, 0.0]])
    out = sample_next_token(logits, 0.0, 0, 1.0, prev_ids=torch.tensor([[0, 2]]), repetition_penalty=2.0, protect_id=0)
    assert out.item() == 0   # eos (id 0) was generated before but must not be penalised
    out = sample_next_token(logits, 0.0, 0, 1.0, prev_ids=torch.tensor([[0]]), repetition_penalty=2.0)
    assert out.item() == 1


def test_generate_penalises_only_generated_tokens(cfg, rt, monkeypatch):
    import khmer_llm.model.lm as lm
    seen = []
    real = lm.sample_next_token

    def spy(logits, *args, **kwargs):
        prev = args[3] if len(args) > 3 else kwargs.get("prev_ids")
        seen.append(None if prev is None else prev.tolist())
        return real(logits, *args, **kwargs)

    monkeypatch.setattr(lm, "sample_next_token", spy)
    model = KimiMoELM(cfg).to(rt.device)
    model.generate(torch.tensor([[5, 6, 7, 8]], device=rt.device), 3, repetition_penalty=1.3)
    assert seen[0] is None and len(seen[1][0]) == 1 and len(seen[2][0]) == 2   # prompt never included


def test_overflow_step_skips_both_optimizers(cfg, rt):
    model = KimiMoELM(cfg).to(rt.device)
    data = np.random.default_rng(0).integers(4, cfg.vocab_size, size=5000).astype(np.uint16)
    loader = TokenLoader(data, cfg.batch_size, cfg.context_length, rt.device, seed=0)
    muon, adam = build_optimizers(model, cfg, rt)
    model.tok_emb.weight.register_hook(lambda g: g * float("inf"))   # inf only in an AdamW param
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    _, _, grad_norm, _, _ = train_step(model, model, muon, adam, make_scaler(rt), loader, loader.get(), 5, cfg, rt)
    assert not torch.isfinite(grad_norm)
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters()), "a step was applied"


def test_atomic_save_replaces_symlink_to_read_only_file(tmp_path):
    ro_dir = tmp_path / "input"
    ro_dir.mkdir()
    (ro_dir / "x.pt").write_bytes(b"old")
    os.chmod(ro_dir / "x.pt", 0o444)
    os.chmod(ro_dir, 0o555)
    link = tmp_path / "x.pt"
    os.symlink(ro_dir / "x.pt", link)
    try:
        atomic_torch_save({"a": 1}, str(link))
        assert not os.path.islink(link) and torch.load(link)["a"] == 1
        assert (ro_dir / "x.pt").read_bytes() == b"old"
    finally:
        os.chmod(ro_dir, 0o755)


def test_inference_weights_keep_router_bias_fp32(cfg, rt, tmp_path):
    model = KimiMoELM(cfg)
    for m in model.moe_layers():
        m.expert_bias += 1e-4 * torch.arange(m.n_experts)
    half_rt = rt.__class__(**{**rt.__dict__, "amp_dtype": torch.float16})
    save_inference_weights(str(tmp_path / "w.pt"), model, cfg, "tok.model", half_rt)
    state = torch.load(tmp_path / "w.pt", weights_only=False)["model"]
    assert all(v.dtype == torch.float32 for k, v in state.items() if k.endswith("expert_bias"))
    assert state["tok_emb.weight"].dtype == torch.float16


def test_custom_only_sft_trains_on_every_example(tmp_path):
    from khmer_llm.stages.sft import prepare_sft_data
    cfg = Config(work_dir=str(tmp_path), vocab_size=128, context_length=64, sft_custom_repeat=3,
                 sft_custom_jsonl=str(tmp_path / "faq.jsonl"))
    open(cfg.sft_jsonl_path, "w").close()   # offline: no downloaded examples
    rows = [{"question": f"q{i}?", "answer": f"a{i}" * 3} for i in range(20)]
    (tmp_path / "faq.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    tok = ToyTokenizer(128)
    prepare_sft_data(cfg, tok)
    with np.load(cfg.sft_packed_path) as d:
        answer_tokens = int(d["train_mask"].sum())
        assert len(d["val_tokens"]) > 0
    assert answer_tokens == 3 * sum(len(r["answer"]) + 1 for r in rows)   # every entry x3, answer + EOS


def test_empty_sft_download_does_not_mark_done(tmp_path, monkeypatch):
    import khmer_llm.stages.sft as sft
    monkeypatch.setattr(sft, "SFT_SOURCES", [])
    cfg = Config(work_dir=str(tmp_path), sft_custom_jsonl="faq.jsonl")
    sft.harvest_sft_examples(cfg)
    assert os.path.exists(cfg.sft_jsonl_path) and not os.path.exists(cfg.sft_jsonl_path + ".done")
