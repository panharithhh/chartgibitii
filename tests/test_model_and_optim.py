import pytest
import torch

from khmer_llm.config import Config
from khmer_llm.model.lm import KimiMoELM, count_params, sample_next_token
from khmer_llm.optim.hybrid import lr_mult
from khmer_llm.runtime import RUNTIME_FLAGS, _TORCH_VERSION
from khmer_llm.selftest import (check_checkpoint_roundtrip, check_generation, check_kv_cache,
                                check_newton_schulz, check_param_groups, check_pretrain_steps)


def test_forward_shapes_and_ignore_index(cfg, rt):
    model = KimiMoELM(cfg).to(rt.device)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.context_length), device=rt.device)
    y = x.clone()
    y[:, : cfg.context_length // 2] = -100
    logits, loss, cache = model(x, y)
    assert logits.shape == (2, cfg.context_length, cfg.vocab_size) and torch.isfinite(loss) and cache is None


def test_newton_schulz(rt):
    check_newton_schulz(rt)


def test_param_groups_cover_everything(cfg):
    check_param_groups(KimiMoELM(cfg))
    check_param_groups(KimiMoELM(Config(**{**cfg.__dict__, "tie_embeddings": True})))


def test_kv_cache_matches_full_forward(cfg, rt):
    check_kv_cache(KimiMoELM(cfg).to(rt.device), rt)


@pytest.mark.skipif(_TORCH_VERSION < (2, 5), reason="SDPA enable_gqa needs torch >= 2.5")
def test_native_gqa_path_matches_repeat_interleave(cfg):
    model = KimiMoELM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 20))
    old = RUNTIME_FLAGS["sdpa_gqa"]
    try:
        with torch.no_grad():
            RUNTIME_FLAGS["sdpa_gqa"] = False
            a = model(x)[0]
            RUNTIME_FLAGS["sdpa_gqa"] = True
            b = model(x)[0]
    finally:
        RUNTIME_FLAGS["sdpa_gqa"] = old
    assert torch.allclose(a, b, atol=1e-5)


def test_training_checkpoint_and_generation(cfg, rt):
    model = KimiMoELM(cfg).to(rt.device)
    _, muon, adam, scaler = check_pretrain_steps(model, rt)
    check_checkpoint_roundtrip(model, muon, adam, scaler, rt)
    check_generation(model, rt)


def test_default_model_size():
    pc = count_params(KimiMoELM(Config()))
    assert 100e6 < pc["total"] < 110e6 and 60e6 < pc["active"] < 70e6


def test_lr_schedules():
    cfg = Config(max_iters=1000, warmup_iters=100, lr_schedule="wsd", wsd_decay_frac=0.2, min_lr_ratio=0.1)
    assert lr_mult(0, cfg) == pytest.approx(0.01) and lr_mult(500, cfg) == 1.0
    assert lr_mult(999, cfg) == pytest.approx(0.1, abs=0.01)
    cfg.lr_schedule = "cosine"
    assert lr_mult(100, cfg) == pytest.approx(1.0) and lr_mult(1000, cfg) == pytest.approx(0.1)


def test_sampling_filters():
    logits = torch.tensor([[5.0, 4.0, -10.0, -10.0]])
    assert sample_next_token(logits, 0.0, 0, 1.0).item() == 0
    for _ in range(20):
        assert sample_next_token(logits, 1.0, 2, 1.0).item() in (0, 1)
        assert sample_next_token(logits, 1.0, 0, 0.5).item() == 0
    penalized = sample_next_token(logits, 0.0, 0, 1.0, prev_ids=torch.tensor([[0]]), repetition_penalty=10.0)
    assert penalized.item() == 1
