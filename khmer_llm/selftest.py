"""Fast runtime self-test (tiny configs, synthetic data, no network). Also used by tests/."""
import os
import tempfile
import time
from dataclasses import replace
from typing import List

import numpy as np
import torch

from .chat import build_chat_example, chat
from .config import Config
from .data.corpus import TokenLoader
from .data.tokenizer import CHAT_SYMBOLS
from .model.lm import KimiMoELM
from .optim.hybrid import build_optimizers, split_param_groups
from .optim.muon import newton_schulz5
from .runtime import Runtime, autocast_ctx, make_scaler
from .stages.rag import BM25Index, rrf_fuse
from .stages.sft import SFTLoader, pack_examples
from .text import chunk_text
from .train.checkpoint import load_checkpoint, save_checkpoint
from .train.loop import train_step


class ToyTokenizer:
    """Character-level stand-in for KhmerTokenizer (tests only)."""
    eos_id, pad_id = 2, 3

    def __init__(self, vocab: int):
        self.vocab = vocab

    def encode(self, text: str) -> List[int]:
        return [8 + (ord(c) % (self.vocab - 8)) for c in text]

    def marker(self, symbol: str) -> List[int]:
        return [4 + CHAT_SYMBOLS.index(symbol)]

    def decode(self, ids: List[int]) -> str:
        return "".join(chr(97 + i % 26) for i in ids)


def tiny_config(**overrides) -> Config:
    base = dict(vocab_size=128, context_length=64, d_model=128, n_layers=3, n_heads=8, n_kv_heads=2,
                n_dense_layers=1, n_routed_experts=8, top_k=2, d_ff_shared=128, d_ff_expert=32,
                batch_size=8, grad_accum_steps=2, max_iters=20, warmup_iters=2, lr=3e-3)
    base.update(overrides)
    return Config(**base)


def check_newton_schulz(rt: Runtime):
    """Output singular values land near 1 for wide / tall / batched inputs."""
    for shape in [(64, 96), (96, 64), (4, 32, 48)]:
        G = torch.randn(*shape, device=rt.device)
        X = newton_schulz5(G, dtype=rt.ns_dtype).float()
        assert X.shape == G.shape
        sv = torch.linalg.svdvals(X.cpu())
        assert 0.5 < sv.min() and sv.max() < 1.5, f"NS singular values out of range: {sv.min()}, {sv.max()}"


def check_param_groups(model: KimiMoELM):
    """Muon/AdamW groups cover every parameter exactly once, with the right owner."""
    muon_p, decay_p, no_decay_p = split_param_groups(model)
    ids = [id(p) for _, p in muon_p + decay_p + no_decay_p]
    assert len(ids) == len(set(ids)) == len(list(model.parameters())), "param groups overlap or miss params"
    muon_names = {n for n, _ in muon_p}
    for n, _ in model.named_parameters():
        expected = any(s in n for s in ("q_proj", "k_proj", "v_proj", "o_proj", "w_gate", "w_up", "w_down"))
        assert (n in muon_names) == expected, f"wrong optimizer for {n}"


def check_kv_cache(model: KimiMoELM, rt: Runtime) -> float:
    """Incremental KV-cache decoding matches a full forward pass (fp32, no TF32)."""
    cfg = model.cfg
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            seq = torch.randint(0, cfg.vocab_size, (2, 24), device=rt.device)
            full = model(seq)[0]
            logits, _, cache = model(seq[:, :10], kv_cache=[None] * cfg.n_layers)
            max_err = (logits - full[:, :10]).abs().max().item()
            for t in range(10, 24):
                step_logits, _, cache = model(seq[:, t:t + 1], kv_cache=cache)
                max_err = max(max_err, (step_logits[:, 0] - full[:, t]).abs().max().item())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        model.train(was_training)
    assert max_err < 1e-3, f"KV-cache mismatch: {max_err}"
    return max_err


def check_pretrain_steps(model: KimiMoELM, rt: Runtime):
    """Real train_step (Muon + AdamW + scaler + bias balancing) learns a repeating motif."""
    cfg = model.cfg
    motif = np.random.default_rng(0).integers(4, cfg.vocab_size, size=37)
    loader = TokenLoader(np.tile(motif, 600).astype(np.uint16), cfg.batch_size, cfg.context_length, rt.device, seed=0)
    muon, adam = build_optimizers(model, cfg, rt)
    scaler = make_scaler(rt)
    batch, losses, load = loader.get(), [], None
    for step in range(cfg.max_iters):
        batch, loss_sum, _, load, _ = train_step(model, model, muon, adam, scaler, loader, batch, step, cfg, rt)
        losses.append(loss_sum.item())
    assert losses[-1] < 0.7 * losses[0], f"loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"
    assert load is not None and torch.allclose(load.sum(-1), torch.ones_like(load.sum(-1)))
    assert all(m.expert_bias.abs().sum() > 0 for m in model.moe_layers()), "router bias not updated"
    return losses, muon, adam, scaler


def check_checkpoint_roundtrip(model: KimiMoELM, muon, adam, scaler, rt: Runtime):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.pt")
        save_checkpoint(path, model, muon, adam, scaler, 20, model.cfg)
        model2 = KimiMoELM(model.cfg).to(rt.device)
        m2, a2 = build_optimizers(model2, model.cfg, rt)
        assert load_checkpoint(path, model2, m2, a2, make_scaler(rt), rt) == 20
        for (k1, v1), (k2, v2) in zip(model.state_dict().items(), model2.state_dict().items()):
            assert k1 == k2 and torch.equal(v1, v2), f"checkpoint mismatch at {k1}"


def check_generation(model: KimiMoELM, rt: Runtime):
    out = model.generate(torch.tensor([[5, 6, 7]], device=rt.device), 100, amp=autocast_ctx(rt),
                         repetition_penalty=1.2)
    assert out.shape == (1, 103)   # 100 > context_length - 3 -> exercises window re-prefill


def check_chat_template(cfg: Config):
    tok = ToyTokenizer(cfg.vocab_size)
    ctx_id, user_id = 4 + CHAT_SYMBOLS.index("<|context|>"), 4 + CHAT_SYMBOLS.index("<|user|>")
    ex, m = build_chat_example(tok, "abc", "xyz", context="ctx")
    assert len(ex) == len(m) and sum(m) == 4 and m[-4:] == [1, 1, 1, 1] and ex[-1] == tok.eos_id
    assert ex[0] == ctx_id
    trimmed = build_chat_example(tok, "q" * 10, "a" * 40, context="c" * 200, max_len=cfg.context_length)
    assert len(trimmed[0]) == cfg.context_length and sum(trimmed[1]) == 41 and trimmed[0][0] == ctx_id
    cut = build_chat_example(tok, "q" * 10, "a" * 60, context="c" * 200, max_len=cfg.context_length)
    assert len(cut[0]) == cfg.context_length and cut[0][0] == user_id   # context dropped, answer trimmed
    assert build_chat_example(tok, "q" * 80, "a", max_len=cfg.context_length) is None


def check_sft_steps(cfg: Config, rt: Runtime):
    """Packing keeps every answer token, the loss mask hides prompts/padding, and SFT loss drops."""
    tok = ToyTokenizer(cfg.vocab_size)
    rng = np.random.default_rng(1)
    letters = "abcdefgh"
    examples = []
    for _ in range(400):
        q = "".join(rng.choice(list(letters), size=rng.integers(4, 9)))
        examples.append(build_chat_example(tok, q, q[::-1].upper(), max_len=cfg.context_length))
    toks, masks = pack_examples(examples, cfg.context_length + 1, tok.pad_id, tok.eos_id)
    assert toks.shape[1] == cfg.context_length + 1 and (toks[:, 0] == tok.eos_id).all()
    assert int(masks.sum()) == sum(sum(mm) for _, mm in examples), "packing lost answer tokens"
    loader = SFTLoader(toks, masks, cfg.batch_size, rt.device, seed=0)
    _, yb = loader.get()
    answer_ids = {tok.encode(c)[0] for c in letters.upper()} | {tok.eos_id}
    graded = set(yb[yb != -100].tolist())
    assert graded and graded <= answer_ids, "loss mask leaks onto prompt / padding tokens"

    scfg = replace(cfg, lr=1e-3, max_iters=25, warmup_iters=2, lr_schedule="cosine")
    model = KimiMoELM(cfg).to(rt.device)
    muon, adam = build_optimizers(model, scfg, rt)
    scaler = make_scaler(rt)
    batch, losses = loader.get(), []
    for step in range(scfg.max_iters):
        batch, loss_sum, _, _, _ = train_step(model, model, muon, adam, scaler, loader, batch, step, scfg, rt)
        losses.append(loss_sum.item())
    assert np.mean(losses[-3:]) < 0.7 * losses[0], f"SFT loss did not drop: {losses[0]:.3f} -> {losses[-1]:.3f}"
    assert isinstance(chat(model, tok, "abcd", rt, context="ctx", max_new_tokens=8), str)
    assert isinstance(chat(model, tok, "q" * 500, rt, max_new_tokens=8), str)   # over-long question
    return losses


def check_rag_components():
    text = "។ ".join(["ប្រយោគ" * 5 + str(i) for i in range(30)]) + "។"
    chunks = chunk_text(text, chunk_chars=120, overlap_chars=40)
    assert len(chunks) > 3 and all(len(c) <= 120 for c in chunks)
    assert any(set(chunks[i].split(" ")) & set(chunks[i + 1].split(" ")) for i in range(len(chunks) - 1)), "no overlap"
    vocab = {w: i for i, w in enumerate("cats sit on mats dogs bark loudly stocks fell sharply today the".split())}
    docs = ["the cats sit on the mats", "dogs bark loudly", "the stocks fell sharply today"]
    bm25 = BM25Index().build([[vocab[w] for w in d.split()] for d in docs], len(vocab))
    assert bm25.search([vocab["dogs"], vocab["bark"]], 3)[0] == 1
    assert bm25.search([vocab["stocks"], vocab["the"]], 3)[0] == 2
    assert BM25Index.from_state(bm25.state()).search([vocab["cats"]], 1) == [0]
    assert rrf_fuse([[1, 2, 3], [3, 1, 4]], top_k=2) == [1, 3]


def self_test(rt: Runtime):
    print("\n[🧪] Self-test (tiny model, synthetic data)...")
    t0 = time.time()
    rt = replace(rt, ddp=False)
    cfg = tiny_config()
    torch.manual_seed(0)
    check_newton_schulz(rt)
    model = KimiMoELM(cfg).to(rt.device)
    logits, loss, _ = model(torch.randint(0, cfg.vocab_size, (2, cfg.context_length), device=rt.device),
                            torch.randint(0, cfg.vocab_size, (2, cfg.context_length), device=rt.device))
    assert logits.shape == (2, cfg.context_length, cfg.vocab_size) and torch.isfinite(loss)
    check_param_groups(model)
    kv_err = check_kv_cache(model, rt)
    losses, muon, adam, scaler = check_pretrain_steps(model, rt)
    check_checkpoint_roundtrip(model, muon, adam, scaler, rt)
    check_generation(model, rt)
    check_chat_template(cfg)
    sft_losses = check_sft_steps(cfg, rt)
    check_rag_components()
    print(f"[✓] Self-test passed in {time.time() - t0:.1f}s | pretrain loss {losses[0]:.3f} -> {losses[-1]:.3f} "
          f"| SFT loss {sft_losses[0]:.3f} -> {sft_losses[-1]:.3f} | KV-cache err {kv_err:.1e}")
