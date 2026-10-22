"""The Kimi-style MoE language model, KV-cache generation and token sampling."""
import math
from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..runtime import Runtime
from .layers import Block, MoE, RMSNorm, precompute_rope


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
        """kv_cache: None (no caching) or a list with one (k, v)/None entry per layer.
        targets equal to -100 are ignored (SFT prompt tokens / padding)."""
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
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=-100)
        return logits, loss, new_cache

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int = 50,
                 top_p: float = 0.95, eos_id: Optional[int] = None, amp=None,
                 repetition_penalty: float = 1.0):
        """KV-cache sampling. When the window fills, re-prefill with the most recent half.
        repetition_penalty applies only to generated tokens (never the prompt / retrieved context)
        and never to eos_id, so it does not discourage copying from passages or stopping."""
        was_training = self.training
        self.eval()
        L = self.cfg.context_length
        idx = idx[:, -L:]
        prompt_len = idx.size(1)
        with (amp if amp is not None else nullcontext()):
            logits, _, cache = self(idx, kv_cache=[None] * len(self.layers), last_only=True)
            for _ in range(max_new_tokens):
                generated = idx[:, prompt_len:] if repetition_penalty != 1.0 and idx.size(1) > prompt_len else None
                nxt = sample_next_token(logits[:, -1].float(), temperature, top_k, top_p,
                                        generated, repetition_penalty, protect_id=eos_id)
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


def sample_next_token(logits, temperature: float, top_k: int, top_p: float,
                      prev_ids: Optional[torch.Tensor] = None, repetition_penalty: float = 1.0,
                      protect_id: Optional[int] = None) -> torch.Tensor:
    if prev_ids is not None and prev_ids.numel() and repetition_penalty != 1.0:   # CTRL-style penalty
        original = logits
        seen = logits.gather(-1, prev_ids)
        seen = torch.where(seen > 0, seen / repetition_penalty, seen * repetition_penalty)
        logits = logits.scatter(-1, prev_ids, seen)
        if protect_id is not None:
            logits[:, protect_id] = original[:, protect_id]
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


def count_params(model: KimiMoELM) -> Dict[str, int]:
    cfg = model.cfg
    total = sum(p.numel() for p in model.parameters())
    n_moe = cfg.n_layers - cfg.n_dense_layers
    inactive = n_moe * (cfg.n_routed_experts - cfg.top_k) * 3 * cfg.d_model * cfg.d_ff_expert
    emb = model.tok_emb.weight.numel() + (0 if cfg.tie_embeddings else model.lm_head.weight.numel())
    return dict(total=total, active=total - inactive, embedding=emb)


def build_model(cfg: Config, rt: Runtime) -> KimiMoELM:
    return KimiMoELM(cfg).to(rt.device)
