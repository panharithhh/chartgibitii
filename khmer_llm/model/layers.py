"""Transformer building blocks: RMSNorm, RoPE, GQA + QK-Norm attention, SwiGLU, shared + routed MoE."""
from contextlib import nullcontext
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..runtime import RUNTIME_FLAGS


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

        if RUNTIME_FLAGS["sdpa_gqa"]:
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
