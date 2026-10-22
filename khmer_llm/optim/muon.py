"""Muon: momentum orthogonalized by a 5th-order Newton-Schulz iteration."""
import math

import torch


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
