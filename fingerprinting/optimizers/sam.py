from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW

from .entry import OptimizerEntry, OptimizerRuntime, get_hparam


class SAMAdamW:
    def __init__(
        self,
        params,
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        rho: float = 0.05,
    ) -> None:
        self.base_optimizer = AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
        self.rho = rho

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def first_step(self) -> None:
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["sam_e_w"] = e_w

    @torch.no_grad()
    def second_step(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = self.state[p].pop("sam_e_w", None)
                if e_w is not None:
                    p.sub_(e_w)
        self.base_optimizer.step()

    def step(self, closure: Callable[[], Any] | None = None):
        if closure is None:
            raise RuntimeError("SAMAdamW requires a closure for the sharpness-aware second gradient")
        self.first_step()
        self.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss = closure()
        self.second_step()
        self.zero_grad(set_to_none=True)
        return loss

    def _grad_norm(self) -> torch.Tensor:
        norms = [
            p.grad.norm(p=2)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not norms:
            return torch.tensor(0.0)
        return torch.norm(torch.stack(norms), p=2)


def build_sam_adamw(model: nn.Module, entry: OptimizerEntry) -> OptimizerRuntime:
    optimizer = SAMAdamW(
        model.parameters(),
        lr=float(get_hparam(entry, "lr", (float, int))),
        betas=tuple(entry.hparams.get("betas", (0.9, 0.95))),
        weight_decay=float(get_hparam(entry, "weight_decay", (float, int))),
        eps=float(entry.hparams.get("eps", 1e-8)),
        rho=float(entry.hparams.get("rho", 0.05)),
    )
    return OptimizerRuntime(optimizer, requires_closure=True)
