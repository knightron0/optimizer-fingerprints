from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer

from .entry import OptimizerEntry, OptimizerRuntime, get_hparam


class Lion(Optimizer):
    def __init__(
        self,
        params,
        *,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                p.add_(update.sign(), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


def build_lion(model: nn.Module, entry: OptimizerEntry) -> OptimizerRuntime:
    optimizer = Lion(
        model.parameters(),
        lr=float(get_hparam(entry, "lr", (float, int))),
        betas=tuple(entry.hparams.get("betas", (0.9, 0.99))),
        weight_decay=float(get_hparam(entry, "weight_decay", (float, int))),
    )
    return OptimizerRuntime(optimizer)
