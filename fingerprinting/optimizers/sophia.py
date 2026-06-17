from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer

from .entry import OptimizerEntry, OptimizerRuntime, get_hparam


class SophiaG(Optimizer):
    def __init__(
        self,
        params,
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 0.0,
        eps: float = 1e-12,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay, eps=eps)
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
            rho = group["rho"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["hessian"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                hessian = state["hessian"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                hessian.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                update = exp_avg.div(rho * hessian + eps).clamp_(-1.0, 1.0)
                p.add_(update, alpha=-lr)

        return loss


def build_sophia_g(model: nn.Module, entry: OptimizerEntry) -> OptimizerRuntime:
    optimizer = SophiaG(
        model.parameters(),
        lr=float(get_hparam(entry, "lr", (float, int))),
        betas=tuple(entry.hparams.get("betas", (0.965, 0.99))),
        rho=float(entry.hparams.get("rho", 0.04)),
        weight_decay=float(get_hparam(entry, "weight_decay", (float, int))),
        eps=float(entry.hparams.get("eps", 1e-12)),
    )
    return OptimizerRuntime(optimizer)
