from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor
from torch import nn
from torch.optim import Optimizer

from .entry import OptimizerEntry, OptimizerRuntime, get_hparam, split_params


def _zeropower_via_newton_schulz(x: Tensor, steps: int) -> Tensor:
    """Approximate the zeroth power using the modded-nanogpt Muon iteration."""
    if x.ndim < 2:
        raise ValueError(f"Muon expects tensors with at least 2 dimensions, got shape {tuple(x.shape)}")

    original_dtype = x.dtype
    original_shape = x.shape
    x = x.reshape(x.shape[0], -1).bfloat16()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T

    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    a, b, c = 2.0, -1.5, 0.5
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x

    if transposed:
        x = x.T
    return x.reshape(original_shape).to(dtype=original_dtype)


class MuonWithAuxAdam(Optimizer):
    """Single-device Muon for matrix params plus AdamW for auxiliary params.

    Muon is applied to params in groups marked ``use_muon=True``. Other groups
    receive a compact AdamW update. This matches the practical split used in
    Muon experiments: matrix weights get orthogonalized updates, biases and
    scalar/vector parameters get AdamW.
    """

    def __init__(
        self,
        param_groups: Iterable[dict],
        *,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        mu: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 12,
        adam_lr: float = 1e-3,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
        adam_weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            mu=mu,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adam_lr=adam_lr,
            adam_betas=adam_betas,
            adam_eps=adam_eps,
            adam_weight_decay=adam_weight_decay,
            use_muon=True,
        )
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", True):
                self._step_muon(group)
            else:
                self._step_adam(group)
        return loss

    def _step_muon(self, group: dict) -> None:
        lr = group["lr"]
        wd = group["weight_decay"]
        mu = group["mu"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]

        for p in group["params"]:
            if p.grad is None:
                continue
            if p.ndim < 2:
                raise ValueError("Muon groups should contain only matrix-like parameters")

            grad = p.grad
            state = self.state[p]
            if "momentum" not in state:
                state["momentum"] = torch.zeros_like(p)
            momentum = state["momentum"]
            momentum.lerp_(grad, 1 - mu)
            update = grad.lerp(momentum, mu) if nesterov else momentum
            update = _zeropower_via_newton_schulz(update, ns_steps)
            matrix_cols = p.numel() // p.size(0)
            update *= max(1.0, p.size(0) / matrix_cols) ** 0.5

            if wd != 0:
                p.mul_(1 - lr * wd)
            p.add_(update, alpha=-lr)

    def _step_adam(self, group: dict) -> None:
        lr = group["adam_lr"]
        beta1, beta2 = group["adam_betas"]
        eps = group["adam_eps"]
        wd = group["adam_weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            if wd != 0:
                p.mul_(1 - lr * wd)

            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]
            step_size = lr / bias_correction1
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
            p.addcdiv_(exp_avg, denom, value=-step_size)


def build_muon(model: nn.Module, entry: OptimizerEntry) -> OptimizerRuntime:
    matrix_params, aux_params = split_params(model)
    optimizer = MuonWithAuxAdam(
        [
            {"params": matrix_params, "use_muon": True},
            {"params": aux_params, "use_muon": False},
        ],
        lr=float(get_hparam(entry, "lr", (float, int))),
        weight_decay=float(get_hparam(entry, "weight_decay", (float, int))),
        mu=float(entry.hparams.get("mu", 0.95)),
        nesterov=bool(entry.hparams.get("nesterov", True)),
        ns_steps=int(entry.hparams.get("ns_steps", 12)),
        adam_lr=float(get_hparam(entry, "adam_lr", (float, int))),
        adam_betas=tuple(entry.hparams.get("adam_betas", (0.9, 0.95))),
        adam_eps=float(entry.hparams.get("adam_eps", 1e-8)),
        adam_weight_decay=float(entry.hparams.get("adam_weight_decay", entry.hparams["weight_decay"])),
    )
    return OptimizerRuntime(optimizer)
