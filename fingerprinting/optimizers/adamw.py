from __future__ import annotations

from torch import nn
from torch.optim import AdamW

from .entry import OptimizerEntry, OptimizerRuntime, get_hparam


def build_adamw(model: nn.Module, entry: OptimizerEntry) -> OptimizerRuntime:
    optimizer = AdamW(
        model.parameters(),
        lr=float(get_hparam(entry, "lr", (float, int))),
        betas=tuple(entry.hparams.get("betas", (0.9, 0.95))),
        weight_decay=float(get_hparam(entry, "weight_decay", (float, int))),
        eps=float(entry.hparams.get("eps", 1e-8)),
    )
    return OptimizerRuntime(optimizer)
