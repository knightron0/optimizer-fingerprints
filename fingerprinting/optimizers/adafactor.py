from __future__ import annotations

from torch import nn
from torch.optim import Adafactor

from .entry import OptimizerEntry, OptimizerRuntime, get_hparam


def build_adafactor(model: nn.Module, entry: OptimizerEntry) -> OptimizerRuntime:
    optimizer = Adafactor(
        model.parameters(),
        lr=float(get_hparam(entry, "lr", (float, int))),
        beta2_decay=float(entry.hparams.get("beta2_decay", -0.8)),
        eps=tuple(entry.hparams.get("eps", (None, 1e-3))),
        d=float(entry.hparams.get("d", 1.0)),
        weight_decay=float(get_hparam(entry, "weight_decay", (float, int))),
    )
    return OptimizerRuntime(optimizer)
