from __future__ import annotations

from torch import nn
from torch.optim import SGD

from .entry import OptimizerEntry, OptimizerRuntime, get_hparam


def build_sgd(model: nn.Module, entry: OptimizerEntry) -> OptimizerRuntime:
    optimizer = SGD(
        model.parameters(),
        lr=float(get_hparam(entry, "lr", (float, int))),
        momentum=float(entry.hparams.get("momentum", 0.0)),
        dampening=float(entry.hparams.get("dampening", 0.0)),
        weight_decay=float(get_hparam(entry, "weight_decay", (float, int))),
        nesterov=bool(entry.hparams.get("nesterov", False)),
    )
    return OptimizerRuntime(optimizer)
