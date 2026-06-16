from __future__ import annotations

from torch import nn

from .entry import (
    OptimizerEntry,
    OptimizerRuntime,
    apply_overrides,
    available_optimizer_names,
    load_optimizer_entry,
)


def build_optimizer_entry(
    model: nn.Module,
    name: str,
    *,
    overrides: list[str] | None = None,
) -> tuple[OptimizerRuntime, OptimizerEntry]:
    entry = load_optimizer_entry(name)
    if overrides:
        entry = apply_overrides(entry, overrides)
    runtime = entry.build(model)
    runtime.assert_covers(model)
    return runtime, entry


__all__ = [
    "OptimizerEntry",
    "OptimizerRuntime",
    "available_optimizer_names",
    "build_optimizer_entry",
    "load_optimizer_entry",
]
