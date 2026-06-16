from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
import yaml


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "optimizers"


@dataclass(frozen=True)
class OptimizerEntry:
    name: str
    family: str
    hparams: dict[str, Any]
    param_groups: dict[str, Any]
    metadata: dict[str, Any]
    config_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "hparams": copy.deepcopy(self.hparams),
            "param_groups": copy.deepcopy(self.param_groups),
            "metadata": copy.deepcopy(self.metadata),
            "config_path": self.config_path,
        }

    def build(self, model: nn.Module) -> OptimizerRuntime:
        from .adamw import build_adamw
        from .muon import build_muon
        from .shampoo import build_shampoo

        builders = {
            "adamw": build_adamw,
            "muon": build_muon,
            "shampoo": build_shampoo,
        }
        try:
            builder = builders[self.family]
        except KeyError as exc:
            raise ValueError(f"Unsupported optimizer family: {self.family}") from exc
        return builder(model, self)


class OptimizerRuntime:
    def __init__(self, optimizers: torch.optim.Optimizer | list[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers if isinstance(optimizers, list) else [optimizers]

    @property
    def param_groups(self) -> list[dict]:
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def assert_covers(self, model: nn.Module) -> None:
        model_params = {p for p in model.parameters() if p.requires_grad}
        opt_params = [p for group in self.param_groups for p in group["params"]]
        if len(opt_params) != len(set(opt_params)):
            raise RuntimeError("A parameter appears in multiple optimizer groups")
        if set(opt_params) != model_params:
            raise RuntimeError("Optimizer parameter groups do not cover trainable model parameters exactly")


def available_optimizer_names(config_dir: Path = CONFIG_DIR) -> list[str]:
    if not config_dir.exists():
        return []
    return sorted(path.stem for path in config_dir.glob("*.yaml"))


def load_optimizer_entry(name: str, config_dir: Path = CONFIG_DIR) -> OptimizerEntry:
    path = config_dir / f"{name}.yaml"
    if not path.exists():
        choices = ", ".join(available_optimizer_names(config_dir))
        raise ValueError(f"Unknown optimizer entry {name!r}. Available entries: {choices}")
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Optimizer config {path} must contain a YAML mapping")
    return _entry_from_payload(payload, path)


def apply_overrides(entry: OptimizerEntry, overrides: list[str]) -> OptimizerEntry:
    payload = entry.to_dict()
    payload.pop("config_path", None)
    for override in overrides:
        key, value = _parse_override(override)
        _set_nested(payload, key.split("."), value)
    return _entry_from_payload(payload, Path(entry.config_path))


def split_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    matrix_params: list[nn.Parameter] = []
    aux_params: list[nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            matrix_params.append(param)
        else:
            aux_params.append(param)
    return matrix_params, aux_params


def get_hparam(entry: OptimizerEntry, name: str, expected_type: type | tuple[type, ...]) -> Any:
    if name not in entry.hparams:
        raise ValueError(f"Optimizer entry {entry.name!r} is missing hparams.{name}")
    value = entry.hparams[name]
    if not isinstance(value, expected_type):
        raise TypeError(f"hparams.{name} must be {expected_type}, got {type(value)}")
    return value


def _entry_from_payload(payload: dict[str, Any], path: Path) -> OptimizerEntry:
    name = payload.get("name")
    family = payload.get("family")
    hparams = payload.get("hparams")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Optimizer config {path} must define string field 'name'")
    if not isinstance(family, str) or not family:
        raise ValueError(f"Optimizer config {path} must define string field 'family'")
    if not isinstance(hparams, dict):
        raise ValueError(f"Optimizer config {path} must define mapping field 'hparams'")
    param_groups = payload.get("param_groups", {})
    metadata = payload.get("metadata", {})
    if not isinstance(param_groups, dict):
        raise ValueError(f"Optimizer config {path} field 'param_groups' must be a mapping")
    if not isinstance(metadata, dict):
        raise ValueError(f"Optimizer config {path} field 'metadata' must be a mapping")
    return OptimizerEntry(
        name=name,
        family=family,
        hparams=copy.deepcopy(hparams),
        param_groups=copy.deepcopy(param_groups),
        metadata=copy.deepcopy(metadata),
        config_path=str(path),
    )


def _parse_override(override: str) -> tuple[str, Any]:
    if "=" not in override:
        raise ValueError(f"Override must have form key=value, got {override!r}")
    key, raw_value = override.split("=", 1)
    if not key:
        raise ValueError(f"Override key cannot be empty: {override!r}")
    value = yaml.safe_load(raw_value)
    if isinstance(value, (dict, list, tuple)):
        raise ValueError(f"Override values must be scalar, got {override!r}")
    return key, value


def _set_nested(payload: dict[str, Any], parts: list[str], value: Any) -> None:
    cursor = payload
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            raise ValueError(f"Override path does not exist or is not a mapping: {'.'.join(parts)}")
        cursor = next_value
    final_key = parts[-1]
    if final_key not in cursor:
        raise ValueError(f"Override key does not exist: {'.'.join(parts)}")
    cursor[final_key] = value
