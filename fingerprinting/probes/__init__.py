from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn


SCHEMA = "optimizer_fingerprint"

METRIC_NAMES = [
    "loss",
    "grad_norm",
    "update_norm",
    "theta_norm",
    "path_length",
    "displacement",
    "directness",
    "update_neg_grad_cos",
    "update_prev_update_cos",
    "update_grad_norm_ratio",
    "update_theta_norm_ratio",
    "cosine_from_initial",
    "cosine_from_previous_snapshot",
]

PARAMETER_METRIC_NAMES = [
    "grad_norm",
    "update_norm",
    "theta_norm",
    "path_length",
    "displacement",
    "directness",
    "update_neg_grad_cos",
    "update_prev_update_cos",
    "update_grad_norm_ratio",
    "update_theta_norm_ratio",
    "cosine_from_initial",
    "cosine_from_previous_snapshot",
    "matrix_update_grad_cos",
    "matrix_update_polar_grad_cos",
    "matrix_update_effective_rank",
    "matrix_update_singular_entropy",
    "matrix_update_nuclear_fro_ratio",
]


@dataclass(frozen=True)
class ProbeConfig:
    max_steps: int = 300
    snapshot_interval: int = 25
    svd_max_dim: int = 512


@dataclass
class ParameterProbeState:
    path_length: float
    prev_update_flat: Tensor | None
    initial_flat: Tensor
    initial_normed: Tensor
    previous_snapshot_normed: Tensor | None


@dataclass(frozen=True)
class ParameterStepObservation:
    grad_flat: Tensor
    theta_norm: float
    update: Tensor
    update_flat: Tensor
    update_norm: float


def _safe_div(num: float, denom: float) -> float:
    if denom <= 1e-20:
        return 0.0
    return num / denom


def _cosine(a: Tensor, b: Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    denom = a.norm() * b.norm()
    if denom.item() <= 1e-20:
        return 0.0
    return torch.dot(a, b).div(denom).clamp(-1.0, 1.0).item()


def _flatten_params(params: Iterable[nn.Parameter]) -> Tensor:
    return torch.cat([p.detach().float().cpu().flatten() for p in params])


def _flatten_param(param: nn.Parameter) -> Tensor:
    return param.detach().float().cpu().flatten().clone()


def _normalize(flat: Tensor) -> Tensor:
    return flat / flat.norm().clamp_min(1e-20)


def _parameter_probe_state(param: nn.Parameter) -> ParameterProbeState:
    initial_flat = _flatten_param(param)
    return ParameterProbeState(
        path_length=0.0,
        prev_update_flat=None,
        initial_flat=initial_flat,
        initial_normed=_normalize(initial_flat),
        previous_snapshot_normed=None,
    )


def _matrix_view(tensor: Tensor) -> Tensor:
    if tensor.ndim == 2:
        return tensor.detach().float()
    return tensor.detach().float().reshape(tensor.shape[0], -1)


def _downsample_matrix(matrix: Tensor, max_dim: int) -> Tensor:
    rows, cols = matrix.shape
    row_idx = torch.linspace(0, rows - 1, min(rows, max_dim), device=matrix.device).long()
    col_idx = torch.linspace(0, cols - 1, min(cols, max_dim), device=matrix.device).long()
    return matrix.index_select(0, row_idx).index_select(1, col_idx)


def _polar_factor(matrix: Tensor) -> Tensor:
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return u @ vh


def _effective_rank(matrix: Tensor) -> tuple[float, float, float]:
    singular_values = torch.linalg.svdvals(matrix)
    total = singular_values.sum()
    fro = matrix.norm()
    if total.item() <= 1e-20:
        return 0.0, 0.0, 0.0
    probs = singular_values / total
    entropy = -(probs * torch.log(probs.clamp_min(1e-20))).sum().item()
    effective_rank = math.exp(entropy)
    nuclear_fro_ratio = _safe_div(total.item(), fro.item())
    return effective_rank, entropy, nuclear_fro_ratio


class FingerprintAccumulator:
    def __init__(
        self,
        *,
        model: nn.Module,
        probe_config: ProbeConfig,
    ) -> None:
        self.model = model
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        self.probe_config = probe_config
        self.path_length = 0.0
        self.prev_update_flat: Tensor | None = None
        self.initial_flat = _flatten_params(self.params)
        self.initial_normed = _normalize(self.initial_flat)
        self.previous_snapshot_normed: Tensor | None = None
        self.parameter_states = [_parameter_probe_state(param) for param in self.params]
        self.snapshots: list[dict] = []

    def capture_before_step(self) -> list[Tensor]:
        return [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def observe_step(self, *, step: int, before_params: list[Tensor], loss: float) -> None:
        update_parts: list[Tensor] = []
        grad_parts: list[Tensor] = []
        parameter_step_observations: list[ParameterStepObservation] = []
        theta_norm_sq = 0.0

        for before, param, state in zip(before_params, self.params, self.parameter_states, strict=True):
            update = param.detach() - before
            update_flat_param = update.float().cpu().flatten()
            before_flat_param = before.float().cpu().flatten()
            theta_norm_param = before_flat_param.norm().item()
            update_parts.append(update_flat_param)
            theta_norm_sq += theta_norm_param**2
            if param.grad is None:
                grad_flat_param = torch.zeros_like(update, dtype=torch.float32).cpu().flatten()
            else:
                grad_flat_param = param.grad.detach().float().cpu().flatten()
            grad_parts.append(grad_flat_param)
            update_norm_param = update_flat_param.norm().item()
            state.path_length += update_norm_param
            parameter_step_observations.append(
                ParameterStepObservation(
                    grad_flat=grad_flat_param,
                    theta_norm=theta_norm_param,
                    update=update,
                    update_flat=update_flat_param,
                    update_norm=update_norm_param,
                )
            )

        update_flat = torch.cat(update_parts)
        grad_flat = torch.cat(grad_parts)
        update_norm = update_flat.norm().item()
        grad_norm = grad_flat.norm().item()
        theta_norm = math.sqrt(theta_norm_sq)
        self.path_length += update_norm

        should_snapshot = step % self.probe_config.snapshot_interval == 0 or step == self.probe_config.max_steps
        if should_snapshot:
            flat = _flatten_params(self.params)
            normalized_flat = _normalize(flat)
            displacement = (flat - self.initial_flat).norm().item()
            metrics = {
                "loss": loss,
                "grad_norm": grad_norm,
                "update_norm": update_norm,
                "theta_norm": theta_norm,
                "path_length": self.path_length,
                "displacement": displacement,
                "directness": _safe_div(displacement, self.path_length),
                "update_neg_grad_cos": _cosine(update_flat, -grad_flat),
                "update_prev_update_cos": (
                    _cosine(update_flat, self.prev_update_flat) if self.prev_update_flat is not None else None
                ),
                "update_grad_norm_ratio": _safe_div(update_norm, grad_norm),
                "update_theta_norm_ratio": _safe_div(update_norm, theta_norm),
                "cosine_from_initial": torch.dot(self.initial_normed, normalized_flat).clamp(-1.0, 1.0).item(),
                "cosine_from_previous_snapshot": (
                    torch.dot(self.previous_snapshot_normed, normalized_flat).clamp(-1.0, 1.0).item()
                    if self.previous_snapshot_normed is not None
                    else None
                ),
            }
            parameters = self._observe_parameters(parameter_step_observations)
            self.snapshots.append({"step": step, "metrics": metrics, "parameters": parameters})
            self.previous_snapshot_normed = normalized_flat

        for state, observation in zip(self.parameter_states, parameter_step_observations, strict=True):
            state.prev_update_flat = observation.update_flat
        self.prev_update_flat = update_flat

    def _observe_parameters(self, parameter_step_observations: list[ParameterStepObservation]) -> list[dict]:
        snapshots: list[dict] = []
        for (name, param), state, observation in zip(
            self.named_params,
            self.parameter_states,
            parameter_step_observations,
            strict=True,
        ):
            update_flat = observation.update_flat
            grad_flat = observation.grad_flat
            flat = param.detach().float().cpu().flatten()
            normalized_flat = _normalize(flat)
            displacement = (flat - state.initial_flat).norm().item()
            update_norm = observation.update_norm
            grad_norm = grad_flat.norm().item()
            theta_norm = observation.theta_norm
            metrics = {
                "grad_norm": grad_norm,
                "update_norm": update_norm,
                "theta_norm": theta_norm,
                "path_length": state.path_length,
                "displacement": displacement,
                "directness": _safe_div(displacement, state.path_length),
                "update_neg_grad_cos": _cosine(update_flat, -grad_flat),
                "update_prev_update_cos": (
                    _cosine(update_flat, state.prev_update_flat) if state.prev_update_flat is not None else None
                ),
                "update_grad_norm_ratio": _safe_div(update_norm, grad_norm),
                "update_theta_norm_ratio": _safe_div(update_norm, theta_norm),
                "cosine_from_initial": torch.dot(state.initial_normed, normalized_flat).clamp(-1.0, 1.0).item(),
                "cosine_from_previous_snapshot": (
                    torch.dot(state.previous_snapshot_normed, normalized_flat).clamp(-1.0, 1.0).item()
                    if state.previous_snapshot_normed is not None
                    else None
                ),
                **self._observe_parameter_matrix_structure(param, observation),
            }
            snapshots.append(
                {
                    "name": name,
                    "shape": list(param.shape),
                    "ndim": param.ndim,
                    "numel": param.numel(),
                    "metrics": metrics,
                }
            )
            state.previous_snapshot_normed = normalized_flat
        return snapshots

    def _observe_parameter_matrix_structure(
        self,
        param: nn.Parameter,
        observation: ParameterStepObservation,
    ) -> dict[str, float | None]:
        metric_names = [
            "matrix_update_grad_cos",
            "matrix_update_polar_grad_cos",
            "matrix_update_effective_rank",
            "matrix_update_singular_entropy",
            "matrix_update_nuclear_fro_ratio",
        ]
        empty = {name: None for name in metric_names}
        if param.ndim < 2 or param.grad is None:
            return empty
        update = _matrix_view(observation.update)
        grad = _matrix_view(param.grad.detach())
        if update.numel() == 0 or grad.numel() == 0:
            return empty
        update_small = _downsample_matrix(update, self.probe_config.svd_max_dim)
        grad_small = _downsample_matrix(grad, self.probe_config.svd_max_dim)
        effective_rank, entropy, nuclear_fro_ratio = _effective_rank(update_small)
        return {
            "matrix_update_grad_cos": _cosine(update_small, grad_small),
            "matrix_update_polar_grad_cos": _cosine(update_small, _polar_factor(grad_small)),
            "matrix_update_effective_rank": effective_rank,
            "matrix_update_singular_entropy": entropy,
            "matrix_update_nuclear_fro_ratio": nuclear_fro_ratio,
        }

    def finalize(self) -> dict:
        return {
            "schema": SCHEMA,
            "metric_names": METRIC_NAMES,
            "parameter_metric_names": PARAMETER_METRIC_NAMES,
            "snapshots": self.snapshots,
        }


def compare_fingerprints(left: dict, right: dict) -> dict:
    raise NotImplementedError("Fingerprint comparison is not implemented for snapshot-only fingerprints.")
