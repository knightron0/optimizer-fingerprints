from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn

from .stats import FractionCounter, NamedStats, RunningStats


SCHEMA_VERSION = "optimizer_fingerprint_v1"


@dataclass(frozen=True)
class ProbeConfig:
    max_steps: int = 300
    log_interval: int = 10
    checkpoint_interval: int = 50
    matrix_probe_interval: int = 25
    svd_max_dim: int = 512


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
        trace_path: Path,
    ) -> None:
        self.model = model
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        self.probe_config = probe_config
        self.trace_path = trace_path
        self.trace_path.touch()
        self.stats = NamedStats()
        self.negative_update_cos = FractionCounter()
        self.update_norm_stats = RunningStats()
        self.path_length = 0.0
        self.prev_update_flat: Tensor | None = None
        self.initial_flat = _flatten_params(self.params)
        self.checkpoints: list[tuple[int, Tensor]] = [(0, self.initial_flat / self.initial_flat.norm().clamp_min(1e-20))]

    def capture_before_step(self) -> list[Tensor]:
        return [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def observe_step(self, *, step: int, before_params: list[Tensor], loss: float) -> None:
        update_parts: list[Tensor] = []
        grad_parts: list[Tensor] = []
        theta_norm_sq = 0.0

        for before, param in zip(before_params, self.params, strict=True):
            update = param.detach() - before
            update_parts.append(update.float().cpu().flatten())
            theta_norm_sq += before.float().norm().item() ** 2
            if param.grad is None:
                grad_parts.append(torch.zeros_like(update, dtype=torch.float32).cpu().flatten())
            else:
                grad_parts.append(param.grad.detach().float().cpu().flatten())

        update_flat = torch.cat(update_parts)
        grad_flat = torch.cat(grad_parts)
        update_norm = update_flat.norm().item()
        grad_norm = grad_flat.norm().item()
        theta_norm = math.sqrt(theta_norm_sq)
        self.path_length += update_norm
        self.update_norm_stats.add(update_norm)

        grad_alignment = _cosine(update_flat, -grad_flat)
        self.stats.add("direction.update_neg_grad_cos", grad_alignment)
        if self.prev_update_flat is not None:
            update_alignment = _cosine(update_flat, self.prev_update_flat)
            self.stats.add("direction.update_prev_update_cos", update_alignment)
            self.negative_update_cos.add(update_alignment < 0.0)
        self.stats.add("scale.update_grad_norm_ratio", _safe_div(update_norm, grad_norm))
        self.stats.add("scale.update_theta_norm_ratio", _safe_div(update_norm, theta_norm))

        matrix_row: dict[str, float] = {}
        if step % self.probe_config.matrix_probe_interval == 0:
            matrix_row = self._observe_matrix_structure(before_params)

        if step % self.probe_config.checkpoint_interval == 0:
            flat = _flatten_params(self.params)
            self.checkpoints.append((step, flat / flat.norm().clamp_min(1e-20)))

        if step % self.probe_config.log_interval == 0 or step == 1:
            row = {
                "step": step,
                "loss": loss,
                "grad_norm": grad_norm,
                "update_norm": update_norm,
                "theta_norm": theta_norm,
                "cos_update_neg_grad": grad_alignment,
                "path_length": self.path_length,
                **matrix_row,
            }
            with self.trace_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")

        self.prev_update_flat = update_flat

    def _observe_matrix_structure(self, before_params: list[Tensor]) -> dict[str, float]:
        local = NamedStats()
        for (_, param), before in zip(self.named_params, before_params, strict=True):
            if param.ndim < 2 or param.grad is None:
                continue
            update = _matrix_view(param.detach() - before)
            grad = _matrix_view(param.grad.detach())
            if update.numel() == 0 or grad.numel() == 0:
                continue
            update_small = _downsample_matrix(update, self.probe_config.svd_max_dim)
            grad_small = _downsample_matrix(grad, self.probe_config.svd_max_dim)
            local.add("matrix.update_grad_cos", _cosine(update_small, grad_small))
            local.add("matrix.update_polar_grad_cos", _cosine(update_small, _polar_factor(grad_small)))
            effective_rank, entropy, nuclear_fro_ratio = _effective_rank(update_small)
            local.add("matrix.update_effective_rank", effective_rank)
            local.add("matrix.update_singular_entropy", entropy)
            local.add("matrix.update_nuclear_fro_ratio", nuclear_fro_ratio)

        for name, stat in local.stats.items():
            self.stats.add(name, stat.mean)

        return {
            "matrix_cos_update_grad": local.mean("matrix.update_grad_cos"),
            "matrix_cos_update_polar_grad": local.mean("matrix.update_polar_grad_cos"),
            "matrix_effective_rank": local.mean("matrix.update_effective_rank"),
        }

    def finalize(self) -> dict:
        final_flat = _flatten_params(self.params)
        displacement = (final_flat - self.initial_flat).norm().item()
        directness = _safe_div(displacement, self.path_length)

        checkpoint_cosines: list[float] = []
        consecutive_cosines: list[float] = []
        for i, (_, left) in enumerate(self.checkpoints):
            for _, right in self.checkpoints[i + 1 :]:
                checkpoint_cosines.append(torch.dot(left, right).clamp(-1.0, 1.0).item())
        for (_, left), (_, right) in zip(self.checkpoints, self.checkpoints[1:], strict=False):
            consecutive_cosines.append(torch.dot(left, right).clamp(-1.0, 1.0).item())

        mds = sum(checkpoint_cosines) / len(checkpoint_cosines) if checkpoint_cosines else 1.0
        consecutive_mean = sum(consecutive_cosines) / len(consecutive_cosines) if consecutive_cosines else 1.0
        consecutive_std = (
            torch.tensor(consecutive_cosines).std(unbiased=True).item() if len(consecutive_cosines) > 1 else 0.0
        )

        features = {
            "direction.update_neg_grad_cos.mean": self.stats.mean("direction.update_neg_grad_cos"),
            "direction.update_neg_grad_cos.std": self.stats.std("direction.update_neg_grad_cos"),
            "direction.update_prev_update_cos.mean": self.stats.mean("direction.update_prev_update_cos"),
            "direction.update_prev_update_cos.std": self.stats.std("direction.update_prev_update_cos"),
            "direction.update_prev_update_cos.frac_negative": self.negative_update_cos.fraction,
            "scale.update_grad_norm_ratio.mean": self.stats.mean("scale.update_grad_norm_ratio"),
            "scale.update_grad_norm_ratio.std": self.stats.std("scale.update_grad_norm_ratio"),
            "scale.update_theta_norm_ratio.mean": self.stats.mean("scale.update_theta_norm_ratio"),
            "scale.update_theta_norm_ratio.std": self.stats.std("scale.update_theta_norm_ratio"),
            "scale.update_norm.cv": self.update_norm_stats.cv,
            "trajectory.mean_directional_similarity": mds,
            "trajectory.directness": directness,
            "trajectory.checkpoint_consecutive_cos.mean": consecutive_mean,
            "trajectory.checkpoint_consecutive_cos.std": consecutive_std,
            "matrix.update_grad_cos.mean": self.stats.mean("matrix.update_grad_cos"),
            "matrix.update_grad_cos.std": self.stats.std("matrix.update_grad_cos"),
            "matrix.update_polar_grad_cos.mean": self.stats.mean("matrix.update_polar_grad_cos"),
            "matrix.update_polar_grad_cos.std": self.stats.std("matrix.update_polar_grad_cos"),
            "matrix.update_effective_rank.mean": self.stats.mean("matrix.update_effective_rank"),
            "matrix.update_singular_entropy.mean": self.stats.mean("matrix.update_singular_entropy"),
            "matrix.update_nuclear_fro_ratio.mean": self.stats.mean("matrix.update_nuclear_fro_ratio"),
        }
        feature_names = list(features)
        raw_vector = [features[name] for name in feature_names]
        normalized_vector = _normalize_vector_by_block(feature_names, raw_vector)
        return {
            "schema_version": SCHEMA_VERSION,
            "feature_names": feature_names,
            "features": features,
            "raw_vector": raw_vector,
            "normalized_vector": normalized_vector,
            "block_weights": {
                "direction": 1.0,
                "scale": 1.0,
                "trajectory": 1.0,
                "matrix": 1.0,
            },
            "probe_config": asdict(self.probe_config),
            "checkpoint_steps": [step for step, _ in self.checkpoints],
            "path_length": self.path_length,
            "displacement": displacement,
        }


def _normalize_vector_by_block(feature_names: list[str], values: list[float]) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    normalized = torch.zeros_like(tensor)
    blocks = sorted({name.split(".", 1)[0] for name in feature_names})
    for block in blocks:
        indices = [idx for idx, name in enumerate(feature_names) if name.startswith(f"{block}.")]
        block_values = tensor[indices]
        block_norm = block_values.norm().clamp_min(1e-12)
        normalized[indices] = block_values.div(block_norm * math.sqrt(len(blocks)))
    return normalized.tolist()


def compare_fingerprints(left: dict, right: dict) -> dict:
    if left["schema_version"] != right["schema_version"]:
        raise ValueError("Cannot compare fingerprints with different schema versions")
    if left["feature_names"] != right["feature_names"]:
        raise ValueError("Cannot compare fingerprints with different feature sets")
    left_world = left.get("world", {}).get("world_id")
    right_world = right.get("world", {}).get("world_id")
    if left_world != right_world:
        raise ValueError(f"Cannot compare different worlds: {left_world!r} vs {right_world!r}")

    lv = torch.tensor(left["normalized_vector"], dtype=torch.float32)
    rv = torch.tensor(right["normalized_vector"], dtype=torch.float32)
    distance = torch.linalg.vector_norm(lv - rv).item()
    cosine = torch.nn.functional.cosine_similarity(lv, rv, dim=0).item()
    return {
        "schema_version": left["schema_version"],
        "world_id": left_world,
        "distance": distance,
        "cosine_similarity": cosine,
        "feature_count": len(left["feature_names"]),
    }
