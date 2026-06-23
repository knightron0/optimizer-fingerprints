"""Minimal effective-update tracing for NanoGPT optimizers."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import torch
import torch.distributed as dist


METRIC_NAMES = [
    "update_neg_grad_cos",
    "update_prev_update_cos",
    "grad_norm",
    "update_norm",
    "theta_norm",
    "update_theta_norm_ratio",
    "update_effective_lr_log_std",
    "update_grad_rms_suppression_corr",
    "update_cumulative_coverage",
    "update_neg_grad_ema_cos",
    "update_rank_fraction",
    "update_orthogonality_error",
]

EPS = 1e-12
HISTORY_BETA = 0.9


def _div(a, b):
    return a / b if abs(b) > 1e-20 else 0.0


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return max(-1.0, min(1.0, _div(torch.dot(a, b).item(), (a.norm() * b.norm()).item())))


def _corr(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    a, b = a - a.mean(), b - b.mean()
    return _cos(a, b)


def _matrix_metrics(name, update):
    if update.ndim != 2 or name in {"embed.weight", "proj.weight"}:
        return None, None
    matrix = update.float()
    singular_values = torch.linalg.svdvals(matrix)
    fro_sq = matrix.square().sum().item()
    top_sq = singular_values[0].square().item() if singular_values.numel() else 0.0
    rank_fraction = _div(_div(fro_sq, top_sq), min(matrix.shape))

    gram = matrix.T @ matrix if matrix.shape[0] >= matrix.shape[1] else matrix @ matrix.T
    dimension = gram.shape[0]
    alpha = gram.trace().item() / max(dimension, 1)
    target = torch.eye(gram.shape[0], dtype=gram.dtype) * alpha
    orthogonality_error = _div((gram - target).norm().item(), alpha * dimension**0.5)
    return rank_fraction, orthogonality_error


def _metrics(name, theta, gradient, update, history):
    update_flat = update.flatten()
    gradient_flat = gradient.flatten()
    theta_norm = theta.norm().item()
    grad_norm = gradient_flat.norm().item()
    update_norm = update_flat.norm().item()
    effective_lr_log = (update_flat.abs() / (gradient_flat.abs() + EPS) + EPS).log()

    history["grad_ema"].mul_(HISTORY_BETA).add_(gradient, alpha=1 - HISTORY_BETA)
    history["grad_sq_ema"].mul_(HISTORY_BETA).addcmul_(
        gradient, gradient, value=1 - HISTORY_BETA
    )
    history["coverage"].add_(update.abs())
    grad_rms_log = (history["grad_sq_ema"].sqrt().flatten() + EPS).log()
    coverage = history["coverage"].flatten()
    rank_fraction, orthogonality_error = _matrix_metrics(name, update)

    return {
        "update_neg_grad_cos": _cos(update_flat, -gradient_flat),
        "update_prev_update_cos": (
            _cos(update_flat, history["previous_update"])
            if history["previous_update"] is not None
            else None
        ),
        "grad_norm": grad_norm,
        "update_norm": update_norm,
        "theta_norm": theta_norm,
        "update_theta_norm_ratio": _div(update_norm, theta_norm + EPS),
        "update_effective_lr_log_std": effective_lr_log.std(unbiased=False).item(),
        "update_grad_rms_suppression_corr": _corr(grad_rms_log, effective_lr_log),
        "update_cumulative_coverage": _div(
            coverage.sum().item() ** 2,
            coverage.numel() * coverage.square().sum().item() + EPS,
        ),
        "update_neg_grad_ema_cos": _cos(update, -history["grad_ema"]),
        "update_rank_fraction": rank_fraction,
        "update_orthogonality_error": orthogonality_error,
    }


def _learning_rates(optimizer):
    direct = []
    missing_direct = False
    for group in optimizer.param_groups:
        if "lr" in group:
            direct.append(float(group["lr"]))
        else:
            missing_direct = True
    if direct and not missing_direct:
        return direct

    inner_optimizer = getattr(optimizer, "inner_optimizer", None)
    if inner_optimizer is not None:
        if isinstance(inner_optimizer, (list, tuple)):
            inner_optimizers = inner_optimizer
        else:
            inner_optimizers = [inner_optimizer]
        inner = [
            lr
            for inner_opt in inner_optimizers
            for lr in _learning_rates(inner_opt)
        ]
        if inner:
            return inner

    return direct


class OptimizerFingerprint:
    """Sample ``parameter_after - parameter_before`` around optimizer steps."""

    def __init__(self, model, optimizers, run_name, snapshot_interval, output_dir, wandb_run=None):
        if not optimizers:
            raise ValueError("At least one optimizer is required")
        if snapshot_interval < 1:
            raise ValueError("snapshot_interval must be positive")

        self.model = model
        self.optimizers = optimizers
        self.run_name = run_name
        self.snapshot_interval = snapshot_interval
        self.output_dir = Path(output_dir)
        self.step = 0
        self.expected_optimizer = 0
        self.sample = False
        self.captures = {}
        self.history = {}
        self.current_parameters = []
        self.snapshots = []
        self.handles = []
        self.path = None
        self.wandb_run = wandb_run
        self.refs = self._parameter_refs()
        self.rank0 = not dist.is_initialized() or dist.get_rank() == 0

        if self.rank0:
            for index, optimizer in enumerate(optimizers):
                self.handles.append(
                    optimizer.register_step_pre_hook(
                        lambda _opt, _args, _kwargs, i=index: self._before(i)
                    )
                )
                self.handles.append(
                    optimizer.register_step_post_hook(
                        lambda _opt, _args, _kwargs, i=index: self._after(i)
                    )
                )

    @classmethod
    def attach(
        cls,
        model,
        optimizers,
        *,
        run_name,
        snapshot_interval=25,
        output_dir="/scratch/gilbreth/mangla/nanogpt/traces",
        wandb_run=None,
    ):
        if isinstance(optimizers, torch.optim.Optimizer):
            optimizers = [optimizers]
        return cls(model, list(optimizers), run_name, snapshot_interval, output_dir, wandb_run)

    def _parameter_refs(self):
        names = {id(p): name for name, p in self.model.named_parameters() if p.requires_grad}
        refs, seen = [[] for _ in self.optimizers], set()
        for opt_i, optimizer in enumerate(self.optimizers):
            for group_i, group in enumerate(optimizer.param_groups):
                for parameter in group["params"]:
                    key = id(parameter)
                    if key in seen:
                        raise ValueError("A trainable parameter appears in more than one optimizer group")
                    if key not in names:
                        raise ValueError("An optimizer parameter is not a trainable model parameter")
                    seen.add(key)
                    refs[opt_i].append((names[key], parameter, group_i))
        missing = set(names) - seen
        if missing:
            missing_names = ", ".join(sorted(names[key] for key in missing))
            raise ValueError(f"Optimizers do not cover trainable parameters: {missing_names}")
        return refs

    def _before(self, optimizer_index):
        if optimizer_index != self.expected_optimizer:
            raise RuntimeError(f"Expected optimizer {self.expected_optimizer}, got {optimizer_index}")
        if optimizer_index == 0:
            self.sample = (self.step + 1) % self.snapshot_interval == 0
            self.current_parameters = []
        if not self.sample:
            return
        self.captures[optimizer_index] = [
            (
                name,
                parameter,
                group_index,
                parameter.detach().cpu().clone(),
                parameter.grad.detach().cpu().clone()
                if parameter.grad is not None
                else torch.zeros_like(parameter, device="cpu"),
            )
            for name, parameter, group_index in self.refs[optimizer_index]
        ]

    @torch.no_grad()
    def _after(self, optimizer_index):
        if optimizer_index != self.expected_optimizer:
            raise RuntimeError(f"Unexpected optimizer completion: {optimizer_index}")
        if self.sample:
            for name, parameter, group_index, before, gradient in self.captures.pop(optimizer_index):
                update = parameter.detach().cpu().float() - before.float()
                key = id(parameter)
                history = self.history.setdefault(
                    key,
                    {
                        "previous_update": None,
                        "grad_ema": torch.zeros_like(update),
                        "grad_sq_ema": torch.zeros_like(update),
                        "coverage": torch.zeros_like(update),
                    },
                )
                self.current_parameters.append(
                    {
                        "name": name,
                        "shape": list(parameter.shape),
                        "optimizer_index": optimizer_index,
                        "group_index": group_index,
                        "metrics": _metrics(
                            name,
                            before.float(),
                            gradient.float(),
                            update,
                            history,
                        ),
                    }
                )
                history["previous_update"] = update

        self.expected_optimizer += 1
        if self.expected_optimizer == len(self.optimizers):
            self.step += 1
            if self.sample:
                self.current_parameters.sort(key=lambda item: item["name"])
                snapshot = {
                    "step": self.step,
                    "learning_rates": [
                        _learning_rates(optimizer) for optimizer in self.optimizers
                    ],
                    "parameters": self.current_parameters,
                }
                self.snapshots.append(snapshot)
                self._log_wandb(snapshot)
            self.expected_optimizer = 0
            self.sample = False

    def _log_wandb(self, snapshot):
        if self.wandb_run is None:
            return
        metrics = {}
        for parameter in snapshot["parameters"]:
            parameter_name = parameter["name"].replace(".", "/")
            for metric_name in METRIC_NAMES:
                value = parameter["metrics"][metric_name]
                if value is not None:
                    metrics[f"{metric_name}/{parameter_name}"] = value
        self.wandb_run.log(metrics, step=snapshot["step"], commit=True)
        print(f"wandb logged {len(metrics)} metrics at step {snapshot['step']}", flush=True)

    def finish(self):
        if not self.rank0:
            return None
        if self.path is not None:
            return self.path
        if self.expected_optimizer:
            raise RuntimeError("Cannot finish during an incomplete optimizer step")
        for handle in self.handles:
            handle.remove()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^A-Za-z0-9_.-]+", "-", self.run_name).strip("-.") or "run"
        self.path = self.output_dir / f"{name}-{uuid.uuid4().hex}.json"
        temporary = self.path.with_suffix(".json.tmp")
        payload = {
            "schema": "nanogpt_optimizer_trace",
            "run_name": self.run_name,
            "completed_steps": self.step,
            "snapshot_interval": self.snapshot_interval,
            "history_beta": HISTORY_BETA,
            "epsilon": EPS,
            "history_semantics": "updated on sampled snapshots",
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "optimizer_classes": [optimizer.__class__.__name__ for optimizer in self.optimizers],
            "metric_names": METRIC_NAMES,
            "snapshots": self.snapshots,
        }
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(self.path)
        self.history.clear()
        return self.path
