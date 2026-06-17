from __future__ import annotations

from torch import nn
from torch.optim import AdamW

from distributed_shampoo import (
    AdamPreconditionerConfig,
    DefaultSOAPConfig,
    DistributedShampoo,
    EigenConfig,
    PseudoInverseConfig,
    RootInvShampooPreconditionerConfig,
    SingleDeviceDistributedConfig,
    WeightDecayType,
)

from .entry import OptimizerEntry, OptimizerRuntime, get_hparam, split_params


def build_shampoo(model: nn.Module, entry: OptimizerEntry) -> OptimizerRuntime:
    matrix_params, aux_params = split_params(model)
    preconditioner_config = _build_preconditioner_config(entry)
    epsilon = float(get_hparam(entry, "epsilon", (float, int)))
    beta2 = float(get_hparam(entry, "beta2", (float, int)))
    weight_decay = float(get_hparam(entry, "weight_decay", (float, int)))

    shampoo = DistributedShampoo(
        matrix_params,
        lr=float(get_hparam(entry, "lr", (float, int))),
        betas=tuple(entry.hparams.get("betas", (0.9, beta2))),
        epsilon=epsilon,
        weight_decay=weight_decay,
        weight_decay_type=WeightDecayType.DECOUPLED,
        max_preconditioner_dim=int(entry.hparams.get("max_preconditioner_dim", 8192)),
        precondition_frequency=int(entry.hparams.get("precondition_frequency", 1)),
        start_preconditioning_step=int(entry.hparams.get("start_preconditioning_step", -1)),
        preconditioner_config=preconditioner_config,
        grafting_config=AdamPreconditionerConfig(beta2=beta2, epsilon=float(entry.hparams.get("grafting_epsilon", 1e-15))),
        distributed_config=SingleDeviceDistributedConfig(),
    )
    aux_adam = AdamW(
        aux_params,
        lr=float(get_hparam(entry, "adam_lr", (float, int))),
        betas=tuple(entry.hparams.get("adam_betas", (0.9, 0.95))),
        weight_decay=weight_decay,
    )
    return OptimizerRuntime([shampoo, aux_adam])


def _build_preconditioner_config(entry: OptimizerEntry) -> RootInvShampooPreconditionerConfig:
    mode = entry.hparams.get("preconditioner", "default")
    if mode == "default":
        return RootInvShampooPreconditionerConfig()
    if mode == "pinv_one_sided":
        return RootInvShampooPreconditionerConfig(
            inverse_exponent_override={2: {0: 0.0, 1: 0.25}},
            amortized_computation_config=EigenConfig(
                rank_deficient_stability_config=PseudoInverseConfig(),
            ),
        )
    if mode == "soap":
        return DefaultSOAPConfig
    raise ValueError(f"Unsupported Shampoo preconditioner mode: {mode!r}")
