# fingerprinting

Minimal optimizer fingerprinting experiments.

The core object is a fixed-size fingerprint for one optimizer in one fixed
experimental world:

```text
optimizer -> trace -> fingerprint.json
```

After a fingerprint is written, comparing two optimizers is just a vector
distance between saved `fingerprint.json` files.

## Structure

```text
fingerprinting/
  cli.py            # run/compare commands
  probes/           # trace collection and fingerprint features
  optimizers/       # OptimizerEntry loader and optimizer builders
  worlds/           # fixed CIFAR-10 ResNet-18 world
configs/
  optimizers/       # YAML optimizer entries
```

## Usage

Install dependencies:

```bash
uv sync
```

Run a short fingerprint:

```bash
uv run python -m fingerprinting run --optimizer adamw --seed 0 --max-steps 20
```

Override optimizer hyperparameters with Hydra-like dot paths:

```bash
uv run python -m fingerprinting run \
  --optimizer muon \
  --set hparams.lr=0.01 \
  --set hparams.weight_decay=0.0
```

Supported optimizers:

```bash
adamw
muon
shampoo_default
shampoo_pinv_one_sided
```

These are loaded from `configs/optimizers/*.yaml`. Each entry defines:

```yaml
name: muon
family: muon
hparams:
  lr: 0.02
param_groups:
  matrix: ndim>=2
metadata:
  description: Matrix-like parameters use Muon.
```

Each run writes:

```text
fingerprints/<world-id>/<optimizer>/<fingerprint-id>.json
web/public/fingerprints.json
```

The `fingerprints/` files and `web/public/fingerprints.json` are intended to be
committed.

Rebuild the centralized web index from committed fingerprints:

```bash
uv run python -m fingerprinting index
```

## Fingerprint schema

Fingerprint JSON artifacts follow `schemas/fingerprint.schema.json`. A
fingerprint is one training run plus an ordered list of scalar metric snapshots:

```text
task + model + optimizer + snapshots
```

Each snapshot is computed after an optimizer step at `snapshot_interval`, plus
the final step if needed. Sampling settings such as `max_steps`,
`snapshot_interval`, and `svd_max_dim` are part of the task definition, not a
separate probe block. The snapshot metrics include direction, scale,
trajectory, and matrix-structure scalars. Curvature/Hessian probes are
intentionally deferred.

Weights, updates, gradients, tensors, aggregate vectors, and normalized vectors
are not serialized. They are only used temporarily while computing scalar
snapshot metrics.
