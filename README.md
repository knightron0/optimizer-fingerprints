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
logs/fingerprints/<run-id>/
  config.json
  trace.jsonl
  fingerprint.json
```

Compare two fingerprints:

```bash
uv run python -m fingerprinting compare \
  logs/fingerprints/<run-a>/fingerprint.json \
  logs/fingerprints/<run-b>/fingerprint.json
```

The v1 fingerprint contains direction, scale, trajectory, and matrix-structure
blocks. Curvature/Hessian probes are intentionally deferred.
