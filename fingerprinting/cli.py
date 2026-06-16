from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .probes import FingerprintAccumulator, ProbeConfig, compare_fingerprints
from .optimizers import available_optimizer_names, build_optimizer_entry
from .worlds import WorldConfig, build_world


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimizer fingerprinting CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run one optimizer and write its fingerprint")
    run.add_argument("--optimizer", choices=available_optimizer_names(), required=True)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--batch-size", type=int, default=128)
    run.add_argument("--max-steps", type=int, default=300)
    run.add_argument("--log-interval", type=int, default=10)
    run.add_argument("--checkpoint-interval", type=int, default=50)
    run.add_argument("--matrix-probe-interval", type=int, default=25)
    run.add_argument("--svd-max-dim", type=int, default=512)
    run.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Apply a YAML-style scalar override, e.g. --set hparams.lr=0.01",
    )
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run.add_argument("--data-dir", type=Path, default=Path("data"))
    run.add_argument("--output-dir", type=Path, default=Path("logs/fingerprints"))
    run.add_argument("--num-workers", type=int, default=0)

    compare = subparsers.add_parser("compare", help="Compare two fingerprint.json files")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)

    return parser.parse_args()


def run_command(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    world_config = WorldConfig(
        batch_size=args.batch_size,
        seed=args.seed,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
    )
    probe_config = ProbeConfig(
        max_steps=args.max_steps,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        matrix_probe_interval=args.matrix_probe_interval,
        svd_max_dim=args.svd_max_dim,
    )
    timestamp = int(time.time())
    run_id = f"{world_config.world_id}_{args.optimizer}_seed{args.seed}_{timestamp}"
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "trace.jsonl"

    model, loader = build_world(world_config, device)
    optimizer, optimizer_entry = build_optimizer_entry(
        model,
        args.optimizer,
        overrides=args.overrides,
    )
    config_payload = {
        "run_id": run_id,
        "world": {**asdict(world_config), "data_dir": str(world_config.data_dir)},
        "optimizer": optimizer_entry.to_dict(),
        "probe": asdict(probe_config),
        "device": str(device),
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2) + "\n")

    accumulator = FingerprintAccumulator(model=model, probe_config=probe_config, trace_path=trace_path)
    model.train()
    step = 0
    progress = tqdm(total=probe_config.max_steps, desc=f"fingerprint:{args.optimizer}")
    while step < probe_config.max_steps:
        for images, targets in loader:
            step += 1
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            before_params = accumulator.capture_before_step()
            optimizer.step()
            accumulator.observe_step(step=step, before_params=before_params, loss=loss.item())
            progress.update(1)
            if step >= probe_config.max_steps:
                break
    progress.close()

    fingerprint = accumulator.finalize()
    fingerprint.update(
        {
            "run_id": run_id,
            "world": config_payload["world"],
            "optimizer": config_payload["optimizer"],
        }
    )
    fingerprint_path = run_dir / "fingerprint.json"
    fingerprint_path.write_text(json.dumps(fingerprint, indent=2) + "\n")

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "fingerprint": str(fingerprint_path),
                "trace": str(trace_path),
            },
            indent=2,
        )
    )


def compare_command(args: argparse.Namespace) -> None:
    left = json.loads(args.left.read_text())
    right = json.loads(args.right.read_text())
    print(json.dumps(compare_fingerprints(left, right), indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "run":
        run_command(args)
    elif args.command == "compare":
        compare_command(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
