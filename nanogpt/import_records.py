"""Import modded-nanogpt record scripts into ``nanogpt/examples``.

The importer prefers explicit ``train*.py`` files inside each result directory.
If none exist, it tries to recover the source code printed at the beginning of
the newest ``.txt``/``.stdout`` log.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_RECORDS = Path("/home/mangla/modded-nanogpt/records/track_3_optimization/results")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "examples"
SOURCE_END_RE = re.compile(r"^={20,}\s*$", re.MULTILINE)
README_LOG_RE = re.compile(r"\[log\]\(results/([^)]*)\)")
FINEWEB_REPLACEMENTS = {
    "data/fineweb10B/fineweb_train_*.bin": "../../../scratch/gilbreth/mangla/fineweb10B/fineweb_train_*.bin",
    "data/fineweb10B/fineweb_val_*.bin": "../../../scratch/gilbreth/mangla/fineweb10B/fineweb_val_*.bin",
    'os.environ.get("DATA_DIR", "data/fineweb10B")': 'os.environ.get("DATA_DIR", "../../../scratch/gilbreth/mangla/fineweb10B")',
}


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    return value or "record"


def newest(paths: list[Path]) -> Path | None:
    return max(paths, key=lambda path: (path.stat().st_mtime, path.name), default=None)


def extract_source_from_log(path: Path) -> str | None:
    text = path.read_text(errors="replace")
    match = SOURCE_END_RE.search(text)
    if match is None:
        return None
    source = text[: match.start()].rstrip() + "\n"
    if "import torch" not in source or "dist.init_process_group" not in source:
        return None
    return source


def readme_entries(records_dir: Path) -> dict[str, dict[str, str]]:
    readme = records_dir.parent / "README.md"
    if not readme.exists():
        return {}

    entries = {}
    for line in readme.read_text(errors="replace").splitlines():
        match = README_LOG_RE.search(line)
        if match is None:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        relative_log = match.group(1)
        if len(cells) >= 6:
            entries[relative_log] = {
                "rank": cells[0],
                "date": cells[4].replace("/", ""),
                "description": re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", cells[3]),
            }
        else:
            entries[relative_log] = {
                "rank": "",
                "date": "",
                "description": "readme_note",
            }
    return entries


def name_for_top_level_log(path: Path, entries: dict[str, dict[str, str]]) -> str:
    relative = path.name
    entry = entries.get(relative)
    if entry is None:
        return f"top_level_{path.stem[:8]}"
    rank = entry["rank"].zfill(2) if entry["rank"].isdigit() else "note"
    date = entry["date"] or "undated"
    description = slugify(entry["description"])[:36]
    return f"entry_{rank}_{date}_{description}_{path.stem[:8]}"


def source_candidates_for_result(result_dir: Path) -> list[tuple[Path, str | None]]:
    candidates: list[tuple[Path, str | None]] = []
    for path in sorted(result_dir.glob("train*.py")):
        candidates.append((path, path.read_text()))
    logs = sorted(
        (
            path
            for path in result_dir.rglob("*")
            if path.is_file() and path.suffix in {".txt", ".stdout", ".out"}
        ),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    for path in logs:
        candidates.append((path, extract_source_from_log(path)))
    return candidates


def source_candidates_for_log(path: Path) -> list[tuple[Path, str | None]]:
    return [(path, extract_source_from_log(path))]


def insert_after(source: str, needle: str, addition: str) -> str:
    index = source.find(needle)
    if index == -1:
        raise ValueError(f"missing insertion point: {needle!r}")
    index += len(needle)
    return source[:index] + addition + source[index:]


def add_imports(source: str) -> str:
    if "import random\n" not in source:
        source = insert_after(source, "import time\n", "import random\n")
    if "from pathlib import Path" not in source:
        source = insert_after(source, "import time\n", "from pathlib import Path\n")
    if "from nanogpt.wrapper import OptimizerFingerprint" not in source:
        source = insert_after(
            source,
            "from pathlib import Path\n",
            "\n# Keep imported records directly runnable via `torchrun nanogpt/examples/...`.\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parents[2]))\n"
            "from nanogpt.wrapper import OptimizerFingerprint\n",
        )
    if "import wandb\n" not in source:
        source = insert_after(source, "import torch.distributed as dist\n", "\nimport wandb\n")
    return source


def add_fingerprint(source: str, run_name: str, snapshot_interval: int) -> str:
    indent = ""
    if "OptimizerFingerprint.attach(" not in source:
        match = re.search(r"(?m)^([ \t]*)# learning rate schedule", source)
        if match is None:
            match = re.search(r"(?m)^([ \t]*)#{2,}\s*$\n^[ \t]*#\s*Training and Validation", source)
        if match is None:
            raise ValueError("missing learning-rate/training insertion marker")
        marker_index = match.start()
        indent = match.group(1)
        if "optimizers" not in source[:marker_index]:
            raise ValueError("missing optimizers setup before learning-rate schedule")
        block = (
            f"{indent}wandb_run = None\n"
            f"{indent}if dist.get_rank() == 0:\n"
            f"{indent}    wandb_run = wandb.init(\n"
            f"{indent}        project=os.environ.get(\"WANDB_PROJECT\", \"nanogpt\"),\n"
            f"{indent}        name={run_name!r},\n"
            f"{indent}    )\n"
            f"{indent}    print0(f\"wandb run:{{wandb_run.url}}\", console=True)\n\n"
            f"{indent}fingerprint = OptimizerFingerprint.attach(\n"
            f"{indent}    model,\n"
            f"{indent}    optimizers,\n"
            f"{indent}    run_name={run_name!r},\n"
            f"{indent}    snapshot_interval={snapshot_interval},\n"
            f"{indent}    wandb_run=wandb_run,\n"
            f"{indent})\n\n"
        )
        source = source[:marker_index] + block + source[marker_index:]
    else:
        match = re.search(r"(?m)^([ \t]*)fingerprint = OptimizerFingerprint\.attach\(", source)
        indent = match.group(1) if match else ""

    if "fingerprint.finish()" not in source:
        finish = (
            f"{indent}fingerprint_path = fingerprint.finish()\n"
            f"{indent}if fingerprint_path is not None:\n"
            f"{indent}    print0(f\"optimizer fingerprint:{{fingerprint_path}}\", console=True)\n\n"
            f"{indent}if wandb_run is not None:\n"
            f"{indent}    wandb_run.finish()\n\n"
        )
        destroy = "\ndist.destroy_process_group()"
        if destroy not in source:
            raise ValueError("missing dist.destroy_process_group()")
        source = source.replace(destroy, "\n" + finish + "dist.destroy_process_group()", 1)
    return source


def add_seed(source: str) -> str:
    seed_marker = "_nanogpt_seed = int(os.environ.get(\"NANOGPT_SEED\", \"0\"))"
    seed_block = (
        f"{seed_marker}\n"
        "random.seed(_nanogpt_seed)\n"
        "try:\n"
        "    np.random.seed(_nanogpt_seed)\n"
        "except NameError:\n"
        "    pass\n"
        "torch.manual_seed(_nanogpt_seed)\n"
        "torch.cuda.manual_seed_all(_nanogpt_seed)\n"
    )
    legacy_patterns = [
        (r"torch\.manual_seed\((.*)\)", "torch.manual_seed"),
        (r"torch\.cuda\.manual_seed_all\((.*)\)", "torch.cuda.manual_seed_all"),
        (r"random\.seed\((.*)\)", "random.seed"),
        (r"(?:np|numpy)\.random\.seed\((.*)\)", "np.random.seed"),
    ]
    for pattern, label in legacy_patterns:
        source = re.sub(
            rf"(?m)^([ \t]*){pattern}\s*$",
            lambda match, label=label: (
                f"{match.group(1)}# Original record seed disabled for array control: "
                f"{label}({match.group(2)})"
            ),
            source,
        )
    if seed_marker in source:
        return source
    match = re.search(r"(?m)^([ \t]*)dist\.init_process_group\(.*\)\n", source)
    if match is None:
        raise ValueError("missing dist.init_process_group() for seed insertion")
    indented_seed_block = "".join(
        f"{match.group(1)}{line}\n" if line else "\n"
        for line in seed_block.splitlines()
    )
    return source[: match.end()] + indented_seed_block + source[match.end():]


def instrument(source: str, run_name: str, snapshot_interval: int) -> str:
    for old, new in FINEWEB_REPLACEMENTS.items():
        source = source.replace(old, new)
    source = add_imports(source)
    source = add_seed(source)
    source = add_fingerprint(source, run_name, snapshot_interval)
    return source


def import_records(
    records_dir: Path,
    output_dir: Path,
    overwrite: bool,
    dry_run: bool,
    snapshot_interval: int,
    limit: int | None,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    imported = 0
    entries = readme_entries(records_dir)
    result_items: list[tuple[str, list[tuple[Path, str | None]]]] = []
    for path in sorted(records_dir.iterdir()):
        if path.is_dir():
            result_items.append((slugify(path.name), source_candidates_for_result(path)))
        elif path.is_file() and path.suffix in {".txt", ".stdout", ".out"}:
            result_items.append((name_for_top_level_log(path, entries), source_candidates_for_log(path)))
    if limit is not None:
        result_items = result_items[:limit]

    for run_name, candidates in result_items:
        destination = output_dir / f"record_{run_name}.py"
        if destination.exists() and not overwrite:
            print(f"skip {run_name}: {destination.name} exists")
            continue

        if not candidates:
            print(f"skip {run_name}: no train*.py or log file")
            continue

        instrumented = None
        source_path = None
        errors = []
        for candidate_path, source in candidates:
            if source is None:
                errors.append(f"{candidate_path.name}: could not extract logged Python source")
                continue
            try:
                instrumented = instrument(source, run_name, snapshot_interval)
                source_path = candidate_path
                break
            except ValueError as exc:
                errors.append(f"{candidate_path.name}: {exc}")
        if instrumented is None or source_path is None:
            print(f"skip {run_name}: {'; '.join(errors[:3])}")
            continue

        print(f"import {run_name}: {source_path} -> {destination}")
        if not dry_run:
            destination.write_text(instrumented)
        imported += 1
    return imported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-dir", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot-interval", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    imported = import_records(
        records_dir=args.records_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        snapshot_interval=args.snapshot_interval,
        limit=args.limit,
    )
    print(f"imported {imported} record scripts")


if __name__ == "__main__":
    main()
