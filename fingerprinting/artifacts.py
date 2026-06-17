from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_FINGERPRINT_DIR = Path("fingerprints")
DEFAULT_INDEX_PATH = Path("web/public/fingerprints.json")
def stable_json_hash(payload: dict[str, Any], length: int = 10) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:length]


def fingerprint_id(*, task: dict[str, Any], optimizer_name: str, optimizer: dict[str, Any]) -> str:
    hash_payload = {
        "task": task,
        "optimizer": optimizer,
    }
    digest = stable_json_hash(hash_payload)
    return f"{task['id']}__{optimizer_name}__seed{task['seed']}__{digest}"


def fingerprint_path(root: Path, fingerprint: dict[str, Any]) -> Path:
    world_id = fingerprint["task"]["id"]
    optimizer_name = fingerprint["optimizer"]["name"]
    return root / world_id / optimizer_name / f"{fingerprint['fingerprint_id']}.json"


def write_fingerprint(root: Path, fingerprint: dict[str, Any]) -> Path:
    path = fingerprint_path(root, fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fingerprint, indent=2, sort_keys=True) + "\n")
    return path


def rebuild_index(fingerprint_root: Path, index_path: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(fingerprint_root.glob("*/*/*.json")):
        fingerprint = json.loads(path.read_text())
        if fingerprint.get("schema") != "optimizer_fingerprint":
            continue
        entries.append(
            {
                "fingerprint_id": fingerprint["fingerprint_id"],
                "schema": fingerprint["schema"],
                "task_id": fingerprint["task"]["id"],
                "optimizer": fingerprint["optimizer"]["name"],
                "optimizer_family": fingerprint["optimizer"]["family"],
                "seed": fingerprint["task"]["seed"],
                "snapshot_count": len(fingerprint["snapshots"]),
                "path": path.as_posix(),
            }
        )

    index = {
        "schema": "fingerprint_index",
        "fingerprint_root": fingerprint_root.as_posix(),
        "fingerprints": entries,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return index
