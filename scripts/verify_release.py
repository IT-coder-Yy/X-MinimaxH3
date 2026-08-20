#!/usr/bin/env python3
"""Offline structural validation for the source release (no GPU required)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    required = (
        "server.py", "h3serve/app.py", "static/index.html", "models/manifest.json",
        "NOTICE", "third_party_licenses/MiniMax-H3-COMMUNITY-LICENSE",
        "scripts/install.sh", "scripts/download_models.py", "scripts/start.sh",
        "scripts/doctor.py", "runtime_sources/PROVENANCE.json",
        "prebuilt/sparge-sm89-py310-torch28-cu126/PROVENANCE.json",
        "setup-and-start.sh", "setup-and-start-windows.ps1",
        "scripts/windows-wsl.sh",
    )
    errors = [f"missing: {path}" for path in required if not (ROOT / path).is_file()]
    legal_hashes = {
        "NOTICE": "154fe7fbdf198395d53c57d6786b2fbf7e63a083ff70843c5ef78c0c7303d91a",
        "third_party_licenses/MiniMax-H3-COMMUNITY-LICENSE": (
            "59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44"
        ),
    }
    for relative, expected in legal_hashes.items():
        path = ROOT / relative
        if path.is_file() and digest(path) != expected:
            errors.append(f"legal notice mismatch: {relative}")

    manifest = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        errors.append("model manifest schema_version must be 2")
    paths = [item["install_path"] for item in manifest.get("artifacts", [])]
    if len(paths) != len(set(paths)):
        errors.append("model install paths are not unique")
    for item in manifest.get("artifacts", []):
        if len(item.get("sha256", "")) != 64 or int(item.get("bytes", 0)) <= 0:
            errors.append(f"invalid model contract: {item.get('role')}")

    source = json.loads((ROOT / "runtime_sources/PROVENANCE.json").read_text(encoding="utf-8"))
    for component in source["components"]:
        for relative, expected in component["sentinels"].items():
            path = ROOT / "runtime_sources" / relative
            if not path.is_file() or digest(path) != expected:
                errors.append(f"runtime source mismatch: {relative}")

    sparse_root = ROOT / "prebuilt/sparge-sm89-py310-torch28-cu126"
    sparse = json.loads((sparse_root / "PROVENANCE.json").read_text(encoding="utf-8"))
    for relative, expected in sparse["sha256"].items():
        path = sparse_root / relative
        if not path.is_file() or digest(path) != expected:
            errors.append(f"sparse binary mismatch: {relative}")

    windows_wrapper = (ROOT / "scripts/windows-wsl.sh").read_text(encoding="utf-8")
    for contract in (
        'export H3_SERVE_RUNTIME_DIR="${state_root}/runtime"',
        'export H3_SERVE_MODEL_DIR="${state_root}/models"',
        'export H3_SERVE_LOCAL_MODEL_CACHE=',
    ):
        if contract not in windows_wrapper:
            errors.append(f"Windows WSL state contract missing: {contract}")

    forbidden_suffixes = {".mp4", ".mov", ".mkv", ".part"}
    ignored_roots = {"runtime", "models", "data", "output", "workspace", ".git"}
    source_files: list[Path] = []
    for directory, names, files in os.walk(ROOT):
        relative_directory = Path(directory).relative_to(ROOT)
        if not relative_directory.parts:
            names[:] = [name for name in names if name not in ignored_roots]
        names[:] = [name for name in names if name != "__pycache__"]
        source_files.extend(Path(directory) / name for name in files)
    # The model contract and empty-directory marker are intentionally tracked
    # even though downloaded weights under models/ are ignored.
    source_files.extend((ROOT / "models/manifest.json", ROOT / "models/.gitkeep"))
    for path in source_files:
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in forbidden_suffixes:
            errors.append(f"generated artifact included: {relative}")
        if path.stat().st_size > 100_000_000:
            errors.append(f"GitHub hard-limit file (>100MB): {relative}")
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            errors.append(f"Python cache included: {relative}")

    if errors:
        print("Release verification failed:", *errors, sep="\n  - ")
        raise SystemExit(1)
    print(f"Release verification passed: {len(paths)} model contracts, {ROOT}")


if __name__ == "__main__":
    main()
