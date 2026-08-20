#!/usr/bin/env python3
"""Read-only installation diagnostics for X-MinimaxH3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = Path(os.environ.get("H3_SERVE_RUNTIME_DIR", ROOT / "runtime")).expanduser().resolve()
VENV_PYTHON = RUNTIME_ROOT / "venv/bin/python"
if VENV_PYTHON.is_file() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__, *sys.argv[1:]])

PROFILE_ROLES = {
    "full": None,
    "core": {"diffusion_model", "reference_diffusion_model", "text_encoder", "video_vae", "audio_vae", "turbo_lora"},
    "fl2va": {"diffusion_model", "text_encoder", "video_vae", "audio_vae", "turbo_lora"},
    "ref2va": {"reference_diffusion_model", "text_encoder", "video_vae", "audio_vae", "turbo_lora"},
    "upscaler": {"flashvsr_diffusion", "flashvsr_lq_projection", "flashvsr_temporal_decoder", "flashvsr_prompt_embedding"},
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILE_ROLES), default="full")
    parser.add_argument("--full-hash", action="store_true",
                        help="重新计算全部大模型 SHA-256（较慢）")
    parser.add_argument("--skip-models", action="store_true")
    args = parser.parse_args()

    checks: dict[str, object] = {}
    checks["linux_x86_64"] = sys.platform.startswith("linux") and platform.machine() == "x86_64"
    checks["python_3_10"] = sys.version_info[:2] == (3, 10)
    checks["ffmpeg"] = command_output(["ffmpeg", "-version"]) is not None
    checks["nvidia_smi"] = command_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])

    try:
        import torch
        checks["torch"] = torch.__version__
        checks["cuda_available"] = torch.cuda.is_available()
        checks["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        checks["sm89"] = torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (8, 9)
    except Exception as error:
        checks["torch_error"] = str(error)
        checks["cuda_available"] = checks["sm89"] = False

    imports: dict[str, str] = {}
    for name in ("aiohttp", "av", "comfy_kitchen", "diffusers", "numpy", "sageattention", "safetensors", "transformers"):
        try:
            module = __import__(name)
            imports[name] = str(getattr(module, "__version__", "ok"))
        except Exception as error:
            imports[name] = f"ERROR: {error}"
    checks["python_packages"] = imports

    provenance = json.loads((ROOT / "runtime_sources/PROVENANCE.json").read_text(encoding="utf-8"))
    source_checks: dict[str, bool] = {}
    for component in provenance["components"]:
        for relative, expected in component["sentinels"].items():
            path = ROOT / "runtime_sources" / relative
            source_checks[relative] = path.is_file() and digest(path) == expected

    sparse_manifest = json.loads((ROOT / "prebuilt/sparge-sm89-py310-torch28-cu126/PROVENANCE.json").read_text(encoding="utf-8"))
    sparse_root = ROOT / "prebuilt/sparge-sm89-py310-torch28-cu126"
    sparse_checks = {
        relative: (sparse_root / relative).is_file() and digest(sparse_root / relative) == expected
        for relative, expected in sparse_manifest["sha256"].items()
    }
    try:
        sys.path.insert(0, str(sparse_root))
        import spas_sage_attn  # noqa: F401
        sparse_import = True
    except Exception as error:
        sparse_import = str(error)

    model_checks: dict[str, object] = {}
    if not args.skip_models:
        model_root = Path(os.environ.get("H3_SERVE_MODEL_DIR", ROOT / "models")).expanduser().resolve()
        manifest = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8"))
        roles = PROFILE_ROLES[args.profile]
        for artifact in manifest["artifacts"]:
            if roles is not None and artifact["role"] not in roles:
                continue
            path = model_root / artifact["install_path"]
            result = {
                "exists": path.is_file(),
                "size_ok": path.is_file() and path.stat().st_size == int(artifact["bytes"]),
            }
            if args.full_hash and result["size_ok"]:
                result["sha256_ok"] = digest(path) == artifact["sha256"]
            model_checks[artifact["role"]] = result

    flash: dict[str, object] = {"required": args.profile in {"full", "upscaler"}}
    if flash["required"]:
        flash_python = RUNTIME_ROOT / "flashvsr-venv/bin/python"
        flash["python"] = flash_python.is_file()
        if flash_python.is_file():
            probe = command_output([
                str(flash_python), "-c",
                # PyTorch must load libc10/libtorch before the prebuilt CUDA
                # extension is imported. This is also the worker's real order.
                "import torch; import av, block_sparse_attn; print(torch.__version__)",
            ])
            flash["packages"] = probe

    report = {
        "profile": args.profile,
        "checks": checks,
        "runtime_sources": source_checks,
        "sparse_prebuilt": {"files": sparse_checks, "importable": sparse_import},
        "models": model_checks,
        "flashvsr": flash,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    scalar_ok = checks["linux_x86_64"] is True and checks["python_3_10"] is True
    gpu_ok = checks.get("sm89") is True
    imports_ok = all(not value.startswith("ERROR:") for value in imports.values())
    sources_ok = all(source_checks.values())
    sparse_ok = all(sparse_checks.values()) and sparse_import is True
    models_ok = all(
        item.get("size_ok") and (not args.full_hash or item.get("sha256_ok"))
        for item in model_checks.values()
    )
    flash_ok = not flash["required"] or bool(flash.get("packages"))
    raise SystemExit(0 if scalar_ok and gpu_ok and imports_ok and sources_ok and sparse_ok and models_ok and flash_ok else 1)


if __name__ == "__main__":
    main()
