#!/usr/bin/env python3
"""Fast, read-only installation check without hashing tens of GiB of weights."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from h3serve.config import ServicePaths  # noqa: E402
from h3serve.models import model_status  # noqa: E402


def main() -> None:
    paths = ServicePaths.defaults(ROOT)
    checks: dict[str, object] = {
        "python_3_10": sys.version_info[:2] == (3, 10),
        "linux_x86_64": sys.platform.startswith("linux")
        and platform.machine() == "x86_64",
        "minimax_source": paths.minimax_source_dir.is_dir(),
        "lightx_source": paths.lightx_source_dir.is_dir(),
        "model_manifest": (paths.model_dir / "manifest.json").is_file()
        or (ROOT / "models/manifest.json").is_file(),
    }
    try:
        import torch

        checks.update({
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "compute_capability": (
                list(torch.cuda.get_device_capability(0))
                if torch.cuda.is_available() else None
            ),
        })
    except Exception as error:  # pragma: no cover - diagnostic path
        checks["torch_error"] = str(error)
        checks["cuda_available"] = False

    models = model_status(paths.model_dir)
    sparse = {"configured": bool(os.environ.get("H3_NATIVE_SPARGE_BUILD_DIR"))}
    if sparse["configured"]:
        try:
            import spas_sage_attn  # noqa: F401
            sparse["importable"] = True
        except Exception as error:  # pragma: no cover - diagnostic path
            sparse.update({"importable": False, "error": str(error)})
    result = {
        "checks": checks,
        "paths": {
            "models": str(paths.model_dir),
            "minimax_source": str(paths.minimax_source_dir),
            "lightx_source": str(paths.lightx_source_dir),
        },
        "models": models,
        "sparse_attention": sparse,
        "ready_for_at_least_one_launcher": models["any_engine_ready"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    required = (
        checks["python_3_10"] is True
        and checks["linux_x86_64"] is True
        and checks.get("cuda_available") is True
        and checks["minimax_source"] is True
        and checks["lightx_source"] is True
        and models["any_engine_ready"]
        and sparse.get("importable", True)
    )
    raise SystemExit(0 if required else 1)


if __name__ == "__main__":
    main()
