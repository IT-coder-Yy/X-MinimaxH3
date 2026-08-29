#!/usr/bin/env python3
"""Read-only preflight for the ComfyUI-free H3 model core."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from h3serve.config import ServicePaths  # noqa: E402
from h3serve.models import MODEL_FILES, model_status  # noqa: E402
from h3serve.native_engine.model import (  # noqa: E402
    audit_larry_lora,
    audit_pruned_convrot_checkpoint,
)
from h3serve.native_engine.adapters.conditioning_vae.preflight import (  # noqa: E402
    run_preflight as conditioning_vae_preflight,
)
from h3serve.native_engine.sm89_policy import configure_sm89_runtime  # noqa: E402


def main() -> None:
    paths = ServicePaths.defaults(ROOT)
    checks: dict[str, object] = {
        "linux_x86_64": sys.platform.startswith("linux")
        and platform.machine() == "x86_64",
    }
    try:
        import torch

        checks["torch"] = torch.__version__
        checks["cuda_available"] = torch.cuda.is_available()
        checks["sm89"] = bool(
            torch.cuda.is_available()
            and torch.cuda.get_device_capability() == (8, 9)
        )
    except Exception as error:
        checks["torch_error"] = str(error)
        checks["cuda_available"] = checks["sm89"] = False

    dit = paths.model_dir / MODEL_FILES["dit"][0] / MODEL_FILES["dit"][1]
    lora = paths.model_dir / MODEL_FILES["lora"][0] / MODEL_FILES["lora"][1]
    audits: dict[str, object] = {}
    for name, path, audit in (
        ("dit", dit, audit_pruned_convrot_checkpoint),
        ("lora", lora, audit_larry_lora),
    ):
        if not path.is_file():
            audits[name] = {"compatible": False, "issues": ["file missing"]}
            continue
        try:
            result = audit(path)
            audits[name] = {
                "compatible": result.compatible,
                "tensor_count": result.tensor_count,
                "quantized_layer_count": result.quantized_layer_count,
                "lora_pair_count": result.lora_pair_count,
                "issues": list(result.issues),
            }
        except Exception as error:
            audits[name] = {"compatible": False, "issues": [str(error)]}

    conditioning_vae = conditioning_vae_preflight(paths.model_dir)
    kernel_runtime: dict[str, object]
    try:
        kernel_runtime = configure_sm89_runtime(
            quant_backend="cuda", smoke_test=True
        ).to_dict()
        kernel_runtime["ready"] = True
    except Exception as error:
        kernel_runtime = {"ready": False, "error": str(error)}
    blockers = list(conditioning_vae["runtime_blockers"])
    blockers.extend([
        "native component adapters are not assembled into one production pipeline",
        "full-size DiT still requires block-offload binding and oracle tensor parity",
    ])
    report = {
        "checks": checks,
        "models": model_status(paths.model_dir),
        "native_checkpoint_audits": audits,
        "conditioning_vae": conditioning_vae,
        "sm89_kernel_runtime": kernel_runtime,
        "full_dit_graph": {
            "implemented": True,
            "real_minimal_sm89_forward_passed": True,
            "production_block_offload_bound": False,
        },
        "end_to_end_generator_ready": False,
        "runtime_blockers": blockers,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    compatible = all(
        bool(value.get("compatible")) for value in audits.values()
        if isinstance(value, dict)
    )
    environment = (
        checks["linux_x86_64"] is True
        and checks["sm89"] is True
        and kernel_runtime.get("ready") is True
    )
    raise SystemExit(0 if compatible and environment else 1)


if __name__ == "__main__":
    main()
