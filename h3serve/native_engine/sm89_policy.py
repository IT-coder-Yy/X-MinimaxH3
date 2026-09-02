"""Pinned SM89 kernel policy for the standalone H3 runtime.

The H3 graph must never inherit an arbitrary process-wide Comfy-Kitchen
installation: the same Python API can select materially different kernels (or
silently fall back to Triton/eager).  This module makes the release-owned
kernel build authoritative and validates the Ada/SageAttention contract before
large weights are loaded.
"""

from __future__ import annotations

import importlib
import hashlib
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


class SM89RuntimeError(RuntimeError):
    """The pinned RTX 4090 runtime is absent, incompatible, or unhealthy."""


@dataclass(frozen=True, slots=True)
class SM89RuntimeReport:
    quant_backend: str
    comfy_kitchen_path: str
    sageattention_path: str
    comfy_kitchen_cuda_sha256: str
    sageattention_sm89_sha256: str
    cuda_capability: tuple[int, int]
    backend_status: dict[str, Any]
    smoke_tested: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def release_vendor_root() -> Path:
    return Path(__file__).resolve().parents[2] / "backends" / "turbo" / "vendor"


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pinned_comfy_kitchen():
    vendor = release_vendor_root().resolve()
    if not (vendor / "comfy_kitchen" / "__init__.py").is_file():
        raise SM89RuntimeError(f"release-owned Comfy-Kitchen is missing: {vendor}")

    loaded = sys.modules.get("comfy_kitchen")
    if loaded is not None:
        loaded_path = Path(getattr(loaded, "__file__", ""))
        if not _inside(loaded_path, vendor):
            raise SM89RuntimeError(
                "an unpinned comfy_kitchen was imported before native runtime "
                f"initialization: {loaded_path}"
            )
        return loaded

    vendor_text = str(vendor)
    if vendor_text in sys.path:
        sys.path.remove(vendor_text)
    sys.path.insert(0, vendor_text)
    module = importlib.import_module("comfy_kitchen")
    module_path = Path(getattr(module, "__file__", ""))
    if not _inside(module_path, vendor):
        raise SM89RuntimeError(
            f"failed to select release-owned comfy_kitchen; loaded {module_path}"
        )
    return module


def _smoke_test(kitchen, sage_module, *, require_w4a8: bool) -> None:
    import torch

    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(4090)
    value = torch.randn((2, 256), device=device, dtype=torch.bfloat16, generator=generator)
    qweight = torch.randint(
        -127, 128, (256, 256), device=device, dtype=torch.int8, generator=generator
    )
    scale = torch.full((1,), 1.0 / 127.0, device=device, dtype=torch.float32)
    with kitchen.use_backend("cuda"):
        projected = kitchen.int8_linear(
            value,
            qweight,
            scale,
            None,
            torch.bfloat16,
            convrot=True,
            convrot_groupsize=256,
        )
    if projected.shape != (2, 256) or not torch.isfinite(projected).all():
        raise SM89RuntimeError("pinned CUDA INT8/ConvRot smoke test returned invalid data")

    w4_projected = None
    if require_w4a8:
        qdata = torch.zeros(
            (256, 128), device=device, dtype=torch.int8
        )
        s_rel = torch.ones(
            (256, 16), device=device, dtype=torch.float8_e4m3fn
        )
        s_channel = torch.full(
            (256,), 1.0 / 127.0, device=device, dtype=torch.float32
        )
        codebook = torch.linspace(
            -1.0, 1.0, 16, device=device, dtype=torch.float32
        )
        with kitchen.use_backend("cuda"):
            w4_projected = kitchen.w4a8_int8_linear(
                value,
                qdata,
                s_rel,
                s_channel,
                codebook=codebook,
                group_size=16,
                convrot_groupsize=256,
                out_dtype=torch.bfloat16,
            )
        if w4_projected.shape != (2, 256) or not torch.isfinite(
            w4_projected
        ).all():
            raise SM89RuntimeError(
                "pinned CUDA W4A8/ConvRot smoke test returned invalid data"
            )

    query = torch.randn((64, 24, 128), device=device, dtype=torch.bfloat16, generator=generator)
    key = torch.randn((64, 24, 128), device=device, dtype=torch.bfloat16, generator=generator)
    val = torch.randn((64, 24, 128), device=device, dtype=torch.bfloat16, generator=generator)
    attended = sage_module.sageattn_qk_int8_pv_fp8_cuda(
        query.unsqueeze(0),
        key.unsqueeze(0),
        val.unsqueeze(0),
        tensor_layout="NHD",
        is_causal=False,
        qk_quant_gran="per_thread",
        pv_accum_dtype="fp32+fp16",
    )
    if attended.shape != (1, 64, 24, 128) or not torch.isfinite(attended).all():
        raise SM89RuntimeError("SageAttention SM89 smoke test returned invalid data")
    torch.cuda.synchronize(device)
    del value, qweight, scale, projected, w4_projected, query, key, val, attended


def configure_sm89_runtime(
    *,
    quant_backend: Literal["cuda", "triton"] = "cuda",
    smoke_test: bool = False,
    require_w4a8: bool = False,
) -> SM89RuntimeReport:
    """Lock native H3 to the release kernel build and fail closed on drift."""

    import torch

    if not torch.cuda.is_available():
        raise SM89RuntimeError("CUDA is unavailable")
    capability = tuple(torch.cuda.get_device_capability(0))
    if capability != (8, 9):
        raise SM89RuntimeError(
            f"an NVIDIA SM89 GPU is required, found SM{capability[0]}{capability[1]}"
        )

    kitchen = _load_pinned_comfy_kitchen()
    kitchen_binary = next(
        (release_vendor_root() / "comfy_kitchen" / "backends" / "cuda").glob(
            "_C.cpython-310-*.so"
        ),
        None,
    )
    if kitchen_binary is None:
        raise SM89RuntimeError("release-owned Comfy-Kitchen CUDA extension is missing")
    kitchen_sha256 = _sha256(kitchen_binary)
    if kitchen_sha256 != "652b1f1aa339742b39cbecb73c51d942bd675063381eefa59fcede5c4da5f322":
        if os.environ.get("H3_ALLOW_UNPINNED_SAGE", "") != "1":
            raise SM89RuntimeError("release-owned Comfy-Kitchen CUDA extension hash changed")
        print(
            "Warning: Comfy-Kitchen CUDA extension hash does not match the "
            "pinned release build; continuing because H3_ALLOW_UNPINNED_SAGE=1.",
            flush=True,
        )
    status = kitchen.list_backends()
    selected = status.get(quant_backend, {})
    if not selected.get("available"):
        reason = selected.get("unavailable_reason") or "not registered"
        raise SM89RuntimeError(f"Comfy-Kitchen {quant_backend} backend unavailable: {reason}")
    capabilities = set(selected.get("capabilities", ()))
    required = {"int8_linear"}
    if require_w4a8:
        required.add("w4a8_int8_linear")
    missing = sorted(required - capabilities)
    if missing:
        raise SM89RuntimeError(
            f"Comfy-Kitchen {quant_backend} backend lacks required operations: {missing}"
        )
    kitchen.enable_backend(quant_backend)
    if quant_backend == "cuda":
        kitchen.set_backend_priority(["cuda", "triton", "eager"])
    else:
        kitchen.disable_backend("cuda")
        kitchen.set_backend_priority(["triton", "eager"])

    try:
        sage = importlib.import_module("sageattention")
        sage_sm89 = importlib.import_module("sageattention._qattn_sm89")
    except (ImportError, OSError) as error:
        raise SM89RuntimeError(
            "the pinned SageAttention SM89 wheel is missing or incompatible"
        ) from error
    if not callable(getattr(sage, "sageattn_qk_int8_pv_fp8_cuda", None)):
        raise SM89RuntimeError("SageAttention lacks the required SM89 FP8-PV entry point")
    sage_binary = Path(sage_sm89.__file__).resolve()
    sage_sha256 = _sha256(sage_binary)
    if sage_sha256 != "abf2a42461561c4780094825e373342f848afb1e73437cf97b4b9f4ce1eff41b":
        if os.environ.get("H3_ALLOW_UNPINNED_SAGE", "") != "1":
            raise SM89RuntimeError("SageAttention SM89 extension hash changed")
        print(
            "Warning: SageAttention SM89 extension hash does not match the "
            "pinned release build; continuing because H3_ALLOW_UNPINNED_SAGE=1.",
            flush=True,
        )
    if smoke_test:
        _smoke_test(kitchen, sage, require_w4a8=require_w4a8)

    return SM89RuntimeReport(
        quant_backend=quant_backend,
        comfy_kitchen_path=str(Path(kitchen.__file__).resolve()),
        sageattention_path=str(Path(sage.__file__).resolve()),
        comfy_kitchen_cuda_sha256=kitchen_sha256,
        sageattention_sm89_sha256=sage_sha256,
        cuda_capability=capability,
        backend_status=kitchen.list_backends(),
        smoke_tested=smoke_test,
    )


__all__ = [
    "SM89RuntimeError",
    "SM89RuntimeReport",
    "configure_sm89_runtime",
    "release_vendor_root",
]
