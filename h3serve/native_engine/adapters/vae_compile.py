"""Regional Video-VAE compilation with an explicit safe CUDA-graph boundary."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


_TRANSFORMER_BLOCK_COMPILE: ContextVar[bool] = ContextVar(
    "h3_native_vae_transformer_block_compile", default=False
)


@contextmanager
def transformer_block_compile(enabled: bool):
    """Select the prebuilt VAE block implementation for one request."""

    token = _TRANSFORMER_BLOCK_COMPILE.set(bool(enabled))
    try:
        yield
    finally:
        _TRANSFORMER_BLOCK_COMPILE.reset(token)


def transformer_block_compile_ready(model: Any) -> bool:
    """Return whether every H3 decoder transformer block has both paths."""

    blocks = [
        module
        for module in model.modules()
        if module.__class__.__name__ == "TransformerBlock"
    ]
    return bool(blocks) and all(
        getattr(module, "_native_block_compile_enabled", False)
        and callable(getattr(module, "_native_eager_forward", None))
        and callable(getattr(module, "_native_compiled_forward", None))
        for module in blocks
    )


def enable_feed_forward_compile(model: Any) -> int:
    """Compile only repeated ViT FFN regions and return the module count.

    Full decoder capture is intentionally avoided: the VAE invokes one graph
    repeatedly for many temporal/spatial tiles, and CUDA-graph output reuse is
    unsafe at that boundary.  Default Inductor mode fuses SiLU/multiply while
    leaving the surrounding tile lifecycle eager.
    """

    import torch

    count = 0
    for module in model.modules():
        implementation = getattr(module, "_forward_impl", None)
        if (
            module.__class__.__name__ != "FeedForward"
            or not callable(implementation)
            or not hasattr(module, "_compile_forward_enabled")
            or not hasattr(module, "_compiled_forward")
        ):
            continue
        module._compiled_forward = torch.compile(
            implementation,
            backend="inductor",
            fullgraph=False,
        )
        module._compile_forward_enabled = True
        count += 1
    if count <= 0:
        raise RuntimeError("no compatible H3 Video-VAE FeedForward modules found")
    return count


def enable_transformer_block_compile(model: Any) -> int:
    """Compile repeated H3 Video-VAE transformer blocks without CUDA graphs.

    The decoder reuses the same 36 modules across many spatial and temporal
    tiles.  Compiling each block independently lets Inductor fuse FP32 norm,
    residual and elementwise work while leaving FlashAttention and GEMMs on
    their existing kernels.  ``triton.cudagraphs`` is disabled deliberately:
    tile outputs escape each invocation and cannot share captured storage.
    """

    import torch

    count = 0
    for module in model.modules():
        if module.__class__.__name__ != "TransformerBlock":
            continue
        if getattr(module, "_native_block_compile_enabled", False):
            count += 1
            continue
        eager_forward = module.forward
        compiled_forward = torch.compile(
            eager_forward,
            backend="inductor",
            fullgraph=False,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
        def dispatch(*args, _eager=eager_forward, _compiled=compiled_forward, **kwargs):
            implementation = _compiled if _TRANSFORMER_BLOCK_COMPILE.get() else _eager
            return implementation(*args, **kwargs)

        module._native_eager_forward = eager_forward
        module._native_compiled_forward = compiled_forward
        module.forward = dispatch
        module._native_block_compile_enabled = True
        count += 1
    if count <= 0:
        raise RuntimeError("no compatible H3 Video-VAE TransformerBlock modules found")
    return count


def prewarm_feed_forward_compile(model: Any) -> None:
    """Compile the fixed 288px/17-frame tile shape used by release profiles."""

    import torch

    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError("Video-VAE compile prewarm requires CUDA residency")
    # H3 uses five body latent tokens plus a two-token temporal overlap for
    # every normal decode chunk (clip_length=17, token_drop=3).
    latent = torch.zeros((1, 24, 7, 18, 18), device=device, dtype=torch.float16)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        output = model.decode(latent)
    if not torch.isfinite(output).all():
        raise RuntimeError("Video-VAE compile prewarm returned non-finite output")
    del latent, output


def prewarm_transformer_block_compile(model: Any) -> None:
    """Compile the fixed 288px/17-frame tile graph used by long routes."""

    with transformer_block_compile(True):
        prewarm_feed_forward_compile(model)


__all__ = [
    "enable_feed_forward_compile",
    "enable_transformer_block_compile",
    "prewarm_feed_forward_compile",
    "prewarm_transformer_block_compile",
    "transformer_block_compile",
    "transformer_block_compile_ready",
]
