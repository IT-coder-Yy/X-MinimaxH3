#!/usr/bin/env python3
"""Install the production acceleration hooks, then run the pinned ComfyUI backend."""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import os
import runpy
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
COMFY = Path(os.environ["H3SERVE_COMFY_DIR"]).resolve()


class _Loader(importlib.abc.Loader):
    def __init__(self, delegate, callback):
        self.delegate = delegate
        self.callback = callback

    def create_module(self, spec):
        creator = getattr(self.delegate, "create_module", None)
        return creator(spec) if creator is not None else None

    def exec_module(self, module):
        self.delegate.exec_module(module)
        self.callback(module)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, matcher, callback):
        self.matcher = matcher
        self.callback = callback

    def find_spec(self, fullname, path, target=None):
        if not self.matcher(fullname):
            return None
        sys.meta_path.remove(self)
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader, self.callback)
        return spec


def patch_quant_ops(module) -> None:
    module.ck.disable_backend("cuda")
    module.ck.enable_backend("triton")
    module.ck.set_backend_priority(["triton", "eager"])


def patch_vae(module) -> None:
    tile_size = int(os.environ.get("H3SERVE_VAE_TILE_SIZE", "288"))
    original_init = module.MiniMaxH3VideoVAE.__init__

    @functools.wraps(original_init)
    def init_with_product_tile(self, *args, **kwargs):
        kwargs.setdefault("tile_size", tile_size)
        original_init(self, *args, **kwargs)

    def fused_feed_forward(self, x):
        return module.comfy.ops.linear_input_act(self.w2, self.w1(x), "swiglu")

    module.MiniMaxH3VideoVAE.__init__ = init_with_product_tile
    module.FeedForward.forward = fused_feed_forward


def patch_sage(module) -> None:
    implementation = module.sageattn_qk_int8_pv_fp8_cuda

    def sageattn(q, k, v, **kwargs):
        kwargs.pop("attn_mask", None)
        kwargs["pv_accum_dtype"] = "fp32+fp16"
        if q.shape[-1] == 64 and q.shape[-2] <= 4096:
            kwargs["qk_quant_gran"] = "per_warp"
        return implementation(q, k, v, **kwargs)

    module.sageattn = sageattn


def patch_native(module) -> None:
    import native_acceleration
    from kernels import adaptive_mlp, chunked_mlp, fused_norm_rope, int8_linear

    state = native_acceleration.install(
        module,
        fused_norm_rope,
        chunked_mlp,
        int8_linear,
        chunk_tokens=int(os.environ.get("H3SERVE_LONG_MLP_CHUNK_TOKENS", "4096")),
    )
    state.update(adaptive_mlp.install_adaptive_mlp(
        module,
        fused_norm_rope,
        chunked_mlp,
        int8_linear,
        short_chunk_tokens=int(os.environ.get("H3SERVE_MLP_CHUNK_TOKENS", "8192")),
        long_chunk_tokens=int(os.environ.get("H3SERVE_LONG_MLP_CHUNK_TOKENS", "4096")),
        long_sequence_threshold=int(os.environ.get("H3SERVE_LONG_SEQUENCE_THRESHOLD", "20000")),
    ))
    print("original engine native hooks: " + ", ".join(f"{k}={v}" for k, v in state.items()), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))

    def patch_forecast(module):
        from spectrum_patches import patch_forecaster
        patch_forecaster(module)

    def patch_runtime(module):
        from forecast_runtime import patch_runtime_module
        from spectrum_patches import patch_runtime_archive

        patch_runtime_archive(module)
        raw_steps = os.environ["H3SERVE_ACTUAL_STEPS"]
        actual_steps = frozenset(int(value) for value in raw_steps.split(",") if value.strip())
        patch_runtime_module(
            module,
            sample_channels=int(os.environ.get("H3SERVE_SAMPLE_CHANNELS", "32")),
            actual_steps=actual_steps,
        )

    def patch_minimax(module):
        from forecast_model import patch_minimax_spectrum_module
        patch_minimax_spectrum_module(module)

    sys.meta_path.insert(0, _Finder(lambda n: n == "comfy.quant_ops", patch_quant_ops))
    sys.meta_path.insert(0, _Finder(lambda n: n == "comfy.ldm.minimax.vae", patch_vae))
    sys.meta_path.insert(0, _Finder(lambda n: n.endswith(".comfyui_spectrum_h3.forecast"), patch_forecast))
    sys.meta_path.insert(0, _Finder(lambda n: n.endswith(".comfyui_spectrum_h3.runtime"), patch_runtime))
    sys.meta_path.insert(0, _Finder(lambda n: n.endswith(".comfyui_spectrum_h3.minimax_h3"), patch_minimax))
    sys.meta_path.insert(0, _Finder(lambda n: n == "sageattention", patch_sage))
    sys.meta_path.insert(0, _Finder(lambda n: n == "comfy.ldm.minimax.model", patch_native))
    sys.path.insert(0, str(COMFY))
    runpy.run_path(str(COMFY / "main.py"), run_name="__main__")

