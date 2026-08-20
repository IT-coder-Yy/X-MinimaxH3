#!/usr/bin/env python3
"""Run the pinned ComfyUI source with the Turbo SM89 kernels."""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import json
import os
import runpy
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
COMFY = Path(os.environ["H3SERVE_COMFY_DIR"]).resolve()


class _PostLoadLoader(importlib.abc.Loader):
    def __init__(self, delegate, callback):
        self.delegate = delegate
        self.callback = callback

    def create_module(self, spec):
        creator = getattr(self.delegate, "create_module", None)
        return creator(spec) if creator is not None else None

    def exec_module(self, module):
        self.delegate.exec_module(module)
        self.callback(module)


class _OneModuleFinder(importlib.abc.MetaPathFinder):
    def __init__(self, fullname, callback):
        self.fullname = fullname
        self.callback = callback

    def find_spec(self, fullname, path, target=None):
        if fullname != self.fullname:
            return None
        sys.meta_path.remove(self)
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _PostLoadLoader(spec.loader, self.callback)
        return spec


def _patch_quant_ops(module) -> None:
    """Enable the locally compiled CUDA 12.8 backend after Comfy's cu13 gate."""
    module.ck.enable_backend("cuda")
    module.ck.set_backend_priority(["cuda", "triton", "eager"])
    state = module.ck.list_backends().get("cuda", {})
    if not state.get("available") or state.get("disabled"):
        raise RuntimeError(f"Turbo CUDA backend did not enable: {state}")
    print(
        "Turbo Comfy Kitchen: local_cuda128_sm89=True, "
        "priority=cuda,triton,eager",
        flush=True,
    )


def _patch_minimax_vae(module) -> None:
    """Reduce redundant 480p tiling and use Comfy's fused SwiGLU dispatch."""
    tile_size = int(os.environ.get("H3SERVE_TURBO_VAE_TILE_SIZE", "288"))
    original_init = module.MiniMaxH3VideoVAE.__init__

    @functools.wraps(original_init)
    def init_with_turbo_tile(self, *args, **kwargs):
        # Checkpoint construction does not pass tile_size. Keep an explicit
        # caller override authoritative for forwards compatibility.
        kwargs.setdefault("tile_size", tile_size)
        original_init(self, *args, **kwargs)

    def fused_feed_forward(self, x):
        return module.comfy.ops.linear_input_act(self.w2, self.w1(x), "swiglu")

    module.MiniMaxH3VideoVAE.__init__ = init_with_turbo_tile
    module.FeedForward.forward = fused_feed_forward
    print(
        f"Turbo VAE hooks: tile_size={tile_size}, fused_swiglu=True",
        flush=True,
    )


def _patch_minimax_model(module) -> None:
    """Install opt-in profiling and compilation at the H3 model boundary."""
    profile_path = os.environ.get("H3SERVE_TURBO_TORCH_PROFILE_PATH")
    if profile_path:
        import torch

        selected_call = int(os.environ.get("H3SERVE_TURBO_TORCH_PROFILE_CALL", "2"))
        output = Path(profile_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        original_forward = module.MiniMaxH3Model._forward
        state = {"calls": 0}

        @functools.wraps(original_forward)
        def profiled_forward(self, *args, **kwargs):
            state["calls"] += 1
            if state["calls"] != selected_call:
                return original_forward(self, *args, **kwargs)
            print(
                f"Turbo torch profiler: capturing DiT call {selected_call} -> {output}",
                flush=True,
            )
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(
                activities=activities,
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            ) as profiler:
                result = original_forward(self, *args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            profiler.export_chrome_trace(str(output))
            rows = []
            for event in profiler.key_averages():
                rows.append({
                    "key": event.key,
                    "count": event.count,
                    "self_cpu_time_total_us": event.self_cpu_time_total,
                    "cpu_time_total_us": event.cpu_time_total,
                    "self_device_time_total_us": getattr(
                        event, "self_device_time_total", 0.0
                    ),
                    "device_time_total_us": getattr(event, "device_time_total", 0.0),
                })
            summary_path = output.with_suffix(output.suffix + ".key_averages.json")
            summary_path.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Turbo torch profiler: trace complete ({len(rows)} operators)", flush=True)
            return result

        module.MiniMaxH3Model._forward = profiled_forward

    if os.environ.get("H3SERVE_TURBO_COMPILE_DIT_BLOCKS", "0") != "1":
        return
    import torch

    original = module.DiTBlock.forward
    module.DiTBlock.forward = torch.compile(
        original,
        dynamic=False,
        fullgraph=False,
        mode=os.environ.get("H3SERVE_TURBO_COMPILE_MODE", "default"),
    )
    print(
        "Turbo DiT hooks: block_compile=True, "
        f"mode={os.environ.get('H3SERVE_TURBO_COMPILE_MODE', 'default')}",
        flush=True,
    )


if __name__ == "__main__":
    sys.meta_path.insert(
        0, _OneModuleFinder("comfy.quant_ops", _patch_quant_ops)
    )
    sys.meta_path.insert(
        0, _OneModuleFinder("comfy.ldm.minimax.vae", _patch_minimax_vae)
    )
    sys.meta_path.insert(
        0, _OneModuleFinder("comfy.ldm.minimax.model", _patch_minimax_model)
    )
    sys.path.insert(0, str(COMFY))
    runpy.run_path(str(COMFY / "main.py"), run_name="__main__")
