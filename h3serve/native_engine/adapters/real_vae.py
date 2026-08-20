"""Real H3 VAE graph loaders used by the production Native session.

Only the model components are imported from pinned Apache source trees. This
module never imports or starts ComfyUI and keeps source provenance explicit in
the runtime configuration.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

from .vae_tiling import install_bounded_tile_batching
from .vae_compile import enable_feed_forward_compile


def load_native_video_vae(
    minimax_source: Path,
    checkpoint: Path,
    *,
    device: str = "cpu",
    tile_size: int | None = None,
    tile_batch_size: int = 1,
    compile_feed_forward: bool = False,
):
    import torch
    from safetensors.torch import load_file

    fl2va_root = minimax_source.resolve() / "FL2VA"
    if not (fl2va_root / "video_vae/klvae.py").is_file():
        raise FileNotFoundError(
            f"MiniMax H3 video VAE source is incomplete: {fl2va_root}"
        )
    source_path = str(fl2va_root)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from video_vae.klvae import AutoencoderKLLegacy
    from video_vae.parallel import get_parallel_state

    parallel_state = get_parallel_state()
    if not parallel_state:
        parallel_state.update(
            {
                "group_size": 1,
                "group_rank": 0,
                "local_process_group": None,
                "sp_size": 1,
                "sp_rank": 0,
                "sp_enabled": False,
                "sp_process_group": None,
                "tp_size": 1,
                "tp_rank": 0,
            }
        )

    source_dir = fl2va_root / "video_vae/source"
    wrapper_config = json.loads(
        (fl2va_root / "video_vae/config.json").read_text(encoding="utf-8")
    )
    config = AutoencoderKLLegacy.load_config(str(source_dir))
    resolved_tile = (
        int(wrapper_config["vae_tile_size"])
        if tile_size is None
        else int(tile_size)
    )
    if resolved_tile < 128 or resolved_tile % 16:
        raise ValueError("video VAE tile size must be >= 128 and divisible by 16")
    model, _ = AutoencoderKLLegacy.from_config(
        config,
        return_unused_kwargs=True,
        clip_length=int(wrapper_config["vae_clip_length"]),
        token_drop=int(wrapper_config["vae_token_drop"]),
        encoder_tiling=int(wrapper_config["vae_encoder_tiling"]),
        decoder_tiling=int(wrapper_config["vae_decoder_tiling"]),
        parallel_tiling=int(wrapper_config["vae_parallel_tiling"]),
        tile_size=resolved_tile,
        tile_overlap_min=int(wrapper_config["vae_tile_overlap_min"]),
        encoder_parallel=0,
        decoder_parallel=0,
        chunk_dim=-1,
    )
    model.half()
    state = load_file(str(checkpoint))
    # Clone the two tiny normalization vectors. Returning mmap-backed views
    # keeps the entire ~4.9 GiB safetensors mapping resident for the lifetime
    # of the service even though model.load_state_dict already copied weights.
    latent_mean = state.pop("latents_mean").clone()
    latent_std = state.pop("latents_std").clone()
    model.load_state_dict(state, strict=True)
    del state
    install_bounded_tile_batching(model, tile_batch_size)
    if compile_feed_forward:
        enable_feed_forward_compile(model)
    return (
        model.eval().requires_grad_(False).to(device),
        latent_mean,
        latent_std,
    )


def decode_native_video(
    model: Any,
    normalized: Any,
    latent_mean: Any,
    latent_std: Any,
    frame_count: int,
    *,
    output_dtype: str = "float32",
):
    import torch

    mean = latent_mean.to(normalized.device).view(1, -1, 1, 1, 1)
    std = latent_std.to(normalized.device).view(1, -1, 1, 1, 1)
    latent = normalized.float() * std + mean
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        decoded = model.decode_base(latent, frame_num=frame_count)
    return postprocess_native_video(decoded, output_dtype=output_dtype)


def postprocess_native_video(decoded: Any, *, output_dtype: str = "float32"):
    """Apply the checkpoint pixel transform and copy the result to the host.

    ``uint8`` is an exact transport optimization for the production MP4 path:
    the native muxer ultimately applies the same clamp, multiply-by-255 and
    round-to-nearest-even operation.  Performing it before the D2H transfer
    reduces host traffic by 4x without changing any VAE or codec input byte.
    The float32 mode remains the audit/reference boundary.
    """

    import torch

    if output_dtype not in {"float32", "uint8"}:
        raise ValueError("video output_dtype must be float32 or uint8")
    pixel_mean = decoded.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
    pixel_std = decoded.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
    pixels = (decoded.float() * pixel_std + pixel_mean).clamp_(0, 1)
    if output_dtype == "uint8":
        return torch.round(pixels.mul_(255.0)).to(torch.uint8).cpu()
    return pixels.cpu()


def load_native_audio_vae(
    lightx_source: Path,
    minimax_source: Path,
    checkpoint: Path,
    *,
    device: str = "cpu",
):
    import torch
    from safetensors.torch import load_file

    lightx_source = lightx_source.resolve()
    module_path = (
        lightx_source
        / "lightx2v/models/audio_encoders/hf/minimax_h3/audio_vae.py"
    )
    if not module_path.is_file():
        raise FileNotFoundError(
            f"LightX2V H3 audio VAE source is incomplete: {lightx_source}"
        )
    # Avoid LightX2V's broad package initializer: only the audited H3 audio
    # component and its relative imports belong in this runtime.
    if "lightx2v" not in sys.modules:
        package = types.ModuleType("lightx2v")
        package.__path__ = [str(lightx_source / "lightx2v")]
        package.__package__ = "lightx2v"
        sys.modules["lightx2v"] = package
    source_path = str(lightx_source)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from lightx2v.models.audio_encoders.hf.minimax_h3.audio_vae import (
        MiniMaxH3AudioVAE,
    )

    config = json.loads(
        (minimax_source.resolve() / "audio_vae/config.json").read_text(
            encoding="utf-8"
        )
    )
    model = MiniMaxH3AudioVAE(config, device=str(device), cpu_offload=False)
    removed = 0
    for module in model.modules():
        try:
            torch.nn.utils.remove_weight_norm(module)
            removed += 1
        except (AttributeError, ValueError):
            pass
    if removed != 172:
        raise RuntimeError(f"audio VAE weight-norm fold count changed: {removed}")
    state = load_file(str(checkpoint))
    latent_mean = state.pop("latents_mean")
    latent_std = state.pop("latents_std")
    model.load_state_dict(state, strict=True)
    del state
    model.latents_mean.copy_(latent_mean)
    model.latents_std.copy_(latent_std)
    return model.eval().requires_grad_(False).to(device)


__all__ = [
    "decode_native_video",
    "load_native_audio_vae",
    "load_native_video_vae",
    "postprocess_native_video",
]
