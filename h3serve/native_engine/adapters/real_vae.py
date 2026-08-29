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


VIDEO_UINT8_STREAMING_MIN_FP32_BYTES = 4 * 1024**3
VIDEO_UINT8_STREAMING_WORKING_SET_BYTES = 256 * 1024**2


class _TemporalUint8HostSink:
    """Consume finalized FP32 VAE time pieces without a full GPU concat.

    H3's temporal decoder has already completed causal overlap blending before
    calling this sink.  The remaining pixel transform is elementwise, so doing
    it piece-by-piece is byte-equivalent to postprocessing the concatenated
    tensor while removing that sequence-length-proportional GPU allocation.
    """

    def __init__(
        self,
        *,
        output: Any,
        frame_chunk: int,
    ) -> None:
        if frame_chunk <= 0:
            raise ValueError("temporal host frame chunk must be positive")
        self.output = output
        self.frame_chunk = int(frame_chunk)
        self._started = False
        self._finished = False

    def begin(self, shape: tuple[int, ...], sample: Any) -> None:
        import torch

        if self._started:
            raise RuntimeError("temporal uint8 sink started more than once")
        if tuple(shape) != tuple(self.output.shape):
            raise RuntimeError(
                "temporal uint8 sink shape mismatch: "
                f"decoder={tuple(shape)} output={tuple(self.output.shape)}"
            )
        if sample.ndim != 5:
            raise RuntimeError("temporal uint8 sink requires one NCTHW tensor")
        if self.output.device.type != "cpu" or self.output.dtype is not torch.uint8:
            raise RuntimeError("temporal uint8 sink output must be CPU uint8")
        self._started = True

    def write(self, part: Any, start: int) -> None:
        import torch

        if not self._started or self._finished:
            raise RuntimeError("temporal uint8 sink is not writable")
        stop = int(start) + int(part.shape[2])
        if start < 0 or stop > int(self.output.shape[2]):
            raise RuntimeError("temporal uint8 sink write exceeds output bounds")
        pixel_mean = part.new_tensor((0.485, 0.456, 0.406)).view(
            1, 3, 1, 1, 1
        )
        pixel_std = part.new_tensor((0.229, 0.224, 0.225)).view(
            1, 3, 1, 1, 1
        )
        for offset in range(0, int(part.shape[2]), self.frame_chunk):
            count = min(self.frame_chunk, int(part.shape[2]) - offset)
            # Always copy to FP32.  The decoder may already return FP32, and
            # in-place normalization must never mutate a clip whose overlap
            # tail can still participate in the next causal blend.
            with torch.inference_mode():
                pixels = part[:, :, offset : offset + count].to(
                    dtype=torch.float32, copy=True
                )
                pixels.mul_(pixel_std).add_(pixel_mean).clamp_(0, 1)
                quantized = torch.round(pixels.mul_(255.0)).to(torch.uint8)
            self.output[:, :, start + offset : start + offset + count].copy_(
                quantized, non_blocking=False
            )
            del pixels, quantized

    def finish(self) -> Any:
        if not self._started or self._finished:
            raise RuntimeError("temporal uint8 sink cannot be finished")
        self._finished = True
        return self.output


def select_uint8_postprocess_frame_chunk(
    shape: tuple[int, int, int, int, int],
) -> int | None:
    """Select a geometry-only temporal chunk for exact uint8 transport.

    Small outputs retain the established single-tensor path.  A 1080p15 H3
    output would otherwise materialize an additional ~8.5 GiB FP32 tensor on
    top of the decoded FP16 video and Video-VAE weights.  Bound only that
    mechanical pixel transform to a 256 MiB FP32 working set; prompt and
    reference content never participate in the decision.
    """

    if len(shape) != 5 or any(int(value) <= 0 for value in shape):
        raise ValueError("decoded video shape must be positive NCTHW")
    batch, channels, frames, height, width = map(int, shape)
    full_fp32_bytes = batch * channels * frames * height * width * 4
    if full_fp32_bytes <= VIDEO_UINT8_STREAMING_MIN_FP32_BYTES:
        return None
    one_frame_bytes = batch * channels * height * width * 4
    return max(
        1,
        min(frames, VIDEO_UINT8_STREAMING_WORKING_SET_BYTES // one_frame_bytes),
    )


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
    temporal_host_chunk_frames: int | None = None,
):
    import torch

    configured_host_chunk = getattr(
        model, "_h3_temporal_host_chunk_frames", temporal_host_chunk_frames
    )
    if configured_host_chunk is not None:
        configured_host_chunk = int(configured_host_chunk)
        if configured_host_chunk <= 0:
            raise ValueError("temporal host chunk must be positive")
        if output_dtype != "uint8":
            raise ValueError("temporal host output is supported only for uint8")
        if frame_count < 5 or (frame_count - 5) % 17:
            raise ValueError("temporal host output requires an H3 17*n+5 frame count")
    mean = latent_mean.to(normalized.device).view(1, -1, 1, 1, 1)
    std = latent_std.to(normalized.device).view(1, -1, 1, 1, 1)
    latent = normalized.float() * std + mean
    sink = None
    sentinel = object()
    previous_sink = getattr(model, "_h3_temporal_output_sink", sentinel)
    if configured_host_chunk is not None:
        # Allocate outside inference_mode so the returned transport tensor is
        # a normal CPU tensor accepted by PyAV and downstream callers.
        output = torch.empty(
            (
                int(normalized.shape[0]),
                3,
                int(frame_count),
                int(normalized.shape[-2]) * 16,
                int(normalized.shape[-1]) * 16,
            ),
            dtype=torch.uint8,
            device="cpu",
        )
        sink = _TemporalUint8HostSink(
            output=output,
            frame_chunk=configured_host_chunk,
        )
        setattr(model, "_h3_temporal_output_sink", sink)
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            decoded = model.decode_base(latent, frame_num=frame_count)
    finally:
        if previous_sink is sentinel:
            try:
                delattr(model, "_h3_temporal_output_sink")
            except AttributeError:
                pass
        else:
            setattr(model, "_h3_temporal_output_sink", previous_sink)
    if sink is not None:
        if decoded is not sink.output:
            raise RuntimeError("H3 temporal decoder did not consume the host sink")
        return decoded
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
    if output_dtype == "uint8":
        frame_chunk = select_uint8_postprocess_frame_chunk(tuple(decoded.shape))
        if frame_chunk is not None:
            output = torch.empty(
                tuple(decoded.shape), dtype=torch.uint8, device="cpu"
            )
            for start in range(0, decoded.shape[2], frame_chunk):
                stop = min(decoded.shape[2], start + frame_chunk)
                # decode_base returns an inference tensor.  Re-enter the
                # local inference scope before in-place pixel arithmetic;
                # the preallocated CPU result remains a normal transport
                # tensor after the scope exits.
                with torch.inference_mode():
                    pixels = decoded[:, :, start:stop].float()
                    pixels.mul_(pixel_std).add_(pixel_mean).clamp_(0, 1)
                    quantized = torch.round(pixels.mul_(255.0)).to(torch.uint8)
                output[:, :, start:stop].copy_(quantized, non_blocking=False)
                del pixels, quantized
            return output
        pixels = (decoded.float() * pixel_std + pixel_mean).clamp_(0, 1)
        return torch.round(pixels.mul_(255.0)).to(torch.uint8).cpu()
    pixels = (decoded.float() * pixel_std + pixel_mean).clamp_(0, 1)
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
    "select_uint8_postprocess_frame_chunk",
    "VIDEO_UINT8_STREAMING_MIN_FP32_BYTES",
    "VIDEO_UINT8_STREAMING_WORKING_SET_BYTES",
]
