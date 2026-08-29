"""MiniMax H3 video-VAE conditioning and decode adapter."""

from __future__ import annotations

import contextlib
from typing import Any, Literal, Sequence

from .contracts import FrameConditioning, KeyframeCondition, ReferenceConditioning
from .preprocess import prepare_keyframes, prepare_reference_images, prepare_reference_videos

KEYFRAME_ENCODE_SEED = 42
VIDEO_LATENT_CHANNELS = 24
EncodePrecision = Literal["full_fp32", "fp16_weights_fp32_posterior"]


def _align_reference_image_for_vae(image: Any, alignment: int = 32):
    """Create a VAE-only aligned copy without changing the public media."""

    import numpy as np
    from PIL import Image

    image = image.convert("RGB")
    width, height = image.size
    aligned_width = ((width + alignment - 1) // alignment) * alignment
    aligned_height = ((height + alignment - 1) // alignment) * alignment
    if (aligned_width, aligned_height) == (width, height):
        return image
    return Image.fromarray(
        np.pad(
            np.asarray(image),
            ((0, aligned_height - height), (0, aligned_width - width), (0, 0)),
            mode="edge",
        ),
        mode="RGB",
    )


def _align_reference_video_for_vae(frames: Any, alignment: int = 32):
    """Create a VAE-only aligned copy of ``[T,H,W,3]`` reference frames."""

    import numpy as np

    height, width = int(frames.shape[1]), int(frames.shape[2])
    aligned_width = ((width + alignment - 1) // alignment) * alignment
    aligned_height = ((height + alignment - 1) // alignment) * alignment
    if (aligned_width, aligned_height) == (width, height):
        return frames
    return np.pad(
        frames,
        ((0, 0), (0, aligned_height - height), (0, aligned_width - width), (0, 0)),
        mode="edge",
    )


def _unwrap_sample(value: Any) -> Any:
    if hasattr(value, "sample") and not callable(value.sample):
        return value.sample
    if isinstance(value, tuple) and len(value) == 1:
        return value[0]
    return value


def _posterior(value: Any) -> Any:
    value = value.latent_dist if hasattr(value, "latent_dist") else value
    if isinstance(value, tuple) and len(value) == 1:
        value = value[0]
    return value


def _module_device(model: Any):
    parameter = next(model.parameters())
    return parameter.device


@contextlib.contextmanager
def _scoped_encode_precision(model: Any, precision: EncodePrecision):
    """Select conditioning-VAE arithmetic without permanently mutating weights.

    ``full_fp32`` is the original, quality-first path used by the 24/16 GiB
    engines.  ``fp16_weights_fp32_posterior`` keeps the resident VAE weights in
    FP16 and lets CUDA autocast feed its convolutions.  MiniMax's
    ``DiagonalGaussianDistribution`` still upcasts the encoded moments to FP32,
    so sampling and normalization retain their numerically sensitive boundary.
    """

    import torch

    parameter = next(model.parameters())
    previous_dtype = parameter.dtype
    if precision == "fp16_weights_fp32_posterior":
        if parameter.device.type == "cuda" and previous_dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            with torch.autocast(
                device_type="cuda",
                dtype=previous_dtype,
            ):
                yield
        else:
            # CPU fakes and already-compatible modules need no dtype mutation.
            yield
        return
    if precision != "full_fp32":
        raise ValueError(f"unsupported video-VAE encode precision: {precision}")

    restore_dtype = previous_dtype != torch.float32
    try:
        # Keep the conversion inside the try block.  Module.to() can fail part
        # way through on a tight device; the finally block then restores any
        # parameters already converted instead of leaving a poisoned session.
        if restore_dtype:
            model.to(torch.float32)
        yield
    finally:
        if restore_dtype:
            model.to(previous_dtype)


def _image_tensor(image: Any, device: Any):
    """RGB PIL -> [1,3,1,H,W] float32 in the public [-1,1] boundary."""

    import torch

    raw = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    pixels = raw.reshape(image.height, image.width, 3).permute(2, 0, 1)
    pixels = pixels.to(device=device, dtype=torch.float32).div_(127.5).sub_(1.0)
    return pixels.unsqueeze(0).unsqueeze(2)


def patchify_keyframe(latent: Any) -> Any:
    """Patchify [B,24,T,H,W] with patch [1,2,2] into fp32 rows."""

    import torch

    if latent.ndim != 5 or int(latent.shape[1]) != VIDEO_LATENT_CHANNELS:
        raise ValueError(f"unexpected keyframe latent shape {tuple(latent.shape)}")
    batch, channels, time, height, width = map(int, latent.shape)
    if height % 2 or width % 2:
        raise ValueError("keyframe latent height/width must be divisible by 2")
    rows = (
        latent.reshape(batch, channels, time, height // 2, 2, width // 2, 2)
        .permute(0, 2, 3, 5, 1, 4, 6)
        .reshape(batch * time * (height // 2) * (width // 2), channels * 4)
    )
    return rows.to(dtype=torch.float32)


class H3VideoVAEAdapter:
    """Adapt a fused SGLang-like or Diffusers-like H3 VAE.

    The wrapped model is injected so the release can vendor the chosen
    Apache implementation without coupling the service to a whole framework.
    """

    def __init__(
        self,
        model: Any,
        *,
        latents_mean: Sequence[float],
        latents_std: Sequence[float],
        encode_precision: EncodePrecision = "full_fp32",
    ) -> None:
        if len(latents_mean) != VIDEO_LATENT_CHANNELS or len(latents_std) != VIDEO_LATENT_CHANNELS:
            raise ValueError("video VAE requires 24 latent means/stds")
        if not callable(getattr(model, "decode", None)) and not callable(
            getattr(model, "decode_base", None)
        ):
            raise TypeError("video VAE must expose decode() or decode_base()")
        if encode_precision not in (
            "full_fp32",
            "fp16_weights_fp32_posterior",
        ):
            raise ValueError(
                f"unsupported video-VAE encode precision: {encode_precision}"
            )
        self.model = model
        self.latents_mean = tuple(float(value) for value in latents_mean)
        self.latents_std = tuple(float(value) for value in latents_std)
        self.encode_precision = encode_precision

    def _stats(self, tensor: Any):
        import torch

        shape = (1, VIDEO_LATENT_CHANNELS) + (1,) * (tensor.ndim - 2)
        mean = torch.as_tensor(self.latents_mean, device=tensor.device, dtype=tensor.dtype).view(shape)
        std = torch.as_tensor(self.latents_std, device=tensor.device, dtype=tensor.dtype).view(shape)
        return mean, std

    def _encode_one(self, image: Any) -> Any:
        import torch

        device = _module_device(self.model)
        with torch.random.fork_rng(
            devices=[device] if device.type == "cuda" else []
        ):
            torch.manual_seed(KEYFRAME_ENCODE_SEED)
            if device.type == "cuda":
                torch.cuda.manual_seed(KEYFRAME_ENCODE_SEED)

            # Fused SGLang VAE: this method performs the checkpoint-declared
            # ImageNet transform and sampled single-frame encode.
            encode_images = getattr(self.model, "encode_images", None)
            if callable(encode_images):
                latent = encode_images(image, use_fp16_latent=True)[0]
            else:
                encode_keyframe = getattr(self.model, "encode_keyframe", None)
                if not callable(encode_keyframe):
                    raise TypeError(
                        "video VAE must expose encode_images() or encode_keyframe()"
                    )
                pixels = _image_tensor(image, device)
                # The adapter boundary is [-1,1].  A Diffusers-style graph may
                # declare ImageNet pixel normalization; convert via [0,1].
                normalize_pixels = getattr(self.model, "normalize_pixels", None)
                if callable(normalize_pixels):
                    pixels = normalize_pixels((pixels + 1.0) * 0.5)
                posterior = _posterior(encode_keyframe(pixels))
                sample = getattr(posterior, "sample", None)
                if not callable(sample):
                    raise TypeError("video VAE encode_keyframe() returned no posterior")
                try:
                    generator = torch.Generator(device=device).manual_seed(
                        KEYFRAME_ENCODE_SEED
                    )
                    latent = sample(generator=generator)
                except TypeError:
                    latent = sample()
        if latent.ndim == 4:
            latent = latent.unsqueeze(0)
        if latent.ndim != 5 or int(latent.shape[1]) != VIDEO_LATENT_CHANNELS:
            raise ValueError(f"unexpected video VAE keyframe latent {tuple(latent.shape)}")
        mean, std = self._stats(latent)
        return (latent - mean) / std

    def encode_conditioning(self, request: Any) -> FrameConditioning:
        import torch

        prepared = prepare_keyframes(request)
        if not prepared:
            raise ValueError("encode_conditioning requires a first or last frame")
        encoded: list[KeyframeCondition] = []
        all_rows = []
        # One dtype transition for a first+last pair, not one per image.
        with _scoped_encode_precision(self.model, self.encode_precision):
            for item in prepared:
                latent = self._encode_one(item.image)
                rows = patchify_keyframe(latent).to("cpu")
                all_rows.append(rows)
                encoded.append(
                    KeyframeCondition(
                        role=item.role,
                        semantic_frame_index=item.semantic_frame_index,
                        resolved_frame_index=item.resolved_frame_index,
                        latent=latent,
                        rows=rows,
                        latent_height=int(latent.shape[-2]),
                        latent_width=int(latent.shape[-1]),
                    )
                )
        return FrameConditioning(
            rows=all_rows[0] if len(all_rows) == 1 else torch.cat(all_rows, dim=0),
            keyframes=tuple(encoded),
            semantic_frame_indices=tuple(item.semantic_frame_index for item in prepared),
            resolved_frame_indices=tuple(item.resolved_frame_index for item in prepared),
            frame_count=int(request.num_frames),
        )

    def encode_references(self, request: Any) -> ReferenceConditioning:
        """Encode ordered images followed by videos; video soundtracks are ignored."""

        import torch

        paths = tuple(getattr(request, "reference_images", ()) or ())
        video_paths = tuple(getattr(request, "reference_videos", ()) or ())
        if not paths and not video_paths:
            raise ValueError("Ref2VA requires reference media")
        images = prepare_reference_images(request) if paths else ()
        videos = prepare_reference_videos(request) if video_paths else ()
        latents = []
        shapes = []
        kinds = []
        with _scoped_encode_precision(self.model, self.encode_precision):
            for image in images:
                latent = self._encode_one(_align_reference_image_for_vae(image))
                latents.append(latent)
                shapes.append(tuple(int(value) for value in latent.shape[-3:]))
                kinds.append("image")
            for video in videos:
                device = _module_device(self.model)
                with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                    torch.manual_seed(KEYFRAME_ENCODE_SEED)
                    if device.type == "cuda":
                        torch.cuda.manual_seed(KEYFRAME_ENCODE_SEED)
                    latent = self.model.encode_videos(
                        [_align_reference_video_for_vae(video.frames)],
                        use_fp16_latent=True,
                    )[0]
                if latent.ndim == 4:
                    latent = latent.unsqueeze(0)
                mean, std = self._stats(latent)
                latent = (latent - mean) / std
                latents.append(latent)
                shapes.append(tuple(int(value) for value in latent.shape[-3:]))
                kinds.append("video")
        return ReferenceConditioning(
            latents=tuple(latents),
            latent_shapes=tuple(shapes),
            kinds=tuple(kinds),
            media=tuple(images) + tuple(videos),
        )

    encode_reference_images = encode_references

    def decode(self, normalized_latents: Any) -> Any:
        """Decode normalized [B,24,T,H,W] to float [T,H,W,3]."""

        import torch

        if normalized_latents.ndim != 5 or int(normalized_latents.shape[1]) != VIDEO_LATENT_CHANNELS:
            raise ValueError("video latents must have shape [B,24,T,H,W]")
        if int(normalized_latents.shape[0]) != 1:
            raise ValueError("native RTX 4090 video adapter supports batch size 1")
        mean, std = self._stats(normalized_latents)
        latents = normalized_latents * std + mean
        decode_base = getattr(self.model, "decode_base", None)
        decoded = decode_base(latents) if callable(decode_base) else self.model.decode(latents)
        decoded = _unwrap_sample(decoded)
        processor = getattr(self.model, "processor", None)
        revert = getattr(processor, "revert_tensor", None)
        if callable(revert):
            decoded = revert(decoded)
        else:
            denormalize = getattr(self.model, "denormalize_pixels", None)
            decoded = denormalize(decoded) if callable(denormalize) else (decoded + 1.0) * 0.5
        if decoded.ndim == 5 and int(decoded.shape[1]) == 3:
            decoded = decoded[0].permute(1, 2, 3, 0)
        elif decoded.ndim == 5 and int(decoded.shape[-1]) == 3:
            decoded = decoded[0]
        else:
            raise ValueError(f"unexpected decoded video shape {tuple(decoded.shape)}")
        return decoded.to(dtype=torch.float32).clamp_(0.0, 1.0).contiguous()


__all__ = ["H3VideoVAEAdapter", "KEYFRAME_ENCODE_SEED", "patchify_keyframe"]
