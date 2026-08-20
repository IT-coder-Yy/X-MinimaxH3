"""Deterministic MiniMax H3 first/last-frame canvas preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import PreparedKeyframe, PreparedReferenceAudio, PreparedReferenceVideo


REFERENCE_SHORT_EDGE_LIMITS = {
    "original": None,
    "360p": 360,
    "480p": 480,
    "720p": 720,
}


def _positive_dimension(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def cover_crop_plan(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> dict[str, Any]:
    """Return the stable LANCZOS resize and integer center-crop geometry.

    ``round`` matches the verified Apache SGLang implementation.  The max
    guards make the resized image cover the canvas even when rounding lands on
    the lower adjacent integer.
    """

    source_width = _positive_dimension("source_width", source_width)
    source_height = _positive_dimension("source_height", source_height)
    target_width = _positive_dimension("target_width", target_width)
    target_height = _positive_dimension("target_height", target_height)
    scale = max(target_width / source_width, target_height / source_height)
    resized_width = max(target_width, int(round(source_width * scale)))
    resized_height = max(target_height, int(round(source_height * scale)))
    left = max(0, (resized_width - target_width) // 2)
    top = max(0, (resized_height - target_height) // 2)
    return {
        "scale": scale,
        "resized_size": (resized_width, resized_height),
        "crop_box": (left, top, left + target_width, top + target_height),
    }


def _open_rgb_first(path: Path):
    from PIL import Image, ImageOps

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"conditioning image does not exist: {path}")
    with Image.open(path) as source:
        # Pillow opens the first frame/page by default.  Copy while the file is
        # open so animated inputs cannot lazily advance after close.
        return ImageOps.exif_transpose(source).convert("RGB").copy()


def stretch_first_frame(image: Any, width: int, height: int):
    """Directly resize a first-frame anchor to the resolved target canvas."""

    from PIL import Image

    width = _positive_dimension("width", width)
    height = _positive_dimension("height", height)
    image = image.convert("RGB")
    if image.size == (width, height):
        return image.copy()
    return image.resize((width, height), Image.Resampling.LANCZOS)


def cover_crop_last_frame(image: Any, width: int, height: int):
    """Aspect-preserving cover resize plus center crop for a last frame."""

    from PIL import Image

    image = image.convert("RGB")
    plan = cover_crop_plan(image.width, image.height, width, height)
    if image.size != plan["resized_size"]:
        image = image.resize(plan["resized_size"], Image.Resampling.LANCZOS)
    return image.crop(plan["crop_box"])


def prepare_keyframes(request: Any) -> tuple[PreparedKeyframe, ...]:
    """Prepare request anchors in the only valid order: first, then last.

    This is intentionally role-based.  A last-frame-only request still uses
    cover/crop; it is not silently treated as the first item in a list.
    """

    width = _positive_dimension("request.width", request.width)
    height = _positive_dimension("request.height", request.height)
    frame_count = int(request.num_frames)
    if frame_count <= 1:
        raise ValueError("keyframe conditioning requires num_frames > 1")

    prepared: list[PreparedKeyframe] = []
    first_path = getattr(request, "first_frame", None)
    if first_path is not None:
        prepared.append(
            PreparedKeyframe(
                role="first",
                semantic_frame_index=0,
                resolved_frame_index=0,
                image=stretch_first_frame(_open_rgb_first(first_path), width, height),
            )
        )
    last_path = getattr(request, "last_frame", None)
    if last_path is not None:
        prepared.append(
            PreparedKeyframe(
                role="last",
                semantic_frame_index=-1,
                resolved_frame_index=frame_count - 1,
                image=cover_crop_last_frame(_open_rgb_first(last_path), width, height),
            )
        )
    return tuple(prepared)


def prepare_reference_images(request: Any) -> tuple[Any, ...]:
    """Load Ref2VA images using the request's lossless-composition policy.

    A ``720p``/``480p``/``360p`` policy is a *short-edge resolution cap*, not
    a new aspect ratio or crop.  Large media is downscaled proportionally,
    small media is never enlarged, and the complete image remains visible.

    This stage never changes the media canvas beyond that proportional
    downscale.  The VAE's private tensor alignment is handled later, inside
    the VAE adapter, so Qwen and every public/API surface see the exact
    aspect-preserving resolution selected here.
    """

    cached = tuple(getattr(request, "prepared_reference_images", ()) or ())
    if cached:
        return cached
    paths = tuple(getattr(request, "reference_images", ()) or ())
    if not 1 <= len(paths) <= 9:
        raise ValueError("Ref2VA requires between 1 and 9 reference images")
    policy = _reference_policy(
        request,
        "reference_image_resolution",
        "720p",
    )
    prepared = []
    for path in paths:
        image = _open_rgb_first(path)
        prepared.append(_resize_reference_image(image, policy))
    return tuple(prepared)


def _reference_policy(request: Any, attribute: str, default: str) -> str:
    policy = str(getattr(request, attribute, default) or default).strip().lower()
    if policy not in REFERENCE_SHORT_EDGE_LIMITS:
        raise ValueError(
            f"{attribute} must be original, 360p, 480p or 720p"
        )
    return policy


def _reference_geometry(
    width: int,
    height: int,
    policy: str,
) -> dict[str, tuple[int, int] | tuple[int, int, int, int] | float]:
    """Resolve the exact proportional resolution cap applied to user media.

    ``content_size`` and ``canvas_size`` intentionally match.  Keeping the
    legacy keys makes diagnostics stable while guaranteeing that this public
    preprocessing stage performs no crop, stretch, letterbox or edge padding.
    """

    width = _positive_dimension("reference width", width)
    height = _positive_dimension("reference height", height)
    if policy not in REFERENCE_SHORT_EDGE_LIMITS:
        raise ValueError("reference policy must be original, 360p, 480p or 720p")
    limit = REFERENCE_SHORT_EDGE_LIMITS[policy]
    scale = 1.0 if limit is None else min(1.0, float(limit) / min(width, height))
    content_width = max(1, int(round(width * scale)))
    content_height = max(1, int(round(height * scale)))
    return {
        "scale": scale,
        "content_size": (content_width, content_height),
        "canvas_size": (content_width, content_height),
        "padding": (0, 0, 0, 0),
    }


def _resize_reference_image(image: Any, policy: str):
    from PIL import Image

    image = image.convert("RGB")
    geometry = _reference_geometry(image.width, image.height, policy)
    content_size = geometry["content_size"]
    if image.size != content_size:
        image = image.resize(content_size, Image.Resampling.LANCZOS)
    return image.copy()


def _resize_reference_array(frame: Any, geometry: dict[str, Any]):
    """Apply one already-resolved geometry while a video is being decoded."""

    import numpy as np
    from PIL import Image

    content_size = geometry["content_size"]
    image = Image.fromarray(frame, mode="RGB")
    if image.size != content_size:
        image = image.resize(content_size, Image.Resampling.LANCZOS)
    return np.asarray(image)


def _resample_24fps(frames: list[Any], fps: float) -> list[Any]:
    if fps <= 0 or not frames:
        raise ValueError("reference video has no decodable timing")
    target_count = max(1, int(round(len(frames) * 24.0 / fps)))
    return [frames[min(len(frames) - 1, int(index * fps / 24.0))] for index in range(target_count)]


def prepare_reference_videos(request: Any) -> tuple[PreparedReferenceVideo, ...]:
    """Decode Ref2VA videos, discard audio, resample and align to H3's 17n+5 grid."""

    import av
    import numpy as np

    cached = tuple(getattr(request, "prepared_reference_videos", ()) or ())
    if cached:
        return cached
    paths = tuple(getattr(request, "reference_videos", ()) or ())
    if not 1 <= len(paths) <= 3:
        raise ValueError("Ref2VA requires between 1 and 3 reference videos")
    policy = _reference_policy(
        request,
        "reference_video_resolution",
        "360p",
    )
    prepared = []
    total_duration = 0.0
    for path in paths:
        decoded = []
        geometry = None
        source_size = None
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError(f"reference video has no video stream: {path}")
            stream = container.streams.video[0]
            fps = float(stream.average_rate or stream.guessed_rate or 0)
            for frame in container.decode(stream):
                pixels = frame.to_ndarray(format="rgb24")
                height, width = pixels.shape[:2]
                if geometry is None or source_size != (width, height):
                    source_size = (width, height)
                    geometry = _reference_geometry(width, height, policy)
                # Compress immediately, before retaining the temporal stack.
                # This keeps high-resolution 15-second references from
                # needlessly occupying their full decoded size in host RAM.
                decoded.append(_resize_reference_array(pixels, geometry))
        duration = len(decoded) / fps if fps else 0.0
        if duration < 1.95 or duration > 15.1:
            raise ValueError("each reference video must be between 2 and 15 seconds")
        total_duration += duration
        frames = _resample_24fps(decoded, fps)[: int(request.num_frames)]
        count = len(frames)
        while count >= 5 and (count - 5) % 17:
            count -= 1
        if count < 5:
            raise ValueError("reference video is too short after temporal alignment")
        frames = frames[:count]
        qwen = [frames[index] for index in range(0, len(frames), 12)]
        timestamps = [index / 2.0 for index in range(len(qwen))]
        if len(qwen) % 2:
            qwen.append(qwen[-1])
            timestamps.append(timestamps[-1])
        block_times = tuple((timestamps[i] + timestamps[i + 1]) / 2 for i in range(0, len(timestamps), 2))
        prepared.append(PreparedReferenceVideo(
            frames=np.stack(frames), qwen_frames=np.stack(qwen),
            qwen_block_timestamps=block_times, source_fps=fps,
            source_duration_seconds=duration,
        ))
    if total_duration > 15.1:
        raise ValueError("total reference video duration must not exceed 15 seconds")
    return tuple(prepared)


def prepare_reference_audios(request: Any) -> tuple[PreparedReferenceAudio, ...]:
    """Decode standalone Ref2VA audio as float stereo at 32 kHz.

    PyAV performs the same band-limited resampling for every input format and
    keeps this release independent of an optional torchaudio installation.
    The returned tensor is ``[1, 2, samples]`` and can be fed directly to the
    native MiniMax H3 audio VAE.
    """

    import av
    import numpy as np
    import torch

    cached = tuple(getattr(request, "prepared_reference_audios", ()) or ())
    if cached:
        return cached
    paths = tuple(getattr(request, "reference_audios", ()) or ())
    if not 1 <= len(paths) <= 3:
        raise ValueError("Ref2VA requires between 1 and 3 reference audios")
    prepared = []
    for path in paths:
        chunks = []
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise ValueError(f"reference audio has no audio stream: {path}")
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=32000)
            for frame in container.decode(stream):
                converted = resampler.resample(frame)
                for item in converted if isinstance(converted, list) else [converted]:
                    if item is not None and item.samples:
                        chunks.append(item.to_ndarray())
            flushed = resampler.resample(None)
            for item in flushed if isinstance(flushed, list) else [flushed]:
                if item is not None and item.samples:
                    chunks.append(item.to_ndarray())
        if not chunks:
            raise ValueError(f"reference audio has no decodable samples: {path}")
        waveform = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
        duration = waveform.shape[1] / 32000.0
        prepared.append(
            PreparedReferenceAudio(
                waveform=torch.from_numpy(waveform.copy()).unsqueeze(0),
                sample_rate=32000,
                source_duration_seconds=duration,
            )
        )
    return tuple(prepared)


__all__ = [
    "cover_crop_last_frame",
    "cover_crop_plan",
    "prepare_keyframes",
    "prepare_reference_images",
    "prepare_reference_audios",
    "prepare_reference_videos",
    "stretch_first_frame",
]
