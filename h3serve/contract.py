from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from typing import Any


FPS = 24
RESOLUTIONS = {"360p": 360, "480p": 480, "720p": 720, "1080p": 1080}
ASPECT_RATIOS = {
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "16:9": (16, 9),
    "9:16": (9, 16),
}
ENGINES = ("original", "lora", "reference", "reference_lora")
SERVICE_FAMILIES = ("first_last", "reference")
MODEL_VARIANTS = ("base", "lora")
PREVIEW_MODES = ("off", "auto", "pause")
EXECUTION_MODES = ("complete", "checkpoint")
CHECKPOINT_PREVIEW_RESOLUTIONS = ("source", "360p", "480p", "720p")
REFERENCE_MEDIA_RESOLUTIONS = ("original", "360p", "480p", "720p")
DEFAULT_REFERENCE_IMAGE_RESOLUTION = "720p"
DEFAULT_REFERENCE_VIDEO_RESOLUTION = "360p"
QUALITY_LEVELS = ("fast", "balanced", "quality", "ultra")
SPARSE_SCOPES = ("middle_only", "guarded", "full")
BASE_SAMPLING_STEPS = (5, 30)
LORA_SAMPLING_STEPS = (4, 10)
ACCELERATION_RANGE = (0.0, 100.0)
MIN_CUSTOM_DIMENSION = 192
MAX_CUSTOM_DIMENSION = 1920
MAX_CUSTOM_SHORT_EDGE = 1088
MAX_CUSTOM_PIXELS = 1920 * 1088
MAX_DURATION_SECONDS = 15
MAX_HIGH_RESOLUTION_DURATION_SECONDS = 8
# The validated high-resolution reference point is 1920x1088 at 192 frames
# (8 seconds). Other aspect ratios share this exact spatial-temporal budget
# instead of inheriting an unnecessarily blunt "1080p = 8 seconds" cap.
MAX_NATIVE_PIXEL_FRAMES = 1920 * 1088 * 192
# H3's nominal 720p preset is aligned to a 736-pixel short edge.
LONG_DURATION_SHORT_EDGE_MAX = 736
# Backward-compatible conservative hints for older clients. New clients use
# duration.max_by_preset / max_native_pixel_frames from public_options().
RESOLUTION_MAX_DURATION_SECONDS = {
    resolution: (
        MAX_HIGH_RESOLUTION_DURATION_SECONDS
        if resolution == "1080p"
        else MAX_DURATION_SECONDS
    )
    for resolution in RESOLUTIONS
}
UPSCALE_LEVELS = {"720p": 720, "1080p": 1080, "2k": 1440}
MIN_UPSCALE_DIMENSION = 256
MAX_UPSCALE_DIMENSION = 3840
MAX_UPSCALE_PIXELS = 3840 * 2160


ORIGINAL_PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "label": "极速",
        "description": "最快，适合预览；复杂运动建议先验片",
        "actual_step_indices": [0, 1, 2, 3, 4, 8, 13, 19],
        "actual_steps": 8,
        "forecast_steps": 12,
        "experimental": True,
    },
    "balanced": {
        "label": "均衡",
        "description": "推荐默认档，兼顾稳定性与等待时间",
        "backend_preset": "balanced",
        "actual_steps": 9,
        "forecast_steps": 11,
    },
    "quality": {
        "label": "高质量",
        "description": "增加完整计算，适合复杂运动和远景",
        "backend_preset": "quality",
        "actual_steps": 12,
        "forecast_steps": 8,
    },
    "ultra": {
        "label": "超高质量",
        "description": "全部调度点执行完整计算，速度最慢",
        "backend_preset": "full",
        "actual_steps": 20,
        "forecast_steps": 0,
    },
}


LORA_PRESETS: dict[str, dict[str, Any]] = {
    "fast": {"label": "极速", "description": "四步快速预览", "steps": 4, "strength": 1.0},
    "balanced": {"label": "均衡", "description": "五步日常生成", "steps": 5, "strength": 1.0},
    "quality": {"label": "高质量", "description": "六步稳定基线", "steps": 6, "strength": 1.0},
    "ultra": {"label": "超高质量", "description": "八步实验档；更多步不保证单调提升", "steps": 8, "strength": 1.0},
}


class ContractError(ValueError):
    pass


def engine_family(engine: str) -> str:
    if engine in ("original", "lora"):
        return "first_last"
    if engine in ("reference", "reference_lora"):
        return "reference"
    raise ContractError(f"unsupported engine: {engine}")


def engine_variant(engine: str) -> str:
    if engine in ("original", "reference"):
        return "base"
    if engine in ("lora", "reference_lora"):
        return "lora"
    raise ContractError(f"unsupported engine: {engine}")


def resolve_engine(family: str, variant: str) -> str:
    if family not in SERVICE_FAMILIES:
        raise ContractError(f"unsupported service_family: {family}")
    if variant not in MODEL_VARIANTS:
        raise ContractError(f"unsupported model_variant: {variant}")
    return {
        ("first_last", "base"): "original",
        ("first_last", "lora"): "lora",
        ("reference", "base"): "reference",
        ("reference", "lora"): "reference_lora",
    }[(family, variant)]


def default_quality(engine: str) -> str:
    """Return the calibrated default for one fixed-engine service."""

    if engine == "original":
        return "balanced"
    if engine == "reference":
        return "balanced"
    if engine in ("lora", "reference_lora"):
        # Six steps are the tested Turbo baseline. Five steps remains an
        # optional quality point, but must not be the implicit product default.
        return "quality"
    raise ContractError(f"unsupported engine: {engine}")


def _nearest_multiple(value: float, multiple: int = 32) -> int:
    return max(multiple, int(math.floor(value / multiple + 0.5)) * multiple)


def resolve_geometry(resolution: str, aspect_ratio: str) -> tuple[int, int]:
    try:
        short_edge = RESOLUTIONS[resolution]
    except KeyError as error:
        raise ContractError(f"unsupported resolution: {resolution}") from error
    try:
        rw, rh = ASPECT_RATIOS[aspect_ratio]
    except KeyError as error:
        raise ContractError(f"unsupported aspect ratio: {aspect_ratio}") from error

    if rw >= rh:
        raw_width = short_edge * rw / rh
        raw_height = short_edge
    else:
        raw_width = short_edge
        raw_height = short_edge * rh / rw
    return _nearest_multiple(raw_width), _nearest_multiple(raw_height)


def resolve_frames(duration_seconds: float) -> tuple[int, float]:
    if not math.isfinite(duration_seconds) or not 1.0 <= duration_seconds <= MAX_DURATION_SECONDS:
        raise ContractError("duration_seconds must be between 1 and 15")
    target = duration_seconds * FPS
    grid_index = max(0, round((target - 5) / 17))
    frames = min(362, 5 + 17 * grid_index)
    return frames, frames / FPS


def max_frames_for_geometry(
    width: int,
    height: int,
    max_native_pixel_frames: int = MAX_NATIVE_PIXEL_FRAMES,
) -> int:
    """Largest legal H3 frame count inside the native pixel-frame budget."""

    pixels = int(width) * int(height)
    if pixels <= 0:
        raise ContractError("width and height must be positive")
    raw_limit = min(362, int(max_native_pixel_frames) // pixels)
    if raw_limit < 5:
        return 0
    return 5 + 17 * ((raw_limit - 5) // 17)


def max_duration_for_geometry(
    width: int,
    height: int,
    max_native_pixel_frames: int = MAX_NATIVE_PIXEL_FRAMES,
) -> float:
    """Requested-duration ceiling for one resolved native canvas."""

    frames = max_frames_for_geometry(width, height, max_native_pixel_frames)
    if frames < 5:
        return 0.0
    return min(MAX_DURATION_SECONDS, frames / FPS)


def validate_native_spatiotemporal_budget(
    width: int,
    height: int,
    frames: int,
    max_native_pixel_frames: int = MAX_NATIVE_PIXEL_FRAMES,
) -> None:
    if int(width) * int(height) * int(frames) > int(max_native_pixel_frames):
        maximum = max_duration_for_geometry(width, height, max_native_pixel_frames)
        raise ContractError(
            "native generation exceeds the validated spatial-temporal budget "
            f"(width*height*frames <= {int(max_native_pixel_frames)}); "
            f"{width}x{height} supports at most {maximum:.3f} seconds"
        )


def _integer(payload: dict[str, Any], name: str) -> int:
    try:
        return int(payload[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError(f"{name} must be an integer") from error


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _nearest_even(value: float) -> int:
    return max(2, int(math.floor(value / 2 + 0.5)) * 2)


def resolve_upscale_geometry(
    width: int,
    height: int,
    level: str,
) -> tuple[int, int]:
    """Resolve a named delivery size while preserving the generated ratio."""

    try:
        short_edge = UPSCALE_LEVELS[level]
    except KeyError as error:
        raise ContractError(f"unsupported upscale resolution: {level}") from error
    if width >= height:
        return _nearest_even(short_edge * width / height), short_edge
    return short_edge, _nearest_even(short_edge * height / width)


def actual_step_schedule(count: int) -> tuple[int, ...]:
    """Build the conservative H3 20-point schedule used by advanced mode."""

    calibrated = {
        8: (0, 1, 2, 3, 4, 8, 13, 19),
        9: (0, 1, 2, 3, 4, 8, 12, 16, 19),
        12: (0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 18, 19),
        20: tuple(range(20)),
    }
    if count in calibrated:
        return calibrated[count]
    if not 5 <= count <= 20:
        raise ContractError("actual_steps must be between 5 and 20")
    anchors = [0, 1, 2, 3, 4]
    remaining = count - len(anchors)
    if remaining:
        import math

        anchors.extend(
            min(19, math.ceil(4 + index * 15 / remaining))
            for index in range(1, remaining + 1)
        )
    return tuple(anchors)


@dataclass(frozen=True)
class GenerationSpec:
    prompt: str
    engine: str
    quality: str
    resolution: str
    aspect_ratio: str
    requested_duration_seconds: float
    seed: int
    width: int
    height: int
    frames: int
    actual_duration_seconds: float
    advanced: bool = False
    custom_actual_steps: int | None = None
    custom_lora_steps: int | None = None
    attention_keep_ratio: float = 1.0
    sparse_scope: str = "full"
    # New two-control execution contract.  ``None`` preserves persisted jobs
    # and legacy API clients; new UI/API requests set both fields together.
    sampling_steps: int | None = None
    acceleration: float | None = None
    upscale_enabled: bool = False
    upscale_resolution: str | None = None
    upscale_target_width: int | None = None
    upscale_target_height: int | None = None
    preview_mode: str = "off"
    preview_step_index: int | None = None
    preview_branch_steps: int = 2
    preview_fast_finish: bool = False
    execution_mode: str = "complete"
    checkpoint_step: int | None = None
    checkpoint_retain: bool = True
    checkpoint_preview: bool = False
    checkpoint_preview_steps: int = 4
    checkpoint_preview_resolution: str = "source"
    # Ref2VA-only service-side pre-compression.  ``original`` skips the extra
    # user-selected cap; the model's mandatory alignment/safety normalization
    # still applies in the conditioning adapter.
    reference_image_resolution: str = DEFAULT_REFERENCE_IMAGE_RESOLUTION
    reference_video_resolution: str = DEFAULT_REFERENCE_VIDEO_RESOLUTION

    @classmethod
    def from_mapping(
        cls,
        payload: dict[str, Any],
        *,
        max_native_pixel_frames: int = MAX_NATIVE_PIXEL_FRAMES,
        max_duration_by_preset: dict[str, dict[str, float]] | None = None,
    ) -> "GenerationSpec":
        # ``prompt`` is the final model-facing text for REST/ComfyUI clients.
        # Validate it without rewriting it: callers may deliberately use
        # whitespace and section boundaries in an H3 prompt template.
        prompt = str(payload.get("prompt", ""))
        if not prompt.strip():
            raise ContractError("prompt is required")
        if len(prompt) > 20_000:
            raise ContractError("prompt is too long (maximum 20000 characters)")

        raw_family = payload.get("service_family")
        raw_variant = payload.get("model_variant")
        if raw_family not in (None, "") or raw_variant not in (None, ""):
            family = str(raw_family or "first_last").strip()
            variant = str(raw_variant or "base").strip()
            engine = resolve_engine(family, variant)
            legacy = payload.get("engine")
            if legacy not in (None, "", engine, family):
                raise ContractError("engine disagrees with service_family/model_variant")
        else:
            engine = str(payload.get("engine", "original"))
        quality = str(payload.get("quality", "balanced"))
        resolution = str(payload.get("resolution", "480p"))
        aspect_ratio = str(payload.get("aspect_ratio", "16:9"))
        if engine not in ENGINES:
            raise ContractError(f"unsupported engine: {engine}")
        if quality not in QUALITY_LEVELS:
            raise ContractError(f"unsupported quality level: {quality}")

        reference_image_resolution = str(
            payload.get(
                "reference_image_resolution",
                DEFAULT_REFERENCE_IMAGE_RESOLUTION,
            )
        ).strip().lower()
        reference_video_resolution = str(
            payload.get(
                "reference_video_resolution",
                DEFAULT_REFERENCE_VIDEO_RESOLUTION,
            )
        ).strip().lower()
        if reference_image_resolution not in REFERENCE_MEDIA_RESOLUTIONS:
            raise ContractError(
                "reference_image_resolution must be original, 360p, 480p or 720p"
            )
        if reference_video_resolution not in REFERENCE_MEDIA_RESOLUTIONS:
            raise ContractError(
                "reference_video_resolution must be original, 360p, 480p or 720p"
            )

        preview_mode = str(payload.get("preview_mode", "off")).strip().lower()
        if preview_mode not in PREVIEW_MODES:
            raise ContractError("preview_mode must be off, auto or pause")
        preview_step_index = None
        if payload.get("preview_step_index") not in (None, "", "auto"):
            preview_step_index = _integer(payload, "preview_step_index")
            if not 0 <= preview_step_index < (20 if engine_variant(engine) == "base" else 8):
                raise ContractError("preview_step_index falls outside the supported schedule")
        try:
            preview_branch_steps = int(payload.get("preview_branch_steps", 2))
        except (TypeError, ValueError) as error:
            raise ContractError("preview_branch_steps must be an integer") from error
        if not 1 <= preview_branch_steps <= 3:
            raise ContractError("preview_branch_steps must be between 1 and 3")
        preview_fast_finish = _boolean(payload.get("preview_fast_finish", False))

        execution_mode = str(
            payload.get("execution_mode", "complete")
        ).strip().lower()
        if execution_mode not in EXECUTION_MODES:
            raise ContractError("execution_mode must be complete or checkpoint")
        checkpoint_step = None
        checkpoint_retain = _boolean(payload.get("checkpoint_retain", True))
        checkpoint_preview = _boolean(payload.get("checkpoint_preview", False))
        try:
            checkpoint_preview_steps = int(
                payload.get("checkpoint_preview_steps", 4)
            )
        except (TypeError, ValueError) as error:
            raise ContractError(
                "checkpoint_preview_steps must be an integer"
            ) from error
        if not 1 <= checkpoint_preview_steps <= 8:
            raise ContractError(
                "checkpoint_preview_steps must be between 1 and 8"
            )
        checkpoint_preview_resolution = str(
            payload.get("checkpoint_preview_resolution", "source")
        ).strip().lower()
        if checkpoint_preview_resolution not in CHECKPOINT_PREVIEW_RESOLUTIONS:
            raise ContractError(
                "checkpoint_preview_resolution must be source, 360p, 480p or 720p"
            )

        try:
            # Public requests use duration_seconds. Persisted records use the
            # explicit requested_duration_seconds name. Accept both so a
            # service restart never silently turns a 15-second job into 5s.
            requested_duration = float(payload.get(
                "duration_seconds", payload.get("requested_duration_seconds", 5)
            ))
        except (TypeError, ValueError) as error:
            raise ContractError("duration_seconds must be numeric") from error
        # Public clients use an explicit mode.  Keep the historical boolean as
        # a backwards-compatible alias for the Web console and persisted jobs.
        raw_mode = payload.get("mode")
        if raw_mode not in (None, ""):
            mode = str(raw_mode).strip().lower()
            if mode not in {"preset", "advanced"}:
                raise ContractError("mode must be preset or advanced")
            advanced = mode == "advanced"
            if "advanced" in payload and _boolean(payload["advanced"]) != advanced:
                raise ContractError("mode and advanced disagree")
        else:
            advanced = _boolean(payload.get("advanced", False))
        if advanced:
            width = _integer(payload, "width")
            height = _integer(payload, "height")
            # Exact frame control remains available, but API clients may send
            # duration_seconds and let the service choose the legal 17*n+5
            # frame grid.  This keeps advanced mode useful without requiring
            # callers to understand H3's latent temporal layout.
            if payload.get("frames") not in (None, ""):
                frames = _integer(payload, "frames")
            else:
                frames, _ = resolve_frames(requested_duration)
            if not (
                MIN_CUSTOM_DIMENSION <= width <= MAX_CUSTOM_DIMENSION
                and MIN_CUSTOM_DIMENSION <= height <= MAX_CUSTOM_DIMENSION
            ):
                raise ContractError(
                    f"width and height must be between {MIN_CUSTOM_DIMENSION} "
                    f"and {MAX_CUSTOM_DIMENSION}"
                )
            if width % 32 or height % 32:
                raise ContractError("width and height must be multiples of 32")
            if min(width, height) > MAX_CUSTOM_SHORT_EDGE:
                raise ContractError(
                    "custom canvas short edge exceeds the validated 1080p envelope"
                )
            if width * height > MAX_CUSTOM_PIXELS:
                raise ContractError("custom canvas exceeds the validated 4090 pixel envelope")
            if frames < 5 or frames > 362 or (frames - 5) % 17:
                raise ContractError("frames must be 5..362 and satisfy 17*n+5")
            requested_duration = frames / FPS
            actual_duration = requested_duration
            validate_native_spatiotemporal_budget(
                width, height, frames, max_native_pixel_frames
            )
        else:
            width, height = resolve_geometry(resolution, aspect_ratio)
            if max_duration_by_preset is not None:
                try:
                    max_duration = float(
                        max_duration_by_preset[resolution][aspect_ratio]
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise ContractError(
                        f"missing configured limit for {resolution} {aspect_ratio}"
                    ) from error
            else:
                max_duration = max_duration_for_geometry(
                    width, height, max_native_pixel_frames
                )
            if requested_duration > max_duration:
                raise ContractError(
                    f"{resolution} {aspect_ratio} generation supports at most "
                    f"{max_duration:.3f} seconds under the configured server limit"
                )
            frames, actual_duration = resolve_frames(requested_duration)
            if max_duration_by_preset is None:
                validate_native_spatiotemporal_budget(
                    width, height, frames, max_native_pixel_frames
                )

        sampling_steps = None
        acceleration = None
        joint_control_fields = tuple(
            name
            for name in ("sampling_steps", "acceleration")
            if payload.get(name) not in (None, "")
        )
        joint_controls = bool(joint_control_fields)
        if joint_controls and len(joint_control_fields) != 2:
            missing = "acceleration" if joint_control_fields == ("sampling_steps",) else "sampling_steps"
            raise ContractError(
                f"sampling_steps and acceleration must be provided together; missing {missing}"
            )
        if joint_controls:
            mixed = tuple(
                name
                for name in (
                    "actual_steps",
                    "lora_steps",
                    "attention_keep_ratio",
                    "sparse_scope",
                )
                if payload.get(name) not in (None, "")
            )
            if mixed:
                raise ContractError(
                    "sampling_steps/acceleration cannot be mixed with legacy "
                    + ", ".join(mixed)
                )
            sampling_steps = _integer(payload, "sampling_steps")
            lower, upper = (
                BASE_SAMPLING_STEPS
                if engine_variant(engine) == "base"
                else LORA_SAMPLING_STEPS
            )
            if not lower <= sampling_steps <= upper:
                raise ContractError(
                    f"sampling_steps must be between {lower} and {upper} for this model"
                )
            try:
                acceleration = float(payload.get("acceleration"))
            except (TypeError, ValueError) as error:
                raise ContractError("acceleration must be numeric") from error
            if not math.isfinite(acceleration) or not (
                ACCELERATION_RANGE[0] <= acceleration <= ACCELERATION_RANGE[1]
            ):
                raise ContractError("acceleration must be between 0 and 100")
            acceleration = round(acceleration, 1)

        total_solver_steps = (
            int(sampling_steps)
            if sampling_steps is not None
            else (
                20
                if engine_variant(engine) == "base"
                else int(LORA_PRESETS[quality]["steps"])
            )
        )
        if (
            not joint_controls
            and advanced
            and engine_variant(engine) == "lora"
            and payload.get("lora_steps") not in (None, "")
        ):
            total_solver_steps = _integer(payload, "lora_steps")
        if execution_mode == "checkpoint":
            checkpoint_step = _integer(payload, "checkpoint_step")
            if not 1 <= checkpoint_step < total_solver_steps:
                raise ContractError(
                    "checkpoint_step must stop after at least one step and before the final step"
                )
            if not checkpoint_retain and not checkpoint_preview:
                raise ContractError(
                    "checkpoint tasks must retain state, generate a preview, or both"
                )
            if checkpoint_preview_resolution != "source":
                preview_short_edge = RESOLUTIONS[checkpoint_preview_resolution]
                if preview_short_edge > min(width, height):
                    raise ContractError(
                        "checkpoint preview resolution cannot exceed the generated canvas"
                    )

        custom_actual_steps = None
        custom_lora_steps = None
        attention_keep_ratio = 1.0
        sparse_scope = "full"
        if not joint_controls and advanced and engine in ("original", "reference"):
            custom_actual_steps = _integer(payload, "actual_steps")
            actual_step_schedule(custom_actual_steps)
        elif not joint_controls and advanced and engine in ("lora", "reference_lora"):
            custom_lora_steps = _integer(payload, "lora_steps")
            if not 4 <= custom_lora_steps <= 8:
                raise ContractError("lora_steps must be between 4 and 8")

        if advanced and not joint_controls:
            try:
                attention_keep_ratio = float(
                    payload.get("attention_keep_ratio", 1.0)
                )
            except (TypeError, ValueError) as error:
                raise ContractError(
                    "attention_keep_ratio must be numeric"
                ) from error
            if not math.isfinite(attention_keep_ratio) or not (
                0.50 <= attention_keep_ratio <= 1.00
            ):
                raise ContractError(
                    "attention_keep_ratio must be between 0.50 and 1.00"
                )
            # Keep the public value stable and avoid insignificant JSON/HTML
            # floating-point tails from becoming distinct execution profiles.
            attention_keep_ratio = round(attention_keep_ratio, 2)
            sparse_scope = str(
                payload.get("sparse_scope", "full")
            ).strip()
            if sparse_scope not in SPARSE_SCOPES:
                raise ContractError(
                    "sparse_scope must be middle_only, guarded or full"
                )

        upscale_enabled = _boolean(payload.get("upscale_enabled", False))
        upscale_resolution = None
        upscale_target_width = None
        upscale_target_height = None
        if upscale_enabled:
            upscale_mode = str(payload.get("upscale_mode", "basic")).strip()
            if upscale_mode == "basic":
                upscale_resolution = str(
                    payload.get("upscale_resolution", "1080p")
                ).strip()
                upscale_target_width, upscale_target_height = resolve_upscale_geometry(
                    width, height, upscale_resolution
                )
            elif upscale_mode == "advanced":
                upscale_target_width = _integer(payload, "upscale_target_width")
                upscale_target_height = _integer(payload, "upscale_target_height")
            else:
                raise ContractError("upscale_mode must be basic or advanced")

            assert upscale_target_width is not None
            assert upscale_target_height is not None
            if not (
                MIN_UPSCALE_DIMENSION <= upscale_target_width <= MAX_UPSCALE_DIMENSION
                and MIN_UPSCALE_DIMENSION <= upscale_target_height <= MAX_UPSCALE_DIMENSION
            ):
                raise ContractError(
                    "upscale target dimensions must be between 256 and 3840"
                )
            if upscale_target_width * upscale_target_height > MAX_UPSCALE_PIXELS:
                raise ContractError("upscale target exceeds the 4K pixel envelope")
            if upscale_target_width % 2 or upscale_target_height % 2:
                raise ContractError("upscale target dimensions must be even")
            if upscale_target_width < width or upscale_target_height < height:
                raise ContractError("upscale target cannot be smaller than the generated video")
            if upscale_target_width == width and upscale_target_height == height:
                raise ContractError("upscale target must be larger than the generated video")
            source_ratio = width / height
            target_ratio = upscale_target_width / upscale_target_height
            if abs(target_ratio / source_ratio - 1.0) > 0.01:
                raise ContractError(
                    "upscale target must preserve the generated video aspect ratio"
                )

        raw_seed = payload.get("seed")
        if raw_seed in (None, "", "random"):
            seed = secrets.randbits(63)
        else:
            try:
                seed = int(raw_seed)
            except (TypeError, ValueError) as error:
                raise ContractError("seed must be an integer") from error
            if not 0 <= seed < 2**64:
                raise ContractError("seed must be in [0, 2^64)")

        return cls(
            prompt=prompt,
            engine=engine,
            quality=quality,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            requested_duration_seconds=requested_duration,
            seed=seed,
            width=width,
            height=height,
            frames=frames,
            actual_duration_seconds=actual_duration,
            advanced=advanced,
            custom_actual_steps=custom_actual_steps,
            custom_lora_steps=custom_lora_steps,
            attention_keep_ratio=attention_keep_ratio,
            sparse_scope=sparse_scope,
            sampling_steps=sampling_steps,
            acceleration=acceleration,
            upscale_enabled=upscale_enabled,
            upscale_resolution=upscale_resolution,
            upscale_target_width=upscale_target_width,
            upscale_target_height=upscale_target_height,
            preview_mode=preview_mode,
            preview_step_index=preview_step_index,
            preview_branch_steps=preview_branch_steps,
            preview_fast_finish=preview_fast_finish,
            execution_mode=execution_mode,
            checkpoint_step=checkpoint_step,
            checkpoint_retain=checkpoint_retain,
            checkpoint_preview=checkpoint_preview,
            checkpoint_preview_steps=checkpoint_preview_steps,
            checkpoint_preview_resolution=checkpoint_preview_resolution,
            reference_image_resolution=reference_image_resolution,
            reference_video_resolution=reference_video_resolution,
        )

    @property
    def service_family(self) -> str:
        return engine_family(self.engine)

    @property
    def model_variant(self) -> str:
        return engine_variant(self.engine)

    @property
    def joint_acceleration_enabled(self) -> bool:
        return self.sampling_steps is not None and self.acceleration is not None

    @property
    def preset(self) -> dict[str, Any]:
        source = ORIGINAL_PRESETS if self.engine in ("original", "reference") else LORA_PRESETS
        result = dict(source[self.quality])
        if self.custom_actual_steps is not None:
            result.update({
                "actual_step_indices": list(actual_step_schedule(self.custom_actual_steps)),
                "actual_steps": self.custom_actual_steps,
                "forecast_steps": 20 - self.custom_actual_steps,
                "advanced": True,
            })
        if self.custom_lora_steps is not None:
            result.update({"steps": self.custom_lora_steps, "advanced": True})
        if self.joint_acceleration_enabled:
            for legacy_key in (
                "actual_step_indices", "actual_steps", "forecast_steps",
                "backend_preset",
            ):
                result.pop(legacy_key, None)
            result.update({
                "steps": self.sampling_steps,
                "sampling_steps": self.sampling_steps,
                "acceleration": self.acceleration,
                "joint_acceleration": True,
                "advanced": True,
            })
        if self.advanced:
            if not self.joint_acceleration_enabled:
                result.update({
                    "attention_keep_ratio": self.attention_keep_ratio,
                    "sparse_scope": self.sparse_scope,
                })
        return result

    def to_dict(self, *, include_execution: bool = False) -> dict[str, Any]:
        result = {
            "prompt": self.prompt,
            "engine": self.engine,
            "service_family": self.service_family,
            "model_variant": self.model_variant,
            "quality": self.quality,
            "resolution": self.resolution,
            "aspect_ratio": self.aspect_ratio,
            "requested_duration_seconds": self.requested_duration_seconds,
            "actual_duration_seconds": self.actual_duration_seconds,
            "width": self.width,
            "height": self.height,
            "frames": self.frames,
            "fps": FPS,
            "seed": self.seed,
            "experimental_duration": self.requested_duration_seconds < 5,
            "advanced": self.advanced,
            "mode": "advanced" if self.advanced else "preset",
            "preview_mode": self.preview_mode,
            "preview_step_index": self.preview_step_index,
            "preview_branch_steps": self.preview_branch_steps,
            "preview_fast_finish": self.preview_fast_finish,
            "execution_mode": self.execution_mode,
            "checkpoint_step": self.checkpoint_step,
            "checkpoint_retain": self.checkpoint_retain,
            "checkpoint_preview": self.checkpoint_preview,
            "checkpoint_preview_steps": self.checkpoint_preview_steps,
            "checkpoint_preview_resolution": self.checkpoint_preview_resolution,
            "reference_image_resolution": self.reference_image_resolution,
            "reference_video_resolution": self.reference_video_resolution,
        }
        if self.custom_actual_steps is not None:
            result["actual_steps"] = self.custom_actual_steps
            result["forecast_steps"] = 20 - self.custom_actual_steps
        if self.custom_lora_steps is not None:
            result["lora_steps"] = self.custom_lora_steps
        if self.joint_acceleration_enabled:
            result["sampling_steps"] = self.sampling_steps
            result["acceleration"] = self.acceleration
        elif self.advanced:
            result["attention_keep_ratio"] = self.attention_keep_ratio
            result["sparse_scope"] = self.sparse_scope
        result["upscale_enabled"] = self.upscale_enabled
        if self.upscale_enabled:
            result["upscale_mode"] = (
                "basic" if self.upscale_resolution is not None else "advanced"
            )
            result["upscale_resolution"] = self.upscale_resolution
            result["upscale_target_width"] = self.upscale_target_width
            result["upscale_target_height"] = self.upscale_target_height
        if include_execution:
            result["execution"] = self.preset
        return result


def _public_presets(source: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Expose product choices without leaking launcher schedules or revisions."""
    return {
        name: {
            "label": preset["label"],
            "description": preset["description"],
            **({"experimental": True} if preset.get("experimental") else {}),
        }
        for name, preset in source.items()
    }


def public_options(
    fixed_engine: str | None = None,
    *,
    max_native_pixel_frames: int = MAX_NATIVE_PIXEL_FRAMES,
    max_duration_by_preset: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    if fixed_engine in ENGINES:
        fixed_engine = engine_family(str(fixed_engine))
    if fixed_engine is not None and fixed_engine not in SERVICE_FAMILIES:
        raise ContractError(f"unsupported service family: {fixed_engine}")
    geometry = {
        resolution: {
            ratio: {"width": resolve_geometry(resolution, ratio)[0],
                    "height": resolve_geometry(resolution, ratio)[1]}
            for ratio in ASPECT_RATIOS
        }
        for resolution in RESOLUTIONS
    }
    if max_duration_by_preset is None:
        max_duration_by_preset = {
            resolution: {
                ratio: max_duration_for_geometry(
                    geometry[resolution][ratio]["width"],
                    geometry[resolution][ratio]["height"],
                    max_native_pixel_frames,
                )
                for ratio in ASPECT_RATIOS
            }
            for resolution in RESOLUTIONS
        }
    engines = {
        "original": {
            "label": "H3 Native 高保真",
            "short_label": "高保真",
            "presets": _public_presets(ORIGINAL_PRESETS),
        },
        "lora": {
            "label": "H3 Native Turbo",
            "short_label": "Turbo",
            "presets": _public_presets(LORA_PRESETS),
        },
        "reference": {
            "label": "H3 Native 多参考",
            "short_label": "Ref2VA",
            "presets": _public_presets(ORIGINAL_PRESETS),
        },
        "reference_lora": {
            "label": "H3 Native 多参考 Turbo",
            "short_label": "Ref2VA Turbo",
            "presets": _public_presets(LORA_PRESETS),
        },
    }
    selected_engine = fixed_engine or "first_last"
    selected_variant = "base"
    families = {
        "first_last": {
            "label": "FL2VA · 文本/首尾帧生成",
            "short_label": "FL2VA",
            "description": "支持文生视频、首帧、尾帧和首尾帧约束",
            "variants": ["base", "lora"],
        },
        "reference": {
            "label": "Ref2VA · 多模态参考生成",
            "short_label": "Ref2VA",
            "description": "支持图片、视频和音频参考",
            "variants": ["base", "lora"],
        },
    }
    return {
        "deployment_mode": "fixed_engine" if fixed_engine else "multi_engine",
        "current_engine": selected_engine,
        "current_engine_options": families[selected_engine],
        "engines": {
            key: value for key, value in engines.items()
            if fixed_engine is None or engine_family(key) == fixed_engine
        },
        "service_families": families,
        "model_variants": {
            "base": {"label": "原始权重", "presets": _public_presets(ORIGINAL_PRESETS)},
            "lora": {"label": "LoRA 极速", "presets": _public_presets(LORA_PRESETS)},
        },
        "resolutions": list(RESOLUTIONS),
        "aspect_ratios": list(ASPECT_RATIOS),
        "geometry": geometry,
        "duration": {
            "min": 1,
            "max": MAX_DURATION_SECONDS,
            "default": 5,
            "fps": FPS,
            "max_by_resolution": {
                resolution: max_duration_by_preset[resolution]["16:9"]
                for resolution in RESOLUTIONS
            },
            "max_by_preset": max_duration_by_preset,
            "max_native_pixel_frames": int(max_native_pixel_frames),
            "high_resolution_max": max_duration_by_preset["1080p"]["16:9"],
            "long_duration_short_edge_max": LONG_DURATION_SHORT_EDGE_MAX,
        },
        "defaults": {
            "service_family": selected_engine,
            "model_variant": selected_variant,
            "engine": resolve_engine(selected_engine, selected_variant),
            "quality": default_quality(resolve_engine(selected_engine, selected_variant)),
            "resolution": "480p", "aspect_ratio": "16:9", "duration_seconds": 5,
            "reference_image_resolution": DEFAULT_REFERENCE_IMAGE_RESOLUTION,
            "reference_video_resolution": DEFAULT_REFERENCE_VIDEO_RESOLUTION,
        },
        "reference_media_processing": {
            "levels": list(REFERENCE_MEDIA_RESOLUTIONS),
            "image_default": DEFAULT_REFERENCE_IMAGE_RESOLUTION,
            "video_default": DEFAULT_REFERENCE_VIDEO_RESOLUTION,
            "preserve_aspect_ratio": True,
            "preserve_composition": True,
            "preserve_duration": True,
            "crop": False,
            "stretch": False,
            "pad_user_media": False,
            "upscale_small_inputs": False,
            "internal_vae_alignment": "private_replicated_edge_padding_to_32px",
            "original_meaning": "skip_extra_service_compression",
        },
        "validated_duration_seconds": [5, 15],
        "execution_modes": list(EXECUTION_MODES),
        "checkpoint_preview_resolutions": list(CHECKPOINT_PREVIEW_RESOLUTIONS),
        "advanced_limits": {
            "dimension_min": MIN_CUSTOM_DIMENSION,
            "dimension_max": MAX_CUSTOM_DIMENSION,
            "short_edge_max": MAX_CUSTOM_SHORT_EDGE,
            "max_pixels": MAX_CUSTOM_PIXELS,
            "frames_min": 5,
            "frames_max": 362,
            "frame_grid": "17*n+5",
            "sampling_steps": {
                "base": {"min": BASE_SAMPLING_STEPS[0], "max": BASE_SAMPLING_STEPS[1], "default": 20},
                "lora": {"min": LORA_SAMPLING_STEPS[0], "max": LORA_SAMPLING_STEPS[1], "default": 8},
            },
            "acceleration": {
                "min": ACCELERATION_RANGE[0], "max": ACCELERATION_RANGE[1],
                "step": 1, "default": 0,
                "meaning": "0=Dense计算；100=当前质量保护边界内的最快调度",
            },
            "checkpoint_preview_steps": {"min": 1, "max": 8, "default": 4},
            "quality_protection": "internal_non_disableable",
            "legacy_execution_controls": {
                "status": "accepted_for_persisted_clients_but_not_exposed",
                "fields": ["actual_steps", "lora_steps", "attention_keep_ratio", "sparse_scope"],
            },
            "upscaler": {
                "levels": list(UPSCALE_LEVELS),
                "dimension_min": MIN_UPSCALE_DIMENSION,
                "dimension_max": MAX_UPSCALE_DIMENSION,
                "max_pixels": MAX_UPSCALE_PIXELS,
                "preserve_aspect_ratio": True,
            },
            "preview": {
                "modes": list(PREVIEW_MODES),
                "branch_steps": {"min": 1, "max": 3, "default": 2},
                "default": "off",
            },
        },
    }
