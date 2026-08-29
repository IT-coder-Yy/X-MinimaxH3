"""Declarative MiniMax H3 LoRA profiles shared by the API and native runtime.

The checkpoint remains the source of truth for tensor rank/alpha.  This small
registry only records inference semantics that cannot be recovered from tensor
shapes: task family, distilled NFE, sigma shifts, and audio clock ownership.
Unknown native-name H3 adapters retain the historical Larry-compatible
fallback so installing ordinary community LoRAs does not require editing code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class H3LoRAProfile:
    profile_id: str
    display_name: str
    checkpoint_names: tuple[str, ...]
    task_families: tuple[str, ...]
    recommended_steps: tuple[int, ...]
    default_steps: int
    video_shift: float
    audio_shift: float
    clock_mode: str
    key_format: str
    training_resolution: str

    def public_dict(self) -> dict[str, object]:
        document = asdict(self)
        document.pop("checkpoint_names", None)
        return document


LARRY_TURBO = H3LoRAProfile(
    profile_id="larry_turbo_v4_step600_ema",
    display_name="Larry Turbo v4-600 EMA",
    checkpoint_names=("minimax_h3_turbo_v4_step600_ema.safetensors",),
    task_families=("first_last", "reference"),
    recommended_steps=(4, 5, 6, 7, 8),
    default_steps=6,
    video_shift=12.0,
    audio_shift=3.0,
    clock_mode="shared_video",
    key_format="minimax-h3-native",
    training_resolution="general",
)

LIGHTX2V_FL2V_8STEP_768P = H3LoRAProfile(
    profile_id="lightx2v_fl2v_8step_v1_768p",
    display_name="LightX2V FL2VA Turbo 8-step v1.0 · 768p",
    checkpoint_names=(
        "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors",
    ),
    task_families=("first_last",),
    recommended_steps=(8,),
    default_steps=8,
    video_shift=6.0,
    audio_shift=3.0,
    clock_mode="dual_shift",
    key_format="minimax-h3-diffusers",
    training_resolution="1344x768",
)

LIGHTX2V_FL2V_4STEP_V11_768P = H3LoRAProfile(
    profile_id="lightx2v_fl2v_4step_v1_1_768p",
    display_name="LightX2V FL2VA Turbo 4-step v1.1 · 768p",
    checkpoint_names=(
        "minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16.safetensors",
    ),
    task_families=("first_last",),
    recommended_steps=(4,),
    default_steps=4,
    video_shift=6.0,
    audio_shift=3.0,
    clock_mode="dual_shift",
    key_format="minimax-h3-diffusers",
    training_resolution="1344x768",
)

LIGHTX2V_REF2V_4STEP = H3LoRAProfile(
    profile_id="lightx2v_ref2v_4step_v0_1",
    display_name="LightX2V Ref2VA Turbo 4-step v0.1 · mixed 544p",
    checkpoint_names=(
        "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
    ),
    task_families=("reference",),
    recommended_steps=(4,),
    default_steps=4,
    video_shift=12.0,
    audio_shift=3.0,
    clock_mode="dual_shift",
    key_format="minimax-h3-diffusers",
    training_resolution="mixed-544p",
)


KNOWN_LORA_PROFILES = (
    LARRY_TURBO,
    LIGHTX2V_FL2V_8STEP_768P,
    LIGHTX2V_FL2V_4STEP_V11_768P,
    LIGHTX2V_REF2V_4STEP,
)

_BY_FILENAME = {
    filename: profile
    for profile in KNOWN_LORA_PROFILES
    for filename in profile.checkpoint_names
}


def resolve_lora_profile(path: str | Path) -> H3LoRAProfile:
    """Resolve known inference semantics or use the native H3 fallback."""

    filename = Path(path).name
    return _BY_FILENAME.get(
        filename,
        H3LoRAProfile(
            profile_id=f"community_native::{filename}",
            display_name=Path(filename).stem,
            checkpoint_names=(filename,),
            task_families=("first_last", "reference"),
            recommended_steps=(4, 5, 6, 7, 8),
            default_steps=6,
            video_shift=12.0,
            audio_shift=3.0,
            clock_mode="shared_video",
            key_format="minimax-h3-native",
            training_resolution="unknown",
        ),
    )


__all__ = [
    "H3LoRAProfile",
    "KNOWN_LORA_PROFILES",
    "resolve_lora_profile",
]
