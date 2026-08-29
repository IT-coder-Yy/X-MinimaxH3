"""Canonical CUDA-free release capabilities for all public H3 launchers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ResourceBackendId = Literal["int8_24gb", "int8_16gb", "w4a8_8gb"]
WeightTier = Literal["int8", "w4a8"]
ExecutionGraph = Literal["whole_query", "exact_streaming", "compact_streaming"]
ServiceFamily = Literal["first_last", "reference"]


@dataclass(frozen=True, slots=True)
class ResourceBackendDefinition:
    profile_id: ResourceBackendId
    vram_profile: Literal["24gb", "16gb", "8gb"]
    weight_tier: WeightTier
    provisioned_gib: float
    planner_budget_gib: float
    first_generation_levels: tuple[str, ...]
    maximum_first_generation: str
    second_sampling_levels: tuple[str, ...]
    maximum_duration_seconds: int
    maximum_short_edge: int
    maximum_pixels: int
    maximum_pixel_frames: int | None
    boundary_reference_items: int | None
    execution_preference: tuple[ExecutionGraph, ...]
    release_evidence: str

    def __post_init__(self) -> None:
        if self.planner_budget_gib >= self.provisioned_gib:
            raise ValueError("planner budget must retain device headroom")
        if not self.first_generation_levels or not self.execution_preference:
            raise ValueError("resource backend capabilities cannot be empty")
        if len(set(self.execution_preference)) != len(self.execution_preference):
            raise ValueError("execution preference cannot contain duplicates")


INT8_24GB_BACKEND = ResourceBackendDefinition(
    "int8_24gb", "24gb", "int8", 24.0, 23.25,
    ("360p", "480p", "720p", "1080p"), "1080p_15s",
    ("720p", "1080p", "2k"), 15, 1088, 1920 * 1088, None, None,
    ("exact_streaming", "compact_streaming", "whole_query"),
    "sm89_int8_1080p15_release_boundary_20260827",
)
INT8_16GB_BACKEND = ResourceBackendDefinition(
    "int8_16gb", "16gb", "int8", 16.0, 15.25,
    ("360p", "480p", "720p", "1080p"), "1080p_15s_experimental",
    ("720p", "1080p", "2k"), 15, 1088, 1920 * 1088, None, None,
    ("exact_streaming", "compact_streaming", "whole_query"),
    "sm89_int8_16gb_ref2va_1440p15_second_sampling_gate_20260829",
)
W4A8_8GB_BACKEND = ResourceBackendDefinition(
    "w4a8_8gb", "8gb", "w4a8", 8.0, 7.25,
    ("360p", "480p", "720p"), "720p_15s", ("720p", "1080p"), 15, 736, 1280 * 736,
    1280 * 736 * 362, 1, ("compact_streaming",),
    "sm89_w4a8_hard8_1080p15_second_sampling_gate_20260828",
)
RESOURCE_BACKENDS: dict[ResourceBackendId, ResourceBackendDefinition] = {
    item.profile_id: item
    for item in (INT8_24GB_BACKEND, INT8_16GB_BACKEND, W4A8_8GB_BACKEND)
}


@dataclass(frozen=True, slots=True)
class LauncherDefinition:
    launcher_id: str
    service_family: ServiceFamily
    resource_profile: ResourceBackendId
    label: str
    short_label: str
    description: str

    @property
    def backend(self) -> ResourceBackendDefinition:
        return RESOURCE_BACKENDS[self.resource_profile]

    @property
    def weight_tier(self) -> WeightTier:
        return self.backend.weight_tier

    @property
    def vram_profile(self) -> str:
        return self.backend.vram_profile

    @property
    def required_model_roles(self) -> tuple[str, ...]:
        dit = "ref_dit" if self.service_family == "reference" else "dit"
        if self.weight_tier == "w4a8":
            dit += "_w4a8"
        roles = (dit, "text_encoder", "video_vae", "audio_vae", "lora")
        return roles + (("latent_upscaler",) if self.backend.second_sampling_levels else ())


_LAUNCHERS = (
    LauncherDefinition("fl2va_int8_24gb", "first_last", "int8_24gb", "24GB · FL2VA", "24GB FL2VA", "INT8高速后端；首代最高1080p，二采最高1440P"),
    LauncherDefinition("ref2va_int8_24gb", "reference", "int8_24gb", "24GB · Ref2VA", "24GB Ref2VA", "INT8多参考高速后端；首代最高1080p，二采最高1440P"),
    LauncherDefinition("fl2va_int8_16gb", "first_last", "int8_16gb", "16GB · FL2VA", "16GB FL2VA", "INT8紧凑高速后端；实验性首代最高1080p×15秒，二采最高1440P"),
    LauncherDefinition("ref2va_int8_16gb", "reference", "int8_16gb", "16GB · Ref2VA", "16GB Ref2VA", "INT8多参考紧凑后端；实验性首代最高1080p×15秒，二采最高1440P"),
    LauncherDefinition("fl2va_w4a8_8gb", "first_last", "w4a8_8gb", "8GB · FL2VA", "8GB FL2VA", "8GB低比特权重；首代最高720p×15秒，二采最高1080p"),
    LauncherDefinition("ref2va_w4a8_8gb", "reference", "w4a8_8gb", "8GB · Ref2VA", "8GB Ref2VA", "8GB低比特参考后端；720p×15秒限单参考，二采最高1080p"),
)
LAUNCHER_DEFINITIONS = {item.launcher_id: item for item in _LAUNCHERS}
LEGACY_MODEL_LAUNCHERS = {
    "fl2va_int8": "fl2va_int8_24gb",
    "ref2va_int8": "ref2va_int8_24gb",
    "fl2va_w4a8": "fl2va_w4a8_8gb",
    "ref2va_w4a8": "ref2va_w4a8_8gb",
}


def get_resource_backend(profile_id: str, *, weight_tier: str | None = None) -> ResourceBackendDefinition:
    try:
        backend = RESOURCE_BACKENDS[profile_id]  # type: ignore[index]
    except KeyError as error:
        raise ValueError(f"unknown resource backend: {profile_id}") from error
    if weight_tier is not None and backend.weight_tier != weight_tier:
        raise ValueError(f"{profile_id} requires {backend.weight_tier} weights")
    return backend


__all__ = [
    "ExecutionGraph", "INT8_16GB_BACKEND", "INT8_24GB_BACKEND",
    "LAUNCHER_DEFINITIONS", "LEGACY_MODEL_LAUNCHERS", "LauncherDefinition",
    "RESOURCE_BACKENDS", "ResourceBackendDefinition", "ResourceBackendId",
    "ServiceFamily", "W4A8_8GB_BACKEND", "WeightTier", "get_resource_backend",
]
