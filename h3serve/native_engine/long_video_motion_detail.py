"""Fail-closed service integration for the measured 720p/15s candidate.

This module contains no public quality control.  It converts one exact,
Human-review-pending workload envelope into the same internal mechanics used
by the calibration runner.  The environment switch is deliberately off by
default until continuous-playback review accepts the candidate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..contract import GenerationSpec


ENVIRONMENT_SWITCH = "H3_NATIVE_LONG_VIDEO_REVIEW"


def candidate_requested() -> bool:
    return os.environ.get(ENVIRONMENT_SWITCH, "0") == "1"


@dataclass(frozen=True, slots=True)
class MotionDetailCandidate:
    initial_width: int = 864
    initial_height: int = 480
    refinement_steps: int = 2
    refinement_denoise: float = 0.025
    dense_tail_steps: int = 1


def select_candidate(
    spec: "GenerationSpec",
    *,
    first_frame: object | None,
    last_frame: object | None,
    reference_images: tuple[object, ...],
    reference_videos: tuple[object, ...],
    reference_audios: tuple[object, ...],
) -> MotionDetailCandidate | None:
    """Match only the measured original-weight, unconditioned envelope."""

    if not candidate_requested() or spec.engine != "original":
        return None
    if (spec.width, spec.height, spec.frames) != (1280, 736, 362):
        return None
    preset = spec.preset
    if (
        int(preset.get("actual_steps", -1)) != 12
        or int(preset.get("forecast_steps", -1)) != 8
    ):
        return None
    if (
        first_frame is not None
        or last_frame is not None
        or reference_images
        or reference_videos
        or reference_audios
    ):
        return None
    return MotionDetailCandidate()


# Round98's head-risk lattice.  The compact tiers are also used by the local
# quality-constrained implementation; formulas below reproduce the measured
# CLI vectors exactly without checking 224 anonymous decimal literals into a
# production-facing configuration surface.
_HEAD_RISK_TIERS = (
    1, 0, 1, 2, 2, 2, 1, 2, 2, 2, 1, 0, 1, 2,
    1, 1, 1, 1, 0, 0, 0, 1, 2, 2, 2, 1, 2, 0,
    0, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 1, 0, 0,
    0, 2, 2, 0, 1, 2, 0, 1, 1, 0, 2, 2, 2, 0,
)
_SENSITIVE_LAYERS = (
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45,
)


def make_attention_backend():
    """Build measured MTCR behind a request-local compatibility router.

    Outside the exact long-video envelope, the immutable DiT graph retains
    the established request-routed Dense/Sparge behavior used by advanced
    creator controls.  The measured policy therefore cannot leak merely
    because the process was started with its review switch enabled.
    """

    from .model.kernels import (
        current_long_video_attention_enabled,
        make_routed_sparge_attention_sm89,
        make_trajectory_layer_modality_routed_sparge_attention_sm89,
        sage_attention_sm89,
    )

    cruise_aggressive = (0.0625,) * len(_HEAD_RISK_TIERS)
    cruise_safe = tuple(0.10 if tier == 2 else 0.08 for tier in _HEAD_RISK_TIERS)
    anchor_aggressive = tuple(0.35 + 0.05 * tier for tier in _HEAD_RISK_TIERS)
    anchor_safe = tuple(value + 0.10 for value in anchor_aggressive)
    recovery_aggressive = tuple(
        0.30 if tier == 2 else 0.25 for tier in _HEAD_RISK_TIERS
    )
    recovery_safe = tuple(
        0.35 if tier == 2 else 0.30 for tier in _HEAD_RISK_TIERS
    )
    measured = make_trajectory_layer_modality_routed_sparge_attention_sm89(
        aggressive_topk=cruise_aggressive,
        sensitive_layers=_SENSITIVE_LAYERS,
        dense_step_indices=(),
        safe_topk=cruise_safe,
        anchor_step_indices=(0,),
        anchor_aggressive_topk=anchor_aggressive,
        anchor_safe_topk=anchor_safe,
        recovery_step_indices=(17, 18, 19),
        recovery_aggressive_topk=recovery_aggressive,
        recovery_safe_topk=recovery_safe,
        experimental_minimum_topk=0.0625,
        temporal_correspondence_radius=1,
        temporal_spatial_block_radius=1,
        temporal_global_anchor_stride=8,
        temporal_global_spatial_block_radius=0,
        request_guarded=True,
    )
    generic = make_routed_sparge_attention_sm89()

    class RequestRouter:
        approximate = True
        request_routed = True

        @staticmethod
        def _measured() -> bool:
            return current_long_video_attention_enabled()

        def __call__(self, query, key, value):
            backend = measured if self._measured() else generic
            return backend(query, key, value)

        def protected_queries(self, query, key, value):
            if self._measured():
                return measured.protected_queries(query, key, value)
            return sage_attention_sm89(query, key, value)

        def selected_queries(
            self,
            prefix_query,
            video_query,
            key,
            value,
            *,
            protected_tokens: int,
            video_query_indices=None,
        ):
            if self._measured():
                return measured.selected_queries(
                    prefix_query,
                    video_query,
                    key,
                    value,
                    protected_tokens=protected_tokens,
                    video_query_indices=video_query_indices,
                )
            import torch

            return sage_attention_sm89(
                torch.cat((prefix_query, video_query), dim=0), key, value
            )

    router = RequestRouter()
    # Read-only handles keep policy parity auditable without putting the
    # calibrated vectors on a public API surface.
    router.measured_backend = measured
    router.generic_backend = generic
    return router


__all__ = [
    "ENVIRONMENT_SWITCH",
    "MotionDetailCandidate",
    "candidate_requested",
    "make_attention_backend",
    "select_candidate",
]
