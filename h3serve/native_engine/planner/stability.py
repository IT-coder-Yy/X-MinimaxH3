"""Fail-closed kernel choices for measured long H3 packed sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DenseQKGranularity = Literal["per_thread", "per_warp"]

# Two same-prompt 1280x736x362 runs at 100,163 packed tokens produced a
# localized collapse in the final video-token tail with Sage's per-thread Q/K
# path.  The same request with per-warp Q/K restored the terminal bottom/upper
# latent-energy ratio from 0.766 to 0.998.  Shorter 34,780-token requests do not
# exhibit the failure.  Keep the boundary conservative and evidence anchored.
LONG_SEQUENCE_STABLE_QK_MIN_TOKENS = 100_000

# The streaming route is selected from generated-video geometry rather than
# total text/reference length.  This keeps 720p15 and short 1080p requests on
# their already validated path while catching the 107 x 2,040-token target
# latent that makes 1080p15 exceed physical VRAM with whole-query Attention.
LONG_SEQUENCE_STREAMING_MIN_VIDEO_TOKENS = 200_000
LONG_SEQUENCE_EXTENDED_PREFIX_MIN_PACKED_TOKENS = 230_000
LONG_SEQUENCE_VALIDATED_MAX_PACKED_TOKENS = 250_000


@dataclass(frozen=True, slots=True)
class LongSequenceChunkDecision:
    query_chunk_tokens: int | None
    projection_chunk_tokens: int
    split_qkv_outputs: bool
    single_qknorm_rope: bool
    parallel_sparse_lut: bool
    reason: str | None


def select_long_sequence_chunks(
    *,
    video_tokens: int,
    packed_tokens: int,
) -> LongSequenceChunkDecision:
    """Choose memory-only chunks from request geometry, never prompt content.

    The first two buckets are direct measurements on the same RTX 4090:
    32,768 Query rows are fastest near the 219,659-token FL2VA anchor, while
    16,384 rows reduce both peak and time for 241,981-token Ref2VA-style
    prefixes.  Shapes beyond the measured envelope fail closed instead of
    silently relying on CUDA HMM oversubscription.
    """

    if video_tokens <= 0:
        raise ValueError("video_tokens must be positive")
    if packed_tokens < video_tokens:
        raise ValueError("packed_tokens cannot be smaller than video_tokens")
    if video_tokens < LONG_SEQUENCE_STREAMING_MIN_VIDEO_TOKENS:
        return LongSequenceChunkDecision(None, 8192, False, False, False, None)
    if packed_tokens > LONG_SEQUENCE_VALIDATED_MAX_PACKED_TOKENS:
        raise ValueError(
            "request exceeds the validated long-sequence packed-token envelope"
        )
    if packed_tokens >= LONG_SEQUENCE_EXTENDED_PREFIX_MIN_PACKED_TOKENS:
        return LongSequenceChunkDecision(
            16_384,
            8192,
            True,
            True,
            True,
            "request_geometry_extended_prefix",
        )
    return LongSequenceChunkDecision(
        32_768,
        8192,
        True,
        True,
        True,
        "request_geometry_1080p15",
    )


def select_stable_dense_qk_quantization(
    requested: DenseQKGranularity,
    *,
    packed_tokens: int,
) -> tuple[DenseQKGranularity, str | None]:
    """Return the effective Sage Q/K path and an optional override reason.

    This is an internal correctness policy, not a creative quality control.
    Explicit ``per_thread`` requests are upgraded for measured long shapes;
    normal-sized requests remain unchanged and therefore retain their exact
    historical execution path.
    """

    if requested not in ("per_thread", "per_warp"):
        raise ValueError("dense Q/K quantization must be per_thread or per_warp")
    if packed_tokens <= 0:
        raise ValueError("packed_tokens must be positive")
    if (
        requested == "per_thread"
        and packed_tokens >= LONG_SEQUENCE_STABLE_QK_MIN_TOKENS
    ):
        return "per_warp", "long_sequence_tail_stability"
    return requested, None


__all__ = [
    "DenseQKGranularity",
    "LongSequenceChunkDecision",
    "LONG_SEQUENCE_EXTENDED_PREFIX_MIN_PACKED_TOKENS",
    "LONG_SEQUENCE_STABLE_QK_MIN_TOKENS",
    "LONG_SEQUENCE_STREAMING_MIN_VIDEO_TOKENS",
    "LONG_SEQUENCE_VALIDATED_MAX_PACKED_TOKENS",
    "select_long_sequence_chunks",
    "select_stable_dense_qk_quantization",
]
