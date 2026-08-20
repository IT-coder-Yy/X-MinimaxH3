"""Fail-closed kernel choices for measured long H3 packed sequences."""

from __future__ import annotations

from typing import Literal


DenseQKGranularity = Literal["per_thread", "per_warp"]

# Two same-prompt 1280x736x362 runs at 100,163 packed tokens produced a
# localized collapse in the final video-token tail with Sage's per-thread Q/K
# path.  The same request with per-warp Q/K restored the terminal bottom/upper
# latent-energy ratio from 0.766 to 0.998.  Shorter 34,780-token requests do not
# exhibit the failure.  Keep the boundary conservative and evidence anchored.
LONG_SEQUENCE_STABLE_QK_MIN_TOKENS = 100_000


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
    "LONG_SEQUENCE_STABLE_QK_MIN_TOKENS",
    "select_stable_dense_qk_quantization",
]
