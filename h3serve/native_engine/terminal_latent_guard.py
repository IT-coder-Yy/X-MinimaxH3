"""Fail-closed repair for a localized collapsed H3 terminal video token.

H3 video latents repeat a five-token temporal phase.  A rare long-sequence
failure can leave only the lower spatial part of the final token with sharply
reduced energy while the matching earlier phases and the upper image remain
normal.  The non-causal Video-VAE turns that latent discontinuity into a
visible band across roughly the final five frames.

This guard is deliberately narrow.  It does nothing unless three independent
signals agree: the bottom/top energy ratio is a robust phase-history outlier,
the upper region remains stable, and the lower region alone collapses.  The
repair then feathers only the lower half toward a same-phase linear estimate,
using the minimum strength needed to recover conservative historical energy.
Normal latents are returned bit-identically.
"""

from __future__ import annotations

from typing import Any

import torch


def _standard_deviation(value: torch.Tensor) -> torch.Tensor:
    return value.float().std(unbiased=False).clamp_min(1.0e-8)


def stabilize_terminal_video_latent_(
    video: torch.Tensor,
    *,
    temporal_period: int = 5,
    split_fraction: float = 0.52,
    hard_ratio_limit: float = 0.85,
    robust_mad_multiplier: float = 8.0,
    stable_top_range: tuple[float, float] = (0.8, 1.2),
    collapsed_bottom_limit: float = 0.85,
    target_history_fraction: float = 0.95,
    target_ratio_cap: float = 0.97,
    max_repair_strength: float = 1.0,
) -> dict[str, Any]:
    """Detect and repair one isolated terminal bottom-region collapse in-place."""

    profile: dict[str, Any] = {
        "schema_version": 1,
        "mode": "terminal_phase_guard",
        "triggered": False,
        "reason": "not_evaluated",
    }
    if video.ndim != 5 or video.shape[0] != 1:
        profile["reason"] = "unsupported_shape"
        return profile
    if not video.is_floating_point() or video.shape[2] < temporal_period * 3:
        profile["reason"] = "insufficient_history"
        return profile
    if (
        not 0.0 < split_fraction < 1.0
        or not 0.0 < target_history_fraction <= 1.0
        or not 0.0 < target_ratio_cap <= 2.0
        or not 0.0 < max_repair_strength <= 1.0
    ):
        raise ValueError("terminal latent guard fractions are invalid")

    height = int(video.shape[-2])
    split = min(height - 1, max(1, round(height * split_fraction)))
    terminal = int(video.shape[2]) - 1
    phase = terminal % temporal_period
    history_indices = tuple(range(phase, terminal, temporal_period))
    if len(history_indices) < 2:
        profile["reason"] = "insufficient_phase_history"
        return profile

    ratios = []
    for index in history_indices:
        token = video[0, :, index]
        top = _standard_deviation(token[:, :split])
        bottom = _standard_deviation(token[:, split:])
        ratios.append(bottom / top)
    history = torch.stack(ratios)
    median = history.median()
    mad = (history - median).abs().median()

    current = video[:, :, terminal : terminal + 1]
    previous = video[:, :, terminal - temporal_period : terminal - temporal_period + 1]
    older = video[
        :, :, terminal - 2 * temporal_period : terminal - 2 * temporal_period + 1
    ]
    current_top = _standard_deviation(current[..., :split, :])
    current_bottom = _standard_deviation(current[..., split:, :])
    previous_top = _standard_deviation(previous[..., :split, :])
    previous_bottom = _standard_deviation(previous[..., split:, :])
    current_ratio = current_bottom / current_top
    top_relative = current_top / previous_top
    bottom_relative = current_bottom / previous_bottom
    robust_limit = median - robust_mad_multiplier * mad.clamp_min(0.005)
    trigger_limit = torch.minimum(
        robust_limit,
        current_ratio.new_tensor(float(hard_ratio_limit)),
    )

    values = torch.stack(
        (
            current_ratio,
            median,
            mad,
            trigger_limit,
            top_relative,
            bottom_relative,
        )
    ).detach().cpu().tolist()
    profile.update(
        {
            "terminal_phase": int(phase),
            "history_count": len(history_indices),
            "split_row": split,
            "bottom_top_ratio": float(values[0]),
            "history_ratio_median": float(values[1]),
            "history_ratio_mad": float(values[2]),
            "trigger_limit": float(values[3]),
            "top_relative_to_previous_phase": float(values[4]),
            "bottom_relative_to_previous_phase": float(values[5]),
        }
    )
    top_is_stable = stable_top_range[0] <= values[4] <= stable_top_range[1]
    bottom_collapsed = values[5] < collapsed_bottom_limit
    ratio_is_outlier = values[0] < values[3]
    if not ratio_is_outlier:
        profile["reason"] = "ratio_not_outlier"
        return profile
    if not top_is_stable:
        profile["reason"] = "top_region_not_stable"
        return profile
    if not bottom_collapsed:
        profile["reason"] = "bottom_region_not_collapsed"
        return profile

    estimate = previous + (previous - older)
    y = torch.linspace(0.0, 1.0, height, device=video.device, dtype=torch.float32)
    base_mask = ((y - split_fraction) / (0.76 - split_fraction)).clamp(0.0, 1.0)
    base_mask = base_mask.view(1, 1, 1, height, 1)

    # Use the smallest reconstruction strength that restores a conservative
    # fraction of the same-phase energy. A fixed half-strength left a visible
    # soft band for the original failing sample; always replacing the whole
    # region would alter more motion/content than necessary.
    target_ratio = min(values[1] * target_history_fraction, target_ratio_cap)
    low, high = 0.0, float(max_repair_strength)
    for _ in range(8):
        middle = (low + high) * 0.5
        candidate = current.lerp(estimate, base_mask.mul(middle))
        candidate_top = _standard_deviation(candidate[..., :split, :])
        candidate_bottom = _standard_deviation(candidate[..., split:, :])
        if float((candidate_bottom / candidate_top).item()) >= target_ratio:
            high = middle
        else:
            low = middle
    repair_strength = high
    mask = base_mask.mul(repair_strength)
    current.lerp_(estimate, mask.to(dtype=current.dtype))
    profile.update(
        {
            "triggered": True,
            "reason": "localized_terminal_bottom_collapse",
            "repair": "adaptive_same_phase_linear_feather",
            "repair_strength": float(repair_strength),
            "repair_target_ratio": float(target_ratio),
        }
    )
    return profile


__all__ = ["stabilize_terminal_video_latent_"]
