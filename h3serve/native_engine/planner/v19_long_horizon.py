"""Evidence-bounded long-horizon trajectories for V19 proposal generation.

The creator interface remains ``sampling_steps + acceleration``.  This module
does not inspect prompt text or reference content.  It turns those two user
controls into a deterministic Actual/Forecast skeleton whose fastest currently
eligible endpoint is the Human-reviewed Round188 12/8 trajectory at 20 steps.
Attention allocation remains a separate planner dimension.
"""

from __future__ import annotations

import math

from .v19_candidates import V19CandidateBlueprint
from .v19_human_constraints import (
    evaluate_v19_human_constraints,
    v19_long_horizon_screening_policy,
)
from .v19_planner import V19PlanningError
from .v19_runtime_bridge import (
    ROUND229_FORECAST_ANCHOR,
    blueprint_from_runtime_schedule,
    contiguous_forecast_runs,
)


V19_LONG_HORIZON_TRAJECTORY = "h3_v19_long_horizon_round188_nested_v1"
V19_LONG_HORIZON_REPLAY_SOURCE = "v19_long_horizon_round188_replay_v1"
ROUND188_RUNTIME_PREFIX = "frontier"
ROUND188_REVIEWED_20_STEP_ACTUALS = (0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 18, 19)
ROUND188_CAUSAL_LAYERS = tuple((*range(30, 44), 45))


def _minimum_long_horizon_actual_steps(total_steps: int) -> tuple[int, ...]:
    if total_steps < 4:
        raise V19PlanningError(
            "V19 long-horizon trajectory requires at least four steps"
        )
    if total_steps == 20:
        return ROUND188_REVIEWED_20_STEP_ACTUALS

    minimum_count = int(math.ceil(total_steps * 0.60))
    opening_count = max(1, int(math.ceil(total_steps * 0.25)))
    terminal_count = min(3, total_steps - opening_count)
    actual = set(range(opening_count))
    actual.update(range(total_steps - terminal_count, total_steps))

    # These normalized anchors reproduce 6/8/11/14 for N=20 and give other
    # step counts the same opening/middle/terminal correction geometry.
    for fraction in (0.30, 0.40, 0.55, 0.70):
        actual.add(min(total_steps - 1, int(round(total_steps * fraction))))

    # Enforce at most two consecutive Forecast positions before adding lower
    # priority Actual points to reach the 60% evidence floor.
    while True:
        ordered = sorted(actual)
        oversized = next(
            (
                (left, right)
                for left, right in zip(ordered, ordered[1:])
                if right - left > 3
            ),
            None,
        )
        if oversized is None:
            break
        left, right = oversized
        actual.add(min(right - 1, left + 3))

    while len(actual) < minimum_count:
        forecast = [step for step in range(total_steps) if step not in actual]
        if not forecast:
            break
        # Prefer the position farthest from an existing correction; ties are
        # stable and favor later phases where long-horizon debt is larger.
        chosen = max(
            forecast,
            key=lambda step: (
                min(abs(step - anchor) for anchor in actual),
                step,
            ),
        )
        actual.add(chosen)
    return tuple(sorted(actual))


def v19_long_horizon_actual_steps(
    total_steps: int,
    acceleration: float,
) -> tuple[int, ...]:
    """Map the user controls to a nested, evidence-bounded Actual trajectory.

    Acceleration 0 retains every requested sampling position.  The trajectory
    reaches the current 60% Human evidence floor at acceleration 75 and does
    not become more aggressive above 75 until a new long-video Human result
    authorizes that expansion.  Higher acceleration can still choose cheaper
    Attention actions on a separately certified Pareto branch.
    """

    if not math.isfinite(acceleration) or not 0.0 <= acceleration <= 100.0:
        raise V19PlanningError("V19 acceleration must lie inside [0, 100]")
    minimum = _minimum_long_horizon_actual_steps(total_steps)
    if acceleration == 0.0:
        return tuple(range(total_steps))
    removable = tuple(step for step in range(total_steps) if step not in minimum)
    progress = min(acceleration / 75.0, 1.0)
    remove_count = int(round(progress * len(removable)))
    # Remove isolated positions before forming length-two Forecast runs.  The
    # final set is exactly Round188 at N=20 and every intermediate set is a
    # superset, so reducing acceleration never removes an Actual correction.
    removal_order = tuple(removable[::2]) + tuple(removable[1::2])
    removed = set(removal_order[:remove_count])
    actual = tuple(step for step in range(total_steps) if step not in removed)
    longest = max((len(run) for run in contiguous_forecast_runs(
        total_steps=total_steps,
        actual_step_indices=actual,
    )), default=0)
    if longest > 2:
        raise V19PlanningError(
            "nested long-horizon trajectory exceeded the reviewed Forecast run"
        )
    return actual


def build_v19_long_horizon_round188_replay(
    *,
    candidate_id: str,
    total_steps: int = 20,
    acceleration: float = 75.0,
) -> V19CandidateBlueprint:
    """Build a sealed Round188 physical replay on the current V19 runtime."""

    actual_steps = v19_long_horizon_actual_steps(total_steps, acceleration)
    actual = set(actual_steps)
    schedule: list[tuple[int, int, str]] = []
    for step in range(total_steps):
        if step not in actual:
            schedule.extend(
                (step, layer, ROUND229_FORECAST_ANCHOR)
                for layer in range(3)
            )
            continue
        for layer in range(50):
            canonical = (
                "sparse_topk_0.1"
                if layer in ROUND188_CAUSAL_LAYERS
                else "sparse_topk_0.0625"
            )
            schedule.append(
                (step, layer, f"{ROUND188_RUNTIME_PREFIX}:{canonical}")
            )
    blueprint = blueprint_from_runtime_schedule(
        candidate_id=candidate_id,
        total_steps=total_steps,
        actual_step_indices=actual_steps,
        attention_action_schedule=tuple(schedule),
        source=V19_LONG_HORIZON_REPLAY_SOURCE,
    )
    report = evaluate_v19_human_constraints(
        blueprint,
        v19_long_horizon_screening_policy(total_steps),
    )
    if not report.proposal_eligible:
        raise V19PlanningError(
            "long-horizon replay violates its evidence floor: "
            + "; ".join(report.rejection_reasons)
        )
    return blueprint


__all__ = [
    "ROUND188_CAUSAL_LAYERS",
    "ROUND188_REVIEWED_20_STEP_ACTUALS",
    "V19_LONG_HORIZON_REPLAY_SOURCE",
    "V19_LONG_HORIZON_TRAJECTORY",
    "build_v19_long_horizon_round188_replay",
    "v19_long_horizon_actual_steps",
]
