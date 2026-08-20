"""Lossless translation between V19 action identities and runtime schedules."""

from __future__ import annotations

from typing import Iterable

from .v19_candidates import V19CandidateBlueprint
from .v19_contracts import V19TrajectoryDebt
from .v19_forecast_calibration import V19ForecastCompositeKey
from .v19_planner import V19ActionUse, V19ForecastUse, V19PlanningError


DENSE_ACTION_ID = "h3.attention.dense.sage_per_warp.sm89.v1"
FORECAST_ACTION_ID = "h3.forecast.directional.anchor3.round229.v1"
ROUND229_ACTION_ID = "h3.attention.mtcr_head_rail.round229.v1"
ROUND229_FORECAST_ANCHOR = "forecastfrontier:sparse_topk_0.0625"

_RUNTIME_PREFIX_TO_ACTION = {
    "round215": "h3.attention.interaction_hybrid.round215.v1",
    "frontier": "h3.attention.mtcr_head_rail.round188.v1",
    "fastfrontier": "h3.attention.mtcr_head_rail.round228.v1",
    "forecastfrontier": ROUND229_ACTION_ID,
}
_ACTION_TO_RUNTIME_PREFIX = {
    action_id: prefix for prefix, action_id in _RUNTIME_PREFIX_TO_ACTION.items()
}


def contiguous_forecast_runs(
    *, total_steps: int, actual_step_indices: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    actual = set(actual_step_indices)
    runs: list[tuple[int, ...]] = []
    current: list[int] = []
    for step in range(total_steps):
        if step in actual:
            if current:
                runs.append(tuple(current))
                current = []
        else:
            current.append(step)
    if current:
        runs.append(tuple(current))
    if any(run[0] == 0 or run[-1] == total_steps - 1 for run in runs):
        raise V19PlanningError(
            "forecast runs require immediate actual anchors and corrections"
        )
    return tuple(runs)


def _decode_runtime_action(runtime_action: str) -> tuple[str, str]:
    if runtime_action == "dense":
        return DENSE_ACTION_ID, "dense"
    try:
        prefix, canonical = runtime_action.split(":", 1)
        action_id = _RUNTIME_PREFIX_TO_ACTION[prefix]
    except (KeyError, ValueError) as error:
        raise V19PlanningError(
            f"runtime schedule names an unknown physical action: {runtime_action}"
        ) from error
    if canonical not in (
        "sparse_topk_0.5", "sparse_topk_0.25",
        "sparse_topk_0.1", "sparse_topk_0.0625",
    ):
        raise V19PlanningError("runtime schedule names an unknown canonical action")
    return action_id, canonical


def _encode_runtime_action(action_id: str, canonical: str) -> str:
    if action_id == DENSE_ACTION_ID and canonical == "dense":
        return "dense"
    try:
        prefix = _ACTION_TO_RUNTIME_PREFIX[action_id]
    except KeyError as error:
        raise V19PlanningError(
            f"V19 action has no runtime executor mapping: {action_id}"
        ) from error
    return f"{prefix}:{canonical}"


def _topk(canonical: str) -> float:
    if canonical == "dense":
        return 1.0
    try:
        return float(canonical.removeprefix("sparse_topk_"))
    except ValueError as error:
        raise V19PlanningError(f"cannot decode sparse mass: {canonical}") from error


def blueprint_from_runtime_schedule(
    *,
    candidate_id: str,
    total_steps: int,
    actual_step_indices: tuple[int, ...],
    attention_action_schedule: Iterable[tuple[int, int, str]],
    source: str = "v19_runtime_import",
) -> V19CandidateBlueprint:
    """Import one complete request schedule without collapsing action identity."""

    actual = set(actual_step_indices)
    runs = contiguous_forecast_runs(
        total_steps=total_steps, actual_step_indices=actual_step_indices
    )
    forecast = {step for run in runs for step in run}
    raw = tuple(
        (int(step), int(layer), str(action))
        for step, layer, action in attention_action_schedule
    )
    if len(set(raw)) != len(raw):
        raise V19PlanningError("runtime Attention schedule contains duplicate cells")
    cells = {(step, layer): action for step, layer, action in raw}
    expected_actual = {(step, layer) for step in actual for layer in range(50)}
    missing = expected_actual - set(cells)
    if missing:
        raise V19PlanningError("runtime schedule omits actual-step Attention cells")
    extra = set(cells) - expected_actual
    expected_forecast_anchors = {
        (step, layer) for step in forecast for layer in range(3)
    }
    if extra != expected_forecast_anchors:
        raise V19PlanningError("runtime schedule has invalid forecast anchor coverage")
    if any(cells[cell] != ROUND229_FORECAST_ANCHOR for cell in extra):
        raise V19PlanningError("forecast anchor physical action is not Round229")

    grouped: dict[tuple[str, str, int], list[int]] = {}
    sparse_mass_deficit = 0.0
    for step, layer in sorted(expected_actual):
        action_id, canonical = _decode_runtime_action(cells[(step, layer)])
        grouped.setdefault((action_id, canonical, layer), []).append(step)
        sparse_mass_deficit += 1.0 - _topk(canonical)
    uses: list[V19ActionUse | V19ForecastUse] = [
        V19ActionUse(
            action_id=action_id,
            canonical_action=canonical,
            step_indices=tuple(steps),
            layer_start=layer,
            layer_stop=layer + 1,
        )
        for (action_id, canonical, layer), steps in sorted(grouped.items())
    ]
    for run in runs:
        uses.append(V19ForecastUse(
            action_id=FORECAST_ACTION_ID,
            composite_key=V19ForecastCompositeKey(
                forecast_step_indices=run,
                preceding_actual_step=run[0] - 1,
                following_actual_step=run[-1] + 1,
                anchor_depth=3,
                anchor_action_id=ROUND229_ACTION_ID,
                anchor_canonical_action="sparse_topk_0.0625",
                extrapolator_id="native_depth3_local_directional_v1",
                correction_id="next_actual_full_stack_v1",
            ),
        ))
    forecast_count = len(forecast)
    debt = V19TrajectoryDebt(
        consecutive_forecasts=max((len(run) for run in runs), default=0),
        forecast_debt=float(forecast_count),
        sparse_mass_deficit=sparse_mass_deficit / 50.0,
        # The directional predictor jointly transports video and audio.  Until
        # Human evidence proves recovery, every forecast contributes one unit.
        audio_debt=float(forecast_count),
        last_refresh_step=actual_step_indices[-1],
    )
    return V19CandidateBlueprint(
        candidate_id=candidate_id,
        action_uses=tuple(uses),
        terminal_debt=debt,
        maximum_debt=debt,
        source=source,
    )


def runtime_schedule_from_blueprint(
    blueprint: V19CandidateBlueprint,
) -> tuple[tuple[int, int, str], ...]:
    """Compile a complete V19 blueprint to the immutable hot-session table."""

    schedule: dict[tuple[int, int], str] = {}
    for use in blueprint.action_uses:
        if isinstance(use, V19ActionUse):
            runtime_action = _encode_runtime_action(
                use.action_id, use.canonical_action
            )
            for step in use.step_indices:
                for layer in range(use.layer_start, use.layer_stop):
                    cell = (step, layer)
                    if cell in schedule:
                        raise V19PlanningError("blueprint has overlapping Attention cells")
                    schedule[cell] = runtime_action
        else:
            key = use.composite_key
            if (
                key.anchor_action_id != ROUND229_ACTION_ID
                or key.anchor_canonical_action != "sparse_topk_0.0625"
                or key.anchor_depth != 3
            ):
                raise V19PlanningError("unsupported V19 forecast runtime identity")
            for step in key.forecast_step_indices:
                for layer in range(key.anchor_depth):
                    cell = (step, layer)
                    if cell in schedule:
                        raise V19PlanningError("blueprint overlaps a forecast anchor")
                    schedule[cell] = ROUND229_FORECAST_ANCHOR
    return tuple((step, layer, action) for (step, layer), action in sorted(schedule.items()))


__all__ = [
    "DENSE_ACTION_ID",
    "FORECAST_ACTION_ID",
    "ROUND229_ACTION_ID",
    "ROUND229_FORECAST_ANCHOR",
    "blueprint_from_runtime_schedule",
    "contiguous_forecast_runs",
    "runtime_schedule_from_blueprint",
]
