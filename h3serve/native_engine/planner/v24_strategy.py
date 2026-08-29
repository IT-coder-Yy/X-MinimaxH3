"""Canonical high-dimensional strategy vectors for V24 Human learning.

The deployment planner emits a compressed physical schedule.  This module
expands it into a fixed-shape mixed-discrete vector so historical outputs,
curve proposals and Human reviews all refer to the same mathematical object.
It does not introduce an execution action or alter inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from .v19_planner import V19PlanningError
from .v24_deployment import (
    _ACTION_RISK,
    _ATTENTION_COMPUTE,
    _ATTENTION_COST_RATIO,
    _CANONICAL_RANK,
    _FORECAST_COMPUTE,
    _NON_ATTENTION_COMPUTE,
    _RANK_TO_CANONICAL,
    V24DeploymentSelection,
)


V24_STRATEGY_VECTOR_SCHEMA = "h3_v24_strategy_vector_v1"
V24_STRATEGY_FEATURE_SCHEMA = "h3_v24_strategy_features_v1"

PREDICTED_TAIL = -1
RAIL_PREDICTED = -1
RAIL_FRONTIER = 0
RAIL_FORECAST_FRONTIER = 1
RAIL_DENSE = 2
RAIL_OTHER = 3

V24_STRATEGY_FEATURE_NAMES = (
    "compute_ratio",
    "forecast_fraction",
    "forecast_run_quadratic",
    "forecast_opening_exposure",
    "forecast_terminal_exposure",
    "attention_approximation_mean",
    "attention_opening_exposure",
    "attention_terminal_exposure",
    "attention_layers_00_29",
    "attention_causal_30_43",
    "attention_bridge_44_45",
    "attention_tail_46_49",
    "forecast_attention_interaction",
    "forecast_aware_sparse_fraction",
)


def _sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _longest_forecast_run(step_modes: tuple[int, ...]) -> int:
    run = maximum = 0
    for actual in step_modes:
        run = 0 if actual else run + 1
        maximum = max(maximum, run)
    return maximum


def _rail_code(action: str, canonical: str) -> int:
    if canonical == "dense":
        return RAIL_DENSE
    if ":" not in action:
        return RAIL_OTHER
    prefix = action.rsplit(":", 1)[0]
    if prefix == "frontier":
        return RAIL_FRONTIER
    if prefix == "forecastfrontier":
        return RAIL_FORECAST_FRONTIER
    return RAIL_OTHER


@dataclass(frozen=True, slots=True)
class V24StrategyVector:
    """Fixed-shape mixed-discrete physical strategy.

    ``step_modes`` uses 1 for Actual and 0 for Forecast.  Attention ranks use
    0/1/2/3/4 for 6.25/10/25/50/Dense and -1 for a predicted DiT tail cell.
    Rails are encoded independently so categorical implementations are never
    assigned a false ordinal relationship.
    """

    total_steps: int
    step_modes: tuple[int, ...]
    attention_ranks: tuple[int, ...]
    attention_rails: tuple[int, ...]
    execution_profile_hint: str | None = None
    schema_version: str = V24_STRATEGY_VECTOR_SCHEMA

    def __post_init__(self) -> None:
        if not 4 <= self.total_steps <= 30:
            raise V19PlanningError("V24 strategy steps must lie inside [4,30]")
        if len(self.step_modes) != self.total_steps or any(
            value not in (0, 1) for value in self.step_modes
        ):
            raise V19PlanningError("invalid V24 strategy step modes")
        expected = self.total_steps * 50
        if len(self.attention_ranks) != expected or len(self.attention_rails) != expected:
            raise V19PlanningError("V24 strategy must contain S*50 Attention cells")
        if any(value not in (-1, 0, 1, 2, 3, 4) for value in self.attention_ranks):
            raise V19PlanningError("invalid V24 Attention fidelity rank")
        if any(value not in (-1, 0, 1, 2, 3) for value in self.attention_rails):
            raise V19PlanningError("invalid V24 Attention rail code")
        for step, actual in enumerate(self.step_modes):
            ranks = self.attention_ranks[step * 50:(step + 1) * 50]
            rails = self.attention_rails[step * 50:(step + 1) * 50]
            expected_depth = 50 if actual else 3
            if any(value == PREDICTED_TAIL for value in ranks[:expected_depth]):
                raise V19PlanningError("executed H3 blocks cannot be marked predicted")
            if any(value != PREDICTED_TAIL for value in ranks[expected_depth:]):
                raise V19PlanningError("Forecast tail blocks must be predicted")
            if any(value != RAIL_PREDICTED for value in rails[expected_depth:]):
                raise V19PlanningError("predicted blocks cannot name an Attention rail")

    @property
    def digest(self) -> str:
        return _sha256({
            "schema_version": self.schema_version,
            "total_steps": self.total_steps,
            "step_modes": self.step_modes,
            "attention_ranks": self.attention_ranks,
            "attention_rails": self.attention_rails,
            "execution_profile_hint": self.execution_profile_hint,
        })

    @property
    def actual_step_indices(self) -> tuple[int, ...]:
        return tuple(index for index, value in enumerate(self.step_modes) if value)

    @property
    def forecast_step_indices(self) -> tuple[int, ...]:
        return tuple(index for index, value in enumerate(self.step_modes) if not value)

    def _row(self, step: int) -> str:
        ranks = self.attention_ranks[step * 50:(step + 1) * 50]
        rails = self.attention_rails[step * 50:(step + 1) * 50]
        result: list[str] = []
        symbols = "STQHD"
        for rank, rail in zip(ranks, rails):
            if rank == PREDICTED_TAIL:
                result.append(".")
                continue
            symbol = symbols[rank]
            if rank < 4 and rail == RAIL_FRONTIER:
                symbol = symbol.lower()
            result.append(symbol)
        return "".join(result)

    def to_dict(self, *, include_flat_vectors: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "digest": self.digest,
            "total_steps": self.total_steps,
            "actual_step_indices": list(self.actual_step_indices),
            "forecast_step_indices": list(self.forecast_step_indices),
            "maximum_forecast_run": _longest_forecast_run(self.step_modes),
            "execution_profile_hint": self.execution_profile_hint,
            "row_encoding": {
                "legend": {
                    ".": "predicted DiT tail",
                    "s/t/q/h": "Round188 6.25/10/25/50 percent",
                    "S/T/Q/H": "Round229 6.25/10/25/50 percent",
                    "D": "Dense",
                },
                "rows": [self._row(step) for step in range(self.total_steps)],
            },
        }
        if include_flat_vectors:
            result.update({
                "step_modes": list(self.step_modes),
                "attention_ranks": list(self.attention_ranks),
                "attention_rails": list(self.attention_rails),
            })
        return result

    @classmethod
    def from_selection(
        cls,
        selection: V24DeploymentSelection,
        *,
        total_steps: int,
    ) -> "V24StrategyVector":
        actual = frozenset(selection.actual_step_indices)
        step_modes = tuple(int(step in actual) for step in range(total_steps))
        physical = {
            (step, layer): action
            for step, layer, action in selection.attention_action_schedule
        }
        ranks: list[int] = []
        rails: list[int] = []
        for step in range(total_steps):
            depth = 50 if step in actual else 3
            for layer in range(50):
                if layer >= depth:
                    ranks.append(PREDICTED_TAIL)
                    rails.append(RAIL_PREDICTED)
                    continue
                action = physical.get((step, layer), "dense")
                canonical = action.rsplit(":", 1)[-1]
                try:
                    rank = _CANONICAL_RANK[canonical]
                except KeyError as error:
                    raise V19PlanningError(
                        f"V24 strategy contains unknown action: {action}"
                    ) from error
                ranks.append(rank)
                rails.append(_rail_code(action, canonical))
        return cls(
            total_steps=total_steps,
            step_modes=step_modes,
            attention_ranks=tuple(ranks),
            attention_rails=tuple(rails),
            execution_profile_hint=selection.summary.get("execution_profile_hint"),
        )


def v24_strategy_features(vector: V24StrategyVector) -> dict[str, float]:
    """Project one physical vector onto interpretable learning coordinates."""

    steps = vector.total_steps
    actual_count = sum(vector.step_modes)
    forecast_count = steps - actual_count
    phase = tuple(index / max(1, steps - 1) for index in range(steps))
    opening = tuple(math.exp(-value / 0.16) for value in phase)
    terminal = tuple(math.exp(-(1.0 - value) / 0.13) for value in phase)
    forecast = tuple(1.0 - value for value in vector.step_modes)

    run_square = 0.0
    run = 0
    for value in vector.step_modes:
        if value:
            run_square += float(run * run)
            run = 0
        else:
            run += 1
    run_square += float(run * run)

    risk_rows: list[tuple[int, int, float]] = []
    rail_forecast = sparse = 0
    attention_cost = 0.0
    for step in range(steps):
        if not vector.step_modes[step]:
            continue
        for layer in range(50):
            index = step * 50 + layer
            rank = vector.attention_ranks[index]
            risk_rows.append((step, layer, _ACTION_RISK[rank]))
            canonical = _RANK_TO_CANONICAL[rank]
            attention_cost += _ATTENTION_COST_RATIO[canonical]
            if rank < 4:
                sparse += 1
                if vector.attention_rails[index] == RAIL_FORECAST_FRONTIER:
                    rail_forecast += 1

    denominator = max(1, len(risk_rows))
    def mean_where(predicate) -> float:
        values = [risk for step, layer, risk in risk_rows if predicate(step, layer)]
        return sum(values) / max(1, len(values))

    attention_mean = sum(row[2] for row in risk_rows) / denominator
    forecast_interaction = 0.0
    for step in range(1, steps):
        if vector.step_modes[step] and not vector.step_modes[step - 1]:
            forecast_interaction += mean_where(
                lambda row_step, _layer, target=step: row_step == target
            )
    compute_units = (
        actual_count * _NON_ATTENTION_COMPUTE
        + _ATTENTION_COMPUTE * attention_cost / 50.0
        + forecast_count * _FORECAST_COMPUTE
    )
    features = {
        "compute_ratio": compute_units / steps,
        "forecast_fraction": forecast_count / steps,
        "forecast_run_quadratic": run_square / max(1.0, float(steps * steps)),
        "forecast_opening_exposure": sum(
            value * weight for value, weight in zip(forecast, opening)
        ) / max(1.0, sum(opening)),
        "forecast_terminal_exposure": sum(
            value * weight for value, weight in zip(forecast, terminal)
        ) / max(1.0, sum(terminal)),
        "attention_approximation_mean": attention_mean,
        "attention_opening_exposure": mean_where(
            lambda step, _layer: phase[step] <= 0.25
        ),
        "attention_terminal_exposure": mean_where(
            lambda step, _layer: phase[step] >= 0.75
        ),
        "attention_layers_00_29": mean_where(lambda _step, layer: layer < 30),
        "attention_causal_30_43": mean_where(
            lambda _step, layer: 30 <= layer <= 43
        ),
        "attention_bridge_44_45": mean_where(
            lambda _step, layer: 44 <= layer <= 45
        ),
        "attention_tail_46_49": mean_where(lambda _step, layer: layer >= 46),
        "forecast_attention_interaction": forecast_interaction / max(1, steps),
        "forecast_aware_sparse_fraction": rail_forecast / max(1, sparse),
    }
    if tuple(features) != V24_STRATEGY_FEATURE_NAMES:
        raise AssertionError("V24 strategy feature order changed")
    return features


__all__ = [
    "PREDICTED_TAIL",
    "V24_STRATEGY_FEATURE_NAMES",
    "V24_STRATEGY_FEATURE_SCHEMA",
    "V24_STRATEGY_VECTOR_SCHEMA",
    "V24StrategyVector",
    "v24_strategy_features",
]
