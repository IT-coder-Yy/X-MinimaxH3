"""Evidence-bound proposal optimizer for V19 Attention schedules.

This module deliberately separates *proposal* from *release certification*.
It may use Dense-relative numerical observations to allocate a fixed Attention
budget, but it never converts those observations into Human quality.  A newly
proposed schedule must still be executed, measured end to end and reviewed by
Human before :mod:`v19_risk_calibration` can make it planner eligible.

For a fixed actual/forecast step trajectory the optimization is a finite
multiple-choice knapsack: every actual ``(step, layer)`` cell chooses exactly
one registry action, the sum of conservatively quantised p90 costs stays under
the requested budget, and the sum of a non-compensating per-cell numerical
proxy is minimized.  Dense cells in the reviewed comparator can be frozen as
causal rails, so the optimizer cannot silently remove known protection merely
to win latency.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
from .v19_calibration import (
    V19CalibratedCellEvidence,
    V19CalibrationCatalog,
    V19CalibrationError,
    V19CalibrationWorkload,
    V19RuntimeFingerprint,
    conservative_quantile,
)
from .v19_candidates import V19CandidateBlueprint
from .v19_planner import V19ActionUse, V19PlanningError
from .v19_runtime_bridge import (
    DENSE_ACTION_ID,
    blueprint_from_runtime_schedule,
    runtime_schedule_from_blueprint,
)


V19_OPTIMIZER_SCHEMA = "h3_v19_budgeted_cell_optimizer_v1"


@dataclass(frozen=True, slots=True)
class V19CellAction:
    action_id: str
    canonical_action: str

    def __post_init__(self) -> None:
        if not self.action_id or not self.canonical_action:
            raise V19PlanningError("V19 cell action identity cannot be empty")


@dataclass(frozen=True, slots=True)
class V19NumericalProxy:
    """Worst-observed, non-compensating diagnostic for one physical cell."""

    one_minus_mean_cosine: float
    one_minus_min_cosine: float
    global_relative_rms: float
    mean_head_relative_rms: float
    max_head_relative_rms: float
    max_relative_l1: float

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value < 0.0 for value in self.as_tuple()):
            raise V19PlanningError("V19 numerical proxy must be finite and non-negative")

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.one_minus_mean_cosine,
            self.one_minus_min_cosine,
            self.global_relative_rms,
            self.mean_head_relative_rms,
            self.max_head_relative_rms,
            self.max_relative_l1,
        )

    @property
    def worst_component(self) -> float:
        """A bad component cannot be hidden by averaging it with good ones."""

        return max(self.as_tuple())


@dataclass(frozen=True, slots=True)
class V19BudgetedProposalRequest:
    candidate_id: str
    comparator: V19CandidateBlueprint
    workload: V19CalibrationWorkload
    runtime: V19RuntimeFingerprint
    maximum_attention_p90_ms: float
    actions: tuple[V19CellAction, ...]
    cost_quantum_ms: float = 10.0
    maximum_cell_proxy: float = math.inf
    protect_comparator_dense: bool = True

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.actions:
            raise V19PlanningError("V19 proposal request is empty")
        if (
            not math.isfinite(self.maximum_attention_p90_ms)
            or self.maximum_attention_p90_ms <= 0.0
            or not math.isfinite(self.cost_quantum_ms)
            or self.cost_quantum_ms <= 0.0
        ):
            raise V19PlanningError("V19 Attention budget and quantum must be positive")
        if self.maximum_cell_proxy <= 0.0:
            raise V19PlanningError("V19 maximum cell proxy must be positive")
        identities = {(row.action_id, row.canonical_action) for row in self.actions}
        if len(identities) != len(self.actions):
            raise V19PlanningError("V19 proposal actions must be unique")


@dataclass(frozen=True, slots=True)
class V19BudgetedProposal:
    blueprint: V19CandidateBlueprint
    requested_attention_p90_ms: float
    calibrated_attention_p50_ms: float
    calibrated_attention_p90_ms: float
    conservative_quantized_p90_ms: float
    proxy_sum: float
    proxy_max: float
    protected_cell_count: int
    action_cell_counts: tuple[tuple[str, str, int], ...]
    numerical_proxy_is_human_risk: bool = False
    schema_version: str = V19_OPTIMIZER_SCHEMA


@dataclass(frozen=True, slots=True)
class _Option:
    action: V19CellAction
    evidence: V19CalibratedCellEvidence
    proxy: V19NumericalProxy
    cost_units: int


def _proxy(evidence: V19CalibratedCellEvidence) -> V19NumericalProxy:
    samples = evidence.numerical_error_samples
    if not samples:
        raise V19PlanningError("V19 proposal action lacks Dense-relative diagnostics")
    # Numerical diagnostics are deterministic for a fixed cell but can still
    # have repeated captures.  Nearest-rank p90 prevents an optimistic mean.
    return V19NumericalProxy(
        one_minus_mean_cosine=conservative_quantile(
            (max(0.0, 1.0 - row.mean_cosine) for row in samples), 0.90
        ),
        one_minus_min_cosine=conservative_quantile(
            (max(0.0, 1.0 - row.min_cosine) for row in samples), 0.90
        ),
        global_relative_rms=conservative_quantile(
            (row.global_relative_rms for row in samples), 0.90
        ),
        mean_head_relative_rms=conservative_quantile(
            (row.mean_head_relative_rms for row in samples), 0.90
        ),
        max_head_relative_rms=conservative_quantile(
            (row.max_head_relative_rms for row in samples), 0.90
        ),
        max_relative_l1=conservative_quantile(
            (row.max_relative_l1 for row in samples), 0.90
        ),
    )


def _actual_cells(
    blueprint: V19CandidateBlueprint,
) -> dict[tuple[int, int], tuple[str, str]]:
    result: dict[tuple[int, int], tuple[str, str]] = {}
    for use in blueprint.action_uses:
        if not isinstance(use, V19ActionUse):
            continue
        for step in use.step_indices:
            for layer in range(use.layer_start, use.layer_stop):
                cell = (step, layer)
                if cell in result:
                    raise V19PlanningError("comparator contains overlapping Attention cells")
                result[cell] = (use.action_id, use.canonical_action)
    return result


class V19BudgetedCellOptimizer:
    """Construct a physical proposal without manufacturing Human quality."""

    def __init__(self, catalog: V19CalibrationCatalog) -> None:
        self.catalog = catalog

    def _options(
        self,
        request: V19BudgetedProposalRequest,
        *,
        step: int,
        layer: int,
        protected_identity: tuple[str, str] | None,
    ) -> tuple[_Option, ...]:
        result: list[_Option] = []
        for action in request.actions:
            if protected_identity is not None and (
                action.action_id,
                action.canonical_action,
            ) != protected_identity:
                continue
            try:
                evidence = self.catalog.lookup_cell(
                    action_id=action.action_id,
                    canonical_action=action.canonical_action,
                    step_index=step,
                    layer_index=layer,
                    workload=request.workload,
                    runtime=request.runtime,
                )
            except V19CalibrationError:
                continue
            proxy = _proxy(evidence)
            if proxy.worst_component > request.maximum_cell_proxy:
                continue
            result.append(_Option(
                action=action,
                evidence=evidence,
                proxy=proxy,
                # Ceil every cell independently: the DP cannot spend latency
                # hidden by favourable rounding.
                cost_units=max(1, math.ceil(evidence.p90_ms / request.cost_quantum_ms)),
            ))
        if not result:
            raise V19PlanningError(
                f"no calibrated V19 action remains for cell {(step, layer)}"
            )
        return tuple(result)

    def optimize(self, request: V19BudgetedProposalRequest) -> V19BudgetedProposal:
        comparator_cells = _actual_cells(request.comparator)
        expected = {
            (step, layer)
            for step in request.workload.actual_step_indices
            for layer in range(50)
        }
        if set(comparator_cells) != expected:
            raise V19PlanningError("V19 comparator does not match the exact workload")
        cells = tuple(sorted(expected))
        option_rows: list[tuple[_Option, ...]] = []
        protected = 0
        for step, layer in cells:
            comparator_identity = comparator_cells[(step, layer)]
            protected_identity = None
            if (
                request.protect_comparator_dense
                and comparator_identity == (DENSE_ACTION_ID, "dense")
            ):
                protected_identity = comparator_identity
                protected += 1
            option_rows.append(self._options(
                request,
                step=step,
                layer=layer,
                protected_identity=protected_identity,
            ))

        budget_units = math.floor(
            request.maximum_attention_p90_ms / request.cost_quantum_ms
        )
        minimum_units = sum(min(row.cost_units for row in options) for options in option_rows)
        if minimum_units > budget_units:
            raise V19PlanningError(
                "V19 Attention budget is below the calibrated physical minimum"
            )

        infinity = float("inf")
        scores = [infinity] * (budget_units + 1)
        scores[0] = 0.0
        back_costs: list[array] = []
        back_choices: list[array] = []
        for options in option_rows:
            next_scores = [infinity] * (budget_units + 1)
            previous = array("i", [-1]) * (budget_units + 1)
            choices = array("b", [-1]) * (budget_units + 1)
            for spent, score in enumerate(scores):
                if not math.isfinite(score):
                    continue
                for option_index, option in enumerate(options):
                    updated = spent + option.cost_units
                    if updated > budget_units:
                        continue
                    updated_score = score + option.proxy.worst_component
                    incumbent = next_scores[updated]
                    if updated_score < incumbent - 1.0e-15:
                        next_scores[updated] = updated_score
                        previous[updated] = spent
                        choices[updated] = option_index
            if not any(math.isfinite(value) for value in next_scores):
                raise V19PlanningError("V19 optimizer exhausted all budgeted states")
            scores = next_scores
            back_costs.append(previous)
            back_choices.append(choices)

        feasible = [index for index, value in enumerate(scores) if math.isfinite(value)]
        # Quality proxy is primary at a fixed maximum budget.  If two states
        # have the same proxy, spend less rather than hiding wasted compute.
        selected_units = min(feasible, key=lambda index: (scores[index], index))
        selected: list[_Option] = []
        cursor = selected_units
        for row_index in range(len(cells) - 1, -1, -1):
            option_index = back_choices[row_index][cursor]
            previous = back_costs[row_index][cursor]
            if option_index < 0 or previous < 0:
                raise V19PlanningError("V19 optimizer backtrace is incomplete")
            selected.append(option_rows[row_index][option_index])
            cursor = previous
        if cursor != 0:
            raise V19PlanningError("V19 optimizer backtrace did not reach the origin")
        selected.reverse()

        runtime_rows = list(runtime_schedule_from_blueprint(request.comparator))
        actual_set = set(expected)
        runtime_rows = [row for row in runtime_rows if (row[0], row[1]) not in actual_set]
        runtime_prefix = {
            DENSE_ACTION_ID: "dense",
            "h3.attention.mtcr_head_rail.round229.v1": "forecastfrontier",
        }
        counts: dict[tuple[str, str], int] = {}
        proxy_max = 0.0
        p50_ms = 0.0
        p90_ms = 0.0
        for (step, layer), option in zip(cells, selected):
            try:
                prefix = runtime_prefix[option.action.action_id]
            except KeyError as error:
                raise V19PlanningError(
                    f"optimizer action has no V19 runtime bridge: {option.action.action_id}"
                ) from error
            runtime_action = (
                "dense"
                if option.action.canonical_action == "dense"
                else f"{prefix}:{option.action.canonical_action}"
            )
            runtime_rows.append((step, layer, runtime_action))
            key = (option.action.action_id, option.action.canonical_action)
            counts[key] = counts.get(key, 0) + 1
            proxy_max = max(proxy_max, option.proxy.worst_component)
            p50_ms += option.evidence.p50_ms
            p90_ms += option.evidence.p90_ms

        blueprint = blueprint_from_runtime_schedule(
            candidate_id=request.candidate_id,
            total_steps=request.workload.steps,
            actual_step_indices=request.workload.actual_step_indices,
            attention_action_schedule=tuple(sorted(runtime_rows)),
            source="v19_budgeted_cell_optimizer",
        )
        return V19BudgetedProposal(
            blueprint=blueprint,
            requested_attention_p90_ms=request.maximum_attention_p90_ms,
            calibrated_attention_p50_ms=p50_ms,
            calibrated_attention_p90_ms=p90_ms,
            conservative_quantized_p90_ms=(
                selected_units * request.cost_quantum_ms
            ),
            proxy_sum=scores[selected_units],
            proxy_max=proxy_max,
            protected_cell_count=protected,
            action_cell_counts=tuple(
                (action_id, canonical, count)
                for (action_id, canonical), count in sorted(counts.items())
            ),
        )


__all__ = [
    "V19_OPTIMIZER_SCHEMA",
    "V19BudgetedCellOptimizer",
    "V19BudgetedProposal",
    "V19BudgetedProposalRequest",
    "V19CellAction",
    "V19NumericalProxy",
]
