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
from typing import Iterable
from .v19_calibration import (
    V19CalibratedCellEvidence,
    V19CalibrationCatalog,
    V19CalibrationError,
    V19CalibrationWorkload,
    V19RuntimeFingerprint,
    conservative_quantile,
)
from .v19_candidates import V19CandidateBlueprint
from .v19_forecast_calibration import V19ForecastCalibrationCatalog
from .v19_planner import V19ActionUse, V19ForecastUse, V19PlanningError
from .v19_runtime_bridge import (
    DENSE_ACTION_ID,
    blueprint_from_runtime_schedule,
    runtime_schedule_from_blueprint,
)


V19_OPTIMIZER_SCHEMA = "h3_v19_budgeted_cell_optimizer_v3"
V19_COUPLED_PROPOSAL_SCHEMA = "h3_v19_coupled_proposal_v1"
V19_STRUCTURAL_IMPORTANCE_PROFILE = "h3_v19_structural_causal_importance_v1"
V19_AV_CLARITY_IMPORTANCE_PROFILE = "h3_v19_av_clarity_floor_v1"


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
class V19CellImportanceProfile:
    """Non-Human structural prior used only to rank proposal allocations.

    The profile cannot turn Dense-relative tensor disagreement into Human
    quality. It only prevents the proposal DP from treating an opening or
    causal-interaction cell as interchangeable with every other cell when the
    measured cost and local numerical proxy happen to be similar.
    """

    profile_id: str
    step_weights: tuple[float, ...]
    layer_weights: tuple[float, ...]
    component_weights: tuple[float, ...] = (1.0,) * 6
    minimum_layer_keep_ratios: tuple[float, ...] = (0.0,) * 50

    def __post_init__(self) -> None:
        if not self.profile_id or not self.step_weights:
            raise V19PlanningError("V19 cell-importance profile cannot be empty")
        if (
            len(self.layer_weights) != 50
            or len(self.component_weights) != 6
            or len(self.minimum_layer_keep_ratios) != 50
        ):
            raise V19PlanningError(
                "V19 importance profile requires 50 layer weights, 50 keep floors "
                "and six proxy components"
            )
        for name in ("step_weights", "layer_weights", "component_weights"):
            values = getattr(self, name)
            if any(not math.isfinite(value) or value <= 0.0 for value in values):
                raise V19PlanningError(
                    f"V19 importance {name} must be finite and positive"
                )
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.minimum_layer_keep_ratios
        ):
            raise V19PlanningError(
                "V19 layer keep floors must be finite and lie inside [0, 1]"
            )

    def score(self, proxy: V19NumericalProxy, *, step: int, layer: int) -> float:
        try:
            step_weight = self.step_weights[step]
            layer_weight = self.layer_weights[layer]
        except IndexError as error:
            raise V19PlanningError(
                "V19 importance profile does not cover the requested cell"
            ) from error
        # Proxy dimensions remain non-compensating: weighting changes their
        # declared importance but a bad component still cannot be averaged
        # away by the other five.
        component = max(
            value * weight
            for value, weight in zip(proxy.as_tuple(), self.component_weights)
        )
        return component * step_weight * layer_weight


def v19_structural_causal_importance_profile(
    total_steps: int,
) -> V19CellImportanceProfile:
    """Return the frozen V19 proposal prior derived from known H3 risk regions.

    These multipliers preserve the already documented Round142/151/215/218
    observations: step zero and the terminal three positions carry more
    trajectory debt, while layers 30--43 and 45 carry the strongest causal
    interaction sensitivity. They are proposal weights, not a Human-risk
    calibration and never make a candidate release eligible.
    """

    if total_steps <= 0:
        raise V19PlanningError("V19 importance profile requires positive total steps")
    step_weights = tuple(
        1.60 if step == 0 else (1.40 if step >= total_steps - 3 else 1.0)
        for step in range(total_steps)
    )
    layer_weights = (
        (1.00,) * 15
        + (1.10,) * 15
        + (2.20,) * 10
        + (2.50,) * 4
        + (1.20,)
        + (2.50,)
        + (1.30,) * 4
    )
    return V19CellImportanceProfile(
        profile_id=V19_STRUCTURAL_IMPORTANCE_PROFILE,
        step_weights=step_weights,
        layer_weights=layer_weights,
    )


def v19_av_clarity_importance_profile(
    total_steps: int,
) -> V19CellImportanceProfile:
    """Protect broad visual fidelity after the first Human V19 Pareto review.

    The fixed 10/10 r2.0 candidate was preferred to V18, while schedules with
    widespread 0.0625/0.1 actions were judged visibly soft and a trajectory
    with one fewer full DiT evaluation degraded speech.  This proposal-only
    profile therefore keeps the accepted 10/10 trajectory, prevents the DP
    from buying quality with very low-density ordinary visual layers, retains
    exact attention in the strongest causal island and weights the terminal
    solver positions more strongly.  It is still only a search prior: the
    resulting video requires fresh Human review and is not release evidence.
    """

    if total_steps <= 0:
        raise V19PlanningError("V19 importance profile requires positive total steps")
    step_weights = tuple(
        1.60
        if step == 0
        else 1.20
        if step <= 4
        else 2.00
        if step >= total_steps - 2
        else 1.0
        for step in range(total_steps)
    )
    layer_weights = (
        (1.60,) * 15
        + (1.70,) * 15
        + (2.00,) * 10
        + (2.60,) * 4
        + (1.80,)
        + (2.60,)
        + (2.00,) * 4
    )
    minimum_layer_keep_ratios = (
        (0.25,) * 30
        + (0.50,) * 10
        + (1.00,) * 4
        + (0.25,)
        + (1.00,)
        + (0.25,) * 4
    )
    return V19CellImportanceProfile(
        profile_id=V19_AV_CLARITY_IMPORTANCE_PROFILE,
        step_weights=step_weights,
        layer_weights=layer_weights,
        minimum_layer_keep_ratios=minimum_layer_keep_ratios,
    )


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
    cell_importance: V19CellImportanceProfile | None = None

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
        if (
            self.cell_importance is not None
            and len(self.cell_importance.step_weights) != self.workload.steps
        ):
            raise V19PlanningError(
                "V19 importance profile step count does not match the workload"
            )


@dataclass(frozen=True, slots=True)
class V19BudgetedProposal:
    blueprint: V19CandidateBlueprint
    requested_attention_p90_ms: float
    calibrated_attention_p50_ms: float
    calibrated_attention_p90_ms: float
    calibrated_attention_peak_vram_gib: float
    conservative_quantized_p90_ms: float
    proxy_sum: float
    proxy_max: float
    proxy_component_sums: tuple[float, ...]
    proxy_component_maxima: tuple[float, ...]
    importance_weighted_proxy_sum: float
    importance_weighted_proxy_max: float
    importance_profile_id: str | None
    protected_cell_count: int
    action_cell_counts: tuple[tuple[str, str, int], ...]
    attention_evidence_ids: tuple[str, ...]
    numerical_proxy_is_human_risk: bool = False
    schema_version: str = V19_OPTIMIZER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V19_OPTIMIZER_SCHEMA:
            raise V19PlanningError("unsupported V19 optimizer proposal schema")
        cost_values = (
            self.requested_attention_p90_ms,
            self.calibrated_attention_p50_ms,
            self.calibrated_attention_p90_ms,
            self.calibrated_attention_peak_vram_gib,
            self.conservative_quantized_p90_ms,
        )
        if any(
            not math.isfinite(value) or value < 0.0 for value in cost_values
        ) or self.calibrated_attention_p90_ms < self.calibrated_attention_p50_ms:
            raise V19PlanningError(
                "V19 proposal costs and peak VRAM must be finite and non-negative"
            )
        if tuple(sorted(set(self.attention_evidence_ids))) != (
            self.attention_evidence_ids
        ):
            raise V19PlanningError(
                "V19 proposal evidence ids must be sorted and unique"
            )
        if (
            len(self.proxy_component_sums) != 6
            or len(self.proxy_component_maxima) != 6
            or any(
                not math.isfinite(value) or value < 0.0
                for value in (
                    *self.proxy_component_sums,
                    *self.proxy_component_maxima,
                )
            )
        ):
            raise V19PlanningError(
                "V19 proposal requires six finite non-negative proxy sums/maxima"
            )

    @property
    def numerical_pareto_objective(self) -> tuple[float, ...]:
        """Fixed-trajectory proposal vector; never interpreted as Human risk.

        Forecast cost is constant only while proposals share one exact
        trajectory.  Use :func:`couple_v19_proposal` before comparing proposals
        from different Actual/Forecast schedules.
        """

        return (
            self.calibrated_attention_p90_ms,
            self.calibrated_attention_peak_vram_gib,
            *self.proxy_component_sums,
            *self.proxy_component_maxima,
            *self.blueprint.terminal_debt.as_pareto_tuple(),
            *self.blueprint.maximum_debt.as_pareto_tuple(),
        )


@dataclass(frozen=True, slots=True)
class V19CoupledProposal:
    """One evidence-priced multi-technique proposal.

    Attention is optimized inside a fixed trajectory.  This wrapper then adds
    the exact cost of every registered Forecast composite so proposals from
    different trajectories can be compared without treating skipped DiT work
    as free.  The numerical proxy remains proposal-only and must not be used as
    a substitute for complete-request Human review.
    """

    attention: V19BudgetedProposal
    workload_digest: str
    runtime_digest: str
    forecast_p50_ms: float
    forecast_p90_ms: float
    physical_p50_ms: float
    physical_p90_ms: float
    peak_vram_gib: float
    evidence_ids: tuple[str, ...]
    schema_version: str = V19_COUPLED_PROPOSAL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V19_COUPLED_PROPOSAL_SCHEMA:
            raise V19PlanningError("unsupported V19 coupled proposal schema")
        if len(self.workload_digest) != 64 or len(self.runtime_digest) != 64:
            raise V19PlanningError(
                "V19 coupled proposal requires workload/runtime SHA256 identities"
            )
        values = (
            self.forecast_p50_ms,
            self.forecast_p90_ms,
            self.physical_p50_ms,
            self.physical_p90_ms,
            self.peak_vram_gib,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise V19PlanningError(
                "V19 coupled proposal costs and VRAM must be finite and non-negative"
            )
        if (
            self.forecast_p90_ms < self.forecast_p50_ms
            or self.physical_p90_ms < self.physical_p50_ms
        ):
            raise V19PlanningError("V19 coupled proposal p90 cannot be below p50")
        if tuple(sorted(set(self.evidence_ids))) != self.evidence_ids:
            raise V19PlanningError(
                "V19 coupled proposal evidence ids must be sorted and unique"
            )

    @property
    def blueprint(self) -> V19CandidateBlueprint:
        return self.attention.blueprint

    @property
    def numerical_pareto_objective(self) -> tuple[float, ...]:
        return (
            self.physical_p90_ms,
            self.peak_vram_gib,
            *self.attention.proxy_component_sums,
            *self.attention.proxy_component_maxima,
            *self.blueprint.terminal_debt.as_pareto_tuple(),
            *self.blueprint.maximum_debt.as_pareto_tuple(),
        )


def couple_v19_proposal(
    proposal: V19BudgetedProposal,
    *,
    workload: V19CalibrationWorkload,
    runtime: V19RuntimeFingerprint,
    forecast_catalog: V19ForecastCalibrationCatalog | None,
) -> V19CoupledProposal:
    """Add exact Forecast-composite cost to one Attention proposal."""

    actual_uses = tuple(
        use for use in proposal.blueprint.action_uses
        if isinstance(use, V19ActionUse)
    )
    forecast_uses = tuple(
        use for use in proposal.blueprint.action_uses
        if isinstance(use, V19ForecastUse)
    )
    actual_steps = {
        step for use in actual_uses for step in use.step_indices
    }
    if actual_steps != set(workload.actual_step_indices):
        raise V19PlanningError(
            "V19 proposal trajectory does not match its calibration workload"
        )
    expected_forecast = set(range(workload.steps)) - actual_steps
    observed_forecast = [
        step for use in forecast_uses for step in use.step_indices
    ]
    if (
        len(observed_forecast) != len(set(observed_forecast))
        or set(observed_forecast) != expected_forecast
    ):
        raise V19PlanningError(
            "V19 proposal does not exactly cover its Forecast trajectory"
        )
    if forecast_uses and forecast_catalog is None:
        raise V19PlanningError(
            "cross-trajectory V19 proposal comparison requires Forecast calibration"
        )

    forecast_p50_ms = 0.0
    forecast_p90_ms = 0.0
    peak_vram_gib = proposal.calibrated_attention_peak_vram_gib
    evidence_ids = set(proposal.attention_evidence_ids)
    if forecast_catalog is not None:
        for use in forecast_uses:
            cost = forecast_catalog.estimate(
                action_id=use.action_id,
                key=use.composite_key,
                workload=workload,
                runtime=runtime,
            )
            forecast_p50_ms += cost.p50_ms
            forecast_p90_ms += cost.p90_ms
            peak_vram_gib = max(peak_vram_gib, cost.peak_vram_gib)
            evidence_ids.add(cost.evidence_id)
    return V19CoupledProposal(
        attention=proposal,
        workload_digest=workload.digest,
        runtime_digest=runtime.digest,
        forecast_p50_ms=forecast_p50_ms,
        forecast_p90_ms=forecast_p90_ms,
        physical_p50_ms=(
            proposal.calibrated_attention_p50_ms + forecast_p50_ms
        ),
        physical_p90_ms=(
            proposal.calibrated_attention_p90_ms + forecast_p90_ms
        ),
        peak_vram_gib=peak_vram_gib,
        evidence_ids=tuple(sorted(evidence_ids)),
    )


def _coupled_numerically_dominates(
    left: V19CoupledProposal,
    right: V19CoupledProposal,
) -> bool:
    left_values = left.numerical_pareto_objective
    right_values = right.numerical_pareto_objective
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def v19_coupled_numerical_frontier(
    proposals: Iterable[V19CoupledProposal],
) -> tuple[V19CoupledProposal, ...]:
    """Cross-trajectory non-dominated set over physical cost/VRAM/proxies/debt."""

    rows = tuple(proposals)
    frontier = tuple(
        candidate
        for index, candidate in enumerate(rows)
        if not any(
            other_index != index
            and _coupled_numerically_dominates(other, candidate)
            for other_index, other in enumerate(rows)
        )
    )
    return tuple(sorted(
        frontier,
        key=lambda row: (
            row.numerical_pareto_objective,
            row.blueprint.candidate_id,
        ),
    ))


def _numerically_dominates(
    left: V19BudgetedProposal,
    right: V19BudgetedProposal,
) -> bool:
    left_values = left.numerical_pareto_objective
    right_values = right.numerical_pareto_objective
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def v19_numerical_proposal_frontier(
    proposals: Iterable[V19BudgetedProposal],
) -> tuple[V19BudgetedProposal, ...]:
    """Return numerical non-dominated proposals without claiming Human quality."""

    rows = tuple(proposals)
    frontier = tuple(
        candidate
        for index, candidate in enumerate(rows)
        if not any(
            other_index != index and _numerically_dominates(other, candidate)
            for other_index, other in enumerate(rows)
        )
    )
    return tuple(sorted(
        frontier,
        key=lambda row: (
            row.numerical_pareto_objective,
            row.blueprint.candidate_id,
        ),
    ))


@dataclass(frozen=True, slots=True)
class _Option:
    action: V19CellAction
    evidence: V19CalibratedCellEvidence
    proxy: V19NumericalProxy
    importance_weighted_proxy: float
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


def _canonical_keep_ratio(canonical_action: str) -> float:
    if canonical_action == "dense":
        return 1.0
    prefix = "sparse_topk_"
    if not canonical_action.startswith(prefix):
        raise V19PlanningError(
            f"V19 optimizer cannot read Attention keep ratio: {canonical_action}"
        )
    try:
        value = float(canonical_action[len(prefix):])
    except ValueError as error:
        raise V19PlanningError(
            f"V19 optimizer cannot read Attention keep ratio: {canonical_action}"
        ) from error
    if not 0.0 < value <= 1.0:
        raise V19PlanningError(
            f"V19 optimizer Attention keep ratio is invalid: {canonical_action}"
        )
    return value


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
            if request.cell_importance is not None:
                minimum_keep = request.cell_importance.minimum_layer_keep_ratios[
                    layer
                ]
                if _canonical_keep_ratio(action.canonical_action) < minimum_keep:
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
                importance_weighted_proxy=(
                    proxy.worst_component
                    if request.cell_importance is None
                    else request.cell_importance.score(
                        proxy, step=step, layer=layer
                    )
                ),
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
                    updated_score = score + option.importance_weighted_proxy
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
        proxy_sum = 0.0
        proxy_max = 0.0
        proxy_component_sums = [0.0] * 6
        proxy_component_maxima = [0.0] * 6
        importance_weighted_proxy_max = 0.0
        p50_ms = 0.0
        p90_ms = 0.0
        peak_vram_gib = 0.0
        evidence_ids: set[str] = set()
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
            proxy_sum += option.proxy.worst_component
            proxy_max = max(proxy_max, option.proxy.worst_component)
            for index, value in enumerate(option.proxy.as_tuple()):
                proxy_component_sums[index] += value
                proxy_component_maxima[index] = max(
                    proxy_component_maxima[index], value
                )
            importance_weighted_proxy_max = max(
                importance_weighted_proxy_max,
                option.importance_weighted_proxy,
            )
            p50_ms += option.evidence.p50_ms
            p90_ms += option.evidence.p90_ms
            peak_vram_gib = max(
                peak_vram_gib, option.evidence.peak_vram_gib
            )
            evidence_ids.add(option.evidence.evidence_id)

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
            calibrated_attention_peak_vram_gib=peak_vram_gib,
            conservative_quantized_p90_ms=(
                selected_units * request.cost_quantum_ms
            ),
            proxy_sum=proxy_sum,
            proxy_max=proxy_max,
            proxy_component_sums=tuple(proxy_component_sums),
            proxy_component_maxima=tuple(proxy_component_maxima),
            importance_weighted_proxy_sum=scores[selected_units],
            importance_weighted_proxy_max=importance_weighted_proxy_max,
            importance_profile_id=(
                None
                if request.cell_importance is None
                else request.cell_importance.profile_id
            ),
            protected_cell_count=protected,
            action_cell_counts=tuple(
                (action_id, canonical, count)
                for (action_id, canonical), count in sorted(counts.items())
            ),
            attention_evidence_ids=tuple(sorted(evidence_ids)),
        )


__all__ = [
    "V19_AV_CLARITY_IMPORTANCE_PROFILE",
    "V19_COUPLED_PROPOSAL_SCHEMA",
    "V19_OPTIMIZER_SCHEMA",
    "V19BudgetedCellOptimizer",
    "V19BudgetedProposal",
    "V19BudgetedProposalRequest",
    "V19CellImportanceProfile",
    "V19CellAction",
    "V19NumericalProxy",
    "V19CoupledProposal",
    "V19_STRUCTURAL_IMPORTANCE_PROFILE",
    "couple_v19_proposal",
    "v19_av_clarity_importance_profile",
    "v19_coupled_numerical_frontier",
    "v19_numerical_proposal_frontier",
    "v19_structural_causal_importance_profile",
]
