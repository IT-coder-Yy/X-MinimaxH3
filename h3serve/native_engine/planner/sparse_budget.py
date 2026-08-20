"""Budgeted online sparse-Attention scheduling for the H3 DiT.

This module is deliberately independent from CUDA and model execution.  It
solves the control problem using *measured milliseconds* and calibrated upper
bounds on Human-visible risk.  Kernel-specific routing remains in
``model.kernels``; this layer decides how much of each verified action may be
used at each actual denoising step and H3 layer band.

The first implementation uses an exact multiple-choice knapsack dynamic
program for the additive risk surrogate.  It is small enough for the H3
domain (normally 12 actual steps x 5 layer bands x 3 actions), deterministic,
and auditable.  It is not a learned quality model: calibration data must be
supplied by the caller, and out-of-distribution requests must fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import math
from typing import Callable, Iterable, Literal, Mapping


OverflowPolicy = Literal["strict_budget", "quality_first"]


class SparseBudgetError(ValueError):
    """Base error for an invalid or infeasible sparse scheduling problem."""


class SparseBudgetInfeasible(SparseBudgetError):
    """No schedule satisfies the requested budget and quality constraints."""


@dataclass(frozen=True, slots=True)
class HumanRiskVector:
    """Non-negative incremental Human-visible risk debt.

    These components are telemetry and calibration surfaces, not independent
    objective metrics.  ``reject_risk_ucb`` on an action is the scalar upper
    confidence bound used by the optimizer after the components have been
    calibrated against blind Human review.
    """

    motion: float = 0.0
    clarity: float = 0.0
    identity: float = 0.0
    audio: float = 0.0

    def __post_init__(self) -> None:
        values = (self.motion, self.clarity, self.identity, self.audio)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise SparseBudgetError("Human risk components must be finite and non-negative")

    def __add__(self, other: "HumanRiskVector") -> "HumanRiskVector":
        return HumanRiskVector(
            motion=self.motion + other.motion,
            clarity=self.clarity + other.clarity,
            identity=self.identity + other.identity,
            audio=self.audio + other.audio,
        )

    @property
    def worst_component(self) -> float:
        return max(self.motion, self.clarity, self.identity, self.audio)


@dataclass(frozen=True, slots=True)
class SparseActionEstimate:
    """One measured execution action available for a decision cell.

    ``fidelity_rank`` is ordered: a larger value is never less conservative.
    A typical action set is 0=aggressive sparse, 1=protected sparse,
    2=exact Dense.  Cost is wall-clock milliseconds measured on the target
    RTX 4090 shape; a selected-block fraction is not a valid replacement.
    """

    name: str
    measured_cost_ms: float
    reject_risk_ucb: float
    components: HumanRiskVector = HumanRiskVector()
    fidelity_rank: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise SparseBudgetError("sparse action name cannot be empty")
        if not math.isfinite(self.measured_cost_ms) or self.measured_cost_ms < 0.0:
            raise SparseBudgetError("sparse action cost must be finite and non-negative")
        if not math.isfinite(self.reject_risk_ucb) or self.reject_risk_ucb < 0.0:
            raise SparseBudgetError("sparse action risk must be finite and non-negative")
        if self.fidelity_rank < 0:
            raise SparseBudgetError("sparse action fidelity rank cannot be negative")


class H3LayerBand(str, Enum):
    EARLY = "layers_00_14"
    MIDDLE = "layers_15_29"
    CAUSAL = "layers_30_39"
    CAUSAL_DETAIL = "layers_40_43"
    BRIDGE = "layer_44"
    CAUSAL_TERMINAL = "layer_45"
    TAIL = "layers_46_49"


@dataclass(frozen=True, slots=True)
class H3AttentionCellKey:
    actual_step: int
    layer_band: H3LayerBand
    layer_start: int
    layer_stop: int
    phase: Literal["opening", "ordinary", "terminal"]

    @property
    def cell_id(self) -> str:
        return f"s{self.actual_step}:{self.layer_band.value}"


@dataclass(frozen=True, slots=True)
class SparseDecisionCell:
    key: H3AttentionCellKey
    actions: tuple[SparseActionEstimate, ...]
    minimum_fidelity_rank: int = 0

    def __post_init__(self) -> None:
        if not self.actions:
            raise SparseBudgetError(f"{self.key.cell_id} has no execution actions")
        names = tuple(action.name for action in self.actions)
        if len(set(names)) != len(names):
            raise SparseBudgetError(f"{self.key.cell_id} contains duplicate action names")
        if self.minimum_fidelity_rank < 0:
            raise SparseBudgetError("minimum fidelity rank cannot be negative")
        if not any(
            action.fidelity_rank >= self.minimum_fidelity_rank
            for action in self.actions
        ):
            raise SparseBudgetInfeasible(
                f"{self.key.cell_id} has no action satisfying its fidelity floor"
            )


@dataclass(frozen=True, slots=True)
class SparseScheduleChoice:
    key: H3AttentionCellKey
    action: SparseActionEstimate


@dataclass(frozen=True, slots=True)
class SparseOptimalityCertificate:
    """Machine-checkable certificate for the finite calibrated problem.

    The certificate deliberately makes a narrow claim: optimality holds for
    the declared cells, actions, hard fidelity floors, conservative cost
    quantisation and additive risk surrogate.  It is *not* a certificate of
    optimal Human-visible video quality.
    """

    schema_version: str
    solver: str
    objective: str
    formal_scope: str
    cost_quantum_ms: float
    budget_units: int
    cell_count: int
    candidate_count: int
    conservative_cost_units: int
    optimum_risk: float
    optimum_worst_component: float
    optimum_actual_cost_ms: float
    max_reject_risk_ucb: float | None
    choice_sha256: str
    frontier_sha256: str


@dataclass(frozen=True, slots=True)
class SparseSchedule:
    choices: tuple[SparseScheduleChoice, ...]
    budget_limit_ms: float
    estimated_cost_ms: float
    estimated_reject_risk_ucb: float
    estimated_components: HumanRiskVector
    budget_overrun_ms: float
    used_recovery_reserve: bool
    optimality_certificate: SparseOptimalityCertificate | None = None

    def action_for(self, cell_id: str) -> SparseActionEstimate:
        for choice in self.choices:
            if choice.key.cell_id == cell_id:
                return choice.action
        raise KeyError(cell_id)


@dataclass(frozen=True, slots=True)
class SparseBudgetLedger:
    """Request-local wall-clock budget and recovery reserve."""

    total_budget_ms: float
    recovery_reserve_ms: float = 0.0
    spent_ms: float = 0.0

    def __post_init__(self) -> None:
        values = (self.total_budget_ms, self.recovery_reserve_ms, self.spent_ms)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise SparseBudgetError("budget ledger values must be finite and non-negative")
        if self.recovery_reserve_ms > self.total_budget_ms:
            raise SparseBudgetError("recovery reserve cannot exceed the total budget")

    def available_ms(self, *, open_recovery_reserve: bool) -> float:
        held = 0.0 if open_recovery_reserve else self.recovery_reserve_ms
        return max(0.0, self.total_budget_ms - self.spent_ms - held)

    def after_spend(self, measured_ms: float) -> "SparseBudgetLedger":
        if not math.isfinite(measured_ms) or measured_ms < 0.0:
            raise SparseBudgetError("measured spend must be finite and non-negative")
        return replace(self, spent_ms=self.spent_ms + measured_ms)


@dataclass(frozen=True, slots=True)
class _DynamicState:
    risk: float
    actual_cost_ms: float
    components: HumanRiskVector
    choices: tuple[SparseScheduleChoice, ...]


@dataclass(frozen=True, slots=True)
class _DPState:
    """Compact exact-DP node with a backpointer instead of copied paths."""

    risk: float
    actual_cost_ms: float
    motion: float
    clarity: float
    identity: float
    audio: float
    previous: "_DPState | None"
    choice: SparseScheduleChoice | None

    @property
    def worst_component(self) -> float:
        return max(self.motion, self.clarity, self.identity, self.audio)


@dataclass(frozen=True, slots=True)
class _InsideBudgetResult:
    optimum: _DynamicState | None
    frontier: Mapping[int, _DPState]


@dataclass(frozen=True, slots=True)
class SparseCertificateVerification:
    valid: bool
    reasons: tuple[str, ...]


def _choice_digest(choices: tuple[SparseScheduleChoice, ...]) -> str:
    payload = [
        {
            "cell": choice.key.cell_id,
            "action": choice.action.name,
            "rank": choice.action.fidelity_rank,
        }
        for choice in choices
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _frontier_digest(frontier: Mapping[int, _DPState]) -> str:
    payload = [
        {
            "units": units,
            "risk": round(state.risk, 12),
            "worst": round(state.worst_component, 12),
            "cost_ms": round(state.actual_cost_ms, 9),
        }
        for units, state in sorted(frontier.items())
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _materialize_dp_state(state: _DPState) -> _DynamicState:
    choices: list[SparseScheduleChoice] = []
    current: _DPState | None = state
    while current is not None and current.choice is not None:
        choices.append(current.choice)
        current = current.previous
    choices.reverse()
    return _DynamicState(
        risk=state.risk,
        actual_cost_ms=state.actual_cost_ms,
        components=HumanRiskVector(
            motion=state.motion,
            clarity=state.clarity,
            identity=state.identity,
            audio=state.audio,
        ),
        choices=tuple(choices),
    )


class H3BudgetedRiskScheduler:
    """Solve and re-solve a fixed-budget H3 sparse-Attention trajectory.

    The solver is exact for the supplied additive ``reject_risk_ucb`` model.
    Costs are rounded *up* to ``cost_quantum_ms`` before entering the dynamic
    program, so a schedule accepted by the integer budget cannot exceed the
    real-valued budget due to quantization.
    """

    def __init__(self, *, cost_quantum_ms: float = 5.0) -> None:
        if not math.isfinite(cost_quantum_ms) or cost_quantum_ms <= 0.0:
            raise SparseBudgetError("cost quantum must be finite and positive")
        self.cost_quantum_ms = float(cost_quantum_ms)

    def _cost_units(self, cost_ms: float) -> int:
        if cost_ms == 0.0:
            return 0
        return int(math.ceil(cost_ms / self.cost_quantum_ms - 1e-12))

    @staticmethod
    def _better(candidate: _DPState, current: _DPState | None) -> bool:
        if current is None:
            return True
        candidate_key = (
            candidate.risk,
            candidate.worst_component,
            candidate.actual_cost_ms,
        )
        current_key = (
            current.risk,
            current.worst_component,
            current.actual_cost_ms,
        )
        return candidate_key < current_key

    @staticmethod
    def _prune_dominated(
        states: dict[int, _DPState],
    ) -> dict[int, _DPState]:
        """Drop schedules that cost more without improving the objective.

        The calibrated scalar UCB is the primary optimization objective;
        component values only break exact-risk ties.  Sorting by conservative
        cost units therefore yields a small Pareto frontier and keeps online
        replanning independent of the absolute millisecond budget size.
        """

        frontier: dict[int, _DPState] = {}
        best_risk = math.inf
        best_worst_component = math.inf
        for used_units, state in sorted(states.items()):
            improves_risk = state.risk < best_risk - 1e-12
            ties_risk_and_improves_component = (
                abs(state.risk - best_risk) <= 1e-12
                and state.worst_component
                < best_worst_component - 1e-12
            )
            if not (improves_risk or ties_risk_and_improves_component):
                continue
            frontier[used_units] = state
            if improves_risk:
                best_risk = state.risk
                best_worst_component = state.worst_component
            else:
                best_worst_component = min(
                    best_worst_component, state.worst_component
                )
        return frontier

    def _solve_inside_budget(
        self,
        cells: tuple[SparseDecisionCell, ...],
        *,
        budget_ms: float,
        max_reject_risk_ucb: float | None,
    ) -> _InsideBudgetResult:
        budget_units = int(math.floor(budget_ms / self.cost_quantum_ms + 1e-12))
        # Cell order does not affect the additive finite problem.  Grouping
        # identical action ladders before DP sharply reduces intermediate
        # Pareto-frontier growth; choices are restored to the caller's
        # original lattice order before returning.
        planning_cells = tuple(
            sorted(
                cells,
                key=lambda cell: (
                    tuple(
                        (
                            action.name,
                            action.measured_cost_ms,
                            action.reject_risk_ucb,
                            action.components.motion,
                            action.components.clarity,
                            action.components.identity,
                            action.components.audio,
                            action.fidelity_rank,
                        )
                        for action in cell.actions
                        if action.fidelity_rank >= cell.minimum_fidelity_rank
                    ),
                    cell.key.cell_id,
                ),
            )
        )
        states: dict[int, _DPState] = {
            0: _DPState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None)
        }
        for cell in planning_cells:
            allowed = tuple(
                action
                for action in cell.actions
                if action.fidelity_rank >= cell.minimum_fidelity_rank
            )
            next_states: dict[int, _DPState] = {}
            for used_units, state in states.items():
                for action in allowed:
                    candidate_units = used_units + self._cost_units(
                        action.measured_cost_ms
                    )
                    if candidate_units > budget_units:
                        continue
                    choice = SparseScheduleChoice(cell.key, action)
                    candidate = _DPState(
                        risk=state.risk + action.reject_risk_ucb,
                        actual_cost_ms=state.actual_cost_ms + action.measured_cost_ms,
                        motion=state.motion + action.components.motion,
                        clarity=state.clarity + action.components.clarity,
                        identity=state.identity + action.components.identity,
                        audio=state.audio + action.components.audio,
                        previous=state,
                        choice=choice,
                    )
                    if self._better(candidate, next_states.get(candidate_units)):
                        next_states[candidate_units] = candidate
            states = self._prune_dominated(next_states)
            if not states:
                return _InsideBudgetResult(None, {})

        feasible = tuple(
            state
            for state in states.values()
            if max_reject_risk_ucb is None
            or state.risk <= max_reject_risk_ucb + 1e-12
        )
        if not feasible:
            return _InsideBudgetResult(None, states)
        optimum = min(
                feasible,
                key=lambda state: (
                    state.risk,
                    state.worst_component,
                    state.actual_cost_ms,
                ),
            )
        materialized = _materialize_dp_state(optimum)
        by_cell = {choice.key.cell_id: choice for choice in materialized.choices}
        materialized = replace(
            materialized,
            choices=tuple(by_cell[cell.key.cell_id] for cell in cells),
        )
        return _InsideBudgetResult(
            materialized,
            states,
        )

    @staticmethod
    def _minimum_risk_without_budget(
        cells: tuple[SparseDecisionCell, ...],
    ) -> _DynamicState:
        choices: list[SparseScheduleChoice] = []
        total_cost = 0.0
        total_risk = 0.0
        components = HumanRiskVector()
        for cell in cells:
            allowed = (
                action
                for action in cell.actions
                if action.fidelity_rank >= cell.minimum_fidelity_rank
            )
            action = min(
                allowed,
                key=lambda item: (
                    item.reject_risk_ucb,
                    item.components.worst_component,
                    item.measured_cost_ms,
                ),
            )
            choices.append(SparseScheduleChoice(cell.key, action))
            total_cost += action.measured_cost_ms
            total_risk += action.reject_risk_ucb
            components = components + action.components
        return _DynamicState(total_risk, total_cost, components, tuple(choices))

    def solve(
        self,
        cells: Iterable[SparseDecisionCell],
        *,
        ledger: SparseBudgetLedger,
        open_recovery_reserve: bool = False,
        max_reject_risk_ucb: float | None = None,
        overflow_policy: OverflowPolicy = "strict_budget",
    ) -> SparseSchedule:
        """Return the lowest calibrated risk schedule under the remaining budget.

        ``quality_first`` is an explicit escape hatch: if the requested risk
        envelope and budget conflict, the solver chooses the lowest-risk
        available actions and reports the overrun.  It never silently changes
        sampler steps, model weights, or the requested output shape.
        """

        cells = tuple(cells)
        if not cells:
            raise SparseBudgetError("sparse schedule needs at least one decision cell")
        if overflow_policy not in ("strict_budget", "quality_first"):
            raise SparseBudgetError(f"unknown overflow policy: {overflow_policy}")
        if max_reject_risk_ucb is not None and (
            not math.isfinite(max_reject_risk_ucb) or max_reject_risk_ucb < 0.0
        ):
            raise SparseBudgetError("maximum reject risk must be finite and non-negative")

        budget_ms = ledger.available_ms(
            open_recovery_reserve=open_recovery_reserve
        )
        inside = self._solve_inside_budget(
            cells,
            budget_ms=budget_ms,
            max_reject_risk_ucb=max_reject_risk_ucb,
        )
        state = inside.optimum
        certificate: SparseOptimalityCertificate | None = None
        if state is None:
            if overflow_policy == "strict_budget":
                raise SparseBudgetInfeasible(
                    "no sparse schedule satisfies the measured-time budget and risk envelope"
                )
            state = self._minimum_risk_without_budget(cells)
            if (
                max_reject_risk_ucb is not None
                and state.risk > max_reject_risk_ucb + 1e-12
            ):
                raise SparseBudgetInfeasible(
                    "even the highest-fidelity schedule violates the calibrated risk envelope"
                )
        else:
            budget_units = int(
                math.floor(budget_ms / self.cost_quantum_ms + 1e-12)
            )
            certificate = SparseOptimalityCertificate(
                schema_version="h3_sparse_optimality_certificate_v1",
                solver="exact_multiple_choice_knapsack_dp",
                objective=(
                    "lexicographic(sum_reject_risk_ucb,"
                    "worst_component,actual_cost_ms)"
                ),
                formal_scope=(
                    "finite supplied actions + fidelity floors + additive calibrated "
                    "risk + conservative quantised budget"
                ),
                cost_quantum_ms=self.cost_quantum_ms,
                budget_units=budget_units,
                cell_count=len(cells),
                candidate_count=sum(
                    sum(
                        action.fidelity_rank >= cell.minimum_fidelity_rank
                        for action in cell.actions
                    )
                    for cell in cells
                ),
                conservative_cost_units=sum(
                    self._cost_units(choice.action.measured_cost_ms)
                    for choice in state.choices
                ),
                optimum_risk=state.risk,
                optimum_worst_component=state.components.worst_component,
                optimum_actual_cost_ms=state.actual_cost_ms,
                max_reject_risk_ucb=max_reject_risk_ucb,
                choice_sha256=_choice_digest(state.choices),
                frontier_sha256=_frontier_digest(inside.frontier),
            )

        return SparseSchedule(
            choices=state.choices,
            budget_limit_ms=budget_ms,
            estimated_cost_ms=state.actual_cost_ms,
            estimated_reject_risk_ucb=state.risk,
            estimated_components=state.components,
            budget_overrun_ms=max(0.0, state.actual_cost_ms - budget_ms),
            used_recovery_reserve=open_recovery_reserve,
            optimality_certificate=certificate,
        )


def verify_sparse_optimality_certificate(
    cells: Iterable[SparseDecisionCell],
    schedule: SparseSchedule,
) -> SparseCertificateVerification:
    """Reconstruct the finite problem and verify its exact-DP certificate.

    This is an audit/verifier boundary, not a statistical quality test.  It
    checks the input lattice, hard floors, conservative budget arithmetic,
    selected path, objective and complete Pareto-frontier digest.
    """

    cells = tuple(cells)
    certificate = schedule.optimality_certificate
    if certificate is None:
        return SparseCertificateVerification(False, ("missing certificate",))
    reasons: list[str] = []
    if certificate.schema_version != "h3_sparse_optimality_certificate_v1":
        reasons.append("unsupported certificate schema")
    if certificate.cell_count != len(cells):
        reasons.append("cell count mismatch")
    if tuple(choice.key for choice in schedule.choices) != tuple(
        cell.key for cell in cells
    ):
        reasons.append("choice lattice does not match cells")
    for cell, choice in zip(cells, schedule.choices):
        matching = tuple(
            action
            for action in cell.actions
            if action.name == choice.action.name
            and action.fidelity_rank >= cell.minimum_fidelity_rank
        )
        if not matching or choice.action != matching[0]:
            reasons.append(f"invalid choice for {cell.key.cell_id}")
            break
    if reasons:
        return SparseCertificateVerification(False, tuple(reasons))

    solver = H3BudgetedRiskScheduler(
        cost_quantum_ms=certificate.cost_quantum_ms
    )
    try:
        replay = solver.solve(
            cells,
            ledger=SparseBudgetLedger(schedule.budget_limit_ms),
            max_reject_risk_ucb=certificate.max_reject_risk_ucb,
            overflow_policy="strict_budget",
        )
    except SparseBudgetError as error:
        return SparseCertificateVerification(
            False, (f"certificate replay failed: {error}",)
        )
    expected = replay.optimality_certificate
    if expected is None:
        reasons.append("replay produced no certificate")
    else:
        for field in (
            "budget_units",
            "cell_count",
            "candidate_count",
            "conservative_cost_units",
            "choice_sha256",
            "frontier_sha256",
        ):
            if getattr(certificate, field) != getattr(expected, field):
                reasons.append(f"certificate {field} mismatch")
        for field in (
            "optimum_risk",
            "optimum_worst_component",
            "optimum_actual_cost_ms",
        ):
            if not math.isclose(
                getattr(certificate, field),
                getattr(expected, field),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                reasons.append(f"certificate {field} mismatch")
        if _choice_digest(schedule.choices) != certificate.choice_sha256:
            reasons.append("schedule choice digest mismatch")
    return SparseCertificateVerification(not reasons, tuple(reasons))


_H3_LAYER_BANDS: tuple[tuple[H3LayerBand, int, int], ...] = (
    (H3LayerBand.EARLY, 0, 15),
    (H3LayerBand.MIDDLE, 15, 30),
    (H3LayerBand.CAUSAL, 30, 40),
    (H3LayerBand.CAUSAL_DETAIL, 40, 44),
    (H3LayerBand.BRIDGE, 44, 45),
    (H3LayerBand.CAUSAL_TERMINAL, 45, 46),
    (H3LayerBand.TAIL, 46, 50),
)


def build_h3_sparse_cells(
    actual_steps: tuple[int, ...],
    action_factory: Callable[
        [H3AttentionCellKey], tuple[SparseActionEstimate, ...]
    ],
    *,
    causal_floor_rank: int = 1,
    opening_floor_rank: int = 1,
    terminal_floor_rank: int = 1,
) -> tuple[SparseDecisionCell, ...]:
    """Build the small, H3-specific step x layer-band decision lattice.

    Forecast evaluations are intentionally absent because they do not execute
    the complete 50-layer DiT.  The default floors encode the reusable local
    evidence: opening composition, layers 30--45, and the last three actual
    evaluations cannot use the most aggressive action.
    """

    if not actual_steps or tuple(sorted(set(actual_steps))) != actual_steps:
        raise SparseBudgetError("actual steps must be sorted and unique")
    first = actual_steps[0]
    terminal = frozenset(actual_steps[-min(3, len(actual_steps)):])
    result: list[SparseDecisionCell] = []
    for step in actual_steps:
        phase: Literal["opening", "ordinary", "terminal"]
        if step == first:
            phase = "opening"
        elif step in terminal:
            phase = "terminal"
        else:
            phase = "ordinary"
        for band, start, stop in _H3_LAYER_BANDS:
            key = H3AttentionCellKey(step, band, start, stop, phase)
            floor = 0
            if band in (
                H3LayerBand.CAUSAL,
                H3LayerBand.CAUSAL_DETAIL,
                H3LayerBand.CAUSAL_TERMINAL,
            ):
                floor = max(floor, causal_floor_rank)
            if phase == "opening":
                floor = max(floor, opening_floor_rank)
            if phase == "terminal":
                floor = max(floor, terminal_floor_rank)
            result.append(
                SparseDecisionCell(
                    key=key,
                    actions=tuple(action_factory(key)),
                    minimum_fidelity_rank=floor,
                )
            )
    return tuple(result)


def apply_h3_checkpoint_recovery(
    cells: Iterable[SparseDecisionCell],
    *,
    trigger_step: int,
    current_step_rank: int = 2,
    next_step_rank: int = 1,
) -> tuple[SparseDecisionCell, ...]:
    """Apply the validated complete-band recovery and one-step hysteresis.

    A triggered step raises both causal bands to ``current_step_rank``.  The
    next actual step protects layers 30--39 with ``next_step_rank``.  This is
    the control-plane form of the Round163--175 positive mechanism; it never
    mixes Dense and sparse hidden rows or claims that a late step can erase
    already accumulated trajectory error.
    """

    cells = tuple(cells)
    later_steps = sorted(
        {cell.key.actual_step for cell in cells if cell.key.actual_step > trigger_step}
    )
    next_step = later_steps[0] if later_steps else None
    result: list[SparseDecisionCell] = []
    for cell in cells:
        floor = cell.minimum_fidelity_rank
        if (
            cell.key.actual_step == trigger_step
            and cell.key.layer_band
            in (
                H3LayerBand.CAUSAL,
                H3LayerBand.CAUSAL_DETAIL,
                H3LayerBand.CAUSAL_TERMINAL,
            )
        ):
            floor = max(floor, current_step_rank)
        elif (
            next_step is not None
            and cell.key.actual_step == next_step
            and cell.key.layer_band is H3LayerBand.CAUSAL
        ):
            floor = max(floor, next_step_rank)
        result.append(replace(cell, minimum_fidelity_rank=floor))
    return tuple(result)


def update_action_risk(
    cells: Iterable[SparseDecisionCell],
    updates: Mapping[tuple[str, str], tuple[float, HumanRiskVector]],
) -> tuple[SparseDecisionCell, ...]:
    """Return cells with request-local calibrated risk observations applied."""

    result: list[SparseDecisionCell] = []
    for cell in cells:
        actions = []
        for action in cell.actions:
            update = updates.get((cell.key.cell_id, action.name))
            if update is None:
                actions.append(action)
            else:
                risk, components = update
                actions.append(
                    replace(
                        action,
                        reject_risk_ucb=risk,
                        components=components,
                    )
                )
        result.append(replace(cell, actions=tuple(actions)))
    return tuple(result)


__all__ = [
    "H3AttentionCellKey",
    "H3BudgetedRiskScheduler",
    "H3LayerBand",
    "HumanRiskVector",
    "SparseActionEstimate",
    "SparseBudgetError",
    "SparseBudgetInfeasible",
    "SparseBudgetLedger",
    "SparseCertificateVerification",
    "SparseDecisionCell",
    "SparseOptimalityCertificate",
    "SparseSchedule",
    "SparseScheduleChoice",
    "apply_h3_checkpoint_recovery",
    "build_h3_sparse_cells",
    "update_action_risk",
    "verify_sparse_optimality_certificate",
]
