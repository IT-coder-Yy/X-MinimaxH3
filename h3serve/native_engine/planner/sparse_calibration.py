"""Load real-H3 RTX 4090 sparse action measurements into the budget solver.

The calibration artifact is emitted by ``benchmark_native_hot_session.py``
while a Dense teacher trajectory remains unmodified.  Every candidate timing
therefore covers one complete 56-head H3 Attention call.  This module turns
those raw per-layer observations into the small step x layer-band action table
used by :mod:`sparse_budget`; it never substitutes Top-K fractions for time.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Mapping

from .sparse_budget import (
    H3AttentionCellKey,
    H3BudgetedRiskScheduler,
    H3LayerBand,
    HumanRiskVector,
    SparseActionEstimate,
    SparseBudgetError,
    SparseBudgetLedger,
    SparseDecisionCell,
    SparseSchedule,
    build_h3_sparse_cells,
)


_BAND_RISK_MULTIPLIER: Mapping[H3LayerBand, float] = {
    H3LayerBand.EARLY: 1.00,
    H3LayerBand.MIDDLE: 1.10,
    # Round142/151 Dense-teacher probes and the door/contact Human failures
    # consistently place causal risk in layers 30--43 and 45.  Layer 44 is
    # deliberately split out because the Human-accepted Round143 route did
    # not make it part of the Dense causal island.
    H3LayerBand.CAUSAL: 2.20,
    H3LayerBand.CAUSAL_DETAIL: 2.50,
    H3LayerBand.BRIDGE: 1.20,
    H3LayerBand.CAUSAL_TERMINAL: 2.50,
    H3LayerBand.TAIL: 1.30,
}

_PHASE_RISK_MULTIPLIER = {
    "opening": 1.60,
    "ordinary": 1.00,
    "terminal": 1.40,
}


@dataclass(frozen=True, slots=True)
class MeasuredBandAction:
    name: str
    topk: float | None
    measured_cost_ms: float
    dense_error_upper: float
    fidelity_rank: int


@dataclass(frozen=True, slots=True)
class H3SparseActionCalibration:
    """Robust band-level action costs from one real H3 tensor shape."""

    source: Path
    engine: str
    sequence_tokens: int
    step_index: int
    actions_by_band: Mapping[H3LayerBand, tuple[MeasuredBandAction, ...]]

    @property
    def maximum_fidelity_rank(self) -> int:
        return max(
            action.fidelity_rank
            for actions in self.actions_by_band.values()
            for action in actions
        )

    @property
    def sparse_ranks(self) -> tuple[tuple[float, int], ...]:
        first = self.actions_by_band[H3LayerBand.EARLY]
        return tuple(
            (float(action.topk), action.fidelity_rank)
            for action in first
            if action.topk is not None
        )

    def first_rank_at_or_above(self, topk: float) -> int:
        for candidate_topk, rank in self.sparse_ranks:
            if candidate_topk >= topk - 1.0e-12:
                return rank
        return self.maximum_fidelity_rank


def _upper_observed(values: list[float]) -> float:
    """Return a deterministic small-sample upper observation.

    This is intentionally not called a probability.  With only one teacher
    task in v1, the 80th-percentile observed Dense disagreement plus the
    structural Human prior is a conservative optimization surrogate, not a
    learned Human rejection model.
    """

    if not values:
        raise SparseBudgetError("sparse calibration action has no error observations")
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(0.80 * len(ordered)) - 1)
    return float(ordered[index])


def load_h3_sparse_action_calibration(
    path: str | Path,
    *,
    step_index: int | None = None,
) -> H3SparseActionCalibration:
    """Validate and aggregate a full-head teacher-side calibration artifact."""

    source = Path(path).resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
        contract = document["contract"]
        steps = document["steps"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SparseBudgetError(f"invalid sparse calibration artifact: {source}") from error
    if not document.get("complete"):
        raise SparseBudgetError("sparse calibration artifact is incomplete")
    if contract.get("timing_scope") != "one complete 56-head call":
        raise SparseBudgetError("budget planner requires complete 56-head timings")
    if contract.get("heads") != 56 or not contract.get("dense_result_returned"):
        raise SparseBudgetError("calibration must use the unchanged 56-head Dense teacher")
    if contract.get("weights_modified"):
        raise SparseBudgetError("calibration must not modify H3 weights")
    if not isinstance(steps, list) or not steps:
        raise SparseBudgetError("calibration contains no measured steps")
    if step_index is None:
        if len(steps) != 1:
            raise SparseBudgetError("multi-step calibration requires an explicit step_index")
        selected = steps[0]
    else:
        selected = next(
            (row for row in steps if row.get("step_index") == step_index), None
        )
        if selected is None:
            raise SparseBudgetError(f"calibration does not contain step {step_index}")
    layers = selected.get("layers")
    if not selected.get("complete") or not isinstance(layers, list) or len(layers) != 50:
        raise SparseBudgetError("selected calibration step must contain all 50 H3 layers")
    by_index = {int(row["layer"]): row for row in layers}
    if set(by_index) != set(range(50)):
        raise SparseBudgetError("calibration layer indices must be exactly 0..49")

    first_candidates = by_index[0].get("candidates")
    if not isinstance(first_candidates, list) or not first_candidates:
        raise SparseBudgetError("calibration contains no full-head sparse candidates")
    candidate_names = tuple(str(row["name"]) for row in first_candidates)
    candidate_topks = tuple(float(row["topk"]) for row in first_candidates)
    if tuple(sorted(candidate_topks)) != candidate_topks:
        raise SparseBudgetError("sparse calibration actions must be ordered by Top-K")
    for row in by_index.values():
        candidates = row.get("candidates")
        if tuple(str(item["name"]) for item in candidates) != candidate_names:
            raise SparseBudgetError("all layers must contain the same sparse actions")

    # Match the seven evidence-derived planner bands exactly.
    bounds = {
        H3LayerBand.EARLY: (0, 15),
        H3LayerBand.MIDDLE: (15, 30),
        H3LayerBand.CAUSAL: (30, 40),
        H3LayerBand.CAUSAL_DETAIL: (40, 44),
        H3LayerBand.BRIDGE: (44, 45),
        H3LayerBand.CAUSAL_TERMINAL: (45, 46),
        H3LayerBand.TAIL: (46, 50),
    }
    actions_by_band: dict[H3LayerBand, tuple[MeasuredBandAction, ...]] = {}
    dense_rank = len(candidate_names)
    for band, (start, stop) in bounds.items():
        rows = [by_index[index] for index in range(start, stop)]
        layer_count = stop - start
        actions: list[MeasuredBandAction] = []
        for candidate_index, (name, topk) in enumerate(
            zip(candidate_names, candidate_topks)
        ):
            costs = [
                float(row["candidates"][candidate_index]["full_head_ms"])
                for row in rows
            ]
            errors = [
                float(row["candidates"][candidate_index]["global_relative_rms"])
                for row in rows
            ]
            # Median per-layer latency rejects one-time lazy setup (the first
            # 0.0625 call in the 720p15 artifact is 410.8ms vs ~129ms hot).
            actions.append(
                MeasuredBandAction(
                    name=name,
                    topk=topk,
                    measured_cost_ms=statistics.median(costs) * layer_count,
                    dense_error_upper=_upper_observed(errors),
                    fidelity_rank=candidate_index,
                )
            )
        dense_cost = statistics.median(
            float(row["dense_ms"]) for row in rows
        ) * layer_count
        actions.append(
            MeasuredBandAction(
                name="dense",
                topk=None,
                measured_cost_ms=dense_cost,
                dense_error_upper=0.0,
                fidelity_rank=dense_rank,
            )
        )
        actions_by_band[band] = tuple(actions)

    return H3SparseActionCalibration(
        source=source,
        engine=str(contract["engine"]),
        sequence_tokens=int(contract["sequence_tokens"]),
        step_index=int(selected["step_index"]),
        actions_by_band=actions_by_band,
    )


def build_measured_h3_sparse_cells(
    calibration: H3SparseActionCalibration,
    actual_steps: tuple[int, ...],
    *,
    exact_opening: bool = True,
    exact_causal_island: bool = True,
    terminal_minimum_topk: float = 0.10,
) -> tuple[SparseDecisionCell, ...]:
    """Build v1 cells using measured time and conservative Human priors.

    The hard defaults deliberately reproduce the reusable structure of the
    Human-accepted Round143 endpoint: exact opening computation and an exact
    layers 30--43/45 causal island.  The optimizer remains free to redistribute
    the remaining measured budget across all other layers and steps.
    """

    dense_rank = calibration.maximum_fidelity_rank
    terminal_rank = calibration.first_rank_at_or_above(terminal_minimum_topk)

    def factory(key: H3AttentionCellKey) -> tuple[SparseActionEstimate, ...]:
        result = []
        band_multiplier = _BAND_RISK_MULTIPLIER[key.layer_band]
        phase_multiplier = _PHASE_RISK_MULTIPLIER[key.phase]
        layer_count = key.layer_stop - key.layer_start
        for action in calibration.actions_by_band[key.layer_band]:
            base = action.dense_error_upper * layer_count
            debt = base * band_multiplier * phase_multiplier
            causal = key.layer_band in (
                H3LayerBand.CAUSAL,
                H3LayerBand.CAUSAL_DETAIL,
                H3LayerBand.CAUSAL_TERMINAL,
            )
            result.append(
                SparseActionEstimate(
                    name=action.name,
                    measured_cost_ms=action.measured_cost_ms,
                    reject_risk_ucb=debt,
                    fidelity_rank=action.fidelity_rank,
                    components=HumanRiskVector(
                        motion=debt if causal else debt * 0.45,
                        clarity=base,
                        identity=debt * (0.60 if causal else 0.25),
                        audio=debt * (0.45 if key.phase == "terminal" else 0.20),
                    ),
                )
            )
        return tuple(result)

    return build_h3_sparse_cells(
        actual_steps,
        factory,
        causal_floor_rank=dense_rank if exact_causal_island else 1,
        opening_floor_rank=dense_rank if exact_opening else 1,
        terminal_floor_rank=terminal_rank,
    )


def minimum_measured_schedule_cost(cells: tuple[SparseDecisionCell, ...]) -> float:
    """Return the exact minimum feasible measured cost under cell floors."""

    return sum(
        min(
            action.measured_cost_ms
            for action in cell.actions
            if action.fidelity_rank >= cell.minimum_fidelity_rank
        )
        for cell in cells
    )


def solve_measured_h3_sparse_schedule(
    calibration: H3SparseActionCalibration,
    actual_steps: tuple[int, ...],
    *,
    attention_budget_ms: float,
    exact_opening: bool = True,
    exact_causal_island: bool = True,
    terminal_minimum_topk: float = 0.10,
) -> tuple[SparseSchedule, dict[tuple[int, int], str]]:
    """Solve a measured schedule and expand its bands to physical H3 layers."""

    cells = build_measured_h3_sparse_cells(
        calibration,
        actual_steps,
        exact_opening=exact_opening,
        exact_causal_island=exact_causal_island,
        terminal_minimum_topk=terminal_minimum_topk,
    )
    schedule = H3BudgetedRiskScheduler(cost_quantum_ms=1.0).solve(
        cells,
        ledger=SparseBudgetLedger(attention_budget_ms),
    )
    physical: dict[tuple[int, int], str] = {}
    for choice in schedule.choices:
        for layer in range(choice.key.layer_start, choice.key.layer_stop):
            physical[(choice.key.actual_step, layer)] = choice.action.name
    return schedule, physical


__all__ = [
    "H3SparseActionCalibration",
    "MeasuredBandAction",
    "build_measured_h3_sparse_cells",
    "load_h3_sparse_action_calibration",
    "minimum_measured_schedule_cost",
    "solve_measured_h3_sparse_schedule",
]
