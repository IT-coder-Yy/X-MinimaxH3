"""Request-local risk-budget control for mechanistic H3 plans.

The offline optimizer chooses the coupled Actual/Forecast/Attention path.  At
runtime, existing Actual corrections expose a cheap secant-tail error sample.
This module converts that observation into a continuous uncertainty belief and
solves a bounded recovery-reserve problem over the *remaining* planned
Forecasts.  No prompt category, named candidate or resolution-specific
schedule branch participates in the decision.

Recovery is deliberately one-sided in this first controller: a requested
Forecast may be promoted to an Actual correction, while an offline Actual is
never demoted.  Missing Attention cells already fail closed to Dense in the
request-routed backend, so the controller cannot introduce a new approximate
kernel path.  The offline planner must reserve the measured promotion cost.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import math
from typing import Any

from .mechanistic_control import (
    H3MechanisticControlModel,
    H3MechanisticPlan,
    H3MechanisticRisk,
    MechanisticControlError,
)


MECHANISTIC_RUNTIME_SCHEMA = "h3_mechanistic_runtime_control_v1"
MECHANISTIC_RUNTIME_POLICY_ID = "h3_request_local_risk_reserve_mpc_v1"
MECHANISTIC_RUNTIME_SOLVER_ID = "exact_bounded_future_promotion_enumeration_v1"


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class H3MechanisticRuntimeBelief:
    """Request-local multiplicative UCB for unexecuted Forecast error."""

    audio_scale: float = 1.0
    video_scale: float = 1.0

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value < 1.0
            for value in (self.audio_scale, self.video_scale)
        ):
            raise MechanisticControlError(
                "runtime Forecast belief scales must be finite and at least one"
            )

    @property
    def digest(self) -> str:
        return _sha256_json({
            "audio_scale": self.audio_scale,
            "video_scale": self.video_scale,
        })


@dataclass(frozen=True, slots=True)
class H3MechanisticRuntimeDecision:
    after_step: int
    belief: H3MechanisticRuntimeBelief
    selected_future_promotions: tuple[int, ...]
    executed_promotions: tuple[int, ...]
    projected_risk: H3MechanisticRisk
    risk_limit: float
    projected_extra_cost_ms: float
    recovery_reserve_ms: float
    admitted: bool
    candidate_subsets_evaluated: int
    decision_digest: str
    schema_version: str = MECHANISTIC_RUNTIME_SCHEMA
    policy_id: str = MECHANISTIC_RUNTIME_POLICY_ID
    solver_id: str = MECHANISTIC_RUNTIME_SOLVER_ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "solver_id": self.solver_id,
            "after_step": self.after_step,
            "belief": {
                "audio_scale": self.belief.audio_scale,
                "video_scale": self.belief.video_scale,
                "digest": self.belief.digest,
            },
            "selected_future_promotions": list(self.selected_future_promotions),
            "executed_promotions": list(self.executed_promotions),
            "projected_risk": self.projected_risk.to_dict(),
            "risk_limit": self.risk_limit,
            "projected_extra_cost_ms": self.projected_extra_cost_ms,
            "recovery_reserve_ms": self.recovery_reserve_ms,
            "admitted": self.admitted,
            "candidate_subsets_evaluated": self.candidate_subsets_evaluated,
            "decision_digest": self.decision_digest,
            "historical_schedule_used": False,
        }


class H3MechanisticRuntimeController:
    """Receding risk-budget controller over a bounded promotion reserve."""

    def __init__(
        self,
        *,
        model: H3MechanisticControlModel,
        plan: H3MechanisticPlan,
        recovery_reserve_ms: float,
        maximum_promotions: int = 1,
        risk_limit: float | None = None,
    ) -> None:
        verification = model.verify(plan)
        if not verification.valid:
            raise MechanisticControlError(
                "runtime controller requires a replayable mechanistic plan: "
                + "; ".join(verification.reasons)
            )
        try:
            reserve = float(recovery_reserve_ms)
        except (TypeError, ValueError) as error:
            raise MechanisticControlError("runtime reserve must be numeric") from error
        if not math.isfinite(reserve) or reserve < 0.0:
            raise MechanisticControlError(
                "runtime reserve must be finite and non-negative"
            )
        if not 0 <= int(maximum_promotions) <= 2:
            raise MechanisticControlError(
                "exact runtime enumeration supports zero to two promotions"
            )
        boundary = plan.admitted_risk_limit if risk_limit is None else risk_limit
        if boundary is None:
            raise MechanisticControlError(
                "runtime controller requires an explicit modeled-risk limit"
            )
        boundary = float(boundary)
        if not math.isfinite(boundary) or boundary < 0.0:
            raise MechanisticControlError("runtime risk limit is invalid")
        self.model = model
        self.plan = plan
        self.recovery_reserve_ms = reserve
        self.maximum_promotions = int(maximum_promotions)
        self.risk_limit = boundary
        self.belief = H3MechanisticRuntimeBelief()
        self._choice_by_step = {
            choice.step_index: choice for choice in plan.choices
        }
        self._executed_promotions: set[int] = set()
        self._selected_future_promotions: set[int] = set()
        self._realized_forecast_beliefs: dict[
            int, H3MechanisticRuntimeBelief
        ] = {}
        self._realized_actual_beliefs: dict[
            int, H3MechanisticRuntimeBelief
        ] = {}
        self._last_observed_step = -1
        self._records: list[dict[str, Any]] = []

    @property
    def selected_future_promotions(self) -> tuple[int, ...]:
        return tuple(sorted(self._selected_future_promotions))

    @property
    def executed_promotions(self) -> tuple[int, ...]:
        return tuple(sorted(self._executed_promotions))

    def should_promote(self, step_index: int) -> bool:
        return int(step_index) in self._selected_future_promotions

    def _promotion_actions(self) -> tuple[str, ...]:
        # The planned Forecast already executes the Round229 three-block
        # sparse anchor.  Missing tail cells fail closed to Dense at runtime.
        return ("sparse_topk_0.0625",) * 3 + ("dense",) * 47

    def _promotion_increment_ms(self, step: int) -> float:
        actions = self._promotion_actions()
        actual = self.model._physical.non_attention_ms + sum(
            self.model._layer_cost_ms(layer, action)
            for layer, action in enumerate(actions)
        )
        return max(0.0, actual - self.model._physical.forecast_ms)

    @staticmethod
    def _scale_forecast_risk(
        risk: H3MechanisticRisk,
        belief: H3MechanisticRuntimeBelief,
    ) -> H3MechanisticRisk:
        return H3MechanisticRisk(
            forecast_audio_energy=(
                risk.forecast_audio_energy * belief.audio_scale**2
            ),
            forecast_video_energy=(
                risk.forecast_video_energy * belief.video_scale**2
            ),
            epistemic_energy=risk.epistemic_energy,
        )

    @staticmethod
    def _scale_actual_risk(
        risk: H3MechanisticRisk,
        belief: H3MechanisticRuntimeBelief,
    ) -> H3MechanisticRisk:
        return H3MechanisticRisk(
            attention_energy=risk.attention_energy,
            interaction_energy=(
                risk.interaction_energy
                * max(belief.audio_scale, belief.video_scale)
            ),
            epistemic_energy=risk.epistemic_energy,
        )

    def _project_risk(
        self,
        *,
        after_step: int,
        future_promotions: frozenset[int],
    ) -> H3MechanisticRisk:
        promotions = frozenset(self._executed_promotions) | future_promotions
        risk = H3MechanisticRisk()
        forecast_run = 0
        promotion_actions = self._promotion_actions()
        for step in range(self.plan.workload.total_steps):
            planned = self._choice_by_step[step]
            actual = planned.actual or step in promotions
            if not actual:
                forecast_run += 1
                _cost, local = self.model._forecast_local(
                    step=step,
                    horizon=forecast_run,
                )
                belief = (
                    self.belief
                    if step > after_step
                    else self._realized_forecast_beliefs.get(
                        step, H3MechanisticRuntimeBelief()
                    )
                )
                risk += self._scale_forecast_risk(local, belief)
                continue
            actions = planned.attention_actions if planned.actual else promotion_actions
            belief = (
                self.belief
                if step > after_step
                else self._realized_actual_beliefs.get(
                    step, H3MechanisticRuntimeBelief()
                )
            )
            for layer, action in enumerate(actions):
                local = self.model._actual_local_action(
                    step=step,
                    layer=layer,
                    action=action,
                    prior_forecast_run=forecast_run,
                ).risk
                risk += self._scale_actual_risk(local, belief)
            forecast_run = 0
        return risk

    def _extra_cost_ms(self, promotions: frozenset[int]) -> float:
        return sum(self._promotion_increment_ms(step) for step in promotions)

    def _solve(self, *, after_step: int) -> H3MechanisticRuntimeDecision:
        future = tuple(
            choice.step_index
            for choice in self.plan.choices
            if not choice.actual and choice.step_index > after_step
        )
        already = frozenset(self._executed_promotions)
        remaining_slots = max(0, self.maximum_promotions - len(already))
        candidates: list[
            tuple[frozenset[int], float, H3MechanisticRisk]
        ] = []
        for count in range(min(remaining_slots, len(future)) + 1):
            for values in combinations(future, count):
                selected = frozenset(values)
                all_promotions = already | selected
                extra_cost = self._extra_cost_ms(all_promotions)
                if extra_cost > self.recovery_reserve_ms + 1.0e-9:
                    continue
                projected = self._project_risk(
                    after_step=after_step,
                    future_promotions=selected,
                )
                candidates.append((selected, extra_cost, projected))
        if not candidates:
            raise MechanisticControlError(
                "runtime recovery reserve cannot replay the executed promotions"
            )
        admitted = [
            row for row in candidates if row[2].total <= self.risk_limit + 1.0e-12
        ]
        if admitted:
            selected, extra_cost, projected = min(
                admitted,
                key=lambda row: (row[1], row[2].total, tuple(sorted(row[0]))),
            )
        else:
            selected, extra_cost, projected = min(
                candidates,
                key=lambda row: (row[2].total, row[1], tuple(sorted(row[0]))),
            )
        document = {
            "policy_id": MECHANISTIC_RUNTIME_POLICY_ID,
            "plan_choice_digest": self.plan.certificate.choice_digest,
            "after_step": after_step,
            "belief_digest": self.belief.digest,
            "selected_future_promotions": tuple(sorted(selected)),
            "executed_promotions": self.executed_promotions,
            "projected_risk": projected.to_dict(),
            "risk_limit": self.risk_limit,
            "projected_extra_cost_ms": extra_cost,
            "recovery_reserve_ms": self.recovery_reserve_ms,
            "candidate_subsets_evaluated": len(candidates),
            "historical_schedule_used": False,
        }
        return H3MechanisticRuntimeDecision(
            after_step=after_step,
            belief=self.belief,
            selected_future_promotions=tuple(sorted(selected)),
            executed_promotions=self.executed_promotions,
            projected_risk=projected,
            risk_limit=self.risk_limit,
            projected_extra_cost_ms=extra_cost,
            recovery_reserve_ms=self.recovery_reserve_ms,
            admitted=projected.total <= self.risk_limit + 1.0e-12,
            candidate_subsets_evaluated=len(candidates),
            decision_digest=_sha256_json(document),
        )

    def observe_actual(
        self,
        *,
        step_index: int,
        audio_risk_ratio: float,
        video_risk_ratio: float,
    ) -> H3MechanisticRuntimeDecision:
        """Update the UCB belief and re-optimize the bounded future reserve."""

        step_index = int(step_index)
        if not self._last_observed_step < step_index < self.plan.workload.total_steps:
            raise MechanisticControlError(
                "runtime Actual observations must be strictly increasing"
            )
        ratios = (float(audio_risk_ratio), float(video_risk_ratio))
        if any(not math.isfinite(value) or value <= 0.0 for value in ratios):
            raise MechanisticControlError("runtime error ratios must be positive")
        if step_index in self._selected_future_promotions:
            self._executed_promotions.add(step_index)
        observed_belief = H3MechanisticRuntimeBelief(
            audio_scale=max(1.0, ratios[0]),
            video_scale=max(1.0, ratios[1]),
        )
        # The correction sample measures the Forecast run that immediately
        # preceded this Actual.  Charging that already-incurred uncertainty is
        # necessary for an honest final-risk projection; future promotions
        # cannot make past numerical debt disappear from the ledger.
        preceding_forecasts: list[int] = []
        cursor = step_index - 1
        while cursor >= 0:
            choice = self._choice_by_step[cursor]
            if choice.actual or cursor in self._executed_promotions:
                break
            preceding_forecasts.append(cursor)
            cursor -= 1
        for forecast_step in preceding_forecasts:
            self._realized_forecast_beliefs[forecast_step] = observed_belief
        if preceding_forecasts:
            self._realized_actual_beliefs[step_index] = observed_belief
        # A running upper envelope is a parameter-free, one-sided UCB update;
        # it cannot make the remainder more approximate after anomalous data.
        self.belief = H3MechanisticRuntimeBelief(
            audio_scale=max(self.belief.audio_scale, ratios[0]),
            video_scale=max(self.belief.video_scale, ratios[1]),
        )
        self._last_observed_step = step_index
        decision = self._solve(after_step=step_index)
        self._selected_future_promotions = set(
            decision.selected_future_promotions
        )
        self._records.append({
            "step_index": step_index,
            "audio_risk_ratio": ratios[0],
            "video_risk_ratio": ratios[1],
            "decision": decision.to_dict(),
        })
        return decision

    def export(self) -> dict[str, Any]:
        return {
            "schema_version": MECHANISTIC_RUNTIME_SCHEMA,
            "policy_id": MECHANISTIC_RUNTIME_POLICY_ID,
            "solver_id": MECHANISTIC_RUNTIME_SOLVER_ID,
            "plan_choice_digest": self.plan.certificate.choice_digest,
            "risk_limit": self.risk_limit,
            "recovery_reserve_ms": self.recovery_reserve_ms,
            "maximum_promotions": self.maximum_promotions,
            "belief": {
                "audio_scale": self.belief.audio_scale,
                "video_scale": self.belief.video_scale,
            },
            "selected_future_promotions": list(self.selected_future_promotions),
            "executed_promotions": list(self.executed_promotions),
            "records": list(self._records),
            "realized_forecast_beliefs": {
                str(step): {
                    "audio_scale": belief.audio_scale,
                    "video_scale": belief.video_scale,
                }
                for step, belief in sorted(
                    self._realized_forecast_beliefs.items()
                )
            },
            "realized_actual_beliefs": {
                str(step): {
                    "audio_scale": belief.audio_scale,
                    "video_scale": belief.video_scale,
                }
                for step, belief in sorted(
                    self._realized_actual_beliefs.items()
                )
            },
            "adds_teacher_evaluations": False,
            "historical_schedule_used": False,
        }

    def checkpoint_state(self) -> dict[str, Any]:
        """Serialize only request-local state; the sealed plan is external."""

        return {
            "schema_version": 1,
            "policy_id": MECHANISTIC_RUNTIME_POLICY_ID,
            "plan_choice_digest": self.plan.certificate.choice_digest,
            "risk_limit": self.risk_limit,
            "recovery_reserve_ms": self.recovery_reserve_ms,
            "maximum_promotions": self.maximum_promotions,
            "belief": {
                "audio_scale": self.belief.audio_scale,
                "video_scale": self.belief.video_scale,
            },
            "executed_promotions": list(self.executed_promotions),
            "selected_future_promotions": list(self.selected_future_promotions),
            "last_observed_step": self._last_observed_step,
            "records": list(self._records),
            "realized_forecast_beliefs": {
                str(step): {
                    "audio_scale": belief.audio_scale,
                    "video_scale": belief.video_scale,
                }
                for step, belief in sorted(
                    self._realized_forecast_beliefs.items()
                )
            },
            "realized_actual_beliefs": {
                str(step): {
                    "audio_scale": belief.audio_scale,
                    "video_scale": belief.video_scale,
                }
                for step, belief in sorted(
                    self._realized_actual_beliefs.items()
                )
            },
        }

    def restore_checkpoint_state(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != 1:
            raise MechanisticControlError(
                "unsupported mechanistic runtime checkpoint schema"
            )
        expected = {
            "policy_id": MECHANISTIC_RUNTIME_POLICY_ID,
            "plan_choice_digest": self.plan.certificate.choice_digest,
            "risk_limit": self.risk_limit,
            "recovery_reserve_ms": self.recovery_reserve_ms,
            "maximum_promotions": self.maximum_promotions,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise MechanisticControlError(
                    f"mechanistic runtime checkpoint mismatch for {key}"
                )
        belief = state.get("belief")
        executed = state.get("executed_promotions")
        selected = state.get("selected_future_promotions")
        records = state.get("records")
        realized_forecast = state.get("realized_forecast_beliefs", {})
        realized_actual = state.get("realized_actual_beliefs", {})
        if (
            not isinstance(belief, dict)
            or not isinstance(executed, list)
            or not isinstance(selected, list)
            or not isinstance(records, list)
            or not isinstance(realized_forecast, dict)
            or not isinstance(realized_actual, dict)
        ):
            raise MechanisticControlError(
                "mechanistic runtime checkpoint is malformed"
            )
        restored_executed = {int(step) for step in executed}
        restored_selected = {int(step) for step in selected}
        forecast_steps = set(self.plan.forecast_step_indices)
        if (
            not restored_executed.issubset(forecast_steps)
            or not restored_selected.issubset(forecast_steps)
            or len(restored_executed) > self.maximum_promotions
            or len(restored_executed | restored_selected)
            > self.maximum_promotions
        ):
            raise MechanisticControlError(
                "mechanistic runtime checkpoint promotions are invalid"
            )
        last = int(state.get("last_observed_step", -1))
        if not -1 <= last < self.plan.workload.total_steps:
            raise MechanisticControlError(
                "mechanistic runtime checkpoint step is invalid"
            )
        self.belief = H3MechanisticRuntimeBelief(
            audio_scale=float(belief["audio_scale"]),
            video_scale=float(belief["video_scale"]),
        )
        self._executed_promotions = restored_executed
        self._selected_future_promotions = restored_selected
        self._last_observed_step = last
        self._records = list(records)
        try:
            restored_forecast_beliefs = {
                int(step): H3MechanisticRuntimeBelief(
                    audio_scale=float(value["audio_scale"]),
                    video_scale=float(value["video_scale"]),
                )
                for step, value in realized_forecast.items()
            }
            restored_actual_beliefs = {
                int(step): H3MechanisticRuntimeBelief(
                    audio_scale=float(value["audio_scale"]),
                    video_scale=float(value["video_scale"]),
                )
                for step, value in realized_actual.items()
            }
        except (KeyError, TypeError, ValueError) as error:
            raise MechanisticControlError(
                "mechanistic runtime checkpoint beliefs are malformed"
            ) from error
        if (
            not set(restored_forecast_beliefs).issubset(forecast_steps)
            or any(step not in range(self.plan.workload.total_steps)
                   for step in restored_actual_beliefs)
        ):
            raise MechanisticControlError(
                "mechanistic runtime checkpoint belief steps are invalid"
            )
        self._realized_forecast_beliefs = restored_forecast_beliefs
        self._realized_actual_beliefs = restored_actual_beliefs


__all__ = [
    "H3MechanisticRuntimeBelief",
    "H3MechanisticRuntimeController",
    "H3MechanisticRuntimeDecision",
    "MECHANISTIC_RUNTIME_POLICY_ID",
    "MECHANISTIC_RUNTIME_SCHEMA",
    "MECHANISTIC_RUNTIME_SOLVER_ID",
]
