"""Mechanism-driven joint control for the H3 Base inference trajectory.

This module is intentionally independent of the historical V1--V24 candidate
tables.  It solves a declared model of the H3 computation itself:

* an Actual transition evaluates the full 50-block residual stack and chooses
  one measured Attention implementation for every block;
* a Forecast transition evaluates the depth-3 directional anchor and incurs a
  request-shape/phase/horizon secant-tail error;
* sparse Attention error is the measured global relative-RMS disagreement to
  Dense at the same H3 layer and normalized sigma position;
* an Actual evaluation following Forecast debt receives the first-order cross
  term between state uncertainty and local Attention approximation; and
* wall cost and modeled perturbation energy share one Lagrangian objective.

For any fixed multiplier the trajectory dynamic program is exact: it jointly
chooses Actual/Forecast placement and all per-layer Attention actions.  A plan
returned for a positive multiplier is a supported Pareto point of the declared
finite problem.  This is a formal compute/numerical-risk statement, not a claim
that finite tensor probes perfectly predict Human video preference.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .joint_global_dp import (
    ACTION_NAMES,
    LAYER_BANDS,
    ROUND215_ACTION_IMPLEMENTATION,
    ROUND218_PHASE_LAYER_RISK_MODEL,
    JointWorkloadContext,
    _profile,
)


MECHANISTIC_CONTROL_SCHEMA = "h3_mechanistic_control_plan_v1"
MECHANISTIC_CONTROL_POLICY_ID = "h3_mechanistic_joint_optimal_control_v1"
MECHANISTIC_SOLVER_ID = "exact_lagrangian_trajectory_dp_v1"
MECHANISTIC_COST_MODEL_ID = "round215_measured_wall_cost_v1"
MECHANISTIC_ATTENTION_MODEL_ID = "round218_continuous_phase_layer_rms_v1"
MECHANISTIC_FORECAST_MODEL_ID = "v24_secant_tail_log_response_v1"
MECHANISTIC_PROPAGATION_MODEL_ID = (
    "h3_gronwall_phase_gain_720p5_identification_v1"
)

_FORECAST_EVIDENCE_PATH = (
    Path(__file__).with_name("evidence") / "v24_forecast_error_model_v1.json"
)
_ROUND218_EVIDENCE_PATH = (
    Path(__file__).with_name("evidence") / "round218_phase_layer_risk_v1.json"
)
_PROPAGATION_EVIDENCE_PATH = (
    Path(__file__).with_name("evidence")
    / "h3_mechanistic_downstream_impulse_response_720p5_v1.json"
)
_ACTION_RUNTIME_PREFIX = "round215"
_FORECAST_ANCHOR_ACTION = "forecastfrontier:sparse_topk_0.0625"


class MechanisticControlError(ValueError):
    """The declared H3 control problem is invalid or cannot be admitted."""


@dataclass(frozen=True, slots=True)
class H3MechanisticWorkload:
    """Exact request geometry available after native H3 tokenisation."""

    total_steps: int
    packed_tokens: int
    video_tokens: int
    condition_count: int = 0
    allow_forecast: bool = True
    required_actual_steps: tuple[int, ...] = ()
    service_family: str = "first_last"
    model_variant: str = "base"

    def __post_init__(self) -> None:
        if not 4 <= self.total_steps <= 30:
            raise MechanisticControlError("H3 steps must lie inside [4, 30]")
        if self.packed_tokens <= 0 or self.video_tokens <= 0:
            raise MechanisticControlError("H3 token counts must be positive")
        if self.video_tokens > self.packed_tokens:
            raise MechanisticControlError("video tokens cannot exceed packed tokens")
        if self.condition_count < 0:
            raise MechanisticControlError("condition count cannot be negative")
        required = tuple(sorted(set(self.required_actual_steps)))
        if required != self.required_actual_steps or any(
            step < 0 or step >= self.total_steps for step in required
        ):
            raise MechanisticControlError(
                "required Actual steps must be sorted, unique and in range"
            )
        if self.model_variant != "base":
            raise MechanisticControlError(
                "mechanistic control is currently calibrated only for H3 Base"
            )
        if self.service_family not in ("first_last", "reference"):
            raise MechanisticControlError("unsupported H3 service family")

    @property
    def model_context(self) -> JointWorkloadContext:
        return JointWorkloadContext(
            packed_tokens=self.packed_tokens,
            condition_count=self.condition_count,
            service_family=self.service_family,
            model_variant=self.model_variant,
        )

    @property
    def digest(self) -> str:
        return _sha256_json({
            "total_steps": self.total_steps,
            "packed_tokens": self.packed_tokens,
            "video_tokens": self.video_tokens,
            "condition_count": self.condition_count,
            "allow_forecast": self.allow_forecast,
            "required_actual_steps": self.required_actual_steps,
            "service_family": self.service_family,
            "model_variant": self.model_variant,
        })


@dataclass(frozen=True, slots=True)
class H3MechanisticAdmission:
    """Human-calibrated risk boundary, never a historical schedule."""

    calibration_id: str
    maximum_modeled_risk: float
    evidence_ids: tuple[str, ...]
    held_out_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.calibration_id:
            raise MechanisticControlError("admission calibration id is required")
        if (
            not math.isfinite(self.maximum_modeled_risk)
            or self.maximum_modeled_risk < 0.0
        ):
            raise MechanisticControlError("admitted risk must be finite and non-negative")
        if not self.evidence_ids:
            raise MechanisticControlError("admission requires calibration evidence")


@dataclass(frozen=True, slots=True)
class H3MechanisticRisk:
    """Additive perturbation-energy ledger used by the optimizer."""

    attention_energy: float = 0.0
    forecast_audio_energy: float = 0.0
    forecast_video_energy: float = 0.0
    interaction_energy: float = 0.0
    epistemic_energy: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.attention_energy,
            self.forecast_audio_energy,
            self.forecast_video_energy,
            self.interaction_energy,
            self.epistemic_energy,
        )
        if any(not math.isfinite(value) or value < -1.0e-12 for value in values):
            raise MechanisticControlError(
                "mechanistic risk components must be finite and non-negative"
            )

    def __add__(self, other: "H3MechanisticRisk") -> "H3MechanisticRisk":
        return H3MechanisticRisk(
            attention_energy=self.attention_energy + other.attention_energy,
            forecast_audio_energy=(
                self.forecast_audio_energy + other.forecast_audio_energy
            ),
            forecast_video_energy=(
                self.forecast_video_energy + other.forecast_video_energy
            ),
            interaction_energy=self.interaction_energy + other.interaction_energy,
            epistemic_energy=self.epistemic_energy + other.epistemic_energy,
        )

    @property
    def total(self) -> float:
        # Every physical component already uses its conservative/UCB value.
        # Epistemic energy is diagnostic provenance for the UCB inflation and
        # is therefore not counted a second time.
        return (
            self.attention_energy
            + self.forecast_audio_energy
            + self.forecast_video_energy
            + self.interaction_energy
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "attention_energy": self.attention_energy,
            "forecast_audio_energy": self.forecast_audio_energy,
            "forecast_video_energy": self.forecast_video_energy,
            "interaction_energy": self.interaction_energy,
            "epistemic_energy": self.epistemic_energy,
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class H3MechanisticStepChoice:
    step_index: int
    actual: bool
    attention_actions: tuple[str, ...] = ()
    forecast_horizon: int = 0

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise MechanisticControlError("negative trajectory step")
        if self.actual:
            if len(self.attention_actions) != 50:
                raise MechanisticControlError(
                    "an Actual H3 step requires exactly 50 Attention actions"
                )
            if self.forecast_horizon:
                raise MechanisticControlError("Actual step cannot have a forecast horizon")
        else:
            if self.attention_actions or self.forecast_horizon <= 0:
                raise MechanisticControlError(
                    "Forecast step requires a positive horizon and no full-stack actions"
                )


@dataclass(frozen=True, slots=True)
class H3MechanisticCertificate:
    solver_id: str
    formal_scope: str
    lagrange_multiplier: float | str
    objective_value: float
    model_digest: str
    workload_digest: str
    choice_digest: str
    historical_schedule_used: bool
    exact_for_declared_lagrangian_problem: bool


@dataclass(frozen=True, slots=True)
class H3MechanisticPlan:
    workload: H3MechanisticWorkload
    choices: tuple[H3MechanisticStepChoice, ...]
    predicted_cost_ms: float
    modeled_risk: H3MechanisticRisk
    certificate: H3MechanisticCertificate
    acceleration: float | None = None
    target_cost_ms: float | None = None
    admitted_risk_limit: float | None = None
    admission_id: str | None = None
    schema_version: str = MECHANISTIC_CONTROL_SCHEMA
    policy_id: str = MECHANISTIC_CONTROL_POLICY_ID

    def __post_init__(self) -> None:
        if len(self.choices) != self.workload.total_steps:
            raise MechanisticControlError("plan does not cover the complete trajectory")
        if tuple(choice.step_index for choice in self.choices) != tuple(
            range(self.workload.total_steps)
        ):
            raise MechanisticControlError("plan steps are not contiguous")
        if self.predicted_cost_ms <= 0.0:
            raise MechanisticControlError("plan cost must be positive")

    @property
    def actual_step_indices(self) -> tuple[int, ...]:
        return tuple(choice.step_index for choice in self.choices if choice.actual)

    @property
    def forecast_step_indices(self) -> tuple[int, ...]:
        return tuple(choice.step_index for choice in self.choices if not choice.actual)

    @property
    def maximum_forecast_run(self) -> int:
        run = maximum = 0
        for choice in self.choices:
            run = 0 if choice.actual else run + 1
            maximum = max(maximum, run)
        return maximum

    def runtime_action_schedule(self) -> tuple[tuple[int, int, str], ...]:
        result: list[tuple[int, int, str]] = []
        for choice in self.choices:
            if choice.actual:
                for layer, action in enumerate(choice.attention_actions):
                    result.append((
                        choice.step_index,
                        layer,
                        action if action == "dense" else f"{_ACTION_RUNTIME_PREFIX}:{action}",
                    ))
            else:
                result.extend(
                    (choice.step_index, layer, _FORECAST_ANCHOR_ACTION)
                    for layer in range(3)
                )
        return tuple(result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "acceleration": self.acceleration,
            "target_cost_ms": self.target_cost_ms,
            "admitted_risk_limit": self.admitted_risk_limit,
            "admission_id": self.admission_id,
            "predicted_cost_ms": self.predicted_cost_ms,
            "modeled_risk": self.modeled_risk.to_dict(),
            "actual_step_indices": list(self.actual_step_indices),
            "forecast_step_indices": list(self.forecast_step_indices),
            "maximum_forecast_run": self.maximum_forecast_run,
            "workload_digest": self.workload.digest,
            "certificate": {
                "solver_id": self.certificate.solver_id,
                "formal_scope": self.certificate.formal_scope,
                "lagrange_multiplier": self.certificate.lagrange_multiplier,
                "objective_value": self.certificate.objective_value,
                "model_digest": self.certificate.model_digest,
                "workload_digest": self.certificate.workload_digest,
                "choice_digest": self.certificate.choice_digest,
                "historical_schedule_used": self.certificate.historical_schedule_used,
                "exact_for_declared_lagrangian_problem": (
                    self.certificate.exact_for_declared_lagrangian_problem
                ),
            },
            "choices": [
                {
                    "step_index": choice.step_index,
                    "actual": choice.actual,
                    "forecast_horizon": choice.forecast_horizon,
                    "attention_actions": list(choice.attention_actions),
                }
                for choice in self.choices
            ],
        }


@dataclass(frozen=True, slots=True)
class H3MechanisticVerification:
    valid: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class H3MechanisticScheduleEvaluation:
    """Risk/cost projection for external evidence, never an optimizer result."""

    evaluation_id: str
    predicted_cost_ms: float
    modeled_risk: H3MechanisticRisk
    actual_step_indices: tuple[int, ...]
    forecast_step_indices: tuple[int, ...]
    maximum_forecast_run: int
    source_implementation_ids: tuple[str, ...]
    calibration_implementation_match: bool
    schedule_digest: str
    model_digest: str


@dataclass(frozen=True, slots=True)
class _ForecastResponse:
    mean: float
    upper: float


@dataclass(frozen=True, slots=True)
class _LocalAction:
    name: str
    cost_ms: float
    risk: H3MechanisticRisk


@dataclass(frozen=True, slots=True)
class _DPState:
    forecast_run: int
    history_depth: int
    cost_ms: float
    risk: H3MechanisticRisk
    path: tuple[H3MechanisticStepChoice, ...]


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _linear_surface(
    anchors: tuple[float, ...],
    values: tuple[float, ...],
    position: float,
) -> float:
    if len(anchors) != len(values) or not anchors:
        raise MechanisticControlError("invalid continuous evidence surface")
    if position <= anchors[0]:
        return values[0]
    if position >= anchors[-1]:
        return values[-1]
    for left_index in range(len(anchors) - 1):
        left, right = anchors[left_index], anchors[left_index + 1]
        if left <= position <= right:
            mix = (position - left) / (right - left)
            return values[left_index] + (values[left_index + 1] - values[left_index]) * mix
    raise AssertionError("unreachable continuous surface interval")


class H3MechanisticControlModel:
    """Measured H3 local-error/cost model plus exact Lagrangian solver."""

    def __init__(self, workload: H3MechanisticWorkload) -> None:
        self.workload = workload
        self._physical = _profile(
            workload.model_context,
            ROUND218_PHASE_LAYER_RISK_MODEL,
        )
        self._forecast_evidence = json.loads(
            _FORECAST_EVIDENCE_PATH.read_text(encoding="utf-8")
        )
        self._propagation_evidence = json.loads(
            _PROPAGATION_EVIDENCE_PATH.read_text(encoding="utf-8")
        )
        if (
            self._forecast_evidence.get("schema_version")
            != "h3_v24_forecast_error_model_v1"
        ):
            raise MechanisticControlError("unexpected Forecast evidence schema")
        fit = self._forecast_evidence.get("fit")
        if not isinstance(fit, dict) or fit.get("sample_count") != 24:
            raise MechanisticControlError("Forecast evidence fit is incomplete")
        if (
            self._propagation_evidence.get("schema_version")
            != "h3_mechanistic_downstream_impulse_response_v1"
        ):
            raise MechanisticControlError("unexpected propagation evidence schema")
        gain_fits = self._propagation_evidence.get("continuous_phase_gain_fits")
        if not isinstance(gain_fits, dict) or not all(
            mechanism in gain_fits
            for mechanism in ("single_forecast", "single_attention")
        ):
            raise MechanisticControlError("propagation evidence fit is incomplete")
        self._required_actual = frozenset(workload.required_actual_steps)
        self.model_digest = _sha256_json({
            "policy_id": MECHANISTIC_CONTROL_POLICY_ID,
            "cost_model_id": MECHANISTIC_COST_MODEL_ID,
            "attention_model_id": MECHANISTIC_ATTENTION_MODEL_ID,
            "forecast_model_id": MECHANISTIC_FORECAST_MODEL_ID,
            "propagation_model_id": MECHANISTIC_PROPAGATION_MODEL_ID,
            "attention_implementation_id": ROUND215_ACTION_IMPLEMENTATION,
            "physical_model_digest": self._physical.model_sha256,
            "forecast_evidence_sha256": _file_sha256(_FORECAST_EVIDENCE_PATH),
            "round218_evidence_sha256": _file_sha256(_ROUND218_EVIDENCE_PATH),
            "propagation_evidence_sha256": _file_sha256(
                _PROPAGATION_EVIDENCE_PATH
            ),
            "risk_composition": (
                "coherent_forecast_run_amplitude_plus_squared_attention_rms_"
                "plus_isotropic_first_order_cross"
            ),
            "historical_schedule_used": False,
        })
        self._lambda_cache: dict[tuple[str, float], H3MechanisticPlan] = {}
        self._admission_cache: dict[float, tuple[float, H3MechanisticPlan]] = {}

    @property
    def calibrated_token_interval(self) -> tuple[int, int]:
        endpoints = self._forecast_evidence["samples"]
        tokens = [int(row["packed_tokens"]) for row in endpoints]
        return min(tokens), max(tokens)

    def _token_ood_distance(self) -> float:
        lower, upper = self.calibrated_token_interval
        if self.workload.packed_tokens < lower:
            return math.log(lower / self.workload.packed_tokens)
        if self.workload.packed_tokens > upper:
            return math.log(self.workload.packed_tokens / upper)
        return 0.0

    def _propagation_shape_ood_distance(self) -> float:
        """Continuous uncertainty radius away from the impulse workload.

        The first impulse experiment identifies phase shape at one geometry;
        it is not yet a production admission surface.  Log token distance is
        therefore carried as epistemic inflation instead of selecting a
        special long-video schedule.
        """

        reference = self._propagation_evidence["workload"]
        packed_distance = abs(math.log(
            self.workload.packed_tokens / float(reference["packed_tokens"])
        ))
        video_distance = abs(math.log(
            self.workload.video_tokens / float(reference["video_tokens"])
        ))
        return max(packed_distance, video_distance)

    @lru_cache(maxsize=1024)
    def _propagation_gain(
        self,
        *,
        mechanism: str,
        modality: str,
        step: int,
    ) -> _ForecastResponse:
        """Final-latent gain from one local approximation impulse.

        Gronwall propagation gives ``G(p)=exp(integral_p^1 L(s) ds)``.  The
        evidence fits ``log G`` with a low-order phase polynomial, rather than
        interpolating any historical acceleration schedule.
        """

        if mechanism not in ("single_forecast", "single_attention"):
            raise MechanisticControlError("unknown propagation mechanism")
        if modality not in ("audio", "video"):
            raise MechanisticControlError("unknown propagation modality")
        if not 0 <= step < self.workload.total_steps:
            raise MechanisticControlError("propagation step is out of range")
        fit = self._propagation_evidence["continuous_phase_gain_fits"][
            mechanism
        ][modality]
        coefficients = tuple(float(value) for value in fit["coefficients"])
        progress = step / max(1, self.workload.total_steps - 1)
        log_mean = sum(
            coefficient * progress ** power
            for power, coefficient in enumerate(coefficients)
        )
        mean = math.exp(log_mean)
        upper = mean * math.exp(
            float(fit["absolute_log_residual_max"])
            + self._propagation_shape_ood_distance()
        )
        return _ForecastResponse(mean=mean, upper=upper)

    @lru_cache(maxsize=64)
    def _integration_weights(self) -> tuple[float, ...]:
        """Normalized rectified-flow step widths for the H3 simple clock."""

        table = tuple(
            12.0 * ((index + 1) / 1000.0)
            / (1.0 + 11.0 * ((index + 1) / 1000.0))
            for index in range(1000)
        )
        stride = 1000.0 / self.workload.total_steps
        sigmas = tuple(
            table[-(1 + int(index * stride))]
            for index in range(self.workload.total_steps)
        ) + (0.0,)
        widths = tuple(
            abs(left - right) for left, right in zip(sigmas, sigmas[1:])
        )
        mean_width = sum(widths) / len(widths)
        return tuple(width / mean_width for width in widths)

    @lru_cache(maxsize=4096)
    def _forecast_response(
        self,
        *,
        modality: str,
        step: int,
        horizon: int,
    ) -> _ForecastResponse:
        if modality not in ("audio", "video") or horizon <= 0:
            raise MechanisticControlError("invalid Forecast response request")
        progress = step / max(1, self.workload.total_steps - 1)
        reference_tokens = float(self._forecast_evidence["reference_packed_tokens"])
        features = (
            1.0,
            progress,
            progress * progress,
            math.log(float(horizon)),
            progress * math.log(float(horizon)),
            math.log(self.workload.packed_tokens / reference_tokens),
        )
        fit = self._forecast_evidence["fit"][modality]
        coefficients = tuple(float(value) for value in fit["coefficients"])
        if len(coefficients) != len(features):
            raise MechanisticControlError("Forecast response coefficient mismatch")
        mean = math.exp(sum(c * x for c, x in zip(coefficients, features)))
        # q95 handles interpolation residual.  Leaving the measured token
        # interval adds a continuous multiplicative epistemic radius rather
        # than choosing a special long-video schedule.
        upper = mean * math.exp(
            float(fit["absolute_log_residual_q95"])
            + self._token_ood_distance()
        )
        return _ForecastResponse(mean=mean, upper=upper)

    @lru_cache(maxsize=32768)
    def _attention_error(
        self,
        *,
        step: int,
        layer: int,
        action: str,
    ) -> _ForecastResponse:
        if action == "dense":
            return _ForecastResponse(0.0, 0.0)
        if action not in ACTION_NAMES or not 0 <= layer < 50:
            raise MechanisticControlError("invalid Attention error request")
        progress = step / max(1, self.workload.total_steps - 1)
        values = tuple(
            float(anchor[action][layer])
            for anchor in self._physical.temporal_layer_errors
        )
        mean = _linear_surface(
            self._physical.temporal_progress_anchors,
            values,
            progress,
        )
        upper = mean * math.exp(self._token_ood_distance())
        return _ForecastResponse(mean=mean, upper=upper)

    @lru_cache(maxsize=512)
    def _layer_cost_ms(self, layer: int, action: str) -> float:
        for band, start, stop in LAYER_BANDS:
            if start <= layer < stop:
                return self._physical.actions[band][action].cost_ms / (stop - start)
        raise MechanisticControlError("H3 layer lies outside [0, 50)")

    @lru_cache(maxsize=131072)
    def _actual_local_action(
        self,
        *,
        step: int,
        layer: int,
        action: str,
        prior_forecast_run: int,
    ) -> _LocalAction:
        error = self._attention_error(step=step, layer=layer, action=action)
        audio_gain = self._propagation_gain(
            mechanism="single_attention", modality="audio", step=step
        )
        video_gain = self._propagation_gain(
            mechanism="single_attention", modality="video", step=step
        )
        audio_mean = error.mean * audio_gain.mean
        audio_upper = error.upper * audio_gain.upper
        video_mean = error.mean * video_gain.mean
        video_upper = error.upper * video_gain.upper
        attention_energy = audio_upper * audio_upper + video_upper * video_upper
        interaction = 0.0
        if prior_forecast_run:
            audio = self._run_forecast_amplitude(
                modality="audio",
                end_step=step - 1,
                horizon=prior_forecast_run,
            )[1]
            video = self._run_forecast_amplitude(
                modality="video",
                end_step=step - 1,
                horizon=prior_forecast_run,
            )[1]
            # Isotropic first-order composition: distribute each modality's
            # accumulated state-error norm across the 50 residual blocks.
            interaction = 2.0 * (
                audio * audio_upper + video * video_upper
            ) / math.sqrt(50.0)
        mean_energy = audio_mean * audio_mean + video_mean * video_mean
        return _LocalAction(
            name=action,
            cost_ms=self._layer_cost_ms(layer, action),
            risk=H3MechanisticRisk(
                attention_energy=attention_energy,
                interaction_energy=interaction,
                epistemic_energy=max(0.0, attention_energy - mean_energy),
            ),
        )

    def _choose_actual(
        self,
        *,
        step: int,
        prior_forecast_run: int,
        lagrange_multiplier: float | None,
    ) -> tuple[float, H3MechanisticRisk, tuple[str, ...]]:
        actions: list[str] = []
        risk = H3MechanisticRisk()
        cost_ms = self._physical.non_attention_ms
        for layer in range(50):
            candidates = tuple(
                self._actual_local_action(
                    step=step,
                    layer=layer,
                    action=action,
                    prior_forecast_run=prior_forecast_run,
                )
                for action in ACTION_NAMES
            )
            if lagrange_multiplier is None:
                selected = min(
                    candidates,
                    key=lambda item: (item.cost_ms, item.risk.total, item.name),
                )
            else:
                selected = min(
                    candidates,
                    key=lambda item: (
                        item.risk.total + lagrange_multiplier * item.cost_ms,
                        item.risk.total,
                        -item.cost_ms,
                        item.name,
                    ),
                )
            actions.append(selected.name)
            risk += selected.risk
            cost_ms += selected.cost_ms
        return cost_ms, risk, tuple(actions)

    @lru_cache(maxsize=4096)
    def _run_forecast_amplitude(
        self,
        *,
        modality: str,
        end_step: int,
        horizon: int,
    ) -> tuple[float, float]:
        """Coherent error amplitude accumulated inside one Forecast run.

        Consecutive secant predictions share the same tail histories, so
        treating their errors as independent squared noise is unjustified.
        The conservative correlated model sums step-width-weighted amplitudes
        and squares only when charging trajectory energy.
        """

        if horizon <= 0 or end_step < horizon - 1:
            raise MechanisticControlError("invalid Forecast run geometry")
        start = end_step - horizon + 1
        mean = upper = 0.0
        weights = self._integration_weights()
        for offset, step in enumerate(range(start, end_step + 1), start=1):
            response = self._forecast_response(
                modality=modality,
                step=step,
                horizon=offset,
            )
            gain = self._propagation_gain(
                mechanism="single_forecast",
                modality=modality,
                step=step,
            )
            weight = weights[step]
            mean += weight * response.mean * gain.mean
            upper += weight * response.upper * gain.upper
        return mean, upper

    @lru_cache(maxsize=4096)
    def _forecast_local(
        self,
        *,
        step: int,
        horizon: int,
    ) -> tuple[float, H3MechanisticRisk]:
        audio_mean, audio_upper = self._run_forecast_amplitude(
            modality="audio", end_step=step, horizon=horizon
        )
        video_mean, video_upper = self._run_forecast_amplitude(
            modality="video", end_step=step, horizon=horizon
        )
        if horizon == 1:
            prior_audio_mean = prior_audio_upper = 0.0
            prior_video_mean = prior_video_upper = 0.0
        else:
            prior_audio_mean, prior_audio_upper = self._run_forecast_amplitude(
                modality="audio", end_step=step - 1, horizon=horizon - 1
            )
            prior_video_mean, prior_video_upper = self._run_forecast_amplitude(
                modality="video", end_step=step - 1, horizon=horizon - 1
            )
        audio_energy = max(
            0.0,
            audio_upper * audio_upper - prior_audio_upper * prior_audio_upper,
        )
        video_energy = max(
            0.0,
            video_upper * video_upper - prior_video_upper * prior_video_upper,
        )
        audio_mean_energy = max(
            0.0,
            audio_mean * audio_mean - prior_audio_mean * prior_audio_mean,
        )
        video_mean_energy = max(
            0.0,
            video_mean * video_mean - prior_video_mean * prior_video_mean,
        )
        return self._physical.forecast_ms, H3MechanisticRisk(
            forecast_audio_energy=audio_energy,
            forecast_video_energy=video_energy,
            epistemic_energy=(
                max(0.0, audio_energy - audio_mean_energy)
                + max(0.0, video_energy - video_mean_energy)
            ),
        )

    @staticmethod
    def _state_objective(
        state: _DPState,
        lagrange_multiplier: float | None,
    ) -> tuple[float, float, float, tuple[tuple[Any, ...], ...]]:
        path_key = tuple(
            (
                choice.step_index,
                choice.actual,
                choice.forecast_horizon,
                *choice.attention_actions,
            )
            for choice in state.path
        )
        if lagrange_multiplier is None:
            return (state.cost_ms, state.risk.total, -state.cost_ms, path_key)
        return (
            state.risk.total + lagrange_multiplier * state.cost_ms,
            state.risk.total,
            -state.cost_ms,
            path_key,
        )

    def solve_lagrangian(
        self,
        lagrange_multiplier: float | None,
    ) -> H3MechanisticPlan:
        """Solve exactly for risk + lambda*wall-cost; ``None`` is cost-only."""

        if lagrange_multiplier is not None and (
            not math.isfinite(lagrange_multiplier) or lagrange_multiplier < 0.0
        ):
            raise MechanisticControlError(
                "Lagrange multiplier must be finite and non-negative"
            )
        cache_lambda = "cost_only" if lagrange_multiplier is None else repr(
            float(lagrange_multiplier)
        )
        cache_key = (self.workload.digest, cache_lambda)
        cached = self._lambda_cache.get(cache_key)
        if cached is not None:
            return cached

        initial = _DPState(
            forecast_run=0,
            history_depth=0,
            cost_ms=0.0,
            risk=H3MechanisticRisk(),
            path=(),
        )
        states: dict[tuple[int, int], _DPState] = {(0, 0): initial}
        for step in range(self.workload.total_steps):
            proposed: dict[tuple[int, int], _DPState] = {}
            for state in states.values():
                actual_cost, actual_risk, actions = self._choose_actual(
                    step=step,
                    prior_forecast_run=state.forecast_run,
                    lagrange_multiplier=lagrange_multiplier,
                )
                actual = _DPState(
                    forecast_run=0,
                    history_depth=min(2, state.history_depth + 1),
                    cost_ms=state.cost_ms + actual_cost,
                    risk=state.risk + actual_risk,
                    path=state.path + (H3MechanisticStepChoice(
                        step_index=step,
                        actual=True,
                        attention_actions=actions,
                    ),),
                )
                self._retain_state(
                    proposed,
                    actual,
                    lagrange_multiplier=lagrange_multiplier,
                )

                forecast_allowed = (
                    self.workload.allow_forecast
                    and state.history_depth >= 2
                    and step not in self._required_actual
                    # The directional tail requires a following full-stack
                    # correction; a terminal Forecast has no physical anchor.
                    and step < self.workload.total_steps - 1
                )
                if forecast_allowed:
                    horizon = state.forecast_run + 1
                    forecast_cost, forecast_risk = self._forecast_local(
                        step=step,
                        horizon=horizon,
                    )
                    forecast = _DPState(
                        forecast_run=horizon,
                        history_depth=state.history_depth,
                        cost_ms=state.cost_ms + forecast_cost,
                        risk=state.risk + forecast_risk,
                        path=state.path + (H3MechanisticStepChoice(
                            step_index=step,
                            actual=False,
                            forecast_horizon=horizon,
                        ),),
                    )
                    self._retain_state(
                        proposed,
                        forecast,
                        lagrange_multiplier=lagrange_multiplier,
                    )
            states = proposed
            if not states:
                raise MechanisticControlError(
                    f"mechanistic trajectory became infeasible at step {step}"
                )

        optimum = min(
            states.values(),
            key=lambda state: self._state_objective(
                state, lagrange_multiplier
            ),
        )
        choice_document = [
            {
                "step": choice.step_index,
                "actual": choice.actual,
                "forecast_horizon": choice.forecast_horizon,
                "attention_actions": choice.attention_actions,
            }
            for choice in optimum.path
        ]
        choice_digest = _sha256_json(choice_document)
        objective_value = (
            optimum.cost_ms
            if lagrange_multiplier is None
            else optimum.risk.total + lagrange_multiplier * optimum.cost_ms
        )
        certificate = H3MechanisticCertificate(
            solver_id=MECHANISTIC_SOLVER_ID,
            formal_scope=(
                "exact finite DP for the declared additive UCB perturbation-energy "
                "plus measured-wall-cost Lagrangian; every Attention layer and "
                "Actual/Forecast trajectory transition is jointly optimized"
            ),
            lagrange_multiplier=(
                "cost_only" if lagrange_multiplier is None else lagrange_multiplier
            ),
            objective_value=objective_value,
            model_digest=self.model_digest,
            workload_digest=self.workload.digest,
            choice_digest=choice_digest,
            historical_schedule_used=False,
            exact_for_declared_lagrangian_problem=True,
        )
        plan = H3MechanisticPlan(
            workload=self.workload,
            choices=optimum.path,
            predicted_cost_ms=optimum.cost_ms,
            modeled_risk=optimum.risk,
            certificate=certificate,
        )
        self._lambda_cache[cache_key] = plan
        return plan

    def _retain_state(
        self,
        states: dict[tuple[int, int], _DPState],
        candidate: _DPState,
        *,
        lagrange_multiplier: float | None,
    ) -> None:
        key = (candidate.forecast_run, candidate.history_depth)
        incumbent = states.get(key)
        if incumbent is None or self._state_objective(
            candidate, lagrange_multiplier
        ) < self._state_objective(incumbent, lagrange_multiplier):
            states[key] = candidate

    def _find_finite_cost_only_lambda(self) -> tuple[float, H3MechanisticPlan]:
        cost_only = self.solve_lagrangian(None)
        multiplier = 1.0e-12
        for _ in range(96):
            plan = self.solve_lagrangian(multiplier)
            if plan.certificate.choice_digest == cost_only.certificate.choice_digest:
                return multiplier, plan
            multiplier *= 2.0
        raise MechanisticControlError(
            "failed to bracket the finite cost-only Lagrangian endpoint"
        )

    def _fastest_admitted(
        self,
        maximum_risk: float,
    ) -> tuple[float, H3MechanisticPlan]:
        cache_key = round(float(maximum_risk), 15)
        cached = self._admission_cache.get(cache_key)
        if cached is not None:
            return cached
        dense = self.solve_lagrangian(0.0)
        if maximum_risk + 1.0e-12 < dense.modeled_risk.total:
            raise MechanisticControlError(
                "admitted risk lies below the minimum-risk Dense endpoint"
            )
        high_lambda, cost_only = self._find_finite_cost_only_lambda()
        if cost_only.modeled_risk.total <= maximum_risk + 1.0e-12:
            result = (high_lambda, cost_only)
            self._admission_cache[cache_key] = result
            return result

        low_lambda = 0.0
        low_plan = dense
        for _ in range(48):
            middle = (low_lambda + high_lambda) * 0.5
            candidate = self.solve_lagrangian(middle)
            if candidate.modeled_risk.total <= maximum_risk + 1.0e-12:
                low_lambda, low_plan = middle, candidate
            else:
                high_lambda = middle
        result = (low_lambda, low_plan)
        self._admission_cache[cache_key] = result
        return result

    def plan_for_acceleration(
        self,
        *,
        acceleration: float,
        admission: H3MechanisticAdmission,
    ) -> H3MechanisticPlan:
        """Select one supported Pareto point from a Human-admitted risk region."""

        try:
            acceleration = float(acceleration)
        except (TypeError, ValueError) as error:
            raise MechanisticControlError("acceleration must be numeric") from error
        if not math.isfinite(acceleration) or not 0.0 <= acceleration <= 100.0:
            raise MechanisticControlError("acceleration must lie inside [0, 100]")
        acceleration = round(acceleration, 1)
        dense = self.solve_lagrangian(0.0)
        if acceleration == 0.0:
            return H3MechanisticPlan(
                workload=dense.workload,
                choices=dense.choices,
                predicted_cost_ms=dense.predicted_cost_ms,
                modeled_risk=dense.modeled_risk,
                certificate=dense.certificate,
                acceleration=acceleration,
                target_cost_ms=dense.predicted_cost_ms,
                admitted_risk_limit=admission.maximum_modeled_risk,
                admission_id=admission.calibration_id,
            )
        fast_lambda, fastest = self._fastest_admitted(
            admission.maximum_modeled_risk
        )
        if dense.certificate.choice_digest == fastest.certificate.choice_digest:
            selected = dense
            target_cost = dense.predicted_cost_ms
        elif acceleration == 100.0:
            selected = fastest
            target_cost = fastest.predicted_cost_ms
        else:
            # Public semantics: acceleration is exactly the fraction of the
            # Human-admitted compute-saving interval requested by the user.
            # It does not select a named schedule or a hand-shaped quality
            # curve; the joint optimizer decides how to spend that budget.
            progress = acceleration / 100.0
            target_cost = (
                dense.predicted_cost_ms
                - (dense.predicted_cost_ms - fastest.predicted_cost_ms) * progress
            )
            low_lambda = 0.0
            high_lambda = fast_lambda
            high_plan = fastest
            for _ in range(48):
                middle = (low_lambda + high_lambda) * 0.5
                candidate = self.solve_lagrangian(middle)
                if candidate.predicted_cost_ms <= target_cost + 1.0e-9:
                    high_lambda, high_plan = middle, candidate
                else:
                    low_lambda = middle
            selected = high_plan
        return H3MechanisticPlan(
            workload=selected.workload,
            choices=selected.choices,
            predicted_cost_ms=selected.predicted_cost_ms,
            modeled_risk=selected.modeled_risk,
            certificate=selected.certificate,
            acceleration=acceleration,
            target_cost_ms=target_cost,
            admitted_risk_limit=admission.maximum_modeled_risk,
            admission_id=admission.calibration_id,
        )

    def plan_for_cost_budget(
        self,
        *,
        maximum_cost_ms: float,
    ) -> H3MechanisticPlan:
        """Best supported numerical-risk point meeting an exploratory budget.

        This method is used for same-speed mechanism A/B experiments.  It has
        no Human admission and therefore cannot be published as a creator-dial
        endpoint by itself.
        """

        try:
            maximum_cost_ms = float(maximum_cost_ms)
        except (TypeError, ValueError) as error:
            raise MechanisticControlError("cost budget must be numeric") from error
        if not math.isfinite(maximum_cost_ms) or maximum_cost_ms <= 0.0:
            raise MechanisticControlError("cost budget must be finite and positive")
        dense = self.solve_lagrangian(0.0)
        if dense.predicted_cost_ms <= maximum_cost_ms + 1.0e-9:
            selected = dense
        else:
            high_lambda, cheapest = self._find_finite_cost_only_lambda()
            if cheapest.predicted_cost_ms > maximum_cost_ms + 1.0e-9:
                raise MechanisticControlError(
                    "cost budget lies below the minimum physical H3 path"
                )
            low_lambda = 0.0
            high_plan = cheapest
            for _ in range(48):
                middle = (low_lambda + high_lambda) * 0.5
                candidate = self.solve_lagrangian(middle)
                if candidate.predicted_cost_ms <= maximum_cost_ms + 1.0e-9:
                    high_lambda, high_plan = middle, candidate
                else:
                    low_lambda = middle
            selected = high_plan
        return H3MechanisticPlan(
            workload=selected.workload,
            choices=selected.choices,
            predicted_cost_ms=selected.predicted_cost_ms,
            modeled_risk=selected.modeled_risk,
            certificate=selected.certificate,
            target_cost_ms=maximum_cost_ms,
        )

    def verify(self, plan: H3MechanisticPlan) -> H3MechanisticVerification:
        reasons: list[str] = []
        certificate = plan.certificate
        if certificate.model_digest != self.model_digest:
            reasons.append("model digest mismatch")
        if certificate.workload_digest != self.workload.digest:
            reasons.append("workload digest mismatch")
        if certificate.historical_schedule_used:
            reasons.append("historical schedule unexpectedly entered solver")
        multiplier = certificate.lagrange_multiplier
        replay = self.solve_lagrangian(
            None if multiplier == "cost_only" else float(multiplier)
        )
        if replay.certificate.choice_digest != certificate.choice_digest:
            reasons.append("choice digest does not replay")
        if not math.isclose(
            replay.predicted_cost_ms,
            plan.predicted_cost_ms,
            rel_tol=0.0,
            abs_tol=1.0e-8,
        ):
            reasons.append("predicted cost does not replay")
        if not math.isclose(
            replay.modeled_risk.total,
            plan.modeled_risk.total,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):
            reasons.append("modeled risk does not replay")
        if plan.admitted_risk_limit is not None and (
            plan.modeled_risk.total > plan.admitted_risk_limit + 1.0e-12
        ):
            reasons.append("plan exceeds admitted Human-calibrated risk")
        return H3MechanisticVerification(not reasons, tuple(reasons))

    def evaluate_external_schedule(
        self,
        *,
        evaluation_id: str,
        actual_step_indices: tuple[int, ...],
        attention_action_schedule: Iterable[tuple[int, int, str]],
    ) -> H3MechanisticScheduleEvaluation:
        """Project one historical schedule through the mechanism model.

        Historical actions are canonicalised only to study whether the model
        explains Human evidence.  A schedule produced by another kernel rail
        is explicitly marked implementation-mismatched and cannot calibrate a
        production admission boundary for the Round215 solver.
        """

        if not evaluation_id:
            raise MechanisticControlError("external evaluation id is required")
        actual = tuple(sorted(set(int(step) for step in actual_step_indices)))
        if actual != actual_step_indices or any(
            step < 0 or step >= self.workload.total_steps for step in actual
        ):
            raise MechanisticControlError("external Actual schedule is invalid")
        actual_set = frozenset(actual)
        raw_schedule = tuple(
            (int(step), int(layer), str(action))
            for step, layer, action in attention_action_schedule
        )
        canonical: dict[tuple[int, int], str] = {}
        implementation_ids: set[str] = set()
        for step, layer, action in raw_schedule:
            if step not in actual_set:
                continue
            if not 0 <= layer < 50:
                raise MechanisticControlError("external Attention layer is invalid")
            if ":" in action:
                implementation, name = action.rsplit(":", 1)
                implementation_ids.add(implementation)
            else:
                name = action
                implementation_ids.add("dense" if name == "dense" else "unbound")
            if name not in ACTION_NAMES:
                raise MechanisticControlError(
                    f"external Attention action is unsupported: {action}"
                )
            key = (step, layer)
            if key in canonical:
                raise MechanisticControlError("external schedule contains duplicate cells")
            canonical[key] = name
        missing = tuple(
            (step, layer)
            for step in actual
            for layer in range(50)
            if (step, layer) not in canonical
        )
        if missing:
            raise MechanisticControlError(
                f"external schedule omits {len(missing)} Actual Attention cells"
            )
        if not {0, 1}.issubset(actual_set):
            raise MechanisticControlError(
                "external Forecast schedule lacks the two physical history Actuals"
            )

        risk = H3MechanisticRisk()
        cost_ms = 0.0
        run = maximum_run = 0
        for step in range(self.workload.total_steps):
            if step not in actual_set:
                run += 1
                maximum_run = max(maximum_run, run)
                local_cost, local_risk = self._forecast_local(
                    step=step,
                    horizon=run,
                )
                cost_ms += local_cost
                risk += local_risk
                continue
            cost_ms += self._physical.non_attention_ms
            for layer in range(50):
                local = self._actual_local_action(
                    step=step,
                    layer=layer,
                    action=canonical[(step, layer)],
                    prior_forecast_run=run,
                )
                cost_ms += local.cost_ms
                risk += local.risk
            run = 0

        source_ids = tuple(sorted(implementation_ids))
        implementation_match = all(
            identity in ("dense", _ACTION_RUNTIME_PREFIX)
            for identity in source_ids
        )
        digest = _sha256_json({
            "actual_step_indices": actual,
            "canonical_attention": tuple(
                (step, layer, canonical[(step, layer)])
                for step in actual
                for layer in range(50)
            ),
        })
        forecasts = tuple(
            step for step in range(self.workload.total_steps) if step not in actual_set
        )
        return H3MechanisticScheduleEvaluation(
            evaluation_id=evaluation_id,
            predicted_cost_ms=cost_ms,
            modeled_risk=risk,
            actual_step_indices=actual,
            forecast_step_indices=forecasts,
            maximum_forecast_run=maximum_run,
            source_implementation_ids=source_ids,
            calibration_implementation_match=implementation_match,
            schedule_digest=digest,
            model_digest=self.model_digest,
        )


def verify_mechanistic_pareto_order(
    plans: Iterable[H3MechanisticPlan],
) -> H3MechanisticVerification:
    """Reject cost/risk dominance and non-monotone creator-dial sequences."""

    rows = tuple(plans)
    reasons: list[str] = []
    for previous, current in zip(rows, rows[1:]):
        if previous.acceleration is None or current.acceleration is None:
            reasons.append("Pareto order requires acceleration-tagged plans")
            break
        if current.acceleration <= previous.acceleration:
            reasons.append("accelerations are not strictly increasing")
        if current.predicted_cost_ms > previous.predicted_cost_ms + 1.0e-8:
            reasons.append("higher acceleration increased predicted cost")
        if current.modeled_risk.total + 1.0e-12 < previous.modeled_risk.total:
            reasons.append("higher acceleration unexpectedly reduced modeled risk")
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            left_dominates = (
                left.predicted_cost_ms <= right.predicted_cost_ms + 1.0e-8
                and left.modeled_risk.total <= right.modeled_risk.total + 1.0e-12
                and (
                    left.predicted_cost_ms < right.predicted_cost_ms - 1.0e-8
                    or left.modeled_risk.total < right.modeled_risk.total - 1.0e-12
                )
            )
            if left_dominates and left.acceleration != right.acceleration:
                reasons.append("creator-dial sequence contains a dominated plan")
                break
    return H3MechanisticVerification(not reasons, tuple(dict.fromkeys(reasons)))


__all__ = [
    "H3MechanisticAdmission",
    "H3MechanisticCertificate",
    "H3MechanisticControlModel",
    "H3MechanisticPlan",
    "H3MechanisticRisk",
    "H3MechanisticScheduleEvaluation",
    "H3MechanisticStepChoice",
    "H3MechanisticVerification",
    "H3MechanisticWorkload",
    "MECHANISTIC_ATTENTION_MODEL_ID",
    "MECHANISTIC_CONTROL_POLICY_ID",
    "MECHANISTIC_CONTROL_SCHEMA",
    "MECHANISTIC_COST_MODEL_ID",
    "MECHANISTIC_FORECAST_MODEL_ID",
    "MECHANISTIC_PROPAGATION_MODEL_ID",
    "MECHANISTIC_SOLVER_ID",
    "MechanisticControlError",
    "verify_mechanistic_pareto_order",
]
