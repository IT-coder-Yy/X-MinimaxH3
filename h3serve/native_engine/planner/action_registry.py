"""Versioned execution-action registry for the V19 H3 planner.

V18 could optimise measurements produced by one Attention implementation and
execute a different implementation with the same nominal Top-K fraction.  A
V19 action is therefore identified by its physical implementation, not by a
friendly action name.  Cost, risk, executor and certificate bindings must all
name the same registry entry and registry digest.

This module is control-plane only.  It deliberately has no CUDA dependency so
registry/certificate validation can run before a model is loaded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import Iterable, Mapping


ACTION_REGISTRY_SCHEMA = "h3_v19_action_registry_v1"


class ActionRegistryError(ValueError):
    """An action or evidence binding is ambiguous, stale, or unsupported."""


class ActionKind(str, Enum):
    DENSE_ATTENTION = "dense_attention"
    SPARSE_ATTENTION = "sparse_attention"
    FORECAST_COMPOSITE = "forecast_composite"
    SEGMENT_REUSE = "segment_reuse"


class EvidenceStatus(str, Enum):
    UNBOUND = "unbound"
    CALIBRATING = "calibrating"
    CALIBRATED = "calibrated"
    HUMAN_REVIEWED = "human_reviewed"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ActionWorkloadEnvelope:
    """The calibration envelope, never the model input capability contract."""

    model_variants: tuple[str, ...] = ("base", "lora")
    service_families: tuple[str, ...] = ("first_last", "reference")
    min_packed_tokens: int = 1
    max_packed_tokens: int | None = None
    condition_count_min: int = 0
    condition_count_max: int | None = None
    device_arches: tuple[str, ...] = ("sm89",)

    def __post_init__(self) -> None:
        if not self.model_variants or len(set(self.model_variants)) != len(
            self.model_variants
        ):
            raise ActionRegistryError("model variants must be non-empty and unique")
        if not self.service_families or len(set(self.service_families)) != len(
            self.service_families
        ):
            raise ActionRegistryError("service families must be non-empty and unique")
        if self.min_packed_tokens <= 0:
            raise ActionRegistryError("minimum packed tokens must be positive")
        if (
            self.max_packed_tokens is not None
            and self.max_packed_tokens < self.min_packed_tokens
        ):
            raise ActionRegistryError("packed-token envelope is inverted")
        if self.condition_count_min < 0:
            raise ActionRegistryError("minimum condition count cannot be negative")
        if (
            self.condition_count_max is not None
            and self.condition_count_max < self.condition_count_min
        ):
            raise ActionRegistryError("condition-count envelope is inverted")

    def contains(
        self,
        *,
        model_variant: str,
        service_family: str,
        packed_tokens: int,
        condition_count: int,
        device_arch: str,
    ) -> bool:
        return (
            model_variant in self.model_variants
            and service_family in self.service_families
            and packed_tokens >= self.min_packed_tokens
            and (
                self.max_packed_tokens is None
                or packed_tokens <= self.max_packed_tokens
            )
            and condition_count >= self.condition_count_min
            and (
                self.condition_count_max is None
                or condition_count <= self.condition_count_max
            )
            and device_arch in self.device_arches
        )


@dataclass(frozen=True, slots=True)
class RegisteredAction:
    """One physically distinct execution action.

    ``implementation_id`` must identify the path actually dispatched by the
    model backend.  Two actions with the same nominal Top-K but different
    masks, compensation, Head rails, or kernels are different implementations.
    """

    action_id: str
    implementation_id: str
    kind: ActionKind
    executor_id: str
    canonical_actions: tuple[str, ...]
    exact: bool
    evidence_status: EvidenceStatus
    envelope: ActionWorkloadEnvelope = ActionWorkloadEnvelope()
    calibration_ids: tuple[str, ...] = ()
    risk_model_ids: tuple[str, ...] = ()
    human_evidence_ids: tuple[str, ...] = ()
    planner_eligible: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("action_id", self.action_id),
            ("implementation_id", self.implementation_id),
            ("executor_id", self.executor_id),
        ):
            if not value or value.strip() != value or any(ch.isspace() for ch in value):
                raise ActionRegistryError(f"{name} must be a non-empty stable identifier")
        if not self.canonical_actions:
            raise ActionRegistryError("an action must expose at least one canonical action")
        if len(set(self.canonical_actions)) != len(self.canonical_actions):
            raise ActionRegistryError("canonical actions must be unique")
        if self.evidence_status is EvidenceStatus.REJECTED and self.planner_eligible:
            raise ActionRegistryError("a rejected action cannot be planner eligible")
        if self.planner_eligible and not self.calibration_ids:
            raise ActionRegistryError("a planner-eligible action requires exact calibration")
        if self.planner_eligible and self.evidence_status not in (
            EvidenceStatus.CALIBRATED,
            EvidenceStatus.HUMAN_REVIEWED,
        ):
            raise ActionRegistryError(
                "a planner-eligible action must be calibrated or Human reviewed"
            )


@dataclass(frozen=True, slots=True)
class ActionEvidenceBinding:
    """Identity carried by a cost, risk, or Human evidence artifact."""

    action_id: str
    implementation_id: str
    registry_digest: str
    evidence_id: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if len(self.registry_digest) != 64 or len(self.evidence_sha256) != 64:
            raise ActionRegistryError("evidence bindings require SHA256 digests")
        try:
            int(self.registry_digest, 16)
            int(self.evidence_sha256, 16)
        except ValueError as error:
            raise ActionRegistryError("evidence binding digest is not hexadecimal") from error


class ActionRegistry:
    """Strict registry with deterministic serialization and no fuzzy lookup."""

    def __init__(self, actions: Iterable[RegisteredAction] = ()) -> None:
        self._by_action: dict[str, RegisteredAction] = {}
        self._by_implementation: dict[str, RegisteredAction] = {}
        for action in actions:
            self.register(action)

    def register(self, action: RegisteredAction) -> None:
        if action.action_id in self._by_action:
            raise ActionRegistryError(f"duplicate action id: {action.action_id}")
        if action.implementation_id in self._by_implementation:
            raise ActionRegistryError(
                f"duplicate implementation id: {action.implementation_id}"
            )
        self._by_action[action.action_id] = action
        self._by_implementation[action.implementation_id] = action

    @property
    def actions(self) -> tuple[RegisteredAction, ...]:
        return tuple(self._by_action[key] for key in sorted(self._by_action))

    def resolve(self, action_id: str) -> RegisteredAction:
        try:
            return self._by_action[action_id]
        except KeyError as error:
            raise ActionRegistryError(f"unregistered action: {action_id}") from error

    def resolve_implementation(self, implementation_id: str) -> RegisteredAction:
        try:
            return self._by_implementation[implementation_id]
        except KeyError as error:
            raise ActionRegistryError(
                f"unregistered implementation: {implementation_id}"
            ) from error

    def to_dict(self) -> dict[str, object]:
        def document(action: RegisteredAction) -> dict[str, object]:
            row = asdict(action)
            row["kind"] = action.kind.value
            row["evidence_status"] = action.evidence_status.value
            return row

        return {
            "schema_version": ACTION_REGISTRY_SCHEMA,
            "actions": [document(action) for action in self.actions],
        }

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def verify_evidence_binding(self, binding: ActionEvidenceBinding) -> RegisteredAction:
        if binding.registry_digest != self.digest:
            raise ActionRegistryError("evidence was produced for a different registry")
        action = self.resolve(binding.action_id)
        if action.implementation_id != binding.implementation_id:
            raise ActionRegistryError(
                "evidence implementation does not match the registered action"
            )
        known = set(action.calibration_ids) | set(action.risk_model_ids) | set(
            action.human_evidence_ids
        )
        if binding.evidence_id not in known:
            raise ActionRegistryError(
                "evidence id is not explicitly bound to the registered action"
            )
        return action

    def planner_actions_for(
        self,
        *,
        model_variant: str,
        service_family: str,
        packed_tokens: int,
        condition_count: int,
        device_arch: str = "sm89",
    ) -> tuple[RegisteredAction, ...]:
        return tuple(
            action
            for action in self.actions
            if action.planner_eligible
            and action.envelope.contains(
                model_variant=model_variant,
                service_family=service_family,
                packed_tokens=packed_tokens,
                condition_count=condition_count,
                device_arch=device_arch,
            )
        )


def build_v19_bootstrap_registry(
    *,
    implementation_ids: Mapping[str, str],
) -> ActionRegistry:
    """Register current implementations without pretending evidence parity.

    The mapping is injected to avoid importing ``joint_global_dp`` here and
    creating a planner import cycle.  Only the Round215 action currently has a
    matching implementation-specific calibration.  Historical Head rails are
    comparators until NG-001/NG-012 produce exact calibration artifacts.
    """

    required = {
        "fixed_topk",
        "round215",
        "round188",
        "round228",
        "round229",
    }
    if set(implementation_ids) != required:
        missing = sorted(required - set(implementation_ids))
        extra = sorted(set(implementation_ids) - required)
        raise ActionRegistryError(
            f"bootstrap implementation mapping mismatch; missing={missing}, extra={extra}"
        )
    canonical = (
        "sparse_topk_0.0625",
        "sparse_topk_0.1",
        "sparse_topk_0.25",
        "sparse_topk_0.5",
        "dense",
    )
    actions = (
        RegisteredAction(
            action_id="h3.attention.dense.sage_per_warp.sm89.v1",
            implementation_id="sage_dense_per_warp_sm89_v1",
            kind=ActionKind.DENSE_ATTENTION,
            executor_id="dense",
            canonical_actions=("dense",),
            exact=False,
            evidence_status=EvidenceStatus.CALIBRATED,
            calibration_ids=("v19_dense_full_head_cost_v1",),
            risk_model_ids=("dense_zero_incremental_acceleration_risk_v1",),
            planner_eligible=True,
            notes=(
                "Dense Attention topology and V19 fallback; numerical parity is "
                "defined against the current per-warp Sage baseline, not FP32."
            ),
        ),
        RegisteredAction(
            action_id="h3.attention.fixed_topk.v1",
            implementation_id=implementation_ids["fixed_topk"],
            kind=ActionKind.SPARSE_ATTENTION,
            executor_id="direct",
            canonical_actions=canonical,
            exact=False,
            evidence_status=EvidenceStatus.CALIBRATING,
            notes="Historical fixed Top-K path; not V19 calibrated.",
        ),
        RegisteredAction(
            action_id="h3.attention.interaction_hybrid.round215.v1",
            implementation_id=implementation_ids["round215"],
            kind=ActionKind.SPARSE_ATTENTION,
            executor_id="round215",
            canonical_actions=canonical,
            exact=False,
            evidence_status=EvidenceStatus.CALIBRATED,
            calibration_ids=("v19_round215_full_head_cost_error_v1",),
            risk_model_ids=("round215_dense_relative_rms_v1",),
            planner_eligible=True,
            notes="Calibration-matched legacy action; Human risk remains provisional.",
        ),
        RegisteredAction(
            action_id="h3.attention.mtcr_head_rail.round188.v1",
            implementation_id=implementation_ids["round188"],
            kind=ActionKind.SPARSE_ATTENTION,
            executor_id="frontier",
            canonical_actions=canonical,
            exact=False,
            evidence_status=EvidenceStatus.HUMAN_REVIEWED,
            calibration_ids=("v19_round188_full_head_cost_error_v1",),
            human_evidence_ids=("round188_human_comparator_v1",),
            notes="Human comparator; exact implementation cost table is still missing.",
        ),
        RegisteredAction(
            action_id="h3.attention.mtcr_head_rail.round228.v1",
            implementation_id=implementation_ids["round228"],
            kind=ActionKind.SPARSE_ATTENTION,
            executor_id="fastfrontier",
            canonical_actions=canonical,
            exact=False,
            evidence_status=EvidenceStatus.UNBOUND,
            calibration_ids=("v19_round228_full_head_cost_error_v1",),
            notes="Execution-tax experiment; not independently calibrated or reviewed.",
        ),
        RegisteredAction(
            action_id="h3.attention.mtcr_head_rail.round229.v1",
            implementation_id=implementation_ids["round229"],
            kind=ActionKind.SPARSE_ATTENTION,
            executor_id="forecastfrontier",
            canonical_actions=canonical,
            exact=False,
            evidence_status=EvidenceStatus.UNBOUND,
            calibration_ids=("v19_round229_attention_cost_error_v1",),
            notes="V18 runtime path; forecast anchor actions remain outside its certificate.",
        ),
        RegisteredAction(
            action_id="h3.forecast.directional.anchor3.round229.v1",
            implementation_id="directional_forecast_anchor3_round229_v1",
            kind=ActionKind.FORECAST_COMPOSITE,
            executor_id="directional_forecast",
            canonical_actions=("forecast",),
            exact=False,
            evidence_status=EvidenceStatus.UNBOUND,
            calibration_ids=("v19_round229_forecast_composite_cost_v1",),
            notes="Must be calibrated as run+anchor+extrapolator+correction before V19 use.",
        ),
        RegisteredAction(
            action_id="h3.cache.coordinate_segment_residual.v1",
            implementation_id="coordinate_segment_residual_v1",
            kind=ActionKind.SEGMENT_REUSE,
            executor_id="coordinate_segment_cache",
            canonical_actions=("reuse", "refresh"),
            exact=False,
            evidence_status=EvidenceStatus.REJECTED,
            human_evidence_ids=("round24_26_human_rejection_v1",),
            notes="Retained for provenance; audio drift, blur and ghosting prohibit planning.",
        ),
    )
    return ActionRegistry(actions)


__all__ = [
    "ACTION_REGISTRY_SCHEMA",
    "ActionEvidenceBinding",
    "ActionKind",
    "ActionRegistry",
    "ActionRegistryError",
    "ActionWorkloadEnvelope",
    "EvidenceStatus",
    "RegisteredAction",
    "build_v19_bootstrap_registry",
]
