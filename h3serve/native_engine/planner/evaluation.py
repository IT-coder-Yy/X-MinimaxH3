"""Executable evaluation contract for the joint H3 acceleration scheduler.

The contract keeps three concerns separate:

* effectiveness: hard invariants, formal surrogate optimality, and Human
  continuous-play evidence;
* efficiency: cold/warm planning overhead and bounded online reserve;
* simplicity: exactly two creator-facing controls and versioned deterministic
  plans.

Automatic gates may prove structural and finite-optimization properties.  A
new policy remains ``pending_human_review`` until its own videos receive Human
labels; historical labels constrain the design but never transfer acceptance
to a new scheduler by analogy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics
import time
from typing import Iterable, Literal

from .joint_acceleration import (
    DEFAULT_JOINT_POLICY,
    H3JointAccelerationScheduler,
    JOINT_POLICY_V1_HEURISTIC,
    JOINT_POLICY_V3_GLOBAL_DP,
    JOINT_POLICY_V4_EVIDENCE_GLOBAL_DP,
    JOINT_POLICY_V5_CALIBRATION_MATCHED_GLOBAL_DP,
    JOINT_POLICY_V6_CAUSAL_ISLAND_GLOBAL_DP,
    JOINT_POLICY_V7_LAYER_RISK_GLOBAL_DP,
    JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP,
    JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP,
    JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP,
    JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
    JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
    JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP,
    JOINT_MECHANICAL_BASELINE_ID,
    ROUND215_ACTION_IMPLEMENTATION,
    ROUND216_CAUSAL_ISLAND_CONSTRAINT,
    ROUND224_ADAPTIVE_LATENCY_CONSTRAINT,
    ROUND215_LAYER_RISK_MODEL,
    ROUND218_PHASE_LAYER_RISK_MODEL,
    ROUND219_BOUNDED_ONLINE_GUARD,
    ROUND220_PHASE_SENTINEL_GUARD,
    ROUND221_CALIBRATED_GROWTH_GUARD,
    ROUND223_RESERVE_REBATE_GUARD,
    JointWorkloadContext,
    JointAccelerationPlan,
    clear_joint_plan_cache,
    verify_joint_plan_certificate,
)
from .online_guard import (
    ROUND221_CALIBRATION_FILE,
    ROUND221_PROBE_GROWTH_CALIBRATION,
    allocate_phase_sentinels,
    load_probe_growth_calibration,
)


GateStatus = Literal["pass", "fail", "pending"]
EVALUATION_SCHEMA = "h3_joint_scheduler_evaluation_v3"

_GLOBAL_DP_POLICIES = frozenset(
    (
        JOINT_POLICY_V3_GLOBAL_DP,
        JOINT_POLICY_V4_EVIDENCE_GLOBAL_DP,
        JOINT_POLICY_V5_CALIBRATION_MATCHED_GLOBAL_DP,
        JOINT_POLICY_V6_CAUSAL_ISLAND_GLOBAL_DP,
        JOINT_POLICY_V7_LAYER_RISK_GLOBAL_DP,
        JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP,
        JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP,
        JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP,
        JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
        JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
        JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP,
    )
)

# V3/V4 predate the Round215 interaction-hybrid calibration and intentionally
# retain their fixed-TopK action identity.  E10 is a hard claim only for the
# measured-action lineage; applying it retroactively would make the evaluator
# reject historically valid, explicitly versioned policies for using the very
# implementation named by their own certificate.
_ROUND215_MEASURED_POLICIES = frozenset(
    (
        JOINT_POLICY_V5_CALIBRATION_MATCHED_GLOBAL_DP,
        JOINT_POLICY_V6_CAUSAL_ISLAND_GLOBAL_DP,
        JOINT_POLICY_V7_LAYER_RISK_GLOBAL_DP,
        JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP,
        JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP,
        JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP,
        JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
        JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
        JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP,
    )
)

_ONLINE_GUARDS = {
    JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP: ROUND219_BOUNDED_ONLINE_GUARD,
    JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP: ROUND220_PHASE_SENTINEL_GUARD,
    JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP: (
        ROUND221_CALIBRATED_GROWTH_GUARD
    ),
    JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP: ROUND223_RESERVE_REBATE_GUARD,
    JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP: (
        ROUND221_CALIBRATED_GROWTH_GUARD
    ),
}


@dataclass(frozen=True, slots=True)
class EvaluationGate:
    gate_id: str
    category: Literal["effective", "efficient", "simple"]
    status: GateStatus
    hard: bool
    observation: object
    requirement: str
    evidence: str


@dataclass(frozen=True, slots=True)
class SchedulerEvaluationReport:
    schema_version: str
    policy_id: str
    overall_status: Literal["pass", "fail", "pending_human_review"]
    gates: tuple[EvaluationGate, ...]
    plan_cases: tuple[dict[str, object], ...]
    human_evidence: dict[str, object]
    elapsed_seconds: float

    @property
    def hard_failures(self) -> tuple[EvaluationGate, ...]:
        return tuple(gate for gate in self.gates if gate.hard and gate.status == "fail")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "overall_status": self.overall_status,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "summary": {
                "pass": sum(gate.status == "pass" for gate in self.gates),
                "fail": sum(gate.status == "fail" for gate in self.gates),
                "pending": sum(gate.status == "pending" for gate in self.gates),
                "hard_failures": [gate.gate_id for gate in self.hard_failures],
            },
            "gates": [asdict(gate) for gate in self.gates],
            "plan_cases": list(self.plan_cases),
            "human_evidence": self.human_evidence,
        }


def _maximum_forecast_run(plan: JointAccelerationPlan) -> int:
    actual = frozenset(plan.actual_step_indices)
    current = 0
    maximum = 0
    for step in range(plan.total_steps):
        if step in actual:
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return maximum


def _nonincreasing(values: Iterable[float], tolerance: float = 1.0e-9) -> bool:
    values = tuple(values)
    return all(
        right <= left + tolerance for left, right in zip(values, values[1:])
    )


def _nondecreasing(values: Iterable[float], tolerance: float = 1.0e-9) -> bool:
    values = tuple(values)
    return all(
        right + tolerance >= left for left, right in zip(values, values[1:])
    )


def _percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _load_human_evidence(serve_root: Path) -> tuple[dict[str, object], list[str]]:
    source = Path(__file__).with_name("evidence") / "human_reviews_v1.json"
    errors: list[str] = []
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"source": str(source), "records": []}, [str(error)]
    records = document.get("records")
    if document.get("schema_version") != "h3_human_scheduler_evidence_v1":
        errors.append("unexpected Human evidence schema")
    if not isinstance(records, list):
        return document, errors + ["Human evidence records must be a list"]
    ids: set[str] = set()
    dimensions = {"motion_physics", "clarity", "identity", "audio"}
    for row in records:
        evidence_id = str(row.get("evidence_id", ""))
        if not evidence_id or evidence_id in ids:
            errors.append(f"invalid or duplicate evidence id: {evidence_id}")
        ids.add(evidence_id)
        if row.get("human_label") not in ("accept", "reject"):
            errors.append(f"{evidence_id}: invalid Human label")
        if set(row.get("dimensions", {})) != dimensions:
            errors.append(f"{evidence_id}: incomplete non-compensating dimensions")
        for field in ("artifact", "telemetry"):
            relative = row.get(field)
            if not isinstance(relative, str) or not (serve_root / relative).is_file():
                errors.append(f"{evidence_id}: missing {field}: {relative}")
    return document, errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence_path(root: Path, relative: object) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _load_online_runtime_evidence(
    serve_root: Path,
    *,
    policy_id: str,
) -> tuple[dict[str, object], list[str], bool]:
    """Validate version-bound runtime probes and their conservative ledger.

    Absence is a pending research state.  Once the manifest exists, malformed
    or over-budget evidence is a failure rather than silently falling back to
    source-code claims.
    """

    expected_guard = _ONLINE_GUARDS.get(policy_id)
    source_name = {
        JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP: (
            "round220_online_runtime_v1.json"
        ),
        JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP: (
            "round221_online_runtime_v1.json"
        ),
        JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP: (
            "round223_online_runtime_v1.json"
        ),
        JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP: (
            "round224_online_runtime_v1.json"
        ),
    }.get(policy_id, "round219_online_runtime_v1.json")
    source = Path(__file__).with_name("evidence") / source_name
    if not source.is_file():
        return {
            "source": str(source),
            "records": 0,
            "normal_probe_records": 0,
            "normal_upgrade_records": 0,
            "diagnostic_upgrade_records": 0,
        }, [], False
    errors: list[str] = []
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"source": str(source), "records": 0}, [str(error)], True
    expected_schema = {
        JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP: (
            "h3_round220_online_runtime_evidence_v1"
        ),
        JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP: (
            "h3_round221_online_runtime_evidence_v1"
        ),
        JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP: (
            "h3_round223_online_runtime_evidence_v1"
        ),
        JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP: (
            "h3_round224_online_runtime_evidence_v1"
        ),
    }.get(policy_id, "h3_round219_online_runtime_evidence_v1")
    if document.get("schema_version") != expected_schema:
        errors.append("unexpected online runtime evidence schema")
    if document.get("policy_id") != policy_id:
        errors.append("online runtime evidence policy mismatch")
    if document.get("online_guard_id") != expected_guard:
        errors.append("online runtime evidence guard mismatch")
    records = document.get("records")
    if not isinstance(records, list) or not records:
        records = []
        errors.append("online runtime evidence records must be a non-empty list")

    normal_probes = 0
    normal_upgrades = 0
    diagnostic_upgrades = 0
    record_summaries: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for row in records:
        if not isinstance(row, dict):
            errors.append("online runtime evidence row must be an object")
            continue
        evidence_id = str(row.get("evidence_id", ""))
        if not evidence_id or evidence_id in seen_ids:
            errors.append(f"invalid or duplicate online evidence id: {evidence_id}")
        seen_ids.add(evidence_id)
        evidence_kind = row.get("evidence_kind")
        if evidence_kind not in ("normal_policy", "forced_diagnostic"):
            errors.append(f"{evidence_id}: invalid evidence kind")
        telemetry_path = _evidence_path(serve_root, row.get("telemetry"))
        if telemetry_path is None or not telemetry_path.is_file():
            errors.append(f"{evidence_id}: missing or unsafe telemetry")
            continue
        declared_digest = row.get("telemetry_sha256")
        observed_digest = _sha256(telemetry_path)
        if declared_digest != observed_digest:
            errors.append(f"{evidence_id}: telemetry digest mismatch")
        artifact_path = _evidence_path(serve_root, row.get("artifact"))
        if artifact_path is None or not artifact_path.is_file():
            errors.append(f"{evidence_id}: missing or unsafe artifact")
        elif row.get("artifact_sha256") != _sha256(artifact_path):
            errors.append(f"{evidence_id}: artifact digest mismatch")
        try:
            telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{evidence_id}: invalid telemetry: {error}")
            continue
        if telemetry.get("schema_version") != "h3_native_scheduler_runtime_v1":
            errors.append(f"{evidence_id}: unexpected telemetry schema")
        profile = telemetry.get("execution_profile")
        if not isinstance(profile, dict):
            errors.append(f"{evidence_id}: missing execution profile")
            continue
        plan = profile.get("joint_acceleration")
        if not isinstance(plan, dict):
            errors.append(f"{evidence_id}: missing serialized joint plan")
            continue
        if plan.get("policy_id") != policy_id:
            errors.append(f"{evidence_id}: runtime policy mismatch")
        if plan.get("online_guard_id") != expected_guard:
            errors.append(f"{evidence_id}: runtime guard mismatch")
        backend = profile.get("attention_backend")
        ledger = profile.get("attention_online_guard")
        if not isinstance(backend, dict) or not isinstance(ledger, dict):
            errors.append(f"{evidence_id}: missing backend or budget telemetry")
            continue
        probe_records = backend.get("probe_records")
        events = ledger.get("events")
        if not isinstance(probe_records, list):
            errors.append(f"{evidence_id}: probe records must be a list")
            probe_records = []
        if not isinstance(events, list):
            errors.append(f"{evidence_id}: budget events must be a list")
            events = []
        if ledger.get("schema_version") != "h3_attention_online_budget_v1":
            errors.append(f"{evidence_id}: unexpected budget schema")
        if ledger.get("policy_id") != expected_guard:
            errors.append(f"{evidence_id}: budget policy mismatch")
        if ledger.get("budget_respected") is not True:
            errors.append(f"{evidence_id}: budget was not respected")
        if ledger.get("upgrade_only") is not True:
            errors.append(f"{evidence_id}: online guard is not upgrade-only")
        expected_rebate = {
            (int(step), int(layer))
            for step, layer in plan.get("online_rebate_schedule", ())
        }
        observed_rebate_schedule = {
            (int(step), int(layer))
            for step, layer in ledger.get("rebate_schedule", ())
        }
        if policy_id == JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP:
            if observed_rebate_schedule != expected_rebate:
                errors.append(f"{evidence_id}: runtime rebate schedule mismatch")
        elif observed_rebate_schedule:
            errors.append(f"{evidence_id}: legacy policy contains rebate cells")
        try:
            limit = float(ledger.get("limit_dense_layers"))
            spent = float(ledger.get("spent_dense_layers"))
        except (TypeError, ValueError):
            limit = spent = math.nan
            errors.append(f"{evidence_id}: invalid budget totals")
        if not math.isfinite(limit) or not math.isfinite(spent) or spent > limit + 1e-9:
            errors.append(f"{evidence_id}: runtime spend exceeds its limit")
        accepted_sum = 0.0
        previous_spent = 0.0
        allowed_kinds = {"probe", "trigger_upgrade", "recovery"}
        if policy_id == JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP:
            allowed_kinds.add("reserve_rebate")
        accepted_upgrade = False
        accepted_rebates = 0
        for event in events:
            if not isinstance(event, dict):
                errors.append(f"{evidence_id}: invalid budget event")
                continue
            kind = event.get("kind")
            if kind not in allowed_kinds:
                errors.append(f"{evidence_id}: unknown budget event kind: {kind}")
            try:
                charge = float(event.get("charge_dense_layers"))
                event_spent = float(event.get("spent_dense_layers"))
            except (TypeError, ValueError):
                errors.append(f"{evidence_id}: invalid budget event totals")
                continue
            if charge <= 0.0 or event_spent + 1e-9 < previous_spent or event_spent > limit + 1e-9:
                errors.append(f"{evidence_id}: non-monotone or over-limit budget event")
            if event.get("accepted") is True:
                accepted_sum += charge
                if kind in ("trigger_upgrade", "recovery"):
                    accepted_upgrade = True
                if kind == "reserve_rebate":
                    accepted_rebates += 1
                    if (int(event.get("step", -1)), int(event.get("layer", -1))) not in expected_rebate:
                        errors.append(
                            f"{evidence_id}: rebate event lies outside its certificate"
                        )
            previous_spent = event_spent
        if math.isfinite(spent) and not math.isclose(accepted_sum, spent, abs_tol=1e-9):
            errors.append(f"{evidence_id}: accepted event sum does not match spend")
        if policy_id == JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP:
            runtime_rebate_calls = int(backend.get("reserve_rebate_calls", -1))
            if runtime_rebate_calls != accepted_rebates:
                errors.append(f"{evidence_id}: rebate call/event count mismatch")
            if accepted_rebates and backend.get("request_had_trigger") is True:
                errors.append(f"{evidence_id}: rebate executed after a trigger")
        if evidence_kind == "normal_policy" and probe_records:
            normal_probes += 1
        if evidence_kind == "normal_policy" and accepted_upgrade:
            normal_upgrades += 1
        if evidence_kind == "forced_diagnostic" and accepted_upgrade:
            diagnostic_upgrades += 1
        record_summaries.append(
            {
                "evidence_id": evidence_id,
                "evidence_kind": evidence_kind,
                "probe_records": len(probe_records),
                "upgrade_executed": accepted_upgrade,
                "limit_dense_layers": limit,
                "spent_dense_layers": spent,
            }
        )
    summary = {
        "source": str(source),
        "records": len(records),
        "normal_probe_records": normal_probes,
        "normal_upgrade_records": normal_upgrades,
        "diagnostic_upgrade_records": diagnostic_upgrades,
        "record_summaries": record_summaries,
    }
    return summary, errors, True


def _validate_probe_growth_calibration(
    serve_root: Path,
) -> tuple[dict[str, object], list[str]]:
    """Recompute Round221 scores from every digest-bound runtime trajectory."""

    errors: list[str] = []
    try:
        calibration = load_probe_growth_calibration()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"source": str(ROUND221_CALIBRATION_FILE)}, [str(error)]
    try:
        document = json.loads(ROUND221_CALIBRATION_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"source": str(ROUND221_CALIBRATION_FILE)}, [str(error)]
    declared_scores = dict(calibration.task_scores)
    observed_scores: dict[str, float] = {}
    for row in document.get("records", []):
        evidence_id = str(row.get("id", ""))
        telemetry_path = _evidence_path(serve_root, row.get("telemetry"))
        artifact_path = _evidence_path(serve_root, row.get("artifact"))
        if telemetry_path is None or not telemetry_path.is_file():
            errors.append(f"{evidence_id}: missing calibration telemetry")
            continue
        if row.get("telemetry_sha256") != _sha256(telemetry_path):
            errors.append(f"{evidence_id}: calibration telemetry digest mismatch")
        if artifact_path is None or not artifact_path.is_file():
            errors.append(f"{evidence_id}: missing calibration artifact")
        elif row.get("artifact_sha256") != _sha256(artifact_path):
            errors.append(f"{evidence_id}: calibration artifact digest mismatch")
        try:
            telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
            backend = telemetry["execution_profile"]["attention_backend"]
            probes = backend["probe_records"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            errors.append(f"{evidence_id}: invalid calibration telemetry: {error}")
            continue
        if not isinstance(probes, list) or not probes:
            errors.append(f"{evidence_id}: calibration probes are missing")
            continue
        by_layer: dict[int, list[tuple[int, float]]] = {}
        for probe in probes:
            try:
                layer = int(probe["layer"])
                step = int(probe["step"])
                rms = float(probe["relative_rms"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{evidence_id}: malformed calibration probe")
                continue
            if probe.get("selection_strategy") != calibration.probe_domain:
                errors.append(
                    f"{evidence_id}: probe domain mismatch: "
                    f"{probe.get('selection_strategy')}"
                )
            by_layer.setdefault(layer, []).append((step, rms))
        ratios: list[float] = []
        for layer, observations in by_layer.items():
            ordered = sorted(observations)
            if len(ordered) < 2:
                errors.append(f"{evidence_id}: layer {layer} lacks a later phase")
                continue
            baseline = max(ordered[0][1], 1.0e-12)
            ratios.extend(value / baseline for _, value in ordered[1:])
        if not ratios:
            errors.append(f"{evidence_id}: no phase-growth ratio could be computed")
            continue
        score = max(ratios)
        observed_scores[evidence_id] = score
        declared = declared_scores.get(evidence_id)
        if declared is None or not math.isclose(score, declared, abs_tol=1.0e-12):
            errors.append(
                f"{evidence_id}: task score mismatch "
                f"declared={declared},observed={score}"
            )
    if set(observed_scores) != set(declared_scores):
        errors.append("calibration task-id coverage mismatch")
    if observed_scores and not math.isclose(
        max(observed_scores.values()),
        calibration.observed_max_task_score,
        abs_tol=1.0e-12,
    ):
        errors.append("recomputed calibration maximum mismatch")
    return {
        "source": str(ROUND221_CALIBRATION_FILE),
        "evidence_sha256": calibration.evidence_sha256,
        "probe_domain": calibration.probe_domain,
        "task_count": calibration.task_count,
        "observed_max_task_score": calibration.observed_max_task_score,
        "runtime_growth_threshold": calibration.runtime_growth_threshold,
        "marginal_coverage_lower_bound": (
            calibration.marginal_coverage_lower_bound
        ),
        "recomputed_tasks": len(observed_scores),
    }, errors


def _validate_probe_growth_holdout(
    serve_root: Path,
) -> tuple[dict[str, object], list[str], bool]:
    """Validate held-out runtime scores without feeding them into the threshold."""

    source = Path(__file__).with_name("evidence") / "round222_probe_growth_holdout_v1.json"
    if not source.is_file():
        return {"source": str(source), "records": 0}, [], False
    errors: list[str] = []
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"source": str(source), "records": 0}, [str(error)], True
    calibration = ROUND221_PROBE_GROWTH_CALIBRATION
    if document.get("schema_version") != "h3_round222_probe_growth_holdout_v1":
        errors.append("unexpected probe-growth holdout schema")
    if document.get("policy_id") != JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP:
        errors.append("probe-growth holdout policy mismatch")
    if document.get("online_guard_id") != ROUND221_CALIBRATED_GROWTH_GUARD:
        errors.append("probe-growth holdout guard mismatch")
    if document.get("calibration_evidence_sha256") != calibration.evidence_sha256:
        errors.append("probe-growth holdout calibration digest mismatch")
    if document.get("probe_domain") != calibration.probe_domain:
        errors.append("probe-growth holdout domain mismatch")
    try:
        threshold = float(document.get("runtime_growth_threshold"))
    except (TypeError, ValueError):
        threshold = math.nan
    if not math.isclose(
        threshold, calibration.runtime_growth_threshold, abs_tol=1.0e-12
    ):
        errors.append("probe-growth holdout threshold mismatch")
    records = document.get("records")
    if not isinstance(records, list) or not records:
        records = []
        errors.append("probe-growth holdout records are missing")
    scores: list[float] = []
    trigger_tasks = 0
    for row in records:
        evidence_id = str(row.get("id", ""))
        telemetry_path = _evidence_path(serve_root, row.get("telemetry"))
        artifact_path = _evidence_path(serve_root, row.get("artifact"))
        if telemetry_path is None or not telemetry_path.is_file():
            errors.append(f"{evidence_id}: missing holdout telemetry")
            continue
        if row.get("telemetry_sha256") != _sha256(telemetry_path):
            errors.append(f"{evidence_id}: holdout telemetry digest mismatch")
        if artifact_path is None or not artifact_path.is_file():
            errors.append(f"{evidence_id}: missing holdout artifact")
        elif row.get("artifact_sha256") != _sha256(artifact_path):
            errors.append(f"{evidence_id}: holdout artifact digest mismatch")
        try:
            telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
            profile = telemetry["execution_profile"]
            plan = profile["joint_acceleration"]
            backend = profile["attention_backend"]
            ledger = profile["attention_online_guard"]
            probes = backend["probe_records"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            errors.append(f"{evidence_id}: invalid holdout telemetry: {error}")
            continue
        if plan.get("policy_id") != JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP:
            errors.append(f"{evidence_id}: runtime policy mismatch")
        if ledger.get("budget_respected") is not True:
            errors.append(f"{evidence_id}: holdout budget exceeded")
        if not isinstance(probes, list) or not probes:
            errors.append(f"{evidence_id}: holdout probes are missing")
            continue
        by_layer: dict[int, list[tuple[int, float]]] = {}
        observed_trigger_count = 0
        for probe in probes:
            try:
                layer = int(probe["layer"])
                step = int(probe["step"])
                rms = float(probe["relative_rms"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{evidence_id}: malformed holdout probe")
                continue
            if probe.get("selection_strategy") != calibration.probe_domain:
                errors.append(f"{evidence_id}: holdout probe domain mismatch")
            observed_trigger_count += int(probe.get("phase_growth_trigger") is True)
            by_layer.setdefault(layer, []).append((step, rms))
        ratios = [
            value / max(ordered[0][1], 1.0e-12)
            for observations in by_layer.values()
            for ordered in (sorted(observations),)
            for _, value in ordered[1:]
        ]
        if not ratios:
            errors.append(f"{evidence_id}: no holdout ratio could be computed")
            continue
        score = max(ratios)
        scores.append(score)
        declared_score = float(row.get("task_score", math.nan))
        declared_triggers = int(row.get("growth_triggers", -1))
        expected_triggers = sum(ratio > threshold for ratio in ratios)
        if not math.isclose(score, declared_score, abs_tol=1.0e-12):
            errors.append(f"{evidence_id}: holdout score mismatch")
        if declared_triggers != expected_triggers or observed_trigger_count != expected_triggers:
            errors.append(f"{evidence_id}: holdout trigger count mismatch")
        trigger_tasks += int(expected_triggers > 0)
    summary = document.get("summary", {})
    if int(summary.get("held_out_tasks", -1)) != len(records):
        errors.append("holdout summary task count mismatch")
    if int(summary.get("tasks_above_threshold", -1)) != trigger_tasks:
        errors.append("holdout summary trigger count mismatch")
    return {
        "source": str(source),
        "records": len(records),
        "recomputed_records": len(scores),
        "tasks_above_threshold": trigger_tasks,
        "maximum_task_score": max(scores) if scores else None,
        "runtime_growth_threshold": threshold,
        "human_labels": sum(row.get("human_review") != "pending" for row in records),
    }, errors, True


def evaluate_joint_scheduler(
    *,
    policy_id: str = DEFAULT_JOINT_POLICY,
    serve_root: str | Path | None = None,
) -> SchedulerEvaluationReport:
    """Run the versioned CPU/control-plane scheduler evaluation contract."""

    started = time.perf_counter()
    root = (
        Path(serve_root).resolve()
        if serve_root is not None
        else Path(__file__).resolve().parents[3]
    )
    scheduler = H3JointAccelerationScheduler(policy_id=policy_id)
    clear_joint_plan_cache()
    gates: list[EvaluationGate] = []
    cases: list[dict[str, object]] = []
    plans: dict[tuple[int, bool, int, str], list[JointAccelerationPlan]] = {}
    cold_ms: list[float] = []
    short_base = JointWorkloadContext(packed_tokens=34_871)
    long_base = JointWorkloadContext(packed_tokens=100_163)
    short_lora = JointWorkloadContext(packed_tokens=34_871, model_variant="lora")
    matrix = (
        (20, True, short_base, (0.0, 25.0, 50.0, 75.0, 100.0)),
        (20, True, long_base, (0.0, 25.0, 50.0, 75.0, 100.0)),
        (8, False, short_lora, (0.0, 25.0, 50.0, 75.0, 100.0)),
        (4, True, short_base, (0.0, 100.0)),
        (30, True, long_base, (0.0, 100.0)),
    )
    planning_errors: list[str] = []
    for total_steps, allow_forecast, workload, accelerations in matrix:
        family: list[JointAccelerationPlan] = []
        for acceleration in accelerations:
            before = time.perf_counter()
            try:
                plan = scheduler.plan(
                    total_steps,
                    acceleration,
                    allow_forecast=allow_forecast,
                    workload=workload,
                )
            except Exception as error:  # evaluation must retain every failure
                planning_errors.append(
                    f"N={total_steps},a={acceleration},forecast={allow_forecast},tokens={workload.packed_tokens}: {error}"
                )
                continue
            elapsed_ms = (time.perf_counter() - before) * 1000.0
            if acceleration > 0.0:
                cold_ms.append(elapsed_ms)
            family.append(plan)
            cases.append(
                {
                    "total_steps": total_steps,
                    "acceleration": acceleration,
                    "allow_forecast": allow_forecast,
                    "workload_packed_tokens": workload.packed_tokens,
                    "model_variant": workload.model_variant,
                    "actual_evaluations": plan.actual_evaluations,
                    "forecast_evaluations": plan.forecast_evaluations,
                    "target_compute_ratio": round(
                        plan.target_compute_units / plan.dense_compute_units, 6
                    ),
                    "estimated_compute_ratio": round(
                        plan.estimated_compute_ratio, 6
                    ),
                    "estimated_risk_debt": round(plan.estimated_risk_debt, 6),
                    "online_recovery_reserve_units": round(
                        plan.online_recovery_reserve_units, 6
                    ),
                    "planning_ms": round(elapsed_ms, 6),
                }
            )
        plans[(total_steps, allow_forecast, workload.packed_tokens, workload.model_variant)] = family

    gates.append(
        EvaluationGate(
            "E01_plan_matrix",
            "effective",
            "pass" if not planning_errors else "fail",
            True,
            planning_errors or f"{len(cases)} plans",
            "All boundary and representative Base/LoRA requests must be plannable.",
            "Executable request matrix; failures are never dropped.",
        )
    )

    all_plans = tuple(plan for family in plans.values() for plan in family)
    dense_ok = all(
        plan.actual_evaluations == plan.total_steps
        and plan.forecast_evaluations == 0
        and set(plan.physical_action_schedule().values()) == {"dense"}
        and plan.estimated_risk_debt == 0.0
        for plan in all_plans
        if plan.acceleration == 0.0
    )
    gates.append(
        EvaluationGate(
            "E02_dense_identity_endpoint",
            "effective",
            "pass" if dense_ok else "fail",
            True,
            dense_ok,
            "Acceleration zero is the exact all-actual, all-Dense endpoint.",
            "Plan structure and serialized risk.",
        )
    )

    budget_ok = all(
        plan.estimated_compute_units <= plan.target_compute_units + 1.0e-6
        for plan in all_plans
    )
    gates.append(
        EvaluationGate(
            "E03_budget_compliance",
            "effective",
            "pass" if budget_ok else "fail",
            True,
            budget_ok,
            "Planned compute including recovery reserve never exceeds the dial budget.",
            "Conservative per-cell cost quantisation and plan totals.",
        )
    )

    monotone_errors: list[str] = []
    for key, family in plans.items():
        if not family:
            continue
        ordered = sorted(family, key=lambda plan: plan.acceleration)
        if not _nonincreasing(plan.target_compute_units for plan in ordered):
            monotone_errors.append(f"{key}: target compute")
        if not _nonincreasing(plan.estimated_compute_units for plan in ordered):
            monotone_errors.append(f"{key}: estimated compute")
        if not _nonincreasing(plan.actual_evaluations for plan in ordered):
            monotone_errors.append(f"{key}: actual evaluations")
        if not _nondecreasing(plan.estimated_risk_debt for plan in ordered):
            monotone_errors.append(f"{key}: risk debt")
    gates.append(
        EvaluationGate(
            "E04_monotone_speed_dial",
            "effective",
            "pass" if not monotone_errors else "fail",
            True,
            monotone_errors or "monotone",
            "More acceleration cannot request more compute or advertise less surrogate risk.",
            "Discrete endpoint families for Base and no-forecast LoRA.",
        )
    )

    safety_errors: list[str] = []
    ranks = {
        "sparse_topk_0.0625": 0,
        "sparse_topk_0.1": 1,
        "sparse_topk_0.25": 2,
        "sparse_topk_0.5": 3,
        "dense": 4,
    }
    causal_layers = frozenset((*range(30, 44), 45))
    for plan in all_plans:
        adaptive_latency = (
            plan.policy_id == JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP
            and plan.acceleration > 0.0
        )
        if _maximum_forecast_run(plan) > (5 if adaptive_latency else 2):
            safety_errors.append(f"N={plan.total_steps},a={plan.acceleration}: forecast run")
        terminal = frozenset(
            plan.actual_step_indices[-min(
                2 if adaptive_latency else 3,
                plan.actual_evaluations,
            ):]
        )
        for (step, layer), action in plan.physical_action_schedule().items():
            rank = ranks[action]
            if layer in causal_layers and rank < 1:
                safety_errors.append(f"N={plan.total_steps},a={plan.acceleration}: causal floor")
                break
            if step in terminal and rank < (1 if adaptive_latency else 2):
                safety_errors.append(f"N={plan.total_steps},a={plan.acceleration}: terminal floor")
                break
    gates.append(
        EvaluationGate(
            "E05_structural_safety_envelope",
            "effective",
            "pass" if not safety_errors else "fail",
            True,
            safety_errors or "all structural floors held",
            "Forecast runs, causal bands and terminal recovery stay inside admitted rails.",
            "Expanded physical layer schedule.",
        )
    )

    certificate_errors: list[str] = []
    representatives = [
        plans[(20, True, 34_871, "base")][2]
        if len(plans[(20, True, 34_871, "base")]) > 2 else None,
        plans[(20, True, 100_163, "base")][-1]
        if plans[(20, True, 100_163, "base")] else None,
        plans[(8, False, 34_871, "lora")][-1]
        if plans[(8, False, 34_871, "lora")] else None,
    ]
    if policy_id != JOINT_POLICY_V1_HEURISTIC:
        for plan in representatives:
            if plan is None:
                continue
            verification = verify_joint_plan_certificate(plan)
            if not verification.valid:
                certificate_errors.extend(verification.reasons)
    certificate_status: GateStatus
    if policy_id == JOINT_POLICY_V1_HEURISTIC:
        certificate_status = "pending"
    else:
        certificate_status = "pass" if not certificate_errors else "fail"
    gates.append(
        EvaluationGate(
            "E06_formal_surrogate_certificate",
            "effective",
            certificate_status,
            policy_id != JOINT_POLICY_V1_HEURISTIC,
            certificate_errors or (
                "not claimed by v1" if policy_id == JOINT_POLICY_V1_HEURISTIC
                else "representative certificates replayed"
            ),
            (
                "Exactness is machine-verified for the finite joint trajectory and Attention problem."
                if policy_id in _GLOBAL_DP_POLICIES
                else "Exactness is machine-verified for the finite Attention allocation problem."
            ),
            "Certificate replay checks the declared finite lattice, shape budget, objective and selected path; it does not claim Human optimality.",
        )
    )

    shape_plans = tuple(
        plan
        for plan in all_plans
        if plan.policy_id in _GLOBAL_DP_POLICIES
        and plan.acceleration > 0.0
        and plan.workload_context is not None
        and plan.workload_context.packed_tokens in (34_871, 100_163)
    )
    shape_errors: list[str] = []
    if policy_id in _GLOBAL_DP_POLICIES:
        observed_tokens = {
            plan.workload_context.packed_tokens for plan in shape_plans
        }
        if observed_tokens != {34_871, 100_163}:
            shape_errors.append(f"missing calibration endpoint: {observed_tokens}")
        if any(plan.workload_extrapolated for plan in shape_plans):
            shape_errors.append("measured endpoint marked as extrapolated")
        for plan in shape_plans:
            expected_mix = 0.0 if plan.workload_context.packed_tokens == 34_871 else 1.0
            if not math.isclose(
                float(plan.workload_calibration_mix or 0.0), expected_mix
            ):
                shape_errors.append(
                    f"bad calibration mix at {plan.workload_context.packed_tokens}"
                )
    gates.append(
        EvaluationGate(
            "E09_shape_matched_cost_model",
            "effective",
            (
                "pass" if not shape_errors else "fail"
            ) if policy_id in _GLOBAL_DP_POLICIES else "pending",
            policy_id in _GLOBAL_DP_POLICIES,
            shape_errors or (
                "both Round215 measured endpoints selected without extrapolation"
                if policy_id in _GLOBAL_DP_POLICIES
                else "shape-aware cost model is a global-DP claim"
            ),
            "A shape-aware policy must reproduce both measured 4090 calibration endpoints and never confuse interpolation with evidence.",
            "Round215 34,871/100,163-token complete-56-head measurements and source hashes in the global certificate.",
        )
    )

    implementation_errors = [
        (
            f"N={plan.total_steps},a={plan.acceleration}: "
            f"{plan.attention_implementation_id}"
        )
        for plan in all_plans
        if plan.acceleration > 0.0
        and plan.policy_id in _ROUND215_MEASURED_POLICIES
        and plan.attention_implementation_id != ROUND215_ACTION_IMPLEMENTATION
    ]
    gates.append(
        EvaluationGate(
            "E10_calibration_runtime_action_identity",
            "effective",
            (
                "pass" if not implementation_errors else "fail"
            ) if policy_id in _ROUND215_MEASURED_POLICIES else "pending",
            policy_id in _ROUND215_MEASURED_POLICIES,
            implementation_errors or (
                ROUND215_ACTION_IMPLEMENTATION
                if policy_id in _ROUND215_MEASURED_POLICIES
                else "no measured global action claim"
            ),
            "Every optimized action must execute the same sparse implementation used by its cost/error calibration.",
            "The implementation id is certificate-bound and the runtime schedule uses implementation-qualified action names.",
        )
    )

    constraint_errors = [
        f"N={plan.total_steps},a={plan.acceleration}: {plan.quality_constraint_id}"
        for plan in all_plans
        if plan.acceleration > 0.0
        and plan.policy_id in (
            JOINT_POLICY_V6_CAUSAL_ISLAND_GLOBAL_DP,
            JOINT_POLICY_V7_LAYER_RISK_GLOBAL_DP,
            JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP,
            JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP,
            JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP,
            JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
            JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
        )
        and plan.quality_constraint_id != ROUND216_CAUSAL_ISLAND_CONSTRAINT
    ]
    gates.append(
        EvaluationGate(
            "E11_human_causal_island_constraint",
            "effective",
            (
                "pass" if not constraint_errors else "fail"
            ) if policy_id in (
                JOINT_POLICY_V6_CAUSAL_ISLAND_GLOBAL_DP,
                JOINT_POLICY_V7_LAYER_RISK_GLOBAL_DP,
                JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP,
                JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP,
                JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP,
                JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
                JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
            ) else "pending",
            policy_id in (
                JOINT_POLICY_V6_CAUSAL_ISLAND_GLOBAL_DP,
                JOINT_POLICY_V7_LAYER_RISK_GLOBAL_DP,
                JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP,
                JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP,
                JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP,
                JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
                JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
            ),
            constraint_errors or (
                ROUND216_CAUSAL_ISLAND_CONSTRAINT
                if policy_id in (
                    JOINT_POLICY_V6_CAUSAL_ISLAND_GLOBAL_DP,
                    JOINT_POLICY_V7_LAYER_RISK_GLOBAL_DP,
                    JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP,
                    JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP,
                    JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP,
                    JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
                    JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
                )
                else "causal-island constraint is a v6 claim"
            ),
            "V6 makes the Human-supported opening and causal island non-compensating hard constraints.",
            "Round143/216 Human accepts versus Round217 reject, bound into every V6 certificate.",
        )
    )

    layer_risk_errors = [
        f"N={plan.total_steps},a={plan.acceleration}: {plan.risk_model_id}"
        for plan in all_plans
        if plan.acceleration > 0.0
        and plan.policy_id == JOINT_POLICY_V7_LAYER_RISK_GLOBAL_DP
        and plan.risk_model_id != ROUND215_LAYER_RISK_MODEL
    ]
    gates.append(
        EvaluationGate(
            "E12_physical_layer_risk_model",
            "effective",
            (
                "pass" if not layer_risk_errors else "fail"
            ) if policy_id == JOINT_POLICY_V7_LAYER_RISK_GLOBAL_DP else "pending",
            policy_id == JOINT_POLICY_V7_LAYER_RISK_GLOBAL_DP,
            layer_risk_errors or (
                ROUND215_LAYER_RISK_MODEL
                if policy_id == JOINT_POLICY_V7_LAYER_RISK_GLOBAL_DP
                else "physical-layer risk allocation is a v7 claim"
            ),
            "V7 allocates non-causal compute using all 50 measured physical-layer Dense disagreements instead of seven band maxima.",
            "Round215 layer-3 teacher probe artifact digests plus certificate-bound risk model id.",
        )
    )

    phase_layer_risk_errors = [
        f"N={plan.total_steps},a={plan.acceleration}: {plan.risk_model_id}"
        for plan in all_plans
        if plan.acceleration > 0.0
        and plan.policy_id in (
            JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP,
            JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP,
            JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP,
            JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
            JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
            JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP,
        )
        and plan.risk_model_id != ROUND218_PHASE_LAYER_RISK_MODEL
    ]
    gates.append(
        EvaluationGate(
            "E13_phase_layer_risk_model",
            "effective",
            (
                "pass" if not phase_layer_risk_errors else "fail"
            ) if policy_id in (
                JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP,
                JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP,
                JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP,
                JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
                JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
                JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP,
            ) else "pending",
            policy_id in (
                JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP,
                JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP,
                JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP,
                JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
                JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
                JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP,
            ),
            phase_layer_risk_errors or (
                ROUND218_PHASE_LAYER_RISK_MODEL
                if policy_id in (
                    JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP,
                    JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP,
                    JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP,
                    JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
                    JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
                    JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP,
                )
                else "phase-layer allocation is a v8 claim"
            ),
            "V8 binds every optimized plan to five measured trajectory-risk anchors at both packed-token endpoints.",
            "Round218 step 1/3/8/14/18 x 50-layer Dense-disagreement evidence, source digests and certificate-bound risk-model id.",
        )
    )

    online_binding_errors = []
    for plan in all_plans:
        expected_guard = _ONLINE_GUARDS.get(plan.policy_id)
        if plan.acceleration <= 0.0:
            expected_guard = None
        if plan.online_guard_id != expected_guard or (
            expected_guard is not None
            and plan.online_recovery_reserve_units <= 0.0
        ):
            online_binding_errors.append(
                f"N={plan.total_steps},a={plan.acceleration}: "
                f"expected={expected_guard},observed={plan.online_guard_id}"
            )
    gates.append(
        EvaluationGate(
            "E14_versioned_online_guard_binding",
            "effective",
            (
                "pass" if not online_binding_errors else "fail"
            ) if policy_id in _ONLINE_GUARDS else "pending",
            policy_id in _ONLINE_GUARDS,
            online_binding_errors or (
                _ONLINE_GUARDS[policy_id]
                if policy_id in _ONLINE_GUARDS
                else "bounded online adaptation is a v9+ claim"
            ),
            "Only explicitly versioned online policies may enable their matching runtime guard, and every accelerated plan must carry a positive certified reserve.",
            "Serialized policy/guard identity and reserve; legacy plans are required to remain guard-free.",
        )
    )

    phase_allocation_errors: list[str] = []
    phase_allocation_observations: list[dict[str, object]] = []
    if policy_id in (
        JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP,
        JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
        JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
        JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP,
    ):
        for plan in all_plans:
            if plan.acceleration <= 0.0:
                continue
            limit = plan.online_recovery_reserve_units * 50.0
            allocation = allocate_phase_sentinels(
                plan.actual_step_indices, limit
            )
            slots_valid = (
                len(allocation.slots) == len(set(allocation.slots))
                and all(
                    step in plan.actual_step_indices
                    and layer in (4, 24, 44)
                    for step, layer in allocation.slots
                )
            )
            if not allocation.budget_respected or not slots_valid:
                phase_allocation_errors.append(
                    f"N={plan.total_steps},a={plan.acceleration}: "
                    f"slots={allocation.slots},limit={limit:.6f},"
                    f"remaining={allocation.remaining_dense_layers:.6f}"
                )
            phase_allocation_observations.append(
                {
                    "total_steps": plan.total_steps,
                    "acceleration": plan.acceleration,
                    "limit_dense_layers": round(limit, 6),
                    "probe_slots": len(allocation.slots),
                    "remaining_dense_layers": round(
                        allocation.remaining_dense_layers, 6
                    ),
                    "required_remaining_dense_layers": round(
                        allocation.required_remaining_dense_layers, 6
                    ),
                }
            )
    gates.append(
        EvaluationGate(
            "E15_formal_online_observation_allocation",
            "effective",
            (
                "pass" if not phase_allocation_errors else "fail"
            ) if policy_id in (
                JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP,
                JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
                JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
                JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP,
            ) else "pending",
            policy_id in (
                JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP,
                JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
                JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
                JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP,
            ),
            phase_allocation_errors or phase_allocation_observations,
            "The phase sentinel schedule is finite, uses only actual steps and certified probe layers, never overspends, and preserves its declared repair balance.",
            "Pure planner allocation replay over the complete evaluation matrix; runtime kernel code calls the same allocation function.",
        )
    )

    growth_calibration, growth_calibration_errors = (
        _validate_probe_growth_calibration(root)
        if policy_id in (
            JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
            JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
            JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP,
        )
        else ({}, [])
    )
    gates.append(
        EvaluationGate(
            "E16_calibrated_phase_growth_integrity",
            "effective",
            (
                "pass" if not growth_calibration_errors else "fail"
            ) if policy_id in (
                JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
                JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
                JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP,
            ) else "pending",
            policy_id in (
                JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
                JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
                JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP,
            ),
            growth_calibration_errors or (
                growth_calibration
                if policy_id in (
                    JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
                    JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
                    JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP,
                )
                else "task-relative phase growth is a v11 claim"
            ),
            "Every V11 calibration score is recomputed from digest-bound runtime probes in one declared sampling domain, and its maximum-order threshold is internally consistent.",
            "Round221 validates numerical novelty only; it is not a perceptual-quality or physics certificate.",
        )
    )

    holdout_evidence, holdout_errors, holdout_exists = (
        _validate_probe_growth_holdout(root)
        if policy_id == JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP
        else ({}, [], False)
    )
    gates.append(
        EvaluationGate(
            "E17_probe_growth_holdout_integrity",
            "effective",
            (
                "fail" if holdout_errors else ("pass" if holdout_exists else "pending")
            ) if policy_id == JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP else "pending",
            policy_id == JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
            holdout_errors or holdout_evidence,
            "A V11 generalization claim requires digest-bound tasks excluded from threshold construction, with scores and trigger counts replayed from runtime telemetry.",
            "The small unlabeled hold-out measures numerical novelty only; it is not a perceptual false-positive estimate.",
        )
    )

    rebate_errors: list[str] = []
    rebate_observations: list[dict[str, object]] = []
    if policy_id == JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP:
        for plan in all_plans:
            if plan.acceleration <= 0.0:
                continue
            allocation = allocate_phase_sentinels(
                plan.actual_step_indices,
                plan.online_recovery_reserve_units * 50.0,
            )
            certificate = plan.online_rebate_certificate
            schedule = plan.online_rebate_schedule
            last_probe = max(
                (step for step, _ in allocation.slots), default=-1
            )
            maximum = max(
                0,
                int(math.floor(
                    plan.online_recovery_reserve_units * 50.0
                    - len(allocation.slots)
                    + 1.0e-9
                )),
            )
            physical = plan.physical_action_schedule()
            valid = (
                not schedule and certificate is None and maximum == 0
                if not allocation.slots
                else (
                    certificate is not None
                    and len(schedule) <= maximum
                    and certificate.maximum_choices == maximum
                    and certificate.selected_count == len(schedule)
                    and (
                        certificate.selected_risk_reduction > 0.0
                        if schedule
                        else certificate.candidate_count == 0
                        and certificate.selected_risk_reduction == 0.0
                    )
                    and all(
                        step > last_probe
                        and physical.get((step, layer), "dense") != "dense"
                        for step, layer in schedule
                    )
                )
            )
            if not valid:
                rebate_errors.append(
                    f"N={plan.total_steps},a={plan.acceleration}: "
                    f"probes={len(allocation.slots)},maximum={maximum},"
                    f"selected={len(schedule)},last_probe={last_probe}"
                )
            rebate_observations.append(
                {
                    "total_steps": plan.total_steps,
                    "acceleration": plan.acceleration,
                    "probe_slots": len(allocation.slots),
                    "maximum_rebate_cells": maximum,
                    "selected_rebate_cells": len(schedule),
                    "last_probe_step": last_probe,
                    "surrogate_risk_reduction": (
                        None
                        if certificate is None
                        else round(certificate.selected_risk_reduction, 6)
                    ),
                }
            )
    gates.append(
        EvaluationGate(
            "E18_exact_no_trigger_reserve_rebate",
            "effective",
            (
                "pass" if not rebate_errors else "fail"
            ) if policy_id == JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP else "pending",
            policy_id == JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
            rebate_errors or rebate_observations,
            "After every sentinel passes without a trigger, V12 must spend only the remaining certified reserve on the exact top-K future sparse cells under its declared additive risk model.",
            "The plan verifier re-solves the finite conditional allocation; runtime cells are tail-only, upgrade-only and charged by the immutable online ledger.",
        )
    )

    adaptive_frontier_errors: list[str] = []
    adaptive_frontier_observation: dict[str, object] = {}
    if policy_id == JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP:
        long_family = plans.get((20, True, 100_163, "base"), [])
        long_fast = next(
            (plan for plan in long_family if plan.acceleration == 100.0),
            None,
        )
        if long_fast is None:
            adaptive_frontier_errors.append("missing 100k-token acceleration=100 plan")
        else:
            adaptive_frontier_observation = {
                "actual_step_indices": list(long_fast.actual_step_indices),
                "actual_evaluations": long_fast.actual_evaluations,
                "forecast_evaluations": long_fast.forecast_evaluations,
                "target_compute_ms": round(float(long_fast.target_compute_ms or 0.0), 6),
                "predicted_compute_ms": round(float(long_fast.predicted_compute_ms or 0.0), 6),
                "estimated_compute_ratio": round(long_fast.estimated_compute_ratio, 6),
                "trajectory_prior_id": long_fast.trajectory_prior_id,
                "quality_constraint_id": long_fast.quality_constraint_id,
            }
            if long_fast.trajectory_prior_id is not None:
                adaptive_frontier_errors.append("historical trajectory prior was injected")
            if long_fast.quality_constraint_id != ROUND224_ADAPTIVE_LATENCY_CONSTRAINT:
                adaptive_frontier_errors.append("adaptive latency constraint is not bound")
            if (long_fast.target_compute_ms or math.inf) > 177_000.0 + 1.0e-6:
                adaptive_frontier_errors.append("100k-token target exceeds 177 seconds")
            if (long_fast.predicted_compute_ms or math.inf) > (
                (long_fast.target_compute_ms or 0.0) + 1.0e-6
            ):
                adaptive_frontier_errors.append("predicted compute exceeds target")
            if long_fast.actual_evaluations + long_fast.forecast_evaluations != 20:
                adaptive_frontier_errors.append("actual/forecast trajectory is incomplete")
            if long_fast.estimated_compute_ratio > 0.22 + 1.0e-9:
                adaptive_frontier_errors.append("fast endpoint exceeds admitted compute ratio")
    gates.append(
        EvaluationGate(
            "E19_independent_adaptive_latency_frontier",
            "effective",
            (
                "pass" if not adaptive_frontier_errors else "fail"
            ) if policy_id == JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP else "pending",
            policy_id == JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP,
            adaptive_frontier_errors or adaptive_frontier_observation,
            "V13 acceleration=100 must independently solve a 100k-token 20-step plan inside the 177-second DiT target without a historical fixed trajectory candidate or fallback.",
            "This is a formal planner-budget gate only. Real 720p15 wall time and Human quality relative to Round188 are separate release gates and cannot be inferred from it.",
        )
    )

    human_document, human_errors = _load_human_evidence(root)
    records = human_document.get("records", [])
    accepts = sum(row.get("human_label") == "accept" for row in records)
    rejects = sum(row.get("human_label") == "reject" for row in records)
    coverage_ok = not human_errors and accepts >= 3 and rejects >= 3
    gates.append(
        EvaluationGate(
            "E07_historical_human_evidence_integrity",
            "effective",
            "pass" if coverage_ok else "fail",
            True,
            human_errors or {"accept": accepts, "reject": rejects},
            "The evaluator retains both accepted and rejected continuous-play evidence across all four non-compensating quality axes.",
            str(Path(__file__).with_name("evidence") / "human_reviews_v1.json"),
        )
    )
    own_labels = [
        row for row in records if row.get("scheduler_policy_id") == policy_id
    ]
    gates.append(
        EvaluationGate(
            "E08_current_policy_human_acceptance",
            "effective",
            "pass" if own_labels and any(row["human_label"] == "accept" for row in own_labels) else "pending",
            False,
            {"labels": len(own_labels), "policy_id": policy_id},
            "A new scheduler needs its own Human continuous-play labels before release acceptance.",
            "Historical policies constrain risk design but do not transfer acceptance.",
        )
    )

    cold_p95_ms = _percentile(cold_ms, 0.95)
    gates.append(
        EvaluationGate(
            "F01_cold_planning_latency",
            "efficient",
            "pass" if cold_p95_ms <= 1500.0 else "fail",
            True,
            round(cold_p95_ms, 6),
            "Cold p95 planning overhead is at most 1.5 seconds and below 0.5% of the 720p15 target regime.",
            "Measured in-process over the evaluation matrix; model import time excluded.",
        )
    )
    warm_samples: list[float] = []
    for _ in range(25):
        before = time.perf_counter()
        scheduler.plan(20, 50.0, allow_forecast=True, workload=long_base)
        warm_samples.append((time.perf_counter() - before) * 1000.0)
    warm_p95_ms = _percentile(warm_samples, 0.95)
    gates.append(
        EvaluationGate(
            "F02_warm_planning_latency",
            "efficient",
            "pass" if warm_p95_ms <= 2.0 else "fail",
            True,
            round(warm_p95_ms, 6),
            "Warm p95 planning overhead is at most 2 ms.",
            "Bounded immutable LRU cache shared by ETA and engine planning.",
        )
    )
    reserve_ok = all(
        plan.online_recovery_reserve_units > 0.0
        for plan in all_plans
        if plan.acceleration > 0.0
    )
    gates.append(
        EvaluationGate(
            "F03_bounded_online_reserve",
            "efficient",
            "pass" if reserve_ok else "fail",
            True,
            reserve_ok,
            "Every accelerated plan withholds an explicit bounded reserve for local protective upgrades.",
            "Serialized plan reserve; this gate does not claim that a runtime probe actually fired.",
        )
    )
    online_evidence, online_errors, online_manifest_exists = (
        _load_online_runtime_evidence(root, policy_id=policy_id)
    )
    normal_probe_records = int(online_evidence.get("normal_probe_records", 0))
    normal_upgrade_records = int(online_evidence.get("normal_upgrade_records", 0))
    online_evidence_status: GateStatus
    if policy_id not in _ONLINE_GUARDS:
        online_evidence_status = "pending"
    elif online_manifest_exists and online_errors:
        online_evidence_status = "fail"
    elif normal_probe_records > 0:
        online_evidence_status = "pass"
    else:
        online_evidence_status = "pending"
    gates.append(
        EvaluationGate(
            "F05_operational_online_probe_evidence",
            "efficient",
            online_evidence_status,
            policy_id in _ONLINE_GUARDS,
            online_errors or online_evidence,
            "An online policy may claim operation only after a normal-policy sparse probe executes and every event remains inside the immutable reserve ledger.",
            "Digest-bound runtime telemetry; planner source strings, unit tests and an unused reserve cannot satisfy this gate.",
        )
    )
    gates.append(
        EvaluationGate(
            "F06_normal_policy_upgrade_evidence",
            "efficient",
            (
                "pass" if normal_upgrade_records > 0 else "pending"
            ) if policy_id in _ONLINE_GUARDS else "pending",
            False,
            online_errors or {
                "normal_upgrade_records": normal_upgrade_records,
                "diagnostic_upgrade_records": int(
                    online_evidence.get("diagnostic_upgrade_records", 0)
                ),
            },
            "A runtime-correction effectiveness claim additionally requires a naturally triggered normal-policy upgrade; forced diagnostics prove mechanics only.",
            "Separating observation from intervention prevents a no-trigger run from being reported as successful correction.",
        )
    )

    mechanical_errors = [
        f"N={plan.total_steps},a={plan.acceleration}: {plan.mechanical_baseline_id}"
        for plan in all_plans
        if plan.mechanical_baseline_id != JOINT_MECHANICAL_BASELINE_ID
    ]
    gates.append(
        EvaluationGate(
            "F04_mechanical_baseline_identity",
            "efficient",
            "pass" if not mechanical_errors else "fail",
            True,
            mechanical_errors or JOINT_MECHANICAL_BASELINE_ID,
            "All dial positions compare against one fused-RMS and compiled-VAE mechanical baseline.",
            "Serialized baseline identity plus engine contract tests; this prevents planner gains from being confused with unrelated runtime switches.",
        )
    )

    signature = inspect.signature(H3JointAccelerationScheduler.plan)
    external_controls = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.name in ("total_steps", "acceleration")
    ]
    gates.append(
        EvaluationGate(
            "S01_two_public_controls",
            "simple",
            "pass" if external_controls == ["total_steps", "acceleration"] else "fail",
            True,
            external_controls,
            "The scheduler exposes exactly sampling steps and one acceleration dial.",
            "Forecast capability and all safety rails are internal model/runtime policy.",
        )
    )
    reference = scheduler.plan(
        20, 50.0, allow_forecast=True, workload=long_base
    )
    deterministic = all(
        scheduler.plan(
            20, 50.0, allow_forecast=True, workload=long_base
        ).to_dict()
        == reference.to_dict()
        for _ in range(3)
    )
    gates.append(
        EvaluationGate(
            "S02_deterministic_versioned_plan",
            "simple",
            "pass" if deterministic and reference.policy_id == policy_id else "fail",
            True,
            {"deterministic": deterministic, "policy_id": reference.policy_id},
            "Equal controls and capability produce an immutable, versioned plan.",
            "Serialized plan equality and explicit policy id.",
        )
    )
    document_size = len(
        json.dumps(reference.to_dict(), sort_keys=True, separators=(",", ":"))
    )
    gates.append(
        EvaluationGate(
            "S03_auditable_plan_surface",
            "simple",
            "pass" if document_size <= 128_000 else "fail",
            True,
            {"serialized_bytes": document_size},
            "A complete physical plan and certificate remain small enough to inspect and persist.",
            "Canonical JSON size threshold 128 KiB.",
        )
    )

    hard_failures = [gate for gate in gates if gate.hard and gate.status == "fail"]
    pending_human = any(gate.gate_id == "E08_current_policy_human_acceptance" and gate.status == "pending" for gate in gates)
    overall: Literal["pass", "fail", "pending_human_review"]
    if hard_failures:
        overall = "fail"
    elif pending_human:
        overall = "pending_human_review"
    else:
        overall = "pass"
    return SchedulerEvaluationReport(
        schema_version=EVALUATION_SCHEMA,
        policy_id=policy_id,
        overall_status=overall,
        gates=tuple(gates),
        plan_cases=tuple(cases),
        human_evidence={
            "schema_version": human_document.get("schema_version"),
            "source": str(Path(__file__).with_name("evidence") / "human_reviews_v1.json"),
            "records": len(records),
            "accept": accepts,
            "reject": rejects,
            "current_policy_labels": len(own_labels),
            "integrity_errors": human_errors,
        },
        elapsed_seconds=time.perf_counter() - started,
    )


__all__ = [
    "EVALUATION_SCHEMA",
    "EvaluationGate",
    "SchedulerEvaluationReport",
    "evaluate_joint_scheduler",
]
