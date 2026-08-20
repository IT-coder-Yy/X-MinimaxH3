"""Digest-sealed deployment bundle for the certified V19 frontier."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from .action_registry import ActionRegistry
from .v19_calibration import V19RuntimeFingerprint
from .v19_candidates import V19MaterializedCandidate
from .v19_contracts import V19HumanRiskVector, V19TrajectoryDebt
from .v19_forecast_calibration import V19ForecastCompositeKey
from .v19_frontier import (
    V19CertifiedEnvelopeEntry,
    V19CertifiedFrontierEntry,
    V19EnvelopeCertificate,
    V19ReferenceProfile,
    V19ReleaseFrontierCatalog,
    V19WorkloadEnvelope,
)
from .v19_planner import (
    V19ActionUse,
    V19CandidatePlan,
    V19ForecastUse,
    V19PlanCertificate,
    V19PlanningError,
    V19PlanningRequest,
    V19WorkloadContext,
)


V19_RELEASE_BUNDLE_SCHEMA = "h3_v19_release_bundle_v2"


def _sha256(document: object) -> str:
    return hashlib.sha256(json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _valid_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class V19LoadedReleaseBundle:
    catalog: V19ReleaseFrontierCatalog
    registry_digest: str
    evidence_sha256: tuple[tuple[str, str], ...]
    bundle_digest: str
    source: Path


def _payload(
    registry: ActionRegistry,
    entries: tuple[V19CertifiedEnvelopeEntry, ...],
    evidence_sha256: Mapping[str, str],
) -> dict[str, object]:
    known = {str(key): str(value) for key, value in evidence_sha256.items()}
    required = {
        evidence_id
        for entry in entries
        for evidence_id in entry.candidate.evidence_ids
    }
    missing = sorted(required - set(known))
    if missing:
        raise V19PlanningError(
            "V19 release bundle lacks evidence digests: " + ", ".join(missing)
        )
    malformed = sorted(
        evidence_id
        for evidence_id, digest in known.items()
        if not evidence_id or not _valid_sha256(digest)
    )
    if malformed:
        raise V19PlanningError(
            "V19 release bundle has invalid evidence digests: "
            + ", ".join(malformed)
        )
    # Re-run admission before sealing.  A bundle is deployment state, not a
    # way to bypass schedule timing, Human risk, or certificate validation.
    validator = V19ReleaseFrontierCatalog(registry)
    for entry in entries:
        validator.add(entry)
    return {
        "schema_version": V19_RELEASE_BUNDLE_SCHEMA,
        "registry_digest": registry.digest,
        "entries": [asdict(entry) for entry in entries],
        "evidence_sha256": dict(sorted(known.items())),
    }


def save_v19_release_bundle(
    path: str | Path,
    *,
    registry: ActionRegistry,
    entries: tuple[V19CertifiedEnvelopeEntry, ...],
    evidence_sha256: Mapping[str, str],
) -> None:
    if not entries:
        raise V19PlanningError("V19 release bundle cannot be empty")
    document = _payload(registry, entries, evidence_sha256)
    document["bundle_digest"] = _sha256(document)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _action_use(document: dict[str, object]) -> V19ActionUse | V19ForecastUse:
    if "composite_key" in document:
        key = dict(document["composite_key"])
        key["forecast_step_indices"] = tuple(
            int(value) for value in key["forecast_step_indices"]
        )
        return V19ForecastUse(
            action_id=str(document["action_id"]),
            canonical_action=str(document.get("canonical_action", "forecast")),
            composite_key=V19ForecastCompositeKey(**key),
        )
    return V19ActionUse(
        action_id=str(document["action_id"]),
        canonical_action=str(document["canonical_action"]),
        step_indices=tuple(int(value) for value in document["step_indices"]),
        layer_start=int(document["layer_start"]),
        layer_stop=int(document["layer_stop"]),
    )


def _candidate(document: dict[str, object]) -> V19CandidatePlan:
    return V19CandidatePlan(
        candidate_id=str(document["candidate_id"]),
        action_uses=tuple(_action_use(dict(row)) for row in document["action_uses"]),
        predicted_cost_p50_ms=float(document["predicted_cost_p50_ms"]),
        predicted_cost_p90_ms=float(document["predicted_cost_p90_ms"]),
        predicted_peak_vram_gib=float(document["predicted_peak_vram_gib"]),
        risk_ucb=V19HumanRiskVector(**document["risk_ucb"]),
        terminal_debt=V19TrajectoryDebt(**document["terminal_debt"]),
        maximum_debt=V19TrajectoryDebt(**document["maximum_debt"]),
        evidence_ids=tuple(str(value) for value in document.get("evidence_ids", ())),
        source=str(document.get("source", "v19_optimizer")),
    )


def _planning_request(document: dict[str, object]) -> V19PlanningRequest:
    workload = dict(document["workload"])
    workload["actual_step_indices"] = tuple(
        int(value) for value in workload.get("actual_step_indices", ())
    )
    runtime = document.get("runtime")
    return V19PlanningRequest(
        workload=V19WorkloadContext(**workload),
        maximum_cost_p90_ms=float(document["maximum_cost_p90_ms"]),
        risk_limits=V19HumanRiskVector(**document["risk_limits"]),
        runtime=(
            None if runtime is None else V19RuntimeFingerprint(**runtime)
        ),
        maximum_peak_vram_gib=float(document["maximum_peak_vram_gib"]),
        debt_limits=V19TrajectoryDebt(**document["debt_limits"]),
    )


def _certificate(document: dict[str, object]) -> V19PlanCertificate:
    return V19PlanCertificate(
        registry_digest=str(document["registry_digest"]),
        workload_digest=str(document["workload_digest"]),
        runtime_digest=str(document["runtime_digest"]),
        candidate_digest=str(document["candidate_digest"]),
        maximum_cost_p90_ms=float(document["maximum_cost_p90_ms"]),
        selected_cost_p90_ms=float(document["selected_cost_p90_ms"]),
        maximum_peak_vram_gib=float(document["maximum_peak_vram_gib"]),
        selected_peak_vram_gib=float(document["selected_peak_vram_gib"]),
        selected_risk_ucb=V19HumanRiskVector(**document["selected_risk_ucb"]),
        debt_limits=V19TrajectoryDebt(**document["debt_limits"]),
        selected_maximum_debt=V19TrajectoryDebt(
            **document["selected_maximum_debt"]
        ),
        action_ids=tuple(str(value) for value in document["action_ids"]),
        certificate_digest=str(document["certificate_digest"]),
        schema_version=str(document["schema_version"]),
        contract_schema=str(document["contract_schema"]),
    )


def _exact_entry(document: dict[str, object]) -> V19CertifiedFrontierEntry:
    materialized = dict(document["materialized"])
    return V19CertifiedFrontierEntry(
        planning_request=_planning_request(dict(document["planning_request"])),
        materialized=V19MaterializedCandidate(
            candidate=_candidate(dict(materialized["candidate"])),
            end_to_end_cost_calibrated=bool(
                materialized["end_to_end_cost_calibrated"]
            ),
            human_risk_calibrated=bool(materialized["human_risk_calibrated"]),
        ),
        certificate=_certificate(dict(document["certificate"])),
    )


def _envelope_certificate(document: dict[str, object]) -> V19EnvelopeCertificate:
    return V19EnvelopeCertificate(
        registry_digest=str(document["registry_digest"]),
        envelope_digest=str(document["envelope_digest"]),
        runtime_digest=str(document["runtime_digest"]),
        execution_digest=str(document["execution_digest"]),
        candidate_digest=str(document["candidate_digest"]),
        source_certificate_digests=tuple(
            str(value) for value in document["source_certificate_digests"]
        ),
        evidence_ids=tuple(str(value) for value in document["evidence_ids"]),
        certificate_digest=str(document["certificate_digest"]),
        schema_version=str(document["schema_version"]),
    )


def _envelope(document: dict[str, object]) -> V19WorkloadEnvelope:
    payload = dict(document)
    payload["reference_profiles"] = tuple(
        V19ReferenceProfile(**dict(row))
        for row in payload["reference_profiles"]
    )
    return V19WorkloadEnvelope(**payload)


def _entry(document: dict[str, object]) -> V19CertifiedEnvelopeEntry:
    materialized = dict(document["materialized"])
    return V19CertifiedEnvelopeEntry(
        envelope=_envelope(dict(document["envelope"])),
        materialized=V19MaterializedCandidate(
            candidate=_candidate(dict(materialized["candidate"])),
            end_to_end_cost_calibrated=bool(
                materialized["end_to_end_cost_calibrated"]
            ),
            human_risk_calibrated=bool(materialized["human_risk_calibrated"]),
        ),
        sources=tuple(_exact_entry(dict(row)) for row in document["sources"]),
        certificate=_envelope_certificate(dict(document["certificate"])),
    )


def load_v19_release_bundle(
    path: str | Path,
    *,
    registry: ActionRegistry,
) -> V19LoadedReleaseBundle:
    source = Path(path).resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
        if document["schema_version"] != V19_RELEASE_BUNDLE_SCHEMA:
            raise V19PlanningError("unsupported V19 release bundle schema")
        if document["registry_digest"] != registry.digest:
            raise V19PlanningError("V19 release bundle registry digest mismatch")
        stored_digest = str(document["bundle_digest"])
        payload = {key: value for key, value in document.items() if key != "bundle_digest"}
        if stored_digest != _sha256(payload):
            raise V19PlanningError("V19 release bundle digest mismatch")
        evidence = {
            str(key): str(value)
            for key, value in dict(document["evidence_sha256"]).items()
        }
        entries = tuple(_entry(dict(row)) for row in document["entries"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, V19PlanningError):
            raise
        raise V19PlanningError(f"invalid V19 release bundle: {source}") from error
    # Reproduce all admission checks and evidence coverage from the decoded
    # payload.  This also catches stale candidate/certificate identities.
    _payload(registry, entries, evidence)
    catalog = V19ReleaseFrontierCatalog(registry)
    for entry in entries:
        catalog.add(entry)
    return V19LoadedReleaseBundle(
        catalog=catalog,
        registry_digest=registry.digest,
        evidence_sha256=tuple(sorted(evidence.items())),
        bundle_digest=stored_digest,
        source=source,
    )


__all__ = [
    "V19_RELEASE_BUNDLE_SCHEMA",
    "V19LoadedReleaseBundle",
    "load_v19_release_bundle",
    "save_v19_release_bundle",
]
