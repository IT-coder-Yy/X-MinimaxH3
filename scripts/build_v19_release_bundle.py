#!/usr/bin/env python3
"""Build the fail-closed V19 deployment bundle from certified exact sources.

Each ``--source-spec`` names one exact workload.  Specs sharing an
``envelope_id`` must execute the same physical plan and provide observed
packed-token endpoints/reference layouts.  The builder replays action,
forecast, complete-schedule and Human-risk admission before issuing either
the exact plan certificate or the conservative workload-envelope certificate.

Paths inside a source spec are relative to the release/serve root.  Example::

  {
    "envelope_id": "base_fl2va_720p15_v1",
    "blueprint": "runtime/calibration/.../blueprint.json",
    "attention_calibrations": ["runtime/calibration/.../attention.json"],
    "forecast_calibration": "runtime/calibration/.../forecast.json",
    "schedule_calibration": "runtime/calibration/.../schedule.json",
    "risk_calibration": "runtime/calibration/.../risk.json",
    "reference_images": 0,
    "reference_audio": 0,
    "reference_videos": 0
  }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SERVE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVE_ROOT))

from h3serve.native_engine.planner import (  # noqa: E402
    FIXED_TOPK_ACTION_IMPLEMENTATION,
    ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ROUND215_ACTION_IMPLEMENTATION,
    ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
    V19CalibrationCatalog,
    V19CandidateFactory,
    V19CertifiedFrontierEntry,
    V19ForecastCalibrationCatalog,
    V19HumanRiskVector,
    V19PlanCertificate,
    V19PlanningError,
    V19PlanningRequest,
    V19ReferenceProfile,
    V19RiskCalibrationCatalog,
    V19ScheduleCostCatalog,
    V19WorkloadContext,
    V19WorkloadEnvelope,
    build_v19_bootstrap_registry,
    build_v19_certified_envelope_entry,
    load_v19_action_calibration,
    load_v19_candidate_blueprint,
    load_v19_forecast_calibration,
    load_v19_human_evidence,
    load_v19_plan_risk_calibration,
    load_v19_schedule_cost_calibration,
    save_v19_release_bundle,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-spec", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _registry():
    return build_v19_bootstrap_registry(implementation_ids={
        "fixed_topk": FIXED_TOPK_ACTION_IMPLEMENTATION,
        "round215": ROUND215_ACTION_IMPLEMENTATION,
        "round188": ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
        "round228": ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
        "round229": ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
    })


def _path(value: object) -> Path:
    path = (SERVE_ROOT / str(value)).resolve()
    if not path.is_relative_to(SERVE_ROOT.resolve()) or not path.is_file():
        raise V19PlanningError(f"V19 evidence path is missing/outside release: {path}")
    return path


def _record_digest(
    evidence: dict[str, str], evidence_id: str, path: Path
) -> None:
    digest = sha256_file(path)
    prior = evidence.get(evidence_id)
    if prior is not None and prior != digest:
        raise V19PlanningError(
            f"V19 evidence id resolves to different payloads: {evidence_id}"
        )
    evidence[evidence_id] = digest


def _record_known_digest(
    evidence: dict[str, str], evidence_id: str, digest: str
) -> None:
    prior = evidence.get(evidence_id)
    if prior is not None and prior != digest:
        raise V19PlanningError(
            f"V19 evidence id resolves to different payloads: {evidence_id}"
        )
    evidence[evidence_id] = digest


def _verify_human_provenance(
    risk,
    *,
    candidate_id: str,
    evidence: dict[str, str],
) -> None:
    records = {}
    for source in risk.sources:
        path = _path(source.relative_path)
        if sha256_file(path) != source.sha256:
            raise V19PlanningError(
                f"V19 Human source digest mismatch: {source.relative_path}"
            )
        _record_digest(
            evidence,
            f"human-source:{source.relative_path}",
            path,
        )
        human_set = load_v19_human_evidence(
            path,
            serve_root=SERVE_ROOT,
            require_artifacts=True,
        )
        for record in human_set.records:
            if record.evidence_id in records:
                raise V19PlanningError(
                    f"duplicate V19 Human case across sources: {record.evidence_id}"
                )
            records[record.evidence_id] = record
    for review in risk.reviews:
        record = records.get(review.case_id)
        if record is None or record.candidate_id != candidate_id:
            raise V19PlanningError(
                f"V19 Human case is missing or names another candidate: {review.case_id}"
            )
        if (
            record.mechanism != review.mechanism
            or record.attribution != review.attribution
            or record.dimensions != review.dimensions
            or record.artifact_sha256 != review.candidate_artifact_sha256
            or record.comparator_artifact_sha256
            != review.comparator_artifact_sha256
        ):
            raise V19PlanningError(
                f"V19 Human case disagrees with sealed risk payload: {review.case_id}"
            )
        _record_known_digest(
            evidence,
            f"human-video:{review.case_id}:candidate",
            review.candidate_artifact_sha256,
        )
        if review.comparator_artifact_sha256 is not None:
            _record_known_digest(
                evidence,
                f"human-video:{review.case_id}:comparator",
                review.comparator_artifact_sha256,
            )


def _exact_source(registry, document: dict, evidence: dict[str, str]):
    blueprint_path = _path(document["blueprint"])
    schedule_path = _path(document["schedule_calibration"])
    risk_path = _path(document["risk_calibration"])
    attention_paths = tuple(
        _path(value) for value in document["attention_calibrations"]
    )
    if not attention_paths:
        raise V19PlanningError("V19 source spec has no Attention calibration")

    schedule = load_v19_schedule_cost_calibration(
        schedule_path, registry=registry
    )
    workload = schedule.workload
    runtime = schedule.runtime
    attention_catalog = V19CalibrationCatalog(registry)
    for path in attention_paths:
        artifact = load_v19_action_calibration(
            path,
            registry=registry,
            expected_workload=workload,
            expected_runtime=runtime,
        )
        attention_catalog.add(artifact)
        _record_digest(evidence, artifact.calibration_id, path)

    forecast_catalog = None
    forecast_value = document.get("forecast_calibration")
    if forecast_value is not None:
        path = _path(forecast_value)
        artifact = load_v19_forecast_calibration(
            path,
            registry=registry,
            expected_workload=workload,
            expected_runtime=runtime,
        )
        forecast_catalog = V19ForecastCalibrationCatalog(registry)
        forecast_catalog.add(artifact)
        _record_digest(evidence, artifact.calibration_id, path)

    schedule_catalog = V19ScheduleCostCatalog(registry)
    schedule_catalog.add(schedule)
    _record_digest(evidence, schedule.calibration_id, schedule_path)

    risk = load_v19_plan_risk_calibration(
        risk_path,
        registry=registry,
        expected_workload=workload,
        expected_runtime=runtime,
    )
    risk_catalog = V19RiskCalibrationCatalog(registry)
    risk_catalog.add(risk)
    blueprint = load_v19_candidate_blueprint(blueprint_path)
    _verify_human_provenance(
        risk,
        candidate_id=blueprint.candidate_id,
        evidence=evidence,
    )
    _record_digest(evidence, risk.binding.risk_model_id, risk_path)

    reference_images = int(document.get("reference_images", 0))
    reference_audio = int(document.get("reference_audio", 0))
    reference_videos = int(document.get("reference_videos", 0))
    if min(reference_images, reference_audio, reference_videos) < 0:
        raise V19PlanningError("V19 reference counts cannot be negative")
    # The current exact schedule artifact records packed/condition tokens but
    # not the individual Ref2VA media inventory.  A source-spec author must
    # therefore not be able to promote FL2VA evidence into a reference-media
    # envelope by merely typing non-zero counts.  Ref2VA remains fully usable
    # through Dense fallback until its v2 schedule provenance is implemented
    # and calibrated separately.
    if workload.service_family == "reference" or any((
        reference_images, reference_audio, reference_videos
    )):
        raise V19PlanningError(
            "the current V19 schedule schema cannot certify reference-media profiles; "
            "leave Ref2VA on the capability-preserving Dense fallback"
        )
    exact_workload = V19WorkloadContext(
        model_variant=workload.model_variant,
        service_family=workload.service_family,
        packed_tokens=workload.packed_tokens,
        condition_count=workload.condition_count,
        reference_images=reference_images,
        reference_audio=reference_audio,
        reference_videos=reference_videos,
        device_arch=workload.device_arch,
        width=workload.width,
        height=workload.height,
        frames=workload.frames,
        steps=workload.steps,
        actual_step_indices=workload.actual_step_indices,
        sampler=workload.sampler,
        scheduler=workload.scheduler,
    )
    request = V19PlanningRequest(
        workload=exact_workload,
        maximum_cost_p90_ms=schedule.p90_ms,
        maximum_peak_vram_gib=24.0,
        risk_limits=V19HumanRiskVector(*(1.0 for _ in range(7))),
        runtime=runtime,
    )
    materialized = V19CandidateFactory(
        registry,
        attention_catalog,
        forecast_catalog=forecast_catalog,
        schedule_cost_catalog=schedule_catalog,
        risk_catalog=risk_catalog,
    ).materialize(
        request,
        blueprint,
        require_end_to_end_cost=True,
        require_human_risk=True,
    )
    if materialized.candidate.predicted_peak_vram_gib > 24.0:
        raise V19PlanningError(
            "certified V19 source exceeds the RTX 4090 24 GiB VRAM budget"
        )
    request = V19PlanningRequest(
        workload=exact_workload,
        maximum_cost_p90_ms=materialized.candidate.predicted_cost_p90_ms,
        maximum_peak_vram_gib=24.0,
        risk_limits=materialized.candidate.risk_ucb,
        runtime=runtime,
    )
    return str(document["envelope_id"]), V19CertifiedFrontierEntry(
        planning_request=request,
        materialized=materialized,
        certificate=V19PlanCertificate.issue(
            registry, request, materialized.candidate
        ),
    )


def main() -> int:
    args = parse_args()
    registry = _registry()
    grouped: dict[str, list[V19CertifiedFrontierEntry]] = {}
    evidence: dict[str, str] = {}
    for spec_path in args.source_spec:
        document = json.loads(spec_path.read_text(encoding="utf-8"))
        envelope_id, source = _exact_source(registry, document, evidence)
        grouped.setdefault(envelope_id, []).append(source)

    envelopes = []
    for envelope_id, source_rows in sorted(grouped.items()):
        sources = tuple(source_rows)
        first = sources[0].planning_request.workload
        profiles = tuple(sorted({
            V19ReferenceProfile.from_workload(row.planning_request.workload)
            for row in sources
        }))
        tokens = tuple(
            row.planning_request.workload.packed_tokens for row in sources
        )
        envelope = V19WorkloadEnvelope(
            envelope_id=envelope_id,
            model_variant=first.model_variant,
            service_family=first.service_family,
            device_arch=first.device_arch,
            width=int(first.width),
            height=int(first.height),
            frames=int(first.frames),
            steps=int(first.steps),
            sampler=first.sampler,
            scheduler=first.scheduler,
            min_packed_tokens=min(tokens),
            max_packed_tokens=max(tokens),
            reference_profiles=profiles,
        )
        envelopes.append(build_v19_certified_envelope_entry(
            registry, envelope=envelope, sources=sources
        ))

    save_v19_release_bundle(
        args.output,
        registry=registry,
        entries=tuple(envelopes),
        evidence_sha256=evidence,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "registry_digest": registry.digest,
        "envelopes": len(envelopes),
        "sources": sum(len(row.sources) for row in envelopes),
        "evidence": len(evidence),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
