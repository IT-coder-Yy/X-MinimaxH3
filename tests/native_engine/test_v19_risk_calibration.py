from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from h3serve.native_engine.planner.action_registry import (
    ActionKind,
    ActionRegistry,
    EvidenceStatus,
    RegisteredAction,
)
from h3serve.native_engine.planner.v19_calibration import (
    V19CalibrationCatalog,
    V19CalibrationWorkload,
    V19RuntimeFingerprint,
    V19SourceRecord,
    V19TimingMeasurement,
    create_v19_action_calibration,
)
from h3serve.native_engine.planner.v19_contracts import V19HumanRiskVector
from h3serve.native_engine.planner.v19_candidates import (
    V19CandidateBlueprint,
    V19CandidateFactory,
)
from h3serve.native_engine.planner.v19_evidence import V19DimensionLabels
from h3serve.native_engine.planner.v19_planner import (
    V19ActionUse,
    V19CandidatePlan,
    V19ParetoPlanner,
    V19PlanningError,
    V19PlanningRequest,
    V19WorkloadContext,
)
from h3serve.native_engine.planner.v19_risk_calibration import (
    V19RiskCalibrationCatalog,
    V19RiskCalibrationError,
    V19RiskReview,
    create_v19_plan_risk_calibration,
    load_v19_plan_risk_calibration,
    wilson_upper_bound,
)


ACTION_ID = "test.dense.v1"
COST_ID = "test_cost_v1"
RISK_ID = "test_human_risk_v1"


def _registry() -> ActionRegistry:
    return ActionRegistry((RegisteredAction(
        action_id=ACTION_ID,
        implementation_id="test_dense_physical_v1",
        kind=ActionKind.DENSE_ATTENTION,
        executor_id="test",
        canonical_actions=("dense",),
        exact=False,
        evidence_status=EvidenceStatus.CALIBRATED,
        calibration_ids=(COST_ID,),
        planner_eligible=True,
    ),))


def _workload() -> V19CalibrationWorkload:
    return V19CalibrationWorkload(
        model_variant="base",
        service_family="first_last",
        width=864,
        height=480,
        frames=124,
        packed_tokens=34_871,
        condition_count=0,
        steps=2,
        actual_step_indices=(0, 1),
    )


def _runtime() -> V19RuntimeFingerprint:
    return V19RuntimeFingerprint(
        gpu_name="NVIDIA GeForce RTX 4090",
        device_arch="sm89",
        torch_version="2.8.0+cu126",
        cuda_runtime="12.6",
        driver_version="580.0",
        quant_backend="cuda",
        comfy_kitchen_cuda_sha256="1" * 64,
        sageattention_sm89_sha256="2" * 64,
        action_source_sha256="3" * 64,
        planner_source_sha256="4" * 64,
    )


def _source() -> V19SourceRecord:
    return V19SourceRecord(
        source_id="human_ab",
        relative_path="calibration/human_ab.json",
        sha256="5" * 64,
    )


def _candidate(*, risk=V19HumanRiskVector(), evidence=(COST_ID,)) -> V19CandidatePlan:
    return V19CandidatePlan(
        candidate_id="candidate",
        action_uses=(V19ActionUse(
            action_id=ACTION_ID,
            canonical_action="dense",
            step_indices=(0, 1),
        ),),
        predicted_cost_p50_ms=100.0,
        predicted_cost_p90_ms=100.0,
        risk_ucb=risk,
        predicted_peak_vram_gib=10.0,
        evidence_ids=evidence,
    )


def _accepted_review(index: int) -> V19RiskReview:
    return V19RiskReview(
        case_id=f"case_{index}",
        mechanism=f"mechanism_{index}",
        attribution="candidate_positive",
        dimensions=V19DimensionLabels(
            prompt_adherence="accept",
            contact_causality="accept",
            trajectory_continuity="accept",
            temporal_clarity="accept",
            identity_binding="accept",
            audio_integrity="accept",
            anomaly="accept",
        ),
        candidate_artifact_sha256=f"{(index % 9) + 1}" * 64,
    )


def _risk_artifact(*, reviews=None):
    candidate = _candidate()
    return create_v19_plan_risk_calibration(
        registry=_registry(),
        execution_digest=candidate.execution_digest,
        risk_model_id=RISK_ID,
        action_ids=(ACTION_ID,),
        workload=_workload(),
        runtime=_runtime(),
        reviews=tuple(reviews or (_accepted_review(0), _accepted_review(1), _accepted_review(2))),
        sources=(_source(),),
        complete=True,
    )


class V19RiskCalibrationTests(unittest.TestCase):
    def test_wilson_bound_is_conservative_and_monotone(self) -> None:
        zero_of_three = wilson_upper_bound(0, 3, z_score=2.45)
        one_of_three = wilson_upper_bound(1, 3, z_score=2.45)
        zero_of_twelve = wilson_upper_bound(0, 12, z_score=2.45)
        self.assertGreater(zero_of_three, 0.0)
        self.assertGreater(one_of_three, zero_of_three)
        self.assertLess(zero_of_twelve, zero_of_three)

    def test_shared_failure_cannot_calibrate_acceleration_risk(self) -> None:
        with self.assertRaises(V19RiskCalibrationError):
            V19RiskReview(
                case_id="shared",
                mechanism="model_limit",
                attribution="shared_failure",
                dimensions=V19DimensionLabels(contact_causality="reject"),
                candidate_artifact_sha256="6" * 64,
            )

    def test_round_trip_seals_raw_reviews_and_exact_identity(self) -> None:
        artifact = _risk_artifact()
        self.assertTrue(artifact.planner_ready)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "risk.json"
            path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")
            loaded = load_v19_plan_risk_calibration(
                path,
                registry=_registry(),
                expected_workload=_workload(),
                expected_runtime=_runtime(),
            )
        self.assertEqual(loaded.payload_sha256, artifact.payload_sha256)
        self.assertEqual(loaded.risk_ucb, artifact.risk_ucb)

    def test_unreported_dimension_or_tamper_fails_closed(self) -> None:
        partial = tuple(
            V19RiskReview(
                case_id=f"partial_{index}",
                mechanism="motion",
                attribution="candidate_positive",
                dimensions=V19DimensionLabels(contact_causality="accept"),
                candidate_artifact_sha256=f"{(index % 9) + 1}" * 64,
            )
            for index in range(3)
        )
        self.assertFalse(_risk_artifact(reviews=partial).planner_ready)

        document = _risk_artifact().to_dict()
        document["reviews"][0]["dimensions"]["audio_integrity"] = "reject"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(V19RiskCalibrationError):
                load_v19_plan_risk_calibration(path, registry=_registry())

        document = _risk_artifact().to_dict()
        document["reviews"][0]["candidate_artifact_sha256"] = "f" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact-tampered.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(V19RiskCalibrationError):
                load_v19_plan_risk_calibration(path, registry=_registry())

    def test_strict_planner_requires_exact_plan_bound_human_risk(self) -> None:
        registry = _registry()
        cost_artifact = create_v19_action_calibration(
            registry=registry,
            action_id=ACTION_ID,
            calibration_id=COST_ID,
            workload=_workload(),
            runtime=_runtime(),
            measurements=tuple(
                V19TimingMeasurement(
                    canonical_action="dense",
                    step_index=step,
                    layer_start=layer,
                    layer_stop=layer + 1,
                    warm_samples_ms=(1.0, 1.0, 1.0),
                    peak_vram_gib_samples=(10.0, 10.0, 10.0),
                )
                for step in (0, 1)
                for layer in range(50)
            ),
            sources=(_source(),),
            timing_scope="attention_layer_call",
            complete=True,
        )
        cost_catalog = V19CalibrationCatalog(registry)
        cost_catalog.add(cost_artifact)
        factory_result = V19CandidateFactory(
            registry, cost_catalog
        ).materialize(
            V19PlanningRequest(
                workload=V19WorkloadContext(
                    model_variant="base",
                    service_family="first_last",
                    packed_tokens=34_871,
                    condition_count=0,
                    width=864,
                    height=480,
                    frames=124,
                    steps=2,
                    actual_step_indices=(0, 1),
                ),
                maximum_cost_p90_ms=101.0,
                risk_limits=V19HumanRiskVector(*(1.0 for _ in range(7))),
                runtime=_runtime(),
            ),
            V19CandidateBlueprint(
                candidate_id="factory",
                action_uses=(_candidate().action_uses[0],),
            ),
        )
        self.assertFalse(factory_result.human_risk_calibrated)
        self.assertEqual(factory_result.candidate.predicted_cost_p50_ms, 100.0)
        self.assertEqual(factory_result.candidate.predicted_peak_vram_gib, 10.0)
        self.assertEqual(
            factory_result.candidate.risk_ucb,
            V19HumanRiskVector(*(1.0 for _ in range(7))),
        )
        risk_artifact = _risk_artifact()
        risk_catalog = V19RiskCalibrationCatalog(registry)
        risk_catalog.add(risk_artifact)
        planner = V19ParetoPlanner(
            registry,
            cost_catalog,
            risk_calibration_catalog=risk_catalog,
            require_human_risk_calibration=True,
        )
        request = V19PlanningRequest(
            workload=V19WorkloadContext(
                model_variant="base",
                service_family="first_last",
                packed_tokens=34_871,
                condition_count=0,
                width=864,
                height=480,
                frames=124,
                steps=2,
                actual_step_indices=(0, 1),
            ),
            maximum_cost_p90_ms=101.0,
            risk_limits=V19HumanRiskVector(*(1.0 for _ in range(7))),
            runtime=_runtime(),
        )
        candidate = _candidate(
            risk=risk_artifact.risk_ucb,
            evidence=(COST_ID, RISK_ID),
        )
        self.assertTrue(planner.certify(request, candidate).certificate_digest)
        with self.assertRaises(V19PlanningError):
            planner.certify(request, replace(candidate, risk_ucb=V19HumanRiskVector()))
        different_schedule = replace(
            candidate,
            action_uses=(V19ActionUse(
                action_id=ACTION_ID,
                canonical_action="dense",
                step_indices=(0,),
            ),),
            predicted_cost_p50_ms=50.0,
            predicted_cost_p90_ms=50.0,
        )
        with self.assertRaises(V19PlanningError):
            planner.certify(request, different_schedule)


if __name__ == "__main__":
    unittest.main()
