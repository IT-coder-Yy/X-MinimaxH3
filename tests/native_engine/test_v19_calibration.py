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
    V19CalibrationError,
    V19CalibrationCatalog,
    V19CalibrationWorkload,
    V19NumericalErrorSample,
    V19RuntimeFingerprint,
    V19SourceRecord,
    V19TimingMeasurement,
    conservative_quantile,
    create_v19_action_calibration,
    load_v19_action_calibration,
)
from h3serve.native_engine.planner.v19_contracts import V19HumanRiskVector
from h3serve.native_engine.planner.v19_planner import (
    V19ActionUse,
    V19CandidatePlan,
    V19ParetoPlanner,
    V19PlanningError,
    V19PlanningRequest,
    V19WorkloadContext,
)


CALIBRATION_ID = "test_physical_cost_v1"


def _registry() -> ActionRegistry:
    return ActionRegistry((RegisteredAction(
        action_id="test.sparse.v1",
        implementation_id="physical_sparse_v1",
        kind=ActionKind.SPARSE_ATTENTION,
        executor_id="test",
        canonical_actions=("sparse_topk_0.1",),
        exact=False,
        evidence_status=EvidenceStatus.CALIBRATED,
        calibration_ids=(CALIBRATION_ID,),
        planner_eligible=True,
    ),))


def _workload() -> V19CalibrationWorkload:
    return V19CalibrationWorkload(
        model_variant="base",
        service_family="first_last",
        width=1280,
        height=736,
        frames=362,
        packed_tokens=100_163,
        condition_count=2,
        steps=20,
        actual_step_indices=(0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 18, 19),
    )


def _runtime() -> V19RuntimeFingerprint:
    return V19RuntimeFingerprint(
        gpu_name="NVIDIA GeForce RTX 4090",
        device_arch="sm89",
        torch_version="2.8.0+cu128",
        cuda_runtime="12.8",
        driver_version="570.0",
        quant_backend="cuda",
        comfy_kitchen_cuda_sha256="1" * 64,
        sageattention_sm89_sha256="2" * 64,
        action_source_sha256="3" * 64,
        planner_source_sha256="4" * 64,
    )


def _artifact(*, samples=(10.0, 12.0, 11.0)):
    return create_v19_action_calibration(
        registry=_registry(),
        action_id="test.sparse.v1",
        calibration_id=CALIBRATION_ID,
        workload=_workload(),
        runtime=_runtime(),
        measurements=(V19TimingMeasurement(
            canonical_action="sparse_topk_0.1",
            step_index=3,
            layer_start=0,
            layer_stop=1,
            warm_samples_ms=samples,
            initialization_samples_ms=(30.0,),
            peak_vram_gib_samples=(20.0, 20.25, 20.125),
            numerical_error_samples=(V19NumericalErrorSample(
                mean_cosine=0.99,
                min_cosine=0.95,
                global_relative_rms=0.1,
                mean_head_relative_rms=0.11,
                max_head_relative_rms=0.3,
                max_relative_l1=0.2,
            ),),
        ),),
        sources=(V19SourceRecord(
            source_id="repeat_set",
            relative_path="calibration/repeat_set.json",
            sha256="5" * 64,
        ),),
        timing_scope="attention_layer_call",
        complete=True,
    )


class V19CalibrationTests(unittest.TestCase):
    def test_nearest_rank_p90_is_conservative_for_three_samples(self) -> None:
        self.assertEqual(conservative_quantile((10.0, 12.0, 11.0), 0.5), 11.0)
        self.assertEqual(conservative_quantile((10.0, 12.0, 11.0), 0.9), 12.0)

    def test_round_trip_binds_registry_workload_runtime_and_raw_samples(self) -> None:
        artifact = _artifact()
        self.assertTrue(artifact.planner_ready)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")
            loaded = load_v19_action_calibration(
                path,
                registry=_registry(),
                expected_workload=_workload(),
                expected_runtime=_runtime(),
            )
        self.assertEqual(loaded.payload_sha256, artifact.payload_sha256)
        self.assertEqual(loaded.measurements[0].p50_ms, 11.0)
        self.assertEqual(loaded.measurements[0].p90_ms, 12.0)
        self.assertEqual(
            loaded.measurements[0].numerical_error_samples[0].global_relative_rms,
            0.1,
        )
        self.assertFalse(loaded.measurements[0].numerical_error_repeated)

    def test_single_historical_sample_is_evidence_but_not_planner_ready(self) -> None:
        artifact = _artifact(samples=(10.0,))
        self.assertFalse(artifact.planner_ready)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")
            loaded = load_v19_action_calibration(
                path, registry=_registry(), require_planner_ready=False
            )
            self.assertFalse(loaded.planner_ready)
            with self.assertRaises(V19CalibrationError):
                load_v19_action_calibration(path, registry=_registry())

    def test_latency_samples_without_peak_vram_are_not_planner_ready(self) -> None:
        artifact = _artifact()
        row = replace(artifact.measurements[0], peak_vram_gib_samples=())
        unsealed = create_v19_action_calibration(
            registry=_registry(),
            action_id="test.sparse.v1",
            calibration_id=CALIBRATION_ID,
            workload=_workload(),
            runtime=_runtime(),
            measurements=(row,),
            sources=artifact.sources,
            timing_scope="attention_layer_call",
            complete=True,
        )
        self.assertFalse(unsealed.planner_ready)
        with self.assertRaises(V19CalibrationError):
            unsealed.require_planner_ready()

    def test_tampered_p90_and_payload_are_rejected(self) -> None:
        document = _artifact().to_dict()
        document["measurements"][0]["p90_ms"] = 9.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(V19CalibrationError):
                load_v19_action_calibration(path, registry=_registry())

        document = _artifact().to_dict()
        samples = list(document["measurements"][0]["warm_samples_ms"])
        samples[0] = 999.0
        document["measurements"][0]["warm_samples_ms"] = samples
        # Remove derived summaries so only the payload seal catches the edit.
        document["measurements"][0].pop("p50_ms")
        document["measurements"][0].pop("p90_ms")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered_payload.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(V19CalibrationError):
                load_v19_action_calibration(path, registry=_registry())

    def test_workload_or_runtime_drift_fails_closed(self) -> None:
        artifact = _artifact()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")
            with self.assertRaises(V19CalibrationError):
                load_v19_action_calibration(
                    path,
                    registry=_registry(),
                    expected_workload=replace(_workload(), packed_tokens=100_164),
                )
            with self.assertRaises(V19CalibrationError):
                load_v19_action_calibration(
                    path,
                    registry=_registry(),
                    expected_runtime=replace(_runtime(), driver_version="571.0"),
                )

    def test_catalog_prices_only_exact_calibrated_cells(self) -> None:
        catalog = V19CalibrationCatalog(_registry())
        catalog.add(_artifact())
        cost = catalog.estimate_schedule(
            (V19ActionUse(
                action_id="test.sparse.v1",
                canonical_action="sparse_topk_0.1",
                step_indices=(3,),
                layer_start=0,
                layer_stop=1,
            ),),
            workload=_workload(),
            runtime=_runtime(),
        )
        self.assertEqual(cost.p50_ms, 11.0)
        self.assertEqual(cost.p90_ms, 12.0)
        self.assertEqual(cost.peak_vram_gib, 20.25)
        self.assertEqual(cost.evidence_ids, (CALIBRATION_ID,))
        cell = catalog.lookup_cell(
            action_id="test.sparse.v1",
            canonical_action="sparse_topk_0.1",
            step_index=3,
            layer_index=0,
            workload=_workload(),
            runtime=_runtime(),
        )
        self.assertEqual(cell.p50_ms, 11.0)
        self.assertEqual(cell.numerical_error_samples[0].global_relative_rms, 0.1)
        with self.assertRaises(V19CalibrationError):
            catalog.estimate_schedule(
                (V19ActionUse(
                    action_id="test.sparse.v1",
                    canonical_action="sparse_topk_0.1",
                    step_indices=(3,),
                    layer_start=1,
                    layer_stop=2,
                ),),
                workload=_workload(),
                runtime=_runtime(),
            )

    def test_strict_planner_rejects_unmeasured_or_repriced_candidate(self) -> None:
        registry = _registry()
        catalog = V19CalibrationCatalog(registry)
        catalog.add(_artifact())
        planner = V19ParetoPlanner(
            registry, catalog, require_complete_schedule=False
        )
        request = V19PlanningRequest(
            workload=V19WorkloadContext(
                model_variant="base",
                service_family="first_last",
                packed_tokens=100_163,
                condition_count=2,
                width=1280,
                height=736,
                frames=362,
                steps=20,
                actual_step_indices=(0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 18, 19),
            ),
            maximum_cost_p90_ms=20.0,
            risk_limits=V19HumanRiskVector(
                prompt_adherence=0.1,
                contact_causality=0.1,
                trajectory_continuity=0.1,
                temporal_clarity=0.1,
                identity_binding=0.1,
                audio_integrity=0.1,
                anomaly=0.1,
            ),
            runtime=_runtime(),
        )
        use = V19ActionUse(
            action_id="test.sparse.v1",
            canonical_action="sparse_topk_0.1",
            step_indices=(3,),
            layer_start=0,
            layer_stop=1,
        )
        candidate = V19CandidatePlan(
            candidate_id="strict",
            action_uses=(use,),
            predicted_cost_p50_ms=11.0,
            predicted_cost_p90_ms=12.0,
            risk_ucb=V19HumanRiskVector(),
            predicted_peak_vram_gib=20.25,
            evidence_ids=(CALIBRATION_ID,),
        )
        self.assertTrue(planner.certify(request, candidate).certificate_digest)
        with self.assertRaises(V19PlanningError):
            planner.certify(
                request, replace(candidate, predicted_cost_p90_ms=11.5)
            )


if __name__ == "__main__":
    unittest.main()
