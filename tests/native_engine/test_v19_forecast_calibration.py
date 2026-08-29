from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from h3serve.native_engine.planner.action_registry import build_v19_bootstrap_registry
from h3serve.native_engine.planner.joint_global_dp import (
    FIXED_TOPK_ACTION_IMPLEMENTATION,
    ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ROUND215_ACTION_IMPLEMENTATION,
    ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
)
from h3serve.native_engine.planner.v19_calibration import (
    V19CalibrationCatalog,
    V19CalibrationError,
    V19CalibrationWorkload,
    V19RuntimeFingerprint,
    V19SourceRecord,
    V19TimingMeasurement,
    create_v19_action_calibration,
)
from h3serve.native_engine.planner.v19_contracts import V19HumanRiskVector
from h3serve.native_engine.planner.v19_planner import (
    V19ActionUse,
    V19CandidatePlan,
    V19ForecastUse,
    V19ParetoPlanner,
    V19PlanningError,
    V19PlanningRequest,
    V19WorkloadContext,
)
from h3serve.native_engine.planner.v19_forecast_calibration import (
    V19ForecastCalibrationCatalog,
    V19ForecastCompositeKey,
    V19ForecastCompositeMeasurement,
    create_v19_forecast_calibration,
    load_v19_forecast_calibration,
)


def _registry():
    return build_v19_bootstrap_registry(implementation_ids={
        "fixed_topk": FIXED_TOPK_ACTION_IMPLEMENTATION,
        "round215": ROUND215_ACTION_IMPLEMENTATION,
        "round188": ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
        "round228": ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
        "round229": ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
    })


def _workload():
    return V19CalibrationWorkload(
        model_variant="base",
        service_family="first_last",
        width=1280,
        height=736,
        frames=362,
        packed_tokens=100_163,
        condition_count=0,
        steps=20,
        actual_step_indices=(0, 1, 2, 3, 4, 8, 12, 15, 18, 19),
    )


def _runtime():
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


def _key():
    return V19ForecastCompositeKey(
        forecast_step_indices=(5, 6, 7),
        preceding_actual_step=4,
        following_actual_step=8,
        anchor_depth=3,
        anchor_action_id="h3.attention.mtcr_head_rail.round229.v1",
        anchor_canonical_action="sparse_topk_0.0625",
        extrapolator_id="native_depth3_local_directional_v1",
        correction_id="next_actual_full_stack_v1",
    )


def _artifact():
    return create_v19_forecast_calibration(
        registry=_registry(),
        action_id="h3.forecast.directional.anchor3.round229.v1",
        calibration_id="v19_round229_forecast_composite_cost_v1",
        workload=_workload(),
        runtime=_runtime(),
        measurements=(V19ForecastCompositeMeasurement(
            key=_key(),
            warm_samples_ms=(3000.0, 3100.0, 3050.0),
            initialization_samples_ms=(3300.0,),
            peak_vram_gib_samples=(17.0, 17.1, 17.05),
        ),),
        sources=(V19SourceRecord(
            source_id="three_warm_runs",
            relative_path="calibration/forecast.json",
            sha256="5" * 64,
        ),),
        complete=True,
    )


class V19ForecastCalibrationTests(unittest.TestCase):
    def test_composite_requires_anchor_run_and_immediate_correction(self) -> None:
        with self.assertRaises(V19CalibrationError):
            V19ForecastCompositeKey(
                forecast_step_indices=(5, 7),
                preceding_actual_step=4,
                following_actual_step=8,
                anchor_depth=3,
                anchor_action_id="anchor",
                anchor_canonical_action="sparse",
                extrapolator_id="directional",
                correction_id="actual",
            )

    def test_round_trip_and_exact_catalog_lookup(self) -> None:
        artifact = _artifact()
        self.assertTrue(artifact.planner_ready)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forecast.json"
            path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")
            loaded = load_v19_forecast_calibration(path, registry=_registry())
        catalog = V19ForecastCalibrationCatalog(_registry())
        catalog.add(loaded)
        cost = catalog.estimate(
            action_id="h3.forecast.directional.anchor3.round229.v1",
            key=_key(),
            workload=_workload(),
            runtime=_runtime(),
        )
        self.assertEqual(cost.p50_ms, 3050.0)
        self.assertEqual(cost.p90_ms, 3100.0)
        self.assertEqual(cost.peak_vram_gib, 17.1)

    def test_different_run_is_not_interpolated(self) -> None:
        catalog = V19ForecastCalibrationCatalog(_registry())
        catalog.add(_artifact())
        with self.assertRaises(V19CalibrationError):
            catalog.estimate(
                action_id="h3.forecast.directional.anchor3.round229.v1",
                key=V19ForecastCompositeKey(
                    forecast_step_indices=(5,),
                    preceding_actual_step=4,
                    following_actual_step=6,
                    anchor_depth=3,
                    anchor_action_id="h3.attention.mtcr_head_rail.round229.v1",
                    anchor_canonical_action="sparse_topk_0.0625",
                    extrapolator_id="native_depth3_local_directional_v1",
                    correction_id="next_actual_full_stack_v1",
                ),
                workload=_workload(),
                runtime=_runtime(),
            )

    def test_strict_plan_prices_actual_cells_and_forecast_composite_together(self) -> None:
        registry = _registry()
        workload = V19CalibrationWorkload(
            model_variant="base",
            service_family="first_last",
            width=1280,
            height=736,
            frames=124,
            packed_tokens=34_871,
            condition_count=0,
            steps=5,
            actual_step_indices=(0, 4),
        )
        key = V19ForecastCompositeKey(
            forecast_step_indices=(1, 2, 3),
            preceding_actual_step=0,
            following_actual_step=4,
            anchor_depth=3,
            anchor_action_id="h3.attention.mtcr_head_rail.round229.v1",
            anchor_canonical_action="sparse_topk_0.0625",
            extrapolator_id="native_depth3_local_directional_v1",
            correction_id="next_actual_full_stack_v1",
        )
        source = V19SourceRecord(
            source_id="unit",
            relative_path="calibration/unit.json",
            sha256="5" * 64,
        )
        dense_artifact = create_v19_action_calibration(
            registry=registry,
            action_id="h3.attention.dense.sage_per_warp.sm89.v1",
            calibration_id="v19_dense_full_head_cost_v1",
            workload=workload,
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
                for step in workload.actual_step_indices
                for layer in range(50)
            ),
            sources=(source,),
            timing_scope="attention_layer_call",
            complete=True,
        )
        forecast_artifact = create_v19_forecast_calibration(
            registry=registry,
            action_id="h3.forecast.directional.anchor3.round229.v1",
            calibration_id="v19_round229_forecast_composite_cost_v1",
            workload=workload,
            runtime=_runtime(),
            measurements=(V19ForecastCompositeMeasurement(
                key=key,
                warm_samples_ms=(30.0, 32.0, 31.0),
                peak_vram_gib_samples=(11.0, 11.0, 11.0),
            ),),
            sources=(source,),
            complete=True,
        )
        attention_catalog = V19CalibrationCatalog(registry)
        attention_catalog.add(dense_artifact)
        forecast_catalog = V19ForecastCalibrationCatalog(registry)
        forecast_catalog.add(forecast_artifact)
        planner = V19ParetoPlanner(
            registry,
            attention_catalog,
            forecast_catalog,
        )
        request = V19PlanningRequest(
            workload=V19WorkloadContext(
                model_variant="base",
                service_family="first_last",
                packed_tokens=34_871,
                condition_count=0,
                width=1280,
                height=736,
                frames=124,
                steps=5,
                actual_step_indices=(0, 4),
            ),
            maximum_cost_p90_ms=200.0,
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
        candidate = V19CandidatePlan(
            candidate_id="actual_plus_forecast",
            action_uses=(
                V19ActionUse(
                    action_id="h3.attention.dense.sage_per_warp.sm89.v1",
                    canonical_action="dense",
                    step_indices=(0, 4),
                ),
                V19ForecastUse(
                    action_id="h3.forecast.directional.anchor3.round229.v1",
                    composite_key=key,
                ),
            ),
            predicted_cost_p50_ms=131.0,
            predicted_cost_p90_ms=132.0,
            risk_ucb=V19HumanRiskVector(),
            predicted_peak_vram_gib=11.0,
            evidence_ids=(
                "v19_dense_full_head_cost_v1",
                "v19_round229_forecast_composite_cost_v1",
            ),
        )
        self.assertTrue(planner.certify(request, candidate).certificate_digest)
        with self.assertRaises(V19PlanningError):
            planner.certify(
                request,
                V19CandidatePlan(
                    candidate_id="missing_forecast",
                    action_uses=(candidate.action_uses[0],),
                    predicted_cost_p50_ms=100.0,
                    predicted_cost_p90_ms=100.0,
                    risk_ucb=V19HumanRiskVector(),
                    predicted_peak_vram_gib=10.0,
                    evidence_ids=("v19_dense_full_head_cost_v1",),
                ),
            )


if __name__ == "__main__":
    unittest.main()
