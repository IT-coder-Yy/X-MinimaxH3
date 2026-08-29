from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from h3serve.native_engine.planner import (
    FIXED_TOPK_ACTION_IMPLEMENTATION,
    ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ROUND215_ACTION_IMPLEMENTATION,
    ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
    V19CalibrationError,
    V19CalibrationWorkload,
    V19RuntimeFingerprint,
    V19ScheduleCostCatalog,
    V19SourceRecord,
    build_v19_bootstrap_registry,
    create_v19_schedule_cost_calibration,
    load_v19_schedule_cost_calibration,
)


def _registry():
    return build_v19_bootstrap_registry(implementation_ids={
        "fixed_topk": FIXED_TOPK_ACTION_IMPLEMENTATION,
        "round215": ROUND215_ACTION_IMPLEMENTATION,
        "round188": ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
        "round228": ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
        "round229": ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
    })


def _runtime():
    return V19RuntimeFingerprint(
        gpu_name="RTX 4090",
        device_arch="sm89",
        torch_version="2.8",
        cuda_runtime="12.6",
        driver_version="580",
        quant_backend="cuda",
        comfy_kitchen_cuda_sha256="1" * 64,
        sageattention_sm89_sha256="2" * 64,
        action_source_sha256="3" * 64,
        planner_source_sha256="4" * 64,
    )


def _workload():
    return V19CalibrationWorkload(
        model_variant="base",
        service_family="first_last",
        width=1280,
        height=736,
        frames=124,
        packed_tokens=34871,
        condition_count=0,
        steps=20,
        actual_step_indices=(0, 1, 2, 3, 4, 8, 12, 15, 18, 19),
    )


class V19ScheduleCalibrationTests(unittest.TestCase):
    def _artifact(self):
        return create_v19_schedule_cost_calibration(
            registry=_registry(),
            calibration_id="v19_candidate_repeat3_e2e",
            execution_digest="a" * 64,
            action_ids=(
                "h3.attention.mtcr_head_rail.round229.v1",
                "h3.forecast.directional.anchor3.round229.v1",
            ),
            workload=_workload(),
            runtime=_runtime(),
            total_samples_ms=(220000.0, 221000.0, 219000.0),
            denoise_samples_ms=(180000.0, 181000.0, 179000.0),
            peak_vram_gib_samples=(17.1, 17.2, 17.15),
            sources=(V19SourceRecord(
                source_id="repeat3",
                relative_path="runtime/repeat3.json",
                sha256="5" * 64,
            ),),
            complete=True,
        )

    def test_round_trip_and_exact_lookup(self) -> None:
        artifact = self._artifact()
        self.assertTrue(artifact.planner_ready)
        self.assertEqual(artifact.p50_ms, 220000.0)
        self.assertEqual(artifact.p90_ms, 221000.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.json"
            path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")
            loaded = load_v19_schedule_cost_calibration(
                path, registry=_registry()
            )
        catalog = V19ScheduleCostCatalog(_registry())
        catalog.add(loaded)
        cost = catalog.estimate(
            execution_digest="a" * 64,
            workload=_workload(),
            runtime=_runtime(),
            action_ids=loaded.binding.action_ids,
        )
        self.assertEqual(cost.p90_ms, 221000.0)
        self.assertEqual(cost.peak_vram_gib, 17.2)

    def test_schedule_identity_never_interpolates(self) -> None:
        catalog = V19ScheduleCostCatalog(_registry())
        catalog.add(self._artifact())
        with self.assertRaises(V19CalibrationError):
            catalog.estimate(
                execution_digest="b" * 64,
                workload=_workload(),
                runtime=_runtime(),
                action_ids=self._artifact().binding.action_ids,
            )


if __name__ == "__main__":
    unittest.main()
