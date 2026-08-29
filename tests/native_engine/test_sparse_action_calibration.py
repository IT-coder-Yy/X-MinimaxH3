from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from h3serve.native_engine.planner import (
    H3LayerBand,
    SparseBudgetError,
    build_measured_h3_sparse_cells,
    load_h3_sparse_action_calibration,
    minimum_measured_schedule_cost,
    solve_measured_h3_sparse_schedule,
)


def _artifact(path: Path, *, full_head: bool = True) -> Path:
    layers = []
    for layer in range(50):
        candidates = [
            {
                "name": "sparse_topk_0.0625",
                "topk": 0.0625,
                "full_head_ms": 1000.0 if layer == 0 else 10.0,
                "global_relative_rms": 0.30 if 30 <= layer <= 45 else 0.10,
            },
            {
                "name": "sparse_topk_0.1",
                "topk": 0.10,
                "full_head_ms": 20.0,
                "global_relative_rms": 0.15 if 30 <= layer <= 45 else 0.05,
            },
        ]
        layers.append(
            {
                "layer": layer,
                "dense_ms": 40.0,
                "candidates": candidates,
            }
        )
    document = {
        "contract": {
            "engine": "original",
            "sequence_tokens": 100_163,
            "heads": 56,
            "timing_scope": (
                "one complete 56-head call" if full_head else "grouped probe"
            ),
            "dense_result_returned": True,
            "weights_modified": False,
        },
        "steps": [{"step_index": 3, "layers": layers, "complete": True}],
        "complete": True,
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class SparseActionCalibrationTests(unittest.TestCase):
    def test_loader_uses_hot_robust_band_cost_not_first_call_outlier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            calibration = load_h3_sparse_action_calibration(
                _artifact(Path(directory) / "actions.json")
            )
        early = calibration.actions_by_band[H3LayerBand.EARLY]
        self.assertEqual(early[0].measured_cost_ms, 150.0)
        self.assertEqual(early[-1].name, "dense")
        self.assertEqual(early[-1].measured_cost_ms, 600.0)

    def test_loader_rejects_grouped_timings_as_budget_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _artifact(Path(directory) / "grouped.json", full_head=False)
            with self.assertRaises(SparseBudgetError):
                load_h3_sparse_action_calibration(path)

    def test_default_v1_floor_reproduces_exact_opening_and_causal_island(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            calibration = load_h3_sparse_action_calibration(
                _artifact(Path(directory) / "actions.json")
            )
        cells = build_measured_h3_sparse_cells(calibration, (0, 1, 2, 3))
        dense_rank = calibration.maximum_fidelity_rank
        opening = [cell for cell in cells if cell.key.actual_step == 0]
        causal = [
            cell
            for cell in cells
            if cell.key.layer_band
            in (
                H3LayerBand.CAUSAL,
                H3LayerBand.CAUSAL_DETAIL,
                H3LayerBand.CAUSAL_TERMINAL,
            )
        ]
        self.assertTrue(all(cell.minimum_fidelity_rank == dense_rank for cell in opening))
        self.assertTrue(all(cell.minimum_fidelity_rank == dense_rank for cell in causal))

    def test_solver_expands_band_choices_to_all_50_physical_layers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            calibration = load_h3_sparse_action_calibration(
                _artifact(Path(directory) / "actions.json")
            )
        actual_steps = (0, 1, 2, 3)
        cells = build_measured_h3_sparse_cells(calibration, actual_steps)
        minimum = minimum_measured_schedule_cost(cells)
        schedule, physical = solve_measured_h3_sparse_schedule(
            calibration,
            actual_steps,
            attention_budget_ms=minimum,
        )
        self.assertLessEqual(schedule.estimated_cost_ms, minimum + 1.0e-9)
        self.assertEqual(len(physical), len(actual_steps) * 50)
        self.assertEqual(physical[(0, 0)], "dense")
        self.assertEqual(physical[(1, 30)], "dense")


if __name__ == "__main__":
    unittest.main()
