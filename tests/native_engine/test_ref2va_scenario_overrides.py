from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.benchmark_native_hot_session import load_scenarios, parse_args


class Ref2VAScenarioOverrideTest(unittest.TestCase):
    def test_hot_batch_preserves_per_request_schedule_and_cache_policy(self) -> None:
        root = Path(__file__).resolve().parents[2]
        manifest = root / "benchmarks/ref2va_extreme/ref2va_enhanced_v1_hot_batch.json"
        with patch.object(
            sys,
            "argv",
            [
                "benchmark_native_hot_session.py",
                "--engine", "reference",
                "--scenario-manifest", str(manifest),
                "--repeat", "1",
            ],
        ):
            args = parse_args()
        scenarios = load_scenarios(args)
        self.assertEqual(len(scenarios), 7)
        self.assertEqual(scenarios[0]["actual_step_indices"], [0, 1])
        self.assertFalse(scenarios[1]["cache_condition_rows"])
        self.assertTrue(scenarios[2]["cache_condition_embeddings"])
        self.assertEqual(len(scenarios[-1]["actual_step_indices"]), 12)


if __name__ == "__main__":
    unittest.main()
