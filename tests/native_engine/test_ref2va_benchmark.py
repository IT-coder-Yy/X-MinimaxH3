from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from h3serve.native_engine.planner import (
    JOINT_POLICY_V18_FORECAST_AWARE_FRONTIER_GLOBAL_DP,
)
from scripts.benchmark_native_hot_session import (
    load_scenarios,
    normalize_joint_policy_id,
    parse_args,
)


class Ref2VABenchmarkContractTest(unittest.TestCase):
    def test_round229_joint_policy_alias_is_normalized(self) -> None:
        self.assertEqual(
            normalize_joint_policy_id("round229"),
            JOINT_POLICY_V18_FORECAST_AWARE_FRONTIER_GLOBAL_DP,
        )

    def test_manifest_repeat_keeps_outputs_distinct_in_one_session(self) -> None:
        serve_root = Path(__file__).resolve().parents[2]
        manifest = serve_root / "benchmarks/ref2va_extreme/ref2va_multiref_8s.json"
        argv = [
            "benchmark_native_hot_session.py",
            "--engine", "reference",
            "--scenario-manifest", str(manifest),
            "--repeat", "2",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        scenarios = load_scenarios(args)
        self.assertEqual(
            [item["name"] for item in scenarios],
            ["campus_bag_dialogue_repeat1", "campus_bag_dialogue_repeat2"],
        )
        self.assertEqual([item["repeat_index"] for item in scenarios], [1, 2])
        self.assertEqual(len(scenarios[0]["reference_images"]), 3)
        self.assertEqual(len(scenarios[0]["reference_audios"]), 2)


if __name__ == "__main__":
    unittest.main()
