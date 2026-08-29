from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.benchmark_native_hot_session import load_scenarios, parse_args


class Ref2VACandidateRegistryBatchTest(unittest.TestCase):
    def test_registry_expands_candidates_in_one_hot_session(self) -> None:
        root = Path(__file__).resolve().parents[2]
        registry = root / "benchmarks/ref2va_extreme/candidates_enhanced_v1_extreme.json"
        with patch.object(sys, "argv", [
            "benchmark_native_hot_session.py", "--engine", "reference",
            "--candidate-registry", str(registry), "--repeat", "1",
        ]):
            args = parse_args()
        scenarios = load_scenarios(args)
        self.assertEqual(
            [item["name"] for item in scenarios],
            [
                "x00_warmup_2_18_not_for_review",
                "x10_forecast10_10",
                "x09_forecast9_11",
                "x08_forecast8_12",
            ],
        )
        self.assertEqual(len(scenarios[-1]["actual_step_indices"]), 8)
        self.assertEqual(scenarios[-1]["resident_block_count"], 0)
        self.assertEqual(len(scenarios[-1]["reference_images"]), 3)
        self.assertEqual(len(scenarios[-1]["reference_audios"]), 2)

    def test_registry_can_batch_exact_execution_variants(self) -> None:
        root = Path(__file__).resolve().parents[2]
        registry = root / "benchmarks/720p10_seed82341_v22_exact_execution_ab.json"
        with patch.object(sys, "argv", [
            "benchmark_native_hot_session.py", "--engine", "original",
            "--candidate-registry", str(registry), "--repeat", "1",
        ]):
            args = parse_args()
        scenarios = load_scenarios(args)
        self.assertEqual(len(scenarios), 3)
        self.assertIsNone(scenarios[0]["long_sequence_query_chunk_tokens"])
        self.assertFalse(scenarios[0]["long_sequence_split_qkv_outputs"])
        self.assertEqual(
            scenarios[1]["long_sequence_query_chunk_tokens"], 32768
        )
        self.assertTrue(scenarios[1]["long_sequence_split_qkv_outputs"])
        self.assertTrue(scenarios[1]["long_sequence_single_qknorm_rope"])
        self.assertFalse(scenarios[1]["long_sequence_parallel_sparse_lut"])
        self.assertTrue(scenarios[2]["long_sequence_parallel_sparse_lut"])


if __name__ == "__main__":
    unittest.main()
