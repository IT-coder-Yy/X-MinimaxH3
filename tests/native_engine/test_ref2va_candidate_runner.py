from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.run_ref2va_candidate import build_command, load_candidate


class Ref2VACandidateRunnerTest(unittest.TestCase):
    def test_v00_replay_disables_both_exact_caches(self) -> None:
        registry = Path(__file__).resolve().parents[2] / "benchmarks/ref2va_extreme/candidates.json"
        document, candidate = load_candidate(registry, "v00_dense20")
        command = build_command(document, candidate, Path("/tmp/ref2va-v00"))
        self.assertIn("--disable-condition-row-cache", command)
        self.assertIn("--disable-reference-latent-cache", command)
        self.assertEqual(command[0], sys.executable)

    def test_v02_repeat_is_preserved(self) -> None:
        registry = Path(__file__).resolve().parents[2] / "benchmarks/ref2va_extreme/candidates.json"
        document, candidate = load_candidate(
            registry, "v02_exact_reference_latents_repeat2"
        )
        command = build_command(document, candidate, Path("/tmp/ref2va-v02"))
        self.assertEqual(command[command.index("--repeat") + 1], "2")
        self.assertNotIn("--disable-condition-row-cache", command)
        self.assertNotIn("--disable-reference-latent-cache", command)

    def test_v03_enables_projected_condition_cache(self) -> None:
        registry = Path(__file__).resolve().parents[2] / "benchmarks/ref2va_extreme/candidates.json"
        document, candidate = load_candidate(
            registry, "v03_equivalent_condition_embeddings"
        )
        command = build_command(document, candidate, Path("/tmp/ref2va-v03"))
        self.assertIn("--cache-condition-embeddings", command)


if __name__ == "__main__":
    unittest.main()
