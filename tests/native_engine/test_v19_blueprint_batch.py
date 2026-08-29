from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from h3serve.native_engine.planner import (
    blueprint_from_runtime_schedule,
    save_v19_candidate_blueprint,
)
from scripts.benchmark_native_hot_session import load_v19_blueprint_batch


def _blueprint(candidate_id: str):
    return blueprint_from_runtime_schedule(
        candidate_id=candidate_id,
        total_steps=20,
        actual_step_indices=tuple(range(20)),
        attention_action_schedule=tuple(
            (step, layer, "dense")
            for step in range(20)
            for layer in range(50)
        ),
    )


class V19BlueprintBatchTests(unittest.TestCase):
    def test_batch_loads_relative_digest_checked_blueprints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_v19_candidate_blueprint(root / "a.json", _blueprint("a"))
            save_v19_candidate_blueprint(root / "b.json", _blueprint("b"))
            manifest = root / "batch.json"
            manifest.write_text(json.dumps({
                "schema_version": 1,
                "candidates": [
                    {"name": "quality", "blueprint": "a.json"},
                    {"name": "speed", "blueprint": "b.json"},
                ],
            }), encoding="utf-8")
            rows = load_v19_blueprint_batch(manifest)
        self.assertEqual(tuple(row[0] for row in rows), ("quality", "speed"))
        self.assertEqual(tuple(row[2].candidate_id for row in rows), ("a", "b"))

    def test_duplicate_batch_names_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_v19_candidate_blueprint(root / "a.json", _blueprint("a"))
            manifest = root / "batch.json"
            manifest.write_text(json.dumps({
                "schema_version": 1,
                "candidates": [
                    {"name": "same", "blueprint": "a.json"},
                    {"name": "same", "blueprint": "a.json"},
                ],
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_v19_blueprint_batch(manifest)


if __name__ == "__main__":
    unittest.main()
