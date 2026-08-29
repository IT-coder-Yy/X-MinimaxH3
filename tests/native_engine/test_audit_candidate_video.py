from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_candidate_video import _load_runtime


class AuditCandidateVideoTests(unittest.TestCase):
    def test_runtime_binding_survives_unique_ext4_staging_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "canonical" / "candidate.mp4"
            canonical.parent.mkdir()
            canonical.touch()
            report = root / "report.json"
            report.write_text(json.dumps({
                "requests": [{
                    "output": "/root/staging/candidate.mp4",
                    "total_seconds": 12.5,
                    "phases": {"denoise": 10.0},
                    "peak_allocated_gib": 9.0,
                    "actual_steps": 11,
                    "forecast_steps": 9,
                    "execution_profile": {
                        "joint_acceleration": {"execution_digest": "a" * 64}
                    },
                }],
            }))
            runtime = _load_runtime(report, video=canonical.resolve())
            self.assertIsNotNone(runtime)
            self.assertEqual(runtime["total_seconds"], 12.5)
            self.assertEqual(runtime["actual_steps"], 11)

    def test_ambiguous_relocated_filename_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "candidate.mp4"
            canonical.touch()
            report = root / "report.json"
            report.write_text(json.dumps({
                "requests": [
                    {"output": "/root/a/candidate.mp4"},
                    {"output": "/root/b/candidate.mp4"},
                ],
            }))
            self.assertIsNone(_load_runtime(report, video=canonical.resolve()))

    def test_batch_attention_telemetry_is_marked_cumulative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.mp4"
            candidate.touch()
            report = root / "report.json"
            report.write_text(json.dumps({
                "requests": [
                    {"output": str(candidate)},
                    {"output": str(root / "control.mp4")},
                ],
                "adaptive_attention": {"draft": {"action_calls": {"dense": 7}}},
            }))
            runtime = _load_runtime(report, video=candidate.resolve())
            self.assertEqual(
                runtime["adaptive_attention"]["scope"],
                "batch_cumulative_not_candidate_specific",
            )
            self.assertEqual(runtime["adaptive_attention"]["request_count"], 2)
            self.assertEqual(
                runtime["adaptive_attention"]["telemetry"]["draft"]
                ["action_calls"]["dense"],
                7,
            )


if __name__ == "__main__":
    unittest.main()
