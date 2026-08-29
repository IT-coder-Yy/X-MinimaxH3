from __future__ import annotations

import unittest
import tempfile
import shutil
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from h3serve.native_engine.detail_restore import (
    FILTER_GRAPH,
    _ffmpeg_chunk_command,
    _frame_partitions,
    _ffmpeg_command,
    restore_intrame_detail,
)


class IntraframeDetailRestoreContractTest(unittest.TestCase):
    def test_filter_has_no_temporal_operator(self):
        self.assertIn("vaguedenoiser", FILTER_GRAPH)
        self.assertIn("cas=", FILTER_GRAPH)
        for forbidden in ("atadenoise", "hqdn3d", "tmix", "minterpolate"):
            self.assertNotIn(forbidden, FILTER_GRAPH)

    def test_command_preserves_audio_and_uses_atomic_mp4_target(self):
        with patch(
            "h3serve.native_engine.detail_restore.shutil.which",
            return_value="/usr/bin/ffmpeg",
        ):
            command = _ffmpeg_command(
                Path("source.mp4"), Path(".temporary.detail.mp4")
            )
        self.assertEqual(command[0], "/usr/bin/ffmpeg")
        self.assertIn(FILTER_GRAPH, command)
        audio_codec = command.index("-c:a")
        self.assertEqual(command[audio_codec + 1], "copy")
        self.assertEqual(command[-1], ".temporary.detail.mp4")

    def test_parallel_partition_covers_each_frame_once(self):
        self.assertEqual(
            _frame_partitions(362, 4),
            ((0, 91), (91, 91), (182, 90), (272, 90)),
        )

    def test_chunk_command_is_frame_bounded_and_video_only(self):
        with patch(
            "h3serve.native_engine.detail_restore.shutil.which",
            return_value="/usr/bin/ffmpeg",
        ):
            command = _ffmpeg_chunk_command(
                Path("source.mp4"),
                Path("chunk.mp4"),
                start_frame=91,
                frame_count=91,
                fps=24,
            )
        self.assertEqual(command[command.index("-ss") + 1], "3.791666667")
        self.assertEqual(command[command.index("-frames:v") + 1], "91")
        self.assertIn("-an", command)
        self.assertEqual(command[-1], "chunk.mp4")

    def test_publication_keeps_raw_and_atomically_replaces_result(self):
        root = Path(tempfile.mkdtemp(prefix="detail-restore-test-"))
        source = root / "result.mp4"
        source.write_bytes(b"raw-h3-video")

        class FakeProcess:
            def __init__(self, command, **_kwargs):
                Path(command[-1]).write_bytes(b"restored-video")
                self.returncode = 0
                self.stderr = StringIO("")

            def poll(self):
                return 0

        try:
            with (
                patch(
                    "h3serve.native_engine.detail_restore.subprocess.Popen",
                    side_effect=FakeProcess,
                ),
                patch(
                    "h3serve.native_engine.detail_restore._probe",
                    return_value=(1280, 736, 362),
                ),
                patch(
                    "h3serve.native_engine.detail_restore.shutil.which",
                    return_value="/usr/bin/ffmpeg",
                ),
            ):
                result = restore_intrame_detail(
                    source,
                    expected_width=1280,
                    expected_height=736,
                    expected_frames=362,
                )
            self.assertEqual(source.read_bytes(), b"restored-video")
            self.assertEqual(result.raw_output_path.read_bytes(), b"raw-h3-video")
            self.assertEqual((result.width, result.height, result.frames), (1280, 736, 362))
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
