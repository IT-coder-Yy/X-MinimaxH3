from __future__ import annotations

import ast
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from h3serve.app import JobRecord, JobService
from h3serve.backend import JobCancelled
from h3serve.contract import GenerationSpec


SERVE_ROOT = Path(__file__).resolve().parents[2]


class RecordingBackend:
    def __init__(self, output: Path, behavior: str = "success") -> None:
        self.output = output
        self.behavior = behavior
        self.key = None
        self.calls = []

    async def generate(
        self, spec, job_id, first_frame, last_frame, reference_images, reference_videos, reference_audios, cancel_event,
        progress_callback=None,
    ):
        self.calls.append({
            "spec": spec,
            "job_id": job_id,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "reference_images": reference_images,
            "reference_videos": reference_videos,
            "reference_audios": reference_audios,
            "cancel_event": cancel_event,
        })
        if self.behavior == "cancel":
            raise JobCancelled("generation cancelled")
        if self.behavior == "error":
            raise RuntimeError("private model path: /do/not/expose/model.safetensors")
        return SimpleNamespace(
            runtime_key=f"native:{spec.engine}:{spec.quality}",
            elapsed_seconds=1.25,
            output_path=self.output,
        )

    async def stop(self) -> None:
        pass


def _attribute_name(node: ast.AST) -> str | None:
    return node.attr if isinstance(node, ast.Attribute) else None


class ServiceSourceBoundaryTest(unittest.TestCase):
    def test_job_service_uses_only_backend_generate(self) -> None:
        source = (SERVE_ROOT / "h3serve/app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        run = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "_run"
            and any(arg.arg == "job" for arg in node.args.args)
        )
        calls = [
            _attribute_name(node.func)
            for node in ast.walk(run)
            if isinstance(node, ast.Call)
        ]
        self.assertEqual(calls.count("generate"), 1)
        self.assertTrue({
            "ensure", "upload_image", "submit_and_wait", "resolve_output",
            "build_workflow",
        }.isdisjoint(calls))

    def test_app_does_not_import_workflow_builder(self) -> None:
        source = (SERVE_ROOT / "h3serve/app.py").read_text(encoding="utf-8")
        imports = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
        self.assertFalse(any(name.endswith("workflows") for name in imports))


class ServiceRuntimeBoundaryTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="h3-native-contract-")
        self.root = Path(self.temporary.name)
        self.output = self.root / "result.mp4"
        self.output.write_bytes(b"contract-output")

    async def asyncTearDown(self) -> None:
        self.temporary.cleanup()

    def make_spec(self) -> GenerationSpec:
        return GenerationSpec.from_mapping({
            "prompt": "A deterministic scene with synchronized audio.",
            "engine": "original",
            "quality": "balanced",
            "resolution": "480p",
            "aspect_ratio": "16:9",
            "duration_seconds": 5,
            "seed": 4404,
        })

    async def run_mode(self, first: bool, last: bool) -> tuple[JobRecord, RecordingBackend]:
        backend = RecordingBackend(self.output)
        service = JobService(self.root / f"data-{first}-{last}", backend)
        first_path = self.root / "first.png" if first else None
        last_path = self.root / "last.png" if last else None
        if first_path:
            first_path.write_bytes(b"first")
        if last_path:
            last_path.write_bytes(b"last")
        job = JobRecord(
            id=f"job-{int(first)}-{int(last)}",
            spec=self.make_spec(),
            first_frame=first_path,
            last_frame=last_path,
        )
        service.jobs[job.id] = job
        service.cancel_events[job.id] = asyncio.Event()
        await service._run(job)
        return job, backend

    async def test_all_four_condition_modes_cross_the_same_generate_boundary(self) -> None:
        expected = {
            (False, False): "text",
            (True, False): "first",
            (False, True): "last",
            (True, True): "first_last",
        }
        for flags, mode in expected.items():
            with self.subTest(mode=mode):
                job, backend = await self.run_mode(*flags)
                self.assertEqual(job.condition_mode, mode)
                self.assertEqual(job.status, "succeeded")
                self.assertEqual(job.output_path, self.output)
                self.assertEqual(len(backend.calls), 1)
                call = backend.calls[0]
                self.assertEqual(call["first_frame"] is not None, flags[0])
                self.assertEqual(call["last_frame"] is not None, flags[1])
                self.assertIs(call["spec"], job.spec)

    async def test_cancel_is_terminal_and_preserves_no_output(self) -> None:
        backend = RecordingBackend(self.output, behavior="cancel")
        service = JobService(self.root / "cancel-data", backend)
        job = JobRecord(id="cancel-job", spec=self.make_spec())
        service.jobs[job.id] = job
        service.cancel_events[job.id] = asyncio.Event()
        await service._run(job)
        self.assertEqual(job.status, "cancelled")
        self.assertEqual(job.error, "generation cancelled")
        self.assertIsNone(job.output_path)

    async def test_failure_masks_details_but_keeps_private_traceback_in_server_log(self) -> None:
        backend = RecordingBackend(self.output, behavior="error")
        data = self.root / "error-data"
        service = JobService(data, backend)
        job = JobRecord(id="12345678-private", spec=self.make_spec())
        service.jobs[job.id] = job
        service.cancel_events[job.id] = asyncio.Event()
        await service._run(job)
        self.assertEqual(job.status, "failed")
        self.assertEqual(job.error, "generation failed (reference 12345678)")
        self.assertNotIn("/do/not/expose", job.error)
        private_log = (data / "logs/job_12345678-private.error.log").read_text(
            encoding="utf-8"
        )
        self.assertIn("/do/not/expose/model.safetensors", private_log)


if __name__ == "__main__":
    unittest.main()
