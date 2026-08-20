from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from h3serve.config import ServicePaths
from h3serve.upscaler import FlashVSRUpscaler


FAKE_DAEMON = r'''
import argparse, json, pathlib, sys
p=argparse.ArgumentParser()
p.add_argument("--serve", action="store_true")
p.add_argument("--source-root")
p.add_argument("--model-root")
p.parse_args()
print('H3_UPSCALE_READY '+json.dumps({'ready':True,'model_load_seconds':1.25}),flush=True)
count=0
for line in sys.stdin:
    req=json.loads(line); request_id=req['request_id']
    if req.get('command') == 'shutdown':
        print('H3_UPSCALE_RESPONSE '+json.dumps({'request_id':request_id,'ok':True}),flush=True)
        break
    count += 1
    print('H3_UPSCALE_PROGRESS '+json.dumps({
        'request_id':request_id,'percent':50,'stage':'upscaling','detail':'fake'
    }),flush=True)
    pathlib.Path(req['output']).write_bytes(pathlib.Path(req['input']).read_bytes()+b'-up')
    print('H3_UPSCALE_RESPONSE '+json.dumps({
        'request_id':request_id,'ok':True,'request_count':count,
        'peak_allocated_mib':100.0,'peak_reserved_mib':120.0
    }),flush=True)
'''


class PersistentUpscalerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="flashvsr-daemon-test-")
        root = Path(self.temporary.name)
        (root / "scripts").mkdir()
        (root / "third_party/flashvsr/diffsynth").mkdir(parents=True)
        (root / "third_party/flashvsr/diffsynth/__init__.py").touch()
        (root / "models/upscalers/flashvsr-v1.1").mkdir(parents=True)
        for name in (
            "diffusion_pytorch_model_streaming_dmd.safetensors",
            "LQ_proj_in.ckpt", "TCDecoder.ckpt", "posi_prompt.pth",
        ):
            (root / "models/upscalers/flashvsr-v1.1" / name).touch()
        worker = root / "scripts/flashvsr_worker.py"
        worker.write_text(textwrap.dedent(FAKE_DAEMON), encoding="utf-8")
        self.paths = ServicePaths.defaults(root, data_dir=root / "data")
        object.__setattr__(self.paths, "flashvsr_python_executable", Path(sys.executable))
        self.upscaler = FlashVSRUpscaler(self.paths)

    async def asyncTearDown(self) -> None:
        await self.upscaler.stop()
        self.temporary.cleanup()

    async def test_daemon_is_reused_across_requests(self) -> None:
        source = self.paths.release_root / "source.mp4"
        source.write_bytes(b"video")
        await self.upscaler.start()
        first_pid = self.upscaler._process.pid
        for index in (1, 2):
            candidate = self.paths.release_root / f"source-{index}.mp4"
            candidate.write_bytes(source.read_bytes())
            events = []
            result = await self.upscaler.upscale(
                candidate, target_width=1280, target_height=720,
                cancel_event=asyncio.Event(), progress_callback=events.append,
            )
            self.assertEqual(result.output_path.read_bytes(), b"video-up")
            self.assertEqual(result.peak_allocated_mib, 100.0)
            self.assertEqual(events[0]["percent"], 50)
            self.assertEqual(self.upscaler._process.pid, first_pid)
        state = self.upscaler.status()
        self.assertEqual(state["resident_state"], "ready")
        self.assertEqual(state["requests_completed"], 2)

    async def test_cancelled_request_terminates_daemon(self) -> None:
        source = self.paths.release_root / "cancel.mp4"
        source.write_bytes(b"video")
        cancel = asyncio.Event()
        cancel.set()
        from h3serve.backend import JobCancelled
        with self.assertRaises(JobCancelled):
            await self.upscaler.upscale(
                source, target_width=1280, target_height=720,
                cancel_event=cancel,
            )
        self.assertIsNone(self.upscaler._process)


if __name__ == "__main__":
    unittest.main()
