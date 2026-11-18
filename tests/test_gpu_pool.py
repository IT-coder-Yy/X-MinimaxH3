import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from h3serve.gpu_pool import GpuLayout, GpuLeases


INVENTORY = [[str(i), f"GPU-{i}", "8.9", "24564"] for i in range(8)]


class LayoutTest(unittest.TestCase):
    def layout(self, **env):
        with patch.dict(os.environ, env, clear=True), patch("h3serve.gpu_pool._smi", return_value=INVENTORY):
            return GpuLayout.from_env()

    def test_defaults_legacy_primary_and_inherited_allocation(self):
        layout = self.layout(CUDA_VISIBLE_DEVICES="7", H3_INHERITED_CUDA_MASK="__unset__")
        self.assertEqual(layout.dit, ((7, 6), (5, 3)))
        self.assertEqual(layout.qwen, (7, 6, 5, 3))
        self.assertIsNone(self.layout(CUDA_VISIBLE_DEVICES="7"))
        self.assertIsNone(self.layout(H3_GPU_MODE="single"))
        with patch.dict(os.environ, {}, clear=True), patch(
            "h3serve.gpu_pool._smi", return_value=[[*row[:3], "12288"] for row in INVENTORY]
        ):
            self.assertIsNone(GpuLayout.from_env())
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.layout(H3_GPU_MODE="dual", CUDA_VISIBLE_DEVICES="GPU-7")

    def test_configured_order_and_invalid_groups(self):
        layout = self.layout(H3_QWEN_GPUS="6,7", H3_DIT_GPUS_FIRST="7",
                             H3_DIT_GPUS_SECOND="3,5", H3_DECODE_GPUS="3,6")
        self.assertEqual((layout.qwen, layout.dit, layout.decode), ((6, 7), ((7,), (3, 5)), (3, 6)))
        for options in [dict(H3_QWEN_GPUS="7,6,5"), dict(H3_DIT_GPUS_SECOND="6,3"),
                        dict(H3_DECODE_GPUS="2,2"), dict(H3_QWEN_GPUS="7,,5")]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.layout(**options)


class LeaseTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.layout = GpuLayout((7, 6, 5, 3), ((7, 6), (5, 3)), (7, 6, 5, 3), {})
        self.leases = GpuLeases(self.layout)
        self.cancel = asyncio.Event()

    def acquire(self, slot, phase, external=()):
        return self.leases.acquire(slot, phase, self.cancel, lambda: set(external))

    async def test_serial_qwen_parallel_dit_and_idle_decoder(self):
        self.leases.phases.update({0: "starting", 1: "starting"})
        second = asyncio.create_task(self.acquire(1, "qwen"))
        await asyncio.sleep(0)
        self.assertFalse(second.done())
        self.assertEqual(await self.acquire(0, "qwen"), self.layout.qwen)
        dit = asyncio.create_task(self.acquire(0, "dit"))
        self.assertEqual(await asyncio.wait_for(second, 1), self.layout.qwen)
        self.assertFalse(dit.done())
        self.assertEqual(await self.acquire(1, "dit"), (5, 3))
        self.assertEqual(await asyncio.wait_for(dit, 1), (7, 6))
        self.assertEqual(await self.acquire(1, "decode", external=(5,)), (3,))
        self.assertEqual(await self.acquire(0, "decode", external=(7,)), (6,))

    async def test_lone_job_late_arrival_and_cancel(self):
        await self.acquire(0, "qwen")
        self.assertEqual(await self.acquire(0, "dit"), (7, 6))
        late = asyncio.create_task(self.acquire(1, "qwen"))
        await asyncio.sleep(0)
        self.assertFalse(late.done())
        await self.leases.finish(0)
        self.assertEqual(await asyncio.wait_for(late, 1), self.layout.qwen)
        wait_decode = asyncio.create_task(self.acquire(0, "decode", external=self.layout.decode))
        await asyncio.sleep(0)
        self.cancel.set()
        await self.leases.finish(1)
        self.assertEqual(await asyncio.wait_for(wait_decode, 1), ())


class DecodeDeviceTest(unittest.TestCase):
    def test_remote_decoder_uses_the_explicit_residency_device_name(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        import torch
        from h3serve.native_engine.hot_session import NativeT2AVHotSession

        remote = torch.device("cuda:2")
        session = SimpleNamespace(gpu_runtime=SimpleNamespace(enter=lambda phase: remote))
        with patch.object(torch.cuda, "empty_cache"), patch.object(
            torch.cuda, "device", return_value=nullcontext()
        ) as use_device:
            with NativeT2AVHotSession._decode_device(session) as target:
                self.assertEqual(target, "cuda:2")
        use_device.assert_called_once_with(remote)


class QueueTest(unittest.IsolatedAsyncioTestCase):
    async def test_third_job_waits_for_a_slot_and_next_lone_job_prefers_first(self):
        from h3serve.app import JobService
        from h3serve.backend import GenerationResult
        from h3serve.contract import GenerationSpec

        entered = asyncio.Queue()
        release = {}

        class Slot:
            async def generate(self, spec, job_id, *args):
                release[job_id] = asyncio.Event()
                entered.put_nowait((job_id, self))
                await release[job_id].wait()
                if spec.seed == 1:
                    raise RuntimeError("deliberate worker failure")
                return GenerationResult("test", 1.0, Path(job_id + ".mp4"))

        class Pool:
            slots = (Slot(), Slot())

            async def stop(self):
                pass

        with tempfile.TemporaryDirectory() as root:
            service = JobService(Path(root), Pool())
            await service.start()
            try:
                async def submit(seed):
                    return await service.submit(GenerationSpec.from_mapping({"prompt": "test", "seed": seed}), {})

                first, second, third = [await submit(i) for i in (1, 2, 3)]
                self.assertEqual(await asyncio.wait_for(entered.get(), 1), (first.id, Pool.slots[0]))
                self.assertEqual(await asyncio.wait_for(entered.get(), 1), (second.id, Pool.slots[1]))
                self.assertEqual(service.pending, [third.id])
                release[first.id].set()
                self.assertEqual(await asyncio.wait_for(entered.get(), 1), (third.id, Pool.slots[0]))
                self.assertEqual(first.status, "failed")
                release[second.id].set()
                release[third.id].set()
                async with service.queue_changed:
                    await asyncio.wait_for(service.queue_changed.wait_for(
                        lambda: second.status == third.status == "succeeded"), 1)
                fourth = await submit(4)
                self.assertEqual(await asyncio.wait_for(entered.get(), 1), (fourth.id, Pool.slots[0]))
            finally:
                await service.close()
