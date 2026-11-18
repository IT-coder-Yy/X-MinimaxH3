import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from h3serve.gpu_pool import GpuLayout, GpuLeases, resolve_service_resources, restrict_cuda_mask


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
            os.environ["H3_DECODE_GPUS"] = "6,7,5,3"
            self.assertIsNone(GpuLayout.from_env())
            os.environ["H3_GPU_MODE"] = "dual"
            os.environ["H3_SERVE_ENGINE"] = "fl2va_w4a8_8gb"
            self.assertEqual(GpuLayout.from_env().decode, (6, 7, 5, 3))
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.layout(H3_GPU_MODE="dual", CUDA_VISIBLE_DEVICES="GPU-7")

    def test_configured_order_and_invalid_groups(self):
        layout = self.layout(H3_QWEN_GPUS="6,7", H3_DIT_GPUS_FIRST="7",
                             H3_DIT_GPUS_SECOND="3,5", H3_DECODE_GPUS="3,6")
        self.assertEqual((layout.qwen, layout.dit, layout.decode), ((6, 7), ((7,), (3, 5)), (3, 6)))
        self.assertEqual(self.layout(H3_QWEN_GPUS="7,6,5").qwen, (7, 6, 5))
        for options in [dict(H3_QWEN_GPUS="0,1,2,3,4,5,6,7,8"), dict(H3_DIT_GPUS_SECOND="6,3"),
                        dict(H3_DECODE_GPUS="2,2"), dict(H3_QWEN_GPUS="7,,5")]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.layout(**options)

    def test_single_mask_keeps_local_order_and_respects_uuid_allocation(self):
        with patch("h3serve.gpu_pool._smi", return_value=[row[:2] for row in INVENTORY]):
            self.assertEqual(restrict_cuda_mask("5,6,7", "GPU-7,GPU-6"), "6,7")
            self.assertEqual(restrict_cuda_mask("GPU-6", "7,6"), "GPU-6")
            self.assertEqual(restrict_cuda_mask("", "7,6"), "")
            with self.assertRaisesRegex(ValueError, "no GPU"):
                restrict_cuda_mask("3", "7,6")

    def test_run_sh_preserves_single_gpu_selected_in_dotenv(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            release = Path(tmp)
            for name in ("run.sh", "scripts/start.sh", "h3serve/gpu_pool.py"):
                (release / name).parent.mkdir(exist_ok=True)
                shutil.copy2(root / name, release / name)
            (release / "scripts/_process.sh").write_text(
                "h3_find_release_server_pids() { :; }\nh3_is_release_server_pid() { return 1; }\n"
            )
            (release / "scripts/_runtime.sh").write_text("h3_configure_runtime() { :; }\n")
            smi = release / "nvidia-smi"
            gpu_uuid = "GPU-12345678-1234-4321-8765-123456789abc"
            smi.write_text(f"#!/bin/sh\nprintf '%s\\n' '6, {gpu_uuid}' '7, GPU-7'\n")
            smi.chmod(0o755)
            (release / "server.py").write_text("import os; print(os.environ['CUDA_VISIBLE_DEVICES'])\n")
            (release / ".env").write_text("H3_GPU_MODE=single\nCUDA_VISIBLE_DEVICES=6\n")
            env = {"PATH": f"{tmp}:{os.environ['PATH']}", "H3_SERVE_PYTHON": sys.executable}
            for mask in ("7,6", f"{gpu_uuid} "):
                with self.subTest(mask=mask):
                    env["CUDA_VISIBLE_DEVICES"] = mask
                    output = subprocess.check_output(["./run.sh"], cwd=release, text=True, env=env)
                    self.assertEqual(output.strip(), "6")
            for mask in ("7,99,6", f"{gpu_uuid},{gpu_uuid} ", f"{gpu_uuid[:12]} "):
                with self.subTest(mask=mask):
                    env["CUDA_VISIBLE_DEVICES"] = mask
                    blocked = subprocess.run(["./run.sh"], cwd=release, text=True, env=env, capture_output=True)
                    self.assertNotEqual(blocked.returncode, 0)
                    self.assertIn("no GPU in the inherited allocation", blocked.stdout + blocked.stderr)

    def test_masks_stop_at_invalid_gpu_in_launcher_and_layout(self):
        with patch("h3serve.gpu_pool._smi", return_value=[row[:2] for row in INVENTORY]):
            for mask in ("7,-1,6", "7,99,6", "7,,6", "7,GPU-6", "GPU-7,6"):
                with self.subTest(mask=mask):
                    self.assertEqual(restrict_cuda_mask("6,7", mask), "7")
                    with self.assertRaisesRegex(ValueError, "unavailable"):
                        self.layout(H3_GPU_MODE="dual", H3_INHERITED_CUDA_MASK=mask,
                                    H3_DIT_GPUS_FIRST="7", H3_DIT_GPUS_SECOND="6")
            self.assertEqual(restrict_cuda_mask("7,99,6", "7,6"), "7")
            with self.assertRaisesRegex(ValueError, "no GPU"):
                restrict_cuda_mask("6", "7,7,6")
            self.assertIsNone(self.layout(CUDA_VISIBLE_DEVICES="7,-1,6", H3_INHERITED_CUDA_MASK="__unset__"))
        with patch("h3serve.gpu_pool._smi", return_value=[["6", "GPU-ab01"], ["7", "GPU-ab12"]]):
            self.assertEqual(restrict_cuda_mask("7,6", "GPU-ab0"), "6")
            with self.assertRaisesRegex(ValueError, "no GPU"):
                restrict_cuda_mask("6", "GPU-ab")

    def test_auto_falls_back_before_workers_start_and_dual_reports_budget(self):
        from h3serve.memory_policy import HostMemoryStatus

        with patch.dict(os.environ, {}, clear=True), patch("h3serve.gpu_pool._smi", return_value=INVENTORY):
            status = HostMemoryStatus(64, 64, 55)
            profile, layout = resolve_service_resources("auto", status)
            self.assertIsNone(layout)
            self.assertEqual(profile.key, "compact")
            self.assertEqual(os.environ.pop("CUDA_VISIBLE_DEVICES"), "GPU-7")
            os.environ["H3_GPU_MODE"] = "dual"
            with self.assertRaisesRegex(RuntimeError, "dual:.*107.86 GiB capacity and 86.78 GiB available"):
                resolve_service_resources("auto", status)
            profile, layout = resolve_service_resources("auto", HostMemoryStatus(140, 140, 90))
            self.assertEqual((profile.key, len(layout.dit)), ("compact", 2))


class LeaseTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.layout = GpuLayout((7, 6, 5, 3), ((7, 6), (5, 3)), (7, 6, 5, 3), {})
        self.leases = GpuLeases(self.layout)
        self.cancel = asyncio.Event()

    def acquire(self, slot, phase, external=()):
        return self.leases.acquire(slot, phase, self.cancel, lambda: set(external))

    async def test_dit_runs_while_another_qwen_waits_for_more_gpus(self):
        self.leases = GpuLeases(GpuLayout((7, 6), ((5,), (3,)), (7, 6), {}))
        busy = {7}
        self.assertEqual(await self.acquire(1, "qwen", busy), (6,))
        retry = asyncio.create_task(self.leases.acquire(1, "qwen", self.cancel, busy.copy, 2))
        try:
            async with self.leases.changed:
                await asyncio.wait_for(self.leases.changed.wait_for(
                    lambda: self.leases.phases[1] == "wait_qwen"), 2)
            self.assertEqual(await asyncio.wait_for(self.acquire(0, "dit"), 2), (5,))
            self.assertFalse(retry.done())
            async with self.leases.changed:
                busy.clear()
                self.leases.changed.notify_all()
            self.assertEqual(await asyncio.wait_for(retry, 2), (7, 6))
        finally:
            self.cancel.set()
            await self.leases.finish(0)
            await retry

    async def test_cancel_waiting_for_load_reports_cold_worker_and_ready_pool(self):
        import multiprocessing
        import threading
        from types import SimpleNamespace
        from h3serve.backend import JobCancelled, NativeBackendManager
        from h3serve.contract import GenerationSpec
        from h3serve.gpu_pool import DualGpuBackend
        from h3serve.gpu_worker import _serve
        from h3serve.native_engine.engine import NativeHotH3Engine

        layout = GpuLayout((7, 6), ((7,), (6,)), (7, 6), {7: "GPU-7", 6: "GPU-6"})
        pool = DualGpuBackend(None, None, None, layout)
        slot, other = pool.slots
        other.warm_state = {"status": "ready"}
        pool.leases.held[1] = (7,)
        parent, child = multiprocessing.Pipe()
        slot.process, slot.connection, slot.cancel = object(), parent, threading.Event()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            def native(paths, *, memory_profile, gpu_runtime):
                factory = SimpleNamespace(build=lambda launcher: gpu_runtime.enter("load"))
                return NativeBackendManager(NativeHotH3Engine(factory, Path(tmp)))

            with patch("h3serve.backend.build_native_backend", side_effect=native):
                spec = GenerationSpec.from_mapping({"prompt": "cancel at load", "sampling_steps": 20, "acceleration": 0})
                worker = asyncio.create_task(asyncio.to_thread(_serve, child, slot.cancel, layout, 0, None, None, None))
                job = asyncio.create_task(slot.generate(spec, "cancel-load", None, None, (), (), (), self.cancel))
                try:
                    async with pool.leases.changed:
                        await asyncio.wait_for(pool.leases.changed.wait_for(
                            lambda: pool.leases.phases.get(0) == "wait_load"), 10)
                    self.assertEqual(pool.warm_state["status"], "loading")
                    self.cancel.set()
                    with self.assertRaises(JobCancelled):
                        await asyncio.wait_for(job, 4)
                    self.assertEqual(slot.warm_state["status"], "cold")
                    self.assertEqual(pool.warm_state["status"], "ready")
                finally:
                    slot.cancel.set()
                    parent.send(None)
                    await asyncio.wait_for(worker, 5)
                    parent.close()

    async def test_serial_qwen_parallel_dit_and_idle_decoder(self):
        self.leases.phases.update({0: "starting", 1: "starting"})
        second = asyncio.create_task(self.acquire(1, "qwen"))
        await asyncio.sleep(0)
        self.assertFalse(second.done())
        self.assertEqual(await self.acquire(0, "qwen"), self.layout.qwen)
        self.assertEqual(await self.acquire(0, "dit"), (7, 6))
        self.assertEqual(await asyncio.wait_for(second, 1), (5, 3))
        self.assertEqual(await self.acquire(1, "dit"), (5, 3))
        self.assertEqual(await self.acquire(1, "decode", external=(5,)), (3,))
        self.assertEqual(await self.acquire(0, "decode", external=(7,)), (6,))

    async def test_lone_job_late_arrival_and_cancel(self):
        await self.acquire(0, "qwen")
        self.assertEqual(await self.acquire(0, "dit"), (7, 6))
        self.assertEqual(await self.acquire(1, "qwen", external=(5,)), (3,))
        await self.leases.finish(0)
        wait_decode = asyncio.create_task(self.acquire(0, "decode", external=self.layout.decode))
        await asyncio.sleep(0)
        self.cancel.set()
        await self.leases.finish(1)
        self.assertEqual(await asyncio.wait_for(wait_decode, 1), ())

    async def test_dit_waits_for_a_gpu_held_by_active_qwen(self):
        self.assertEqual(await self.acquire(1, "qwen", external=(7, 5, 3)), (6,))
        dit = asyncio.create_task(self.acquire(0, "dit"))
        async with self.leases.changed:
            await asyncio.wait_for(self.leases.changed.wait_for(
                lambda: self.leases.phases.get(0) == "wait_dit"), 2)
        self.assertFalse(dit.done())
        await self.leases.finish(1)
        self.assertEqual(await asyncio.wait_for(dit, 2), (7, 6))

    async def test_qwen_waits_for_minimum_and_keeps_configured_order(self):
        self.leases.held[1] = (7, 6)
        waiting = asyncio.create_task(self.leases.acquire(0, "qwen", self.cancel, lambda: {5}, 2))
        await asyncio.sleep(0)
        self.assertFalse(waiting.done())
        await self.leases.finish(1)
        self.assertEqual(await asyncio.wait_for(waiting, 2), (7, 6, 3))
        second = asyncio.create_task(self.acquire(1, "qwen"))
        await asyncio.sleep(0)
        self.assertFalse(second.done())
        await self.leases.finish(0)
        self.assertEqual(await asyncio.wait_for(second, 2), self.layout.qwen)


class QwenAttentionTest(unittest.TestCase):
    def test_one_to_eight_gpus_preserve_kv_groups_and_result(self):
        import torch
        import torch.nn.functional as F
        from h3serve.native_engine.adapters.conditioning_vae.qwen_quantized import _sharded_attention

        rng = torch.Generator().manual_seed(73)
        q, k, v = [torch.randn(1, heads, 11, 8, generator=rng) for heads in (64, 8, 8)]
        mask = torch.full((11, 11), -1000.0).triu_(1)
        sdpa = F.scaled_dot_product_attention
        expected = sdpa(q, k, v, attn_mask=mask, enable_gqa=True)
        for n in range(1, 9):
            with self.subTest(n=n), patch.object(F, "scaled_dot_product_attention", wraps=sdpa) as call:
                actual = _sharded_attention(q, k, v, mask, ["cpu"] * n)
                sizes = [args.args[1].size(1) for args in call.call_args_list]
                self.assertEqual(sizes, [8 // n + (i < 8 % n) for i in range(n)])
                torch.testing.assert_close(actual, expected)

        def limited(q, k, v, **kwargs):
            if k.size(1) > 2:
                raise torch.cuda.OutOfMemoryError("test attention scratch limit")
            return sdpa(q, k, v, **kwargs)

        with patch.object(F, "scaled_dot_product_attention", side_effect=limited):
            torch.testing.assert_close(_sharded_attention(q, k, v, mask, ["cpu"]), expected)


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
