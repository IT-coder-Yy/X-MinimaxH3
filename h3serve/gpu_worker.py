"""A private spawned process per DiT group. Only metadata crosses this pipe."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import traceback
from contextlib import suppress
from dataclasses import replace

from .backend import BackendError, JobCancelled


def _serve(connection, cancel, layout, slot, paths, profile, lora):
    order = tuple(dict.fromkeys((*layout.dit[slot], *layout.qwen, *layout.decode)))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(layout.uuids[card] for card in order)
    import torch
    from .backend import build_native_backend

    class Placement:
        devices = tuple(torch.device("cuda", i) for i in range(len(order)))
        dit = tuple(torch.device("cuda", order.index(card)) for card in layout.dit[slot])
        qwen = tuple(torch.device("cuda", order.index(card)) for card in layout.qwen)

        def enter(self, phase):
            if phase == "qwen":
                connection.send(("warm", backend.warm_state))
            connection.send(("phase", phase))
            group = connection.recv()
            if not group:
                raise JobCancelled("cancelled waiting for GPUs")
            return self.devices[order.index(group[0])]

    backend = build_native_backend(paths, memory_profile=profile, gpu_runtime=Placement())
    if lora is not None:
        backend.configure_lora_checkpoint(lora)
    loop = asyncio.new_event_loop()
    try:
        while True:
            command = connection.recv()
            if command is None:
                loop.run_until_complete(backend.stop())
                return
            method, args, kwargs, preview, decision = command
            try:
                if method != "preload":
                    kwargs["cancel_event"] = cancel
                    kwargs["progress_callback"] = lambda e: connection.send(("progress", e))
                    if preview:
                        kwargs["preview_ready_callback"] = lambda e: connection.send(("preview", e))
                    if decision:
                        def wait_decision():
                            connection.send(("decision", None))
                            return connection.recv()
                        kwargs["preview_decision_wait"] = wait_decision
                result = loop.run_until_complete(getattr(backend, method)(*args, **kwargs))
                connection.send(("done", (result, backend.warm_state, backend.key)))
            except JobCancelled as exc:
                connection.send(("cancelled", str(exc)))
            except Exception:
                connection.send(("error", traceback.format_exc()))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()
        loop.close()


class WorkerSlot:
    def __init__(self, pool, index):
        self.pool = pool
        self.index = index
        self.process = None
        self.connection = None
        self.key = None
        self.warm_state = {"status": "cold"}

    @property
    def pid(self):
        return self.process.pid if self.process else None

    def _start(self):
        ctx = multiprocessing.get_context("spawn")
        parent, child = ctx.Pipe()
        self.cancel = ctx.Event()
        self.process = ctx.Process(
            target=_serve, name=f"h3-gpu-{self.index + 1}", daemon=True,
            args=(child, self.cancel, self.pool.layout, self.index,
                  self.pool.paths, self.pool.profile, self.pool.lora),
        )
        try:
            self.process.start()
        finally:
            child.close()
        self.connection = parent

    async def stop(self, *, failed=False):
        if self.process is not None:
            if self.process.is_alive():
                if failed:
                    self.process.terminate()
                else:
                    with suppress(BrokenPipeError, OSError):
                        self.connection.send(None)
            await asyncio.to_thread(self.process.join)
            self.process.close()
            self.connection.close()
            self.process = self.connection = None
        self.key = None
        self.warm_state = {"status": "cold"}

    async def _call(self, method, args, kwargs, cancel_event=None, progress=None,
                    preview=None, decision=None):
        if self.process is None:
            self._start()
        self.cancel.clear()
        self.pool.leases.phases[self.index] = "starting"
        conn = self.connection
        placement = {}

        async def watch_cancel():
            await cancel_event.wait()
            self.cancel.set()

        watcher = asyncio.create_task(watch_cancel()) if cancel_event is not None else None
        try:
            conn.send((method, args, kwargs, preview is not None, decision is not None))
            while True:
                kind, value = await asyncio.to_thread(conn.recv)
                if kind == "done":
                    result, self.warm_state, self.key = value
                    if result is not None:
                        result = replace(result, inference_plan={
                            **(result.inference_plan or {}), "gpu_placement": placement,
                        })
                    return result
                if kind == "cancelled":
                    raise JobCancelled(value)
                if kind == "error":
                    raise BackendError(value)
                if kind == "phase":
                    if value == "load":
                        self.warm_state = {"status": "loading"}
                    group = await self.pool.leases.acquire(
                        self.index, value, self.cancel, self.pool.external_cards
                    )
                    if group:
                        placement[value] = list(group)
                        print(f"H3 GPU slot={self.index + 1} job={self.job_id} phase={value} cards={group}", flush=True)
                    conn.send(group)
                elif kind == "warm":
                    self.warm_state = value
                elif kind == "progress" and progress:
                    progress(value)
                elif kind == "preview":
                    preview(value)
                elif kind == "decision":
                    conn.send(await asyncio.to_thread(decision))
        except JobCancelled:
            raise
        except BaseException:
            # Reap the process before releasing its leases (including broken pipes).
            await self.stop(failed=True)
            raise
        finally:
            if watcher is not None:
                watcher.cancel()
                with suppress(asyncio.CancelledError):
                    await watcher
            await self.pool.leases.finish(self.index)

    async def preload(self, engine):
        self.job_id = "preload"
        self.warm_state = {"status": "loading", "launcher": engine}
        await self._call("preload", (engine,), {})

    async def generate(self, spec, job_id, first, last, images, videos, audios,
                       cancel_event, progress_callback=None, preview_ready_callback=None,
                       preview_decision_wait=None, **kwargs):
        self.job_id = job_id
        return await self._call("generate", (spec, job_id, first, last, images, videos, audios),
                                kwargs, cancel_event, progress_callback,
                                preview_ready_callback, preview_decision_wait)

    async def second_sample(self, spec, sampling, source, job_id, first, last, images,
                            videos, audios, cancel_event, progress_callback=None):
        self.job_id = job_id
        return await self._call("second_sample", (spec, sampling, source, job_id, first, last,
                                images, videos, audios), {}, cancel_event, progress_callback)
