"""Physical GPU configuration and phase ownership for two native workers."""

from __future__ import annotations

import asyncio
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


def _smi(query: str) -> list[list[str]]:
    output = subprocess.check_output(
        ["nvidia-smi", query, "--format=csv,noheader,nounits"], text=True, timeout=10
    )
    return [[v.strip() for v in line.split(",")] for line in output.splitlines() if line.strip()]


@dataclass(frozen=True)
class GpuLayout:
    qwen: tuple[int, ...]
    dit: tuple[tuple[int, ...], tuple[int, ...]]
    decode: tuple[int, ...]
    uuids: dict[int, str]

    @classmethod
    def from_env(cls):
        env = os.environ
        mode = env.get("H3_GPU_MODE", "auto")
        if mode == "single":
            return None
        if mode not in {"auto", "dual"}:
            raise ValueError("H3_GPU_MODE must be auto, dual or single")
        keys = ("H3_QWEN_GPUS", "H3_DIT_GPUS_FIRST", "H3_DIT_GPUS_SECOND", "H3_DECODE_GPUS")
        explicit = mode == "dual" or any(key in env for key in keys)

        def cards(key, default):
            raw = env.get(key, default).split(",")
            if any(not v.strip().isdigit() for v in raw):
                raise ValueError(f"{key} requires comma-separated physical GPU indices")
            group = tuple(int(v) for v in raw)
            if len(set(group)) != len(group):
                raise ValueError(f"{key} contains duplicate GPUs")
            return group

        first, second = cards(keys[1], "7,6"), cards(keys[2], "5,3")
        if set(first) & set(second) or any(56 % len(group) for group in (first, second)):
            raise ValueError("DiT groups must be disjoint and each size must divide 56 heads")
        ordered = ",".join(map(str, (*first, *second)))
        qwen, decode = cards(keys[0], ordered), cards(keys[3], ordered)
        if 8 % len(qwen):
            raise ValueError("Qwen GPU count must divide its eight KV heads")
        selected = set((*first, *second, *qwen, *decode))
        try:
            inventory = {int(i): (uuid, capability, int(mib)) for i, uuid, capability, mib in
                         _smi("--query-gpu=index,uuid,compute_cap,memory.total")}
        except (OSError, subprocess.SubprocessError):
            if explicit:
                raise ValueError("dual GPU mode requires nvidia-smi to enumerate GPUs") from None
            return None
        from .deployment_profiles import get_resource_backend

        budget_mib = get_resource_backend("int8_24gb").planner_budget_gib * 1024
        available = {i for i, (_, cap, mib) in inventory.items()
                     if cap == "8.9" and (explicit or mib >= budget_mib)}
        # run.sh distinguishes an inherited allocation from the old primary-only .env.
        inherited = env.get("H3_INHERITED_CUDA_MASK", env.get("CUDA_VISIBLE_DEVICES"))
        if inherited is not None and inherited != "__unset__":
            mask = {value.strip() for value in inherited.split(",")}
            available &= {i for i, (uuid, _, _) in inventory.items() if str(i) in mask or uuid in mask}
        local_mask = env.get("CUDA_VISIBLE_DEVICES")
        if not explicit and local_mask is not None:
            mask = {value.strip() for value in local_mask.split(",")}
            allowed = {i for i, (uuid, _, _) in inventory.items() if str(i) in mask or uuid in mask}
            if allowed != {first[0]} and not selected <= allowed:
                return None
        if not selected <= available:
            if explicit:
                raise ValueError(f"selected SM89 GPUs are unavailable: {sorted(selected - available)}")
            return None
        return cls(qwen, (first, second), decode, {i: inventory[i][0] for i in sorted(selected)})


class GpuLeases:
    def __init__(self, layout):
        self.layout = layout
        self.phases = {}
        self.held = {}
        self.changed = asyncio.Condition()

    async def finish(self, slot):
        async with self.changed:
            self.phases.pop(slot, None)
            self.held.pop(slot, None)
            self.changed.notify_all()

    async def acquire(self, slot, phase, cancel, external_cards):
        """Swap leases atomically. An admitted second encoder precedes the first DiT."""
        async with self.changed:
            self.held.pop(slot, None)
            self.phases[slot] = "wait_" + phase
            self.changed.notify_all()
            while not cancel.is_set():
                occupied = {card for other, group in self.held.items() if other != slot for card in group}
                if phase == "decode":
                    candidates = [card for card in self.layout.decode if card not in occupied]
                    if candidates:
                        external = await asyncio.to_thread(external_cards)
                        candidates = [card for card in candidates if card not in external]
                    group = tuple(candidates[:1])
                elif phase == "qwen":
                    group = self.layout.qwen
                    for earlier, state in self.phases.items():
                        if earlier == slot:
                            break
                        if state in {"starting", "wait_load", "load", "wait_qwen"}:
                            group = ()
                            break
                elif phase == "dit":
                    group = self.layout.dit[slot]
                    if any(other != slot and p in {"starting", "wait_load", "load", "wait_qwen", "qwen"}
                           for other, p in self.phases.items()):
                        group = ()
                elif phase == "load":
                    group = self.layout.dit[slot][:1]
                else:
                    raise ValueError(f"unknown GPU phase {phase}")
                if group and occupied.isdisjoint(group):
                    self.held[slot] = group
                    self.phases[slot] = phase
                    self.changed.notify_all()
                    return group
                try:
                    await asyncio.wait_for(self.changed.wait(), timeout=1)
                except asyncio.TimeoutError:
                    pass
            return ()


class DualGpuBackend:
    def __init__(self, native, paths, memory_profile, layout):
        from .gpu_worker import WorkerSlot

        self.native = native
        self.layout = layout
        self.leases = GpuLeases(layout)
        self.paths = paths
        self.profile = memory_profile
        self.lora: Path | None = None
        self.slots = tuple(WorkerSlot(self, i) for i in range(2))

    def external_cards(self):
        ours = {w.pid for w in self.slots}
        busy = {uuid for uuid, pid in _smi("--query-compute-apps=gpu_uuid,pid") if int(pid) not in ours}
        return {i for i, uuid in self.layout.uuids.items() if uuid in busy}

    @property
    def key(self):
        return next((w.key for w in self.slots if w.key), None)

    @property
    def warm_state(self):
        states = [slot.warm_state for slot in self.slots]
        for status in ("failed", "loading", "ready"):
            for state in states:
                if state.get("status") == status:
                    return state
        return states[0]

    def preflight(self, engine):
        return self.native.preflight(engine)

    async def preload(self, engine):
        results = await asyncio.gather(*(w.preload(engine) for w in self.slots), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def stop(self):
        await asyncio.gather(*(w.stop() for w in self.slots))

    def configure_memory_profile(self, profile):
        self.native.configure_memory_profile(profile)
        self.profile = profile

    def configure_lora_checkpoint(self, checkpoint):
        self.native.configure_lora_checkpoint(checkpoint)
        self.lora = checkpoint

    def configure_output_root(self, output_root):
        from dataclasses import replace

        self.native.configure_output_root(output_root)
        self.paths = replace(self.paths, output_dir=output_root)
