"""Physical GPU configuration and phase ownership for two native workers."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


def _smi(query: str) -> list[list[str]]:
    output = subprocess.check_output(
        ["nvidia-smi", query, "--format=csv,noheader,nounits"], text=True, timeout=10
    )
    return [[v.strip() for v in line.split(",")] for line in output.splitlines() if line.strip()]


def _mask_indices(mask, uuids):
    """Physical indices and their input tokens, in CUDA enumeration order."""
    selected = {}
    by_uuid = mask.startswith("GPU-")
    for token in mask.split(","):
        if by_uuid:
            matches = [i for i, uuid in uuids.items()
                       if len(token) > 4 and token.startswith("GPU-") and uuid.lower().startswith(token.lower())]
            if len(matches) != 1:
                break
            index = matches[0]
        else:
            number = re.match(r"\s*([+-]?[0-9]+)", token, re.ASCII)
            index = int(number[1]) if number else -1
            if index not in uuids:
                break
        if index in selected:
            return {}  # A repeated device invalidates the CUDA allocation.
        selected[index] = token.strip()
    return selected


def restrict_cuda_mask(local, inherited):
    """Keep the requested CUDA order inside the caller's allocation."""
    if not local:
        return ""
    uuids = {int(i): uuid for i, uuid in _smi("--query-gpu=index,uuid")}
    allowed = _mask_indices(inherited, uuids)
    filtered = [token for i, token in _mask_indices(local, uuids).items() if i in allowed]
    if not filtered:
        raise ValueError(f"CUDA_VISIBLE_DEVICES={local!r} has no GPU in the inherited allocation {inherited!r}")
    return ",".join(filtered)


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
        force_dual = mode == "dual"

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
        if len(qwen) > 8:
            raise ValueError("Qwen supports up to eight GPUs (one per KV head)")
        selected = set((*first, *second, *qwen, *decode))
        try:
            inventory = {int(i): (uuid, capability, int(mib)) for i, uuid, capability, mib in
                         _smi("--query-gpu=index,uuid,compute_cap,memory.total")}
        except (OSError, subprocess.SubprocessError):
            if force_dual:
                raise ValueError("dual GPU mode requires nvidia-smi to enumerate GPUs") from None
            return None
        from .deployment_profiles import get_resource_backend

        budget_mib = get_resource_backend("int8_24gb").planner_budget_gib * 1024
        available = {i for i, (_, cap, mib) in inventory.items()
                     if cap == "8.9" and (force_dual or mib >= budget_mib)}
        uuids = {i: row[0] for i, row in inventory.items()}
        # run.sh distinguishes an inherited allocation from the old primary-only .env.
        inherited = env.get("H3_INHERITED_CUDA_MASK", env.get("CUDA_VISIBLE_DEVICES"))
        if inherited is not None and inherited != "__unset__":
            available.intersection_update(_mask_indices(inherited, uuids))
        local_mask = env.get("CUDA_VISIBLE_DEVICES")
        if not force_dual and local_mask is not None:
            allowed = _mask_indices(local_mask, uuids).keys()
            legacy_primary = "," not in local_mask and allowed == {first[0]}
            if not legacy_primary and not selected <= allowed:
                return None
        if not selected <= available:
            if force_dual:
                raise ValueError(f"selected SM89 GPUs are unavailable: {sorted(selected - available)}")
            return None
        return cls(qwen, (first, second), decode, {i: inventory[i][0] for i in sorted(selected)})


def resolve_service_resources(requested, status):
    from .memory_policy import resolve_host_memory_profile

    layout = GpuLayout.from_env()
    if layout is not None:
        try:
            profile = resolve_host_memory_profile(requested, status, sessions=len(layout.dit))
        except RuntimeError as error:
            if os.environ.get("H3_GPU_MODE") == "dual":
                raise RuntimeError(f"H3_GPU_MODE=dual: {error}") from error
            print(f"H3 auto: falling back to one worker: {error}", flush=True)
            # The native worker uses cuda:0. Keep an explicit local mask or
            # use the configured primary when the shell did not select one.
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", layout.uuids[layout.dit[0][0]])
        else:
            return profile, layout
    return resolve_host_memory_profile(requested, status), None


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

    async def acquire(self, slot, phase, cancel, external_cards, minimum=1):
        """Swap phase leases atomically; only granted cards are occupied."""
        async with self.changed:
            self.held.pop(slot, None)
            self.phases[slot] = "wait_" + phase
            self.changed.notify_all()
            while not cancel.is_set():
                occupied = {card for other, group in self.held.items() if other != slot for card in group}
                if phase in {"qwen", "decode"}:
                    pool = self.layout.qwen if phase == "qwen" else self.layout.decode
                    candidates = [card for card in pool if card not in occupied]
                    if candidates:
                        external = await asyncio.to_thread(external_cards)
                        candidates = [card for card in candidates if card not in external]
                    group = tuple(candidates[:1] if phase == "decode" else candidates)
                    if phase == "qwen":
                        if len(group) < minimum or "qwen" in self.phases.values():
                            group = ()
                        for earlier, state in self.phases.items():
                            if earlier == slot:
                                break
                            if state in {"starting", "wait_load", "load", "wait_qwen"}:
                                group = ()
                                break
                elif phase == "dit":
                    group = self.layout.dit[slot]
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


if __name__ == "__main__":
    import sys

    try:
        print(restrict_cuda_mask(*sys.argv[1:]))
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        sys.exit(str(error))
