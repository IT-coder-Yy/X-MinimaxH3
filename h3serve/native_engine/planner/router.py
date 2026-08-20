"""Fail-closed selection of the fastest measured strategy that fits memory."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import replace

from .contracts import CalibratedProfile, ExecutionPlan, RouteDecision, WorkloadFeatures
from .vae_tiles import select_vae_tile


class NoFeasibleProfile(RuntimeError):
    pass


class RTX4090Planner:
    def __init__(
        self,
        profiles: Sequence[CalibratedProfile],
        *,
        device_budget_bytes: int = 23 * 1024**3,
        reserve_bytes: int = 1024**3,
        allow_experimental: bool = False,
    ) -> None:
        if not profiles:
            raise ValueError("planner requires at least one measured profile")
        if device_budget_bytes <= 0 or reserve_bytes < 0:
            raise ValueError("invalid device memory policy")
        ids = [profile.profile_id for profile in profiles]
        if len(ids) != len(set(ids)):
            raise ValueError("profile ids must be unique")
        self.profiles = tuple(profiles)
        self.device_budget_bytes = int(device_budget_bytes)
        self.reserve_bytes = int(reserve_bytes)
        self.allow_experimental = bool(allow_experimental)

    def select(
        self,
        features: WorkloadFeatures,
        *,
        free_device_bytes: int,
        current_profile_id: str | None = None,
        cached_shape_keys: Collection[tuple[int, int, int, int]] = (),
    ) -> RouteDecision:
        if free_device_bytes <= 0:
            raise NoFeasibleProfile("CUDA reports no free device memory")
        memory_limit = min(
            self.device_budget_bytes,
            max(0, int(free_device_bytes) - self.reserve_bytes),
        )
        shape_hit = features.shape_key in cached_shape_keys
        candidates: list[
            tuple[float, CalibratedProfile, ExecutionPlan, int, bool]
        ] = []
        for profile in self.profiles:
            if profile.evidence_status != "validated" and not self.allow_experimental:
                continue
            if not profile.supports(features):
                continue
            peak = profile.memory.predict(features)
            if peak > memory_limit:
                continue
            switched = current_profile_id not in (None, profile.profile_id)
            latency = profile.latency.predict(features)
            if switched:
                latency += profile.switch_penalty_seconds
            plan = profile.plan
            if profile.vae_tile_candidates is not None:
                tile = select_vae_tile(
                    width=features.width,
                    height=features.height,
                    candidates=profile.vae_tile_candidates,
                ).tile_size
                plan = replace(plan, vae_spatial_tile=(tile, tile))
            candidates.append((latency, profile, plan, peak, switched))

        if not candidates:
            raise NoFeasibleProfile(
                "no validated RTX4090 execution profile fits this H3 workload and current free VRAM"
            )
        latency, profile, plan, peak, switched = min(
            candidates, key=lambda item: (item[0], item[3], item[1].profile_id)
        )
        return RouteDecision(
            profile_id=profile.profile_id,
            plan=plan,
            predicted_seconds=latency,
            predicted_peak_bytes=peak,
            shape_cache_hit=shape_hit,
            switched_profile=switched,
        )
