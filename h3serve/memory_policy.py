"""Host-RAM policies for the fixed single-RTX4090 service."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .contract import (
    MAX_CUSTOM_DIMENSION,
    MAX_CUSTOM_PIXELS,
    MAX_CUSTOM_SHORT_EDGE,
    MAX_NATIVE_PIXEL_FRAMES,
)


GIB = 1024**3


@dataclass(frozen=True, slots=True)
class HostMemoryProfile:
    key: str
    label: str
    minimum_ram_gib: int
    description: str
    cache_qwen_weights: bool
    pin_model_weights: bool
    copy_model_weights: bool
    preload_upscaler: bool
    exclusive_upscaler: bool
    parallel_model_build: bool
    evidence: str

    def public(self) -> dict[str, object]:
        value = asdict(self)
        for private in (
            "cache_qwen_weights", "pin_model_weights", "copy_model_weights",
            "preload_upscaler",
            "exclusive_upscaler",
            "parallel_model_build",
        ):
            value.pop(private)
        return value


# A profile exists only when residency mechanics change. Larger machines use
# the same fastest eligible profile; there are intentionally no 128/96 aliases.
HOST_MEMORY_PROFILES: dict[str, HostMemoryProfile] = {
    "fullspeed": HostMemoryProfile(
        key="fullspeed",
        label="128GB 火力全开",
        minimum_ram_gib=128,
        description="Qwen与H3保持热态；生成与原生H3二次采样共享同一热引擎。",
        cache_qwen_weights=True,
        pin_model_weights=True,
        copy_model_weights=True,
        preload_upscaler=True,
        exclusive_upscaler=False,
        parallel_model_build=True,
        evidence="validated",
    ),
    "generation_hot": HostMemoryProfile(
        key="generation_hot",
        label="96GB 生成优先",
        minimum_ram_gib=96,
        description="Qwen与H3保持热态；生成与原生H3二次采样共享同一热引擎。",
        cache_qwen_weights=True,
        pin_model_weights=True,
        copy_model_weights=True,
        preload_upscaler=False,
        exclusive_upscaler=True,
        parallel_model_build=True,
        evidence="validated",
    ),
    "compact": HostMemoryProfile(
        key="compact",
        label="64GB 高效兼容",
        minimum_ram_gib=64,
        description="Qwen按执行层流水读取；H3按需驻留并支持原生二次采样。",
        cache_qwen_weights=False,
        pin_model_weights=True,
        copy_model_weights=True,
        preload_upscaler=False,
        exclusive_upscaler=True,
        parallel_model_build=False,
        evidence="validated",
    ),
}


@dataclass(frozen=True, slots=True)
class HostMemoryStatus:
    physical_total_gib: float
    effective_limit_gib: float
    available_gib: float

    def public(self) -> dict[str, float]:
        return {
            "physical_total_gib": round(self.physical_total_gib, 2),
            "effective_limit_gib": round(self.effective_limit_gib, 2),
            "available_gib": round(self.available_gib, 2),
        }


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            value = raw.strip().split()
            if value and value[0].isdigit():
                values[key] = int(value[0]) * 1024
    except (OSError, ValueError):
        pass
    return values


def _current_cgroup_memory_files() -> tuple[tuple[Path, Path | None], ...]:
    """Locate the calling process's cgroup limit, not only the root limit."""

    candidates: list[tuple[Path, Path | None]] = []
    try:
        memberships = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        memberships = []
    for line in memberships:
        try:
            _hierarchy, controllers, relative = line.split(":", 2)
        except ValueError:
            continue
        relative = relative.lstrip("/")
        if controllers == "":  # unified cgroup v2
            root = Path("/sys/fs/cgroup")
            directory = root / relative
            while True:
                candidates.append(
                    (directory / "memory.max", directory / "memory.current")
                )
                if directory == root:
                    break
                directory = directory.parent
        elif "memory" in controllers.split(","):  # cgroup v1
            root = Path("/sys/fs/cgroup/memory")
            directory = root / relative
            while True:
                candidates.append(
                    (
                        directory / "memory.limit_in_bytes",
                        directory / "memory.usage_in_bytes",
                    )
                )
                if directory == root:
                    break
                directory = directory.parent
    # Root files retain compatibility with containers that hide membership.
    candidates.extend((
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    ))
    return tuple(dict.fromkeys(candidates))


def detect_host_memory() -> HostMemoryStatus:
    info = _meminfo()
    total = int(info.get("MemTotal", 0))
    available = int(info.get("MemAvailable", total))
    limits = [total] if total else []
    cgroup_available: list[int] = []
    for candidate, usage_path in _current_cgroup_memory_files():
        try:
            text = candidate.read_text().strip()
            if text != "max" and text.isdigit():
                value = int(text)
                # Some cgroup-v1 hosts publish a sentinel near LONG_MAX.
                if value > 0 and (not total or value < total * 16):
                    limits.append(value)
                    if usage_path is not None:
                        usage_text = usage_path.read_text().strip()
                        if usage_text.isdigit():
                            cgroup_available.append(max(0, value - int(usage_text)))
        except OSError:
            pass
    effective = min(limits) if limits else total
    if cgroup_available:
        available = min(available, *cgroup_available)
    return HostMemoryStatus(total / GIB, effective / GIB, available / GIB)


def current_process_pss_gib() -> float:
    """Physical pages reclaimed when the current hot session is rebuilt."""

    try:
        for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) * 1024 / GIB
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def resolve_host_memory_profile(
    requested: str,
    status: HostMemoryStatus | None = None,
) -> HostMemoryProfile:
    status = detect_host_memory() if status is None else status
    # A retail 48/64 GiB installation reports less usable RAM to Linux/WSL
    # after firmware and host reservations. Five percent is the conventional
    # capacity-label tolerance; operational headroom is enforced by each
    # profile's measured resident target, not by pretending MemTotal is exact.
    measured_peak_targets = {
        "fullspeed": 88.835,
        "generation_hot": 73.141,
        "compact": 49.931,
    }
    cold_build_targets = {
        "fullspeed": 88.835,
        "generation_hot": 73.141,
        "compact": 39.390,
    }

    def fits(profile: HostMemoryProfile) -> bool:
        # Use measured full-product peaks plus 8GiB rather than assuming that
        # Linux/WSL exposes the retail DIMM label verbatim.
        effective_capacity_floor = measured_peak_targets[profile.key] + 8.0
        capacity_ok = status.effective_limit_gib >= effective_capacity_floor
        # At startup the process itself is still small, so MemAvailable must
        # cover the measured construction peak plus an 8GiB OS reserve.
        available_ok = status.available_gib >= cold_build_targets[profile.key] + 8.0
        return capacity_ok and available_ok

    if requested == "auto":
        eligible = [
            profile for profile in HOST_MEMORY_PROFILES.values()
            if fits(profile)
        ]
        if not eligible:
            raise RuntimeError(
                f"effective host RAM {status.effective_limit_gib:.1f} GiB is below "
                "the 64 GiB supported minimum"
            )
        return max(eligible, key=lambda profile: profile.minimum_ram_gib)
    try:
        profile = HOST_MEMORY_PROFILES[requested]
    except KeyError as error:
        raise ValueError(f"unknown host-memory profile: {requested}") from error
    if not fits(profile):
        raise RuntimeError(
            f"{profile.label} requires a {profile.minimum_ram_gib} GiB class host "
            f"and enough free RAM; effective limit is {status.effective_limit_gib:.1f} "
            f"GiB and currently available is {status.available_gib:.1f} GiB"
        )
    return profile


def validate_workload_for_profile(
    profile: HostMemoryProfile,
    *,
    width: int,
    height: int,
    frames: int,
) -> None:
    """Fail before queueing workloads outside a measured host-RAM envelope."""

    del profile
    width = int(width)
    height = int(height)
    frames = int(frames)
    pixels = width * height
    if (
        width > MAX_CUSTOM_DIMENSION
        or height > MAX_CUSTOM_DIMENSION
        or min(width, height) > MAX_CUSTOM_SHORT_EDGE
        or pixels > MAX_CUSTOM_PIXELS
        or frames > 362
        or pixels * frames > MAX_NATIVE_PIXEL_FRAMES
    ):
        raise ValueError(
            "workload exceeds the validated native spatial-temporal envelope "
            f"(width*height*frames <= {MAX_NATIVE_PIXEL_FRAMES})"
        )


__all__ = [
    "HOST_MEMORY_PROFILES",
    "HostMemoryProfile",
    "HostMemoryStatus",
    "detect_host_memory",
    "current_process_pss_gib",
    "_current_cgroup_memory_files",
    "resolve_host_memory_profile",
    "validate_workload_for_profile",
]
