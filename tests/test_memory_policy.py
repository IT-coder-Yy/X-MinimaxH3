from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from h3serve.memory_policy import (
    HOST_MEMORY_PROFILES,
    HostMemoryStatus,
    _current_cgroup_memory_files,
    resolve_host_memory_profile,
    validate_workload_for_profile,
)


class HostMemoryPolicyTests(unittest.TestCase):
    def status(self, total: float) -> HostMemoryStatus:
        return HostMemoryStatus(total, total, total)

    def test_auto_uses_only_meaningful_strategy_boundaries(self) -> None:
        self.assertEqual(resolve_host_memory_profile("auto", self.status(140)).key, "fullspeed")
        # A 128GB Windows host commonly exposes about 110GiB inside WSL.
        self.assertEqual(resolve_host_memory_profile("auto", self.status(110)).key, "fullspeed")
        self.assertEqual(resolve_host_memory_profile("auto", self.status(90)).key, "generation_hot")
        self.assertEqual(resolve_host_memory_profile("auto", self.status(70)).key, "compact")
        with self.assertRaisesRegex(RuntimeError, "64 GiB supported minimum"):
            resolve_host_memory_profile("auto", self.status(48))

    def test_explicit_profile_checks_effective_wsl_limit(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "effective limit"):
            resolve_host_memory_profile("generation_hot", self.status(48))

    def test_auto_downgrades_when_other_processes_consume_ram(self) -> None:
        busy = HostMemoryStatus(140, 140, 50)
        self.assertEqual(resolve_host_memory_profile("auto", busy).key, "compact")

    def test_profiles_fail_close_above_native_generation_envelopes(self) -> None:
        compact = HOST_MEMORY_PROFILES["compact"]
        validate_workload_for_profile(compact, width=864, height=480, frames=362)
        validate_workload_for_profile(compact, width=1280, height=736, frames=362)
        validate_workload_for_profile(compact, width=1920, height=1088, frames=192)
        validate_workload_for_profile(compact, width=1440, height=1088, frames=243)
        validate_workload_for_profile(compact, width=1088, height=1088, frames=328)
        with self.assertRaisesRegex(ValueError, r"width\*height\*frames"):
            validate_workload_for_profile(compact, width=1920, height=1088, frames=209)
        with self.assertRaisesRegex(ValueError, "spatial-temporal"):
            validate_workload_for_profile(compact, width=1920, height=1120, frames=192)
        validate_workload_for_profile(
            HOST_MEMORY_PROFILES["generation_hot"],
            width=1280, height=736, frames=362,
        )

    def test_no_unsupported_below_64gb_tier_is_published(self) -> None:
        self.assertEqual(
            {profile.minimum_ram_gib for profile in HOST_MEMORY_PROFILES.values()},
            {64, 96, 128},
        )

    def test_only_128gb_profile_allows_h3_and_upscaler_to_overlap(self) -> None:
        self.assertFalse(HOST_MEMORY_PROFILES["fullspeed"].exclusive_upscaler)
        self.assertTrue(HOST_MEMORY_PROFILES["generation_hot"].exclusive_upscaler)
        self.assertTrue(HOST_MEMORY_PROFILES["compact"].exclusive_upscaler)

    def test_current_cgroup_membership_is_checked_before_root(self) -> None:
        paths = _current_cgroup_memory_files()
        self.assertTrue(paths)
        # The active process path must be represented on cgroup-v2 systems;
        # checking only /sys/fs/cgroup/memory.max misses systemd/container caps.
        membership = Path("/proc/self/cgroup").read_text(encoding="utf-8")
        if "0::" in membership:
            relative = membership.split("0::", 1)[1].splitlines()[0].lstrip("/")
            expected = Path("/sys/fs/cgroup") / relative / "memory.max"
            self.assertEqual(paths[0][0], expected)

    def test_detected_capacity_honors_leaf_cgroup_limit_and_usage(self) -> None:
        from h3serve.memory_policy import GIB, detect_host_memory

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            limit = root / "memory.max"
            usage = root / "memory.current"
            limit.write_text(str(58 * GIB))
            usage.write_text(str(10 * GIB))
            with patch(
                "h3serve.memory_policy._meminfo",
                return_value={"MemTotal": 128 * GIB, "MemAvailable": 100 * GIB},
            ), patch(
                "h3serve.memory_policy._current_cgroup_memory_files",
                return_value=((limit, usage),),
            ):
                status = detect_host_memory()
        self.assertEqual(status.physical_total_gib, 128)
        self.assertEqual(status.effective_limit_gib, 58)
        self.assertEqual(status.available_gib, 48)
        self.assertEqual(resolve_host_memory_profile("auto", status).key, "compact")


if __name__ == "__main__":
    unittest.main()
