#!/usr/bin/env python3
"""Sample synchronous RAM use of one H3 service tree and write a JSON report.

Linux VmHWM is per-process and cannot prove that two process peaks happened at
the same instant. This sampler sums RSS/PSS for the service and descendants at
each interval, which is the release metric used for future full-job audits.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path


KIB = 1024
GIB = 1024**3


def read_status(pid: int) -> dict[str, int | str]:
    document: dict[str, int | str] = {"pid": pid}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        key, _, raw = line.partition(":")
        value = raw.strip()
        if key in {"Name", "State"}:
            document[key.lower()] = value
        elif key == "PPid":
            document["ppid"] = int(value.split()[0])
        elif key in {"VmRSS", "VmHWM", "VmSwap"}:
            document[key.lower()] = int(value.split()[0]) * KIB
    return document


def read_pss(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) * KIB
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return 0


def descendants(root_pid: int) -> list[int]:
    parents: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            closing = stat.rfind(")")
            fields = stat[closing + 2:].split()
            parent = int(fields[1])
            parents.setdefault(parent, []).append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
            continue
    result, pending = [], [root_pid]
    while pending:
        pid = pending.pop()
        if pid in result:
            continue
        result.append(pid)
        pending.extend(parents.get(pid, ()))
    return sorted(result)


def snapshot(root_pid: int, additional_pids: tuple[int, ...] = ()) -> dict:
    processes = []
    selected: set[int] = set()
    for root in (root_pid, *additional_pids):
        selected.update(descendants(root))
    for pid in sorted(selected):
        try:
            item = read_status(pid)
            item["pss_bytes"] = read_pss(pid)
            processes.append(item)
        except (FileNotFoundError, ProcessLookupError):
            continue
    return {
        "unix_seconds": time.time(),
        "rss_bytes": sum(int(item.get("vmrss", 0)) for item in processes),
        "pss_bytes": sum(int(item.get("pss_bytes", 0)) for item in processes),
        "swap_bytes": sum(int(item.get("vmswap", 0)) for item in processes),
        "processes": processes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True, help="H3 server PID")
    parser.add_argument(
        "--additional-pid", type=int, action="append", default=[],
        help="additional process-tree root; repeat for isolated stage workers",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--duration", type=float, default=0.0,
                        help="seconds; zero samples until SIGINT/SIGTERM")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")

    stopped = False

    def stop(_signal, _frame) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    count = 0
    peak = {"rss_bytes": 0, "pss_bytes": 0, "swap_bytes": 0}
    peak_pss_snapshot = None
    while not stopped:
        if not Path(f"/proc/{args.pid}").exists():
            break
        item = snapshot(args.pid, tuple(args.additional_pid))
        count += 1
        # PSS apportions shared pages and is the defensible whole-tree host-RAM
        # metric. Summed RSS remains diagnostic only because forked workers can
        # map the same shared storage into every process.
        if item["pss_bytes"] > peak["pss_bytes"]:
            peak_pss_snapshot = item
        for key in peak:
            peak[key] = max(peak[key], item[key])
        if args.duration > 0 and time.monotonic() - started >= args.duration:
            break
        time.sleep(args.interval)

    report = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "root_pid": args.pid,
        "additional_root_pids": args.additional_pid,
        "interval_seconds": args.interval,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "sample_count": count,
        "synchronous_peak": {
            **peak,
            "rss_gib": round(peak["rss_bytes"] / GIB, 3),
            "pss_gib": round(peak["pss_bytes"] / GIB, 3),
            "swap_gib": round(peak["swap_bytes"] / GIB, 3),
        },
        "metric_note": (
            "pss_gib is the primary physical-RAM estimate; rss_gib can "
            "double-count shared mappings across child processes"
        ),
        "peak_pss_snapshot": peak_pss_snapshot,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report["synchronous_peak"], indent=2))


if __name__ == "__main__":
    main()
