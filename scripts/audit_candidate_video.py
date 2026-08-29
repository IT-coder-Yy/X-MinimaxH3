#!/usr/bin/env python3
"""Build a reproducible automated audit bundle for one H3 candidate.

This deliberately does not judge motion causality, speaker identity, acting,
or physical correctness.  Those properties require continuous Human playback.
The report only records media integrity, runtime/VRAM, a blur reference, and a
uniform eight-frame contact sheet for catastrophic visual screening.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


BLUR_RE = re.compile(r"blur mean:\s*([0-9.]+)")


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _probe(video: Path) -> dict[str, object]:
    result = _run(
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=index,codec_type,codec_name,width,height,r_frame_rate,duration",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(video),
    )
    return json.loads(result.stdout)


def _blur_mean(video: Path) -> float:
    result = subprocess.run(
        (
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(video),
            "-vf",
            "blurdetect",
            "-f",
            "null",
            "-",
        ),
        check=True,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    match = BLUR_RE.search(result.stderr)
    if match is None:
        raise RuntimeError("ffmpeg blurdetect did not report a blur mean")
    return float(match.group(1))


def _contact_sheet(video: Path, output: Path, duration: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 8.0 / max(duration, 1e-6)
    _run(
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps={sample_rate:.12f},scale=640:-1,tile=4x2",
        "-frames:v",
        "1",
        str(output),
    )


def _load_runtime(
    report_path: Path | None,
    *,
    video: Path,
) -> dict[str, object] | None:
    if report_path is None:
        return None
    document = json.loads(report_path.read_text(encoding="utf-8"))
    requests = document.get("requests") or []
    matched = [
        row
        for row in requests
        if isinstance(row, dict)
        and row.get("output")
        and Path(str(row["output"])).resolve() == video
    ]
    # Production experiments are staged on Linux ext4 and copied once to the
    # canonical project tree.  Preserve the timing binding across that exact
    # relocation when, and only when, the output filename is unique.
    if not matched:
        matched = [
            row
            for row in requests
            if isinstance(row, dict)
            and row.get("output")
            and Path(str(row["output"])).name == video.name
        ]
    request = (
        matched[0]
        if len(matched) == 1
        else requests[0]
        if len(requests) == 1
        else None
    )
    if not isinstance(request, dict):
        return None
    adaptive_attention = document.get("adaptive_attention")
    adaptive_attention_record = (
        None
        if adaptive_attention is None
        else {
            "scope": (
                "single_request"
                if len(requests) == 1
                else "batch_cumulative_not_candidate_specific"
            ),
            "request_count": len(requests),
            "telemetry": adaptive_attention,
        }
    )
    return {
        "report": str(report_path.resolve()),
        "total_seconds": request.get("total_seconds"),
        "denoise_seconds": (request.get("phases") or {}).get("denoise"),
        "peak_allocated_gib": request.get("peak_allocated_gib"),
        "actual_steps": request.get("actual_steps"),
        "forecast_steps": request.get("forecast_steps"),
        "candidate": (request.get("execution_profile") or {}).get(
            "joint_acceleration"
        ),
        # The backend exports one process-level accumulator.  In a multi-
        # candidate hot batch it must never be presented as if every action
        # call belonged to the matched video alone.
        "adaptive_attention": adaptive_attention_record,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--runtime-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--static-gate-status",
        choices=("pending", "pass", "fail"),
        default="pending",
    )
    args = parser.parse_args()

    video = args.video.resolve()
    if not video.is_file():
        raise SystemExit(f"video does not exist: {video}")
    output = (args.output or video.with_suffix(".audit.json")).resolve()
    media = _probe(video)
    duration = float((media.get("format") or {}).get("duration") or 0.0)
    contact_sheet = output.with_name(f"{video.stem}_contact8.jpg")
    _contact_sheet(video, contact_sheet, duration)

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "video": str(video),
        "media": media,
        "automated_observations": {
            "blurdetect_mean": _blur_mean(video),
            "blurdetect_interpretation": (
                "reference only; lower was sharper in this fixed-scene series"
            ),
            "contact_sheet": str(contact_sheet),
            "static_catastrophe_gate": args.static_gate_status,
        },
        "runtime": _load_runtime(
            args.runtime_report.resolve() if args.runtime_report else None,
            video=video,
        ),
        "human_review": {
            "status": "awaiting Human continuous playback",
            "required": [
                "motion causality and contact physics",
                "identity and speaker-timbre continuity",
                "acting and instruction following",
            ],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
