#!/usr/bin/env python3
"""Measure task-independent peripheral flicker signals in one video.

This is a screening metric, not a Human quality score.  It measures temporal
second differences and isolated positive luminance impulses in an outer frame
band after deterministic downscaling.  Smooth constant-velocity luminance
changes cancel; background motion and camera motion may still contribute, so
only matched prompt/seed candidates are comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np


def _probe(video: Path) -> tuple[int, int, int]:
    result = subprocess.run(
        (
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames",
            "-of",
            "json",
            str(video),
        ),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"]), int(stream["nb_frames"])


def _decode_gray(video: Path, *, target_width: int) -> np.ndarray:
    width, height, _frames = _probe(video)
    target_height = max(2, round(height * target_width / width / 2) * 2)
    result = subprocess.run(
        (
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-vf",
            f"scale={target_width}:{target_height}:flags=area,format=gray",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ),
        check=True,
        stdout=subprocess.PIPE,
    )
    frame_bytes = target_width * target_height
    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    if raw.size % frame_bytes:
        raise RuntimeError("decoded grayscale byte count is not frame aligned")
    return raw.reshape((-1, target_height, target_width))


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def analyze_frames(
    frames: np.ndarray,
    *,
    border_fraction: float = 0.18,
) -> dict[str, object]:
    if frames.ndim != 3 or frames.shape[0] < 3:
        raise ValueError("frames must have shape [T,H,W] with T >= 3")
    if not 0.05 <= border_fraction <= 0.45:
        raise ValueError("border fraction must lie inside [0.05, 0.45]")
    _count, height, width = frames.shape
    border_y = max(1, round(height * border_fraction))
    border_x = max(1, round(width * border_fraction))
    border = np.ones((height, width), dtype=bool)
    border[border_y : height - border_y, border_x : width - border_x] = False
    center = ~border

    values = frames.astype(np.float32)
    temporal_second = np.abs(
        values[2:] - 2.0 * values[1:-1] + values[:-2]
    )
    border_second = temporal_second[:, border]
    center_second = temporal_second[:, center]
    border_frame_mean = border_second.mean(axis=1)
    center_frame_mean = center_second.mean(axis=1)
    border_frame_p99 = np.quantile(border_second, 0.99, axis=1)

    positive_impulse = np.maximum(
        values[1:-1] - 0.5 * (values[:-2] + values[2:]),
        0.0,
    )[:, border]
    impulse_frame_p99 = np.quantile(positive_impulse, 0.99, axis=1)
    impulse_fraction_over_20 = (positive_impulse > 20.0).mean(axis=1)
    median = float(np.median(impulse_frame_p99))
    mad = float(np.median(np.abs(impulse_frame_p99 - median)))
    threshold = median + max(6.0 * mad, 2.0)
    outliers = np.flatnonzero(impulse_frame_p99 > threshold) + 1

    return {
        "frame_count": int(frames.shape[0]),
        "analysis_height": int(height),
        "analysis_width": int(width),
        "border_fraction": border_fraction,
        "border_temporal_second_difference": _summary(border_frame_mean),
        "center_temporal_second_difference": _summary(center_frame_mean),
        "border_to_center_mean_ratio": float(
            border_frame_mean.mean() / max(center_frame_mean.mean(), 1e-9)
        ),
        "border_pixel_p99_second_difference": _summary(border_frame_p99),
        "positive_luminance_impulse_p99": _summary(impulse_frame_p99),
        "positive_impulse_fraction_over_20": _summary(
            impulse_fraction_over_20
        ),
        "bright_outlier_threshold": threshold,
        "bright_outlier_frame_indices": [int(index) for index in outliers],
        "screening_limit": (
            "Matched-content screening only; camera/background motion can "
            "raise the metric and Human continuous playback remains required."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-width", type=int, default=320)
    parser.add_argument("--border-fraction", type=float, default=0.18)
    args = parser.parse_args()
    video = args.video.resolve()
    if not video.is_file():
        raise SystemExit(f"video does not exist: {video}")
    if args.target_width < 64:
        raise SystemExit("target width must be at least 64")
    report = {
        "schema_version": "h3_peripheral_temporal_stability_v1",
        "video": str(video),
        "analysis": analyze_frames(
            _decode_gray(video, target_width=args.target_width),
            border_fraction=args.border_fraction,
        ),
    }
    output = args.output or video.with_suffix(".peripheral_temporal.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
