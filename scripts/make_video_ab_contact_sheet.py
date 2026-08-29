#!/usr/bin/env python3
"""Extract matched timestamps from two videos into a compact A/B contact sheet."""

from __future__ import annotations

import argparse
from pathlib import Path

import av
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--times", required=True, help="comma-separated seconds")
    parser.add_argument("--label-a", default="A: eager")
    parser.add_argument("--label-b", default="B: candidate")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_nearest(path: Path, times: list[float]) -> list[Image.Image]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        targets = [round(second * fps) for second in times]
        result: list[Image.Image] = []
        target_index = 0
        for frame_index, frame in enumerate(container.decode(stream)):
            while target_index < len(targets) and frame_index >= targets[target_index]:
                result.append(frame.to_image().convert("RGB"))
                target_index += 1
            if target_index == len(targets):
                break
    if len(result) != len(times):
        raise RuntimeError(f"could not decode every requested timestamp from {path}")
    return result


def main() -> int:
    args = parse_args()
    times = [float(value) for value in args.times.split(",")]
    if not times or any(value < 0 for value in times):
        raise ValueError("times must be non-negative")
    rows = [read_nearest(args.a, times), read_nearest(args.b, times)]
    width, height = rows[0][0].size
    thumb_width = 432
    thumb_height = round(height * thumb_width / width)
    label_height = 32
    sheet = Image.new(
        "RGB", (thumb_width * len(times), (thumb_height + label_height) * 2), "#111111"
    )
    draw = ImageDraw.Draw(sheet)
    for row_index, (label, images) in enumerate(
        ((args.label_a, rows[0]), (args.label_b, rows[1]))
    ):
        top = row_index * (thumb_height + label_height)
        for column, (second, frame) in enumerate(zip(times, images)):
            left = column * thumb_width
            sheet.paste(frame.resize((thumb_width, thumb_height)), (left, top + label_height))
            draw.text((left + 8, top + 8), f"{label}  t={second:.2f}s", fill="white")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=92)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
