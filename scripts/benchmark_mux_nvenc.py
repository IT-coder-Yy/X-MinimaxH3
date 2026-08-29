#!/usr/bin/env python3
"""Compare release CPU H.264 and RTX 4090 NVENC on identical decoded AV."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import av
import numpy as np

from h3serve.native_engine.adapters.sampling_mux import AtomicPyAVMuxer, MuxConfig


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "runtime/calibration/mux_nvenc",
    )
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"source does not exist: {args.source}")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    return args


def decode_source(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    with av.open(str(path)) as container:
        stream = next(item for item in container.streams if item.type == "video")
        fps = int(round(float(stream.average_rate)))
        frames = np.stack(
            [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
        )
    with av.open(str(path)) as container:
        stream = next(item for item in container.streams if item.type == "audio")
        sample_rate = int(stream.codec_context.sample_rate or stream.rate)
        chunks = [frame.to_ndarray() for frame in container.decode(stream)]
        audio = np.ascontiguousarray(np.concatenate(chunks, axis=1), dtype=np.float32)
    return frames, audio, sample_rate, fps


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    frames, audio, sample_rate, fps = decode_source(args.source)
    report = {
        "schema_version": 1,
        "source": str(args.source.resolve()),
        "shape": list(frames.shape),
        "sample_rate": sample_rate,
        "fps": fps,
        "runs": [],
    }
    configs = (
        ("libx264", MuxConfig(video_codec="libx264", video_crf=18, video_preset="veryfast")),
        ("h264_nvenc", MuxConfig(video_codec="h264_nvenc", video_crf=18, video_preset="p6")),
    )
    for name, config in configs:
        muxer = AtomicPyAVMuxer(output_root=args.output_root, config=config)
        for index in range(args.repeat):
            output = args.output_root / f"{args.source.stem}_{name}_r{index + 1}.mp4"
            started = time.perf_counter()
            try:
                result = muxer.write(
                    video=frames,
                    audio=audio,
                    sample_rate=sample_rate,
                    fps=fps,
                    output_path=output,
                )
                item = {
                    "codec": name,
                    "repeat": index + 1,
                    "status": "success",
                    "seconds": time.perf_counter() - started,
                    "bytes": output.stat().st_size,
                    "output": str(output.resolve()),
                    "media": result["media"],
                    "options": config.video_options(),
                }
            except Exception as error:
                item = {
                    "codec": name,
                    "repeat": index + 1,
                    "status": "failed",
                    "seconds": time.perf_counter() - started,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "options": config.video_options(),
                }
            report["runs"].append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
            if item["status"] == "failed":
                break
    report_path = args.output_root / "mux_nvenc_benchmark.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
