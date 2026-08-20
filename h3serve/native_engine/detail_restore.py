"""Bounded intraframe detail restoration for the reviewed long-video route.

The filter deliberately contains no temporal operator: every decoded frame is
processed independently, so it cannot invent a new trajectory, interpolate
motion, or mix identities across time.  Publication is atomic and the raw H3
output is retained beside the final file while the route remains under review.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


FILTER_GRAPH = (
    "vaguedenoiser=threshold=1.2:method=garrote:nsteps=4:"
    "percent=45:planes=7:type=bayes,cas=strength=0.25"
)


class DetailRestoreCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DetailRestoreResult:
    output_path: Path
    raw_output_path: Path
    elapsed_seconds: float
    width: int
    height: int
    frames: int


def _ffmpeg_command(source: Path, temporary: Path) -> list[str]:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("ffmpeg is required for intraframe detail restoration")
    return [
        executable,
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        FILTER_GRAPH,
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p7",
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-cq",
        "16",
        "-b:v",
        "0",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(temporary),
    ]


def _encoder_arguments() -> list[str]:
    return [
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p7",
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-cq",
        "16",
        "-b:v",
        "0",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
    ]


def _frame_partitions(frames: int, shards: int) -> tuple[tuple[int, int], ...]:
    if frames <= 0 or shards <= 0 or shards > frames:
        raise ValueError("detail restoration partitions must be non-empty")
    quotient, remainder = divmod(frames, shards)
    partitions: list[tuple[int, int]] = []
    start = 0
    for index in range(shards):
        count = quotient + (1 if index < remainder else 0)
        partitions.append((start, count))
        start += count
    if start != frames:
        raise AssertionError("detail restoration frame partition is incomplete")
    return tuple(partitions)


def _ffmpeg_chunk_command(
    source: Path,
    target: Path,
    *,
    start_frame: int,
    frame_count: int,
    fps: int,
) -> list[str]:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("ffmpeg is required for intraframe detail restoration")
    if start_frame < 0 or frame_count <= 0 or fps <= 0:
        raise ValueError("invalid detail restoration chunk geometry")
    return [
        executable,
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start_frame / fps:.9f}",
        "-i",
        str(source),
        "-frames:v",
        str(frame_count),
        "-vf",
        FILTER_GRAPH,
        "-an",
        *_encoder_arguments(),
        str(target),
    ]


def _terminate_processes(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 3.0
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _wait_processes(
    processes: list[subprocess.Popen[str]],
    *,
    started: float,
    timeout_seconds: float,
    cancel_check: Callable[[], bool] | None,
) -> None:
    while any(process.poll() is None for process in processes):
        if cancel_check is not None and cancel_check():
            _terminate_processes(processes)
            raise DetailRestoreCancelled("detail restoration cancelled")
        if time.monotonic() - started > timeout_seconds:
            _terminate_processes(processes)
            raise RuntimeError("detail restoration exceeded its bounded timeout")
        time.sleep(0.05)
    failures: list[str] = []
    for process in processes:
        stderr = "" if process.stderr is None else process.stderr.read().strip()
        if process.returncode != 0:
            failures.append(stderr[-500:] if stderr else f"exit {process.returncode}")
    if failures:
        raise RuntimeError("detail restoration failed: " + " | ".join(failures))


def _launch_process(
    command: list[str],
    *,
    stdin: int | None = None,
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        command,
        stdin=stdin,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def _restore_parallel(
    source: Path,
    temporary: Path,
    *,
    token: str,
    expected_frames: int,
    fps: int,
    shards: int,
    started: float,
    timeout_seconds: float,
    cancel_check: Callable[[], bool] | None,
) -> tuple[Path, ...]:
    """Apply the frame-independent filter in parallel temporal shards.

    Each output frame still runs the identical filter graph.  Sharding changes
    only host scheduling and NVENC session parallelism; it never mixes frames.
    The returned paths are scratch artifacts owned by the caller.
    """

    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("ffmpeg is required for intraframe detail restoration")
    chunks = tuple(
        source.with_name(f".{source.stem}.detail-{token}.chunk-{index}.mp4")
        for index in range(shards)
    )
    combined = source.with_name(f".{source.stem}.detail-{token}.video.mp4")
    try:
        processes = [
            _launch_process(
                _ffmpeg_chunk_command(
                    source,
                    target,
                    start_frame=start_frame,
                    frame_count=frame_count,
                    fps=fps,
                )
            )
            for target, (start_frame, frame_count) in zip(
                chunks, _frame_partitions(expected_frames, shards), strict=True
            )
        ]
        _wait_processes(
            processes,
            started=started,
            timeout_seconds=timeout_seconds,
            cancel_check=cancel_check,
        )

        manifest = "".join(f"file 'file:{path.resolve()}'\n" for path in chunks)
        concat = _launch_process(
            [
                executable,
                "-y",
                "-v",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-protocol_whitelist",
                "file,pipe,fd",
                "-i",
                "-",
                "-c",
                "copy",
                str(combined),
            ],
            stdin=subprocess.PIPE,
        )
        if concat.stdin is None:
            raise RuntimeError("failed to open detail restoration concat input")
        concat.stdin.write(manifest)
        concat.stdin.close()
        _wait_processes(
            [concat],
            started=started,
            timeout_seconds=timeout_seconds,
            cancel_check=cancel_check,
        )

        mux = _launch_process(
            [
                executable,
                "-y",
                "-v",
                "error",
                "-i",
                str(combined),
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-map",
                "1:a?",
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(temporary),
            ]
        )
        _wait_processes(
            [mux],
            started=started,
            timeout_seconds=timeout_seconds,
            cancel_check=cancel_check,
        )
        return (*chunks, combined)
    except BaseException:
        for artifact in (*chunks, combined):
            artifact.unlink(missing_ok=True)
        raise


def _probe(path: Path) -> tuple[int, int, int]:
    executable = shutil.which("ffprobe")
    if executable is None:
        raise RuntimeError("ffprobe is required to validate restored video")
    completed = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        stream = json.loads(completed.stdout)["streams"][0]
        return int(stream["width"]), int(stream["height"]), int(stream["nb_frames"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("restored video has no valid primary video stream") from error


def restore_intrame_detail(
    output_path: Path,
    *,
    expected_width: int,
    expected_height: int,
    expected_frames: int,
    cancel_check: Callable[[], bool] | None = None,
    preserve_raw: bool = True,
    timeout_seconds: float = 120.0,
    parallel_shards: int = 1,
    fps: int = 24,
) -> DetailRestoreResult:
    """Restore spatial detail and atomically publish over ``output_path``."""

    source = Path(output_path).resolve()
    if not source.is_file():
        raise RuntimeError("detail restoration source video does not exist")
    token = uuid.uuid4().hex
    temporary = source.with_name(f".{source.stem}.detail-{token}.mp4")
    raw = source.with_name(f"{source.stem}.motion-detail-raw.mp4")
    started = time.monotonic()
    process: subprocess.Popen[str] | None = None
    scratch: tuple[Path, ...] = ()
    try:
        if parallel_shards == 1:
            process = _launch_process(_ffmpeg_command(source, temporary))
            _wait_processes(
                [process],
                started=started,
                timeout_seconds=timeout_seconds,
                cancel_check=cancel_check,
            )
        else:
            scratch = _restore_parallel(
                source,
                temporary,
                token=token,
                expected_frames=expected_frames,
                fps=fps,
                shards=parallel_shards,
                started=started,
                timeout_seconds=timeout_seconds,
                cancel_check=cancel_check,
            )
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError("detail restoration produced no video")
        width, height, frames = _probe(temporary)
        if (width, height, frames) != (
            expected_width,
            expected_height,
            expected_frames,
        ):
            raise RuntimeError(
                "restored video geometry/frame count differs from the H3 output"
            )
        if preserve_raw:
            try:
                os.link(source, raw)
            except OSError:
                shutil.copy2(source, raw)
        os.replace(temporary, source)
        return DetailRestoreResult(
            output_path=source,
            raw_output_path=raw if preserve_raw else source,
            elapsed_seconds=time.monotonic() - started,
            width=width,
            height=height,
            frames=frames,
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        if preserve_raw and source.is_file():
            # A raw link/copy is created only immediately before publication.
            # If publication did not happen, the source itself is authoritative.
            raw.unlink(missing_ok=True)
        raise
    finally:
        for artifact in scratch:
            artifact.unlink(missing_ok=True)


__all__ = [
    "DetailRestoreCancelled",
    "DetailRestoreResult",
    "FILTER_GRAPH",
    "restore_intrame_detail",
]
