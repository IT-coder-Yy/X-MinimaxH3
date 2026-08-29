#!/usr/bin/env python3
"""Deterministic, isolated FasterVQA scoring for generated videos.

FasterVQA predicts general perceptual video quality.  It is useful for ranked
A/B comparisons, but it is not a pure blur detector and does not replace human
review of motion causality, dialogue, identity, or instruction following.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import random
import sys
import time
import types
from pathlib import Path
from typing import Any


SOURCE_COMMIT = "8db452e2caa5d5d4da507bcf577c19b8114f2ebd"
CHECKPOINT_SHA256 = "8c3108647653fd48e31f3bebbe03a344d624c806d3f1af9478a4e9f5aa3038ab"
CHECKPOINT_SIZE = 127343543
MODEL_MEAN = 0.14759505
MODEL_STD = 0.03613452


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_paths() -> tuple[Path, Path, Path]:
    release_root = Path(__file__).resolve().parents[1]
    runtime_root = Path(
        __import__("os").environ.get(
            "H3_FAST_VQA_ROOT", release_root / "runtime" / "quality" / "fastvqa"
        )
    ).expanduser().resolve()
    return (
        runtime_root / "source",
        runtime_root / "python",
        runtime_root / "FAST_VQA_3D_1_1.pth",
    )


def _prepare_imports(source_root: Path, python_root: Path) -> Any:
    sys.path.insert(0, str(python_root))

    # Import only the inference modules.  Upstream package __init__ files also
    # import training-only cv2, skvideo and OpenAI CLIP dependencies.
    fastvqa_pkg = types.ModuleType("fastvqa")
    fastvqa_pkg.__path__ = [str(source_root / "fastvqa")]
    sys.modules["fastvqa"] = fastvqa_pkg
    models_pkg = types.ModuleType("fastvqa.models")
    models_pkg.__path__ = [str(source_root / "fastvqa" / "models")]
    sys.modules["fastvqa.models"] = models_pkg

    # evaluator.py imports the optional X-CLIP backbone at module load.  The
    # FasterVQA route below never constructs it, so an empty compatibility
    # module keeps that unused dependency out of the service environment.
    sys.modules.setdefault("clip", types.ModuleType("clip"))

    import decord  # type: ignore
    import numpy as np
    import torch
    from fastvqa.models.evaluator import DiViDeAddEvaluator

    return decord, np, torch, DiViDeAddEvaluator


def _validate_install(source_root: Path, checkpoint: Path) -> None:
    required = (
        source_root / "fastvqa" / "models" / "evaluator.py",
        source_root / "fastvqa" / "models" / "swin_backbone.py",
    )
    for path in required:
        if not path.is_file():
            raise RuntimeError(
                f"FasterVQA source is missing: {path}. Run scripts/setup_fast_vqa.sh."
            )
    if not checkpoint.is_file():
        raise RuntimeError(
            f"FasterVQA checkpoint is missing: {checkpoint}. Run scripts/setup_fast_vqa.sh."
        )
    if checkpoint.stat().st_size != CHECKPOINT_SIZE or _sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("FasterVQA checkpoint failed size or SHA-256 verification.")


def _seed_all(seed: int, np: Any, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_frame_indices(total_frames: int, np: Any) -> Any:
    """Match upstream FragmentSampleFrames for the FasterVQA checkpoint."""
    fragments_t = 8
    fragment_size_t = 4
    # Although f3dvqa-b.yml contains frame_interval: 2, the official vqa.py
    # does not pass it to FragmentSampleFrames. Its inference default is 1.
    frame_interval = 1
    temporal_grids = np.array(
        [total_frames // fragments_t * index for index in range(fragments_t)],
        dtype=np.int32,
    )
    temporal_length = total_frames // fragments_t
    needed = fragment_size_t * frame_interval
    if temporal_length > needed:
        offsets = np.random.randint(0, temporal_length - needed, size=len(temporal_grids))
    else:
        offsets = np.zeros(len(temporal_grids), dtype=np.int32)
    ranges = (
        np.arange(fragment_size_t)[None, :] * frame_interval
        + offsets[:, None]
        + temporal_grids[:, None]
    )
    return np.mod(np.concatenate(ranges), total_frames).astype(np.int32)


def _spatial_fragments(video: Any, torch: Any) -> Any:
    """Match upstream 7x7 fragment sampling without training-only imports."""
    fragments_h = fragments_w = 7
    fragment_h = fragment_w = 32
    aligned = 8
    target_h = fragments_h * fragment_h
    target_w = fragments_w * fragment_w

    duration, source_h, source_w = video.shape[-3:]
    ratio = min(source_h / target_h, source_w / target_w)
    if ratio < 1:
        original = video
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=1 / ratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(original)
        source_h, source_w = video.shape[-2:]

    if duration % aligned:
        raise RuntimeError(f"FasterVQA sampled duration {duration} is not aligned to {aligned}.")

    h_grids = torch.tensor(
        [min(source_h // fragments_h * i, source_h - fragment_h) for i in range(fragments_h)],
        dtype=torch.long,
    )
    w_grids = torch.tensor(
        [min(source_w // fragments_w * i, source_w - fragment_w) for i in range(fragments_w)],
        dtype=torch.long,
    )
    h_length = source_h // fragments_h
    w_length = source_w // fragments_w
    time_groups = duration // aligned
    if h_length > fragment_h:
        random_h = torch.randint(
            h_length - fragment_h, (fragments_h, fragments_w, time_groups)
        )
    else:
        random_h = torch.zeros((fragments_h, fragments_w, time_groups), dtype=torch.int64)
    if w_length > fragment_w:
        random_w = torch.randint(
            w_length - fragment_w, (fragments_h, fragments_w, time_groups)
        )
    else:
        random_w = torch.zeros((fragments_h, fragments_w, time_groups), dtype=torch.int64)

    target = torch.zeros(video.shape[:-2] + (target_h, target_w), dtype=video.dtype)
    for row, h_grid in enumerate(h_grids):
        for column, w_grid in enumerate(w_grids):
            for group in range(time_groups):
                target_t = slice(group * aligned, (group + 1) * aligned)
                target_h_slice = slice(row * fragment_h, (row + 1) * fragment_h)
                target_w_slice = slice(column * fragment_w, (column + 1) * fragment_w)
                source_h_start = int(h_grid + random_h[row, column, group])
                source_w_start = int(w_grid + random_w[row, column, group])
                target[:, target_t, target_h_slice, target_w_slice] = video[
                    :,
                    target_t,
                    source_h_start : source_h_start + fragment_h,
                    source_w_start : source_w_start + fragment_w,
                ]
    return target


def _load_model(checkpoint: Path, device: str, torch: Any, evaluator_type: Any) -> Any:
    # Upstream constructors print architecture diagnostics to stdout. Keep our
    # stdout machine-readable JSON while retaining the unchanged model graph.
    with contextlib.redirect_stdout(io.StringIO()):
        model = evaluator_type(
            backbone={"fragments": {"checkpoint": False, "pretrained": None}},
            backbone_size="swin_tiny_grpb",
            backbone_preserve_keys="fragments",
            divide_head=False,
            vqa_head={"in_channels": 768, "hidden_channels": 64},
        ).to(device)
    # The file is the trusted, hash-pinned official release checkpoint.  An
    # explicit weights_only=False preserves compatibility with its legacy dict.
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def _score_video(
    path: Path,
    *,
    seed: int,
    device: str,
    model: Any,
    decord: Any,
    np: Any,
    torch: Any,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    _seed_all(seed, np, torch)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
    total_frames = len(reader)
    if total_frames == 0:
        raise RuntimeError(f"Video has no decodable frames: {path}")
    indices = _sample_frame_indices(total_frames, np)
    frames = reader.get_batch(indices.tolist()).asnumpy()
    source_h, source_w = int(frames.shape[1]), int(frames.shape[2])
    sampled = torch.from_numpy(frames).permute(3, 0, 1, 2)
    sampled = _spatial_fragments(sampled, torch)
    mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)
    std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32)
    sampled = ((sampled.permute(1, 2, 3, 0).float() - mean) / std).permute(3, 0, 1, 2)
    sampled = sampled.reshape(sampled.shape[0], 1, -1, *sampled.shape[2:]).transpose(0, 1)

    with torch.inference_mode():
        # Upstream Swin code has one unconditional shape print in its forward.
        # Suppress it so stdout remains valid JSON for automation callers.
        with contextlib.redirect_stdout(io.StringIO()):
            raw = float(model({"fragments": sampled.to(device)}).mean().item())
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_allocated = peak_reserved = 0
    elapsed = time.perf_counter() - started
    quality = 1.0 / (1.0 + math.exp(-((raw - MODEL_MEAN) / MODEL_STD)))

    try:
        fps = float(reader.get_avg_fps())
    except Exception:
        fps = 0.0
    return {
        "path": str(path.resolve()),
        "source_frames": total_frames,
        "sampled_frames": int(len(indices)),
        "width": source_w,
        "height": source_h,
        "fps": fps,
        "raw_score": raw,
        "quality_score": quality,
        "elapsed_seconds": elapsed,
        "peak_gpu_allocated_bytes": peak_allocated,
        "peak_gpu_reserved_bytes": peak_reserved,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score one or more videos with the official FasterVQA checkpoint."
    )
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    source_root, python_root, checkpoint = _runtime_paths()
    _validate_install(source_root, checkpoint)
    decord, np, torch, evaluator_type = _prepare_imports(source_root, python_root)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    load_started = time.perf_counter()
    model = _load_model(checkpoint, device, torch, evaluator_type)
    if device == "cuda":
        torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started

    results = [
        _score_video(
            path,
            seed=args.seed,
            device=device,
            model=model,
            decord=decord,
            np=np,
            torch=torch,
        )
        for path in args.videos
    ]
    document = {
        "schema_version": 1,
        "metric": "FasterVQA",
        "interpretation": "Higher is predicted better general perceptual video quality; compare relatively under the same scorer and seed.",
        "source_commit": SOURCE_COMMIT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "device": device,
        "seed": args.seed,
        "model_load_seconds": model_load_seconds,
        "results": results,
    }
    encoded = json.dumps(document, ensure_ascii=False, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
