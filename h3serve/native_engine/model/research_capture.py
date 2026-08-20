"""Opt-in capture of one real H3 MLP boundary for offline research.

The release runtime never enables this path.  A researcher can set a target
path, denoise step and transformer layer through environment variables.  The
capture contains only generated-video tensors at the MLP boundary; text,
audio and reference conditioning rows are deliberately excluded.
"""

from __future__ import annotations

import os
import atexit
import json
from pathlib import Path

import torch


_QUANTIZED_CHUNKS: dict[Path, list[tuple[torch.Tensor, torch.Tensor]]] = {}
_QUANTIZED_WEIGHTS: dict[Path, tuple[torch.Tensor, torch.Tensor]] = {}
_GATE_ROWS: list[dict[str, object]] = []
_GATE_CAPTURE_REGISTERED = False


class ResearchCaptureComplete(RuntimeError):
    """Raised after a requested capture when early-stop is enabled."""


def _write_gate_capture() -> None:
    raw_path = os.environ.get("H3_NATIVE_CAPTURE_MLP_GATES_PATH", "").strip()
    if not raw_path or not _GATE_ROWS:
        return
    path = Path(raw_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": "h3_adaln_video_mlp_gate_trajectory",
                "records": _GATE_ROWS,
            },
            indent=2,
        )
        + "\n"
    )
    temporary.replace(path)


def record_video_mlp_gate(
    gate: torch.Tensor,
    *,
    row: int,
    layer: int | None,
) -> None:
    """Record one tiny AdaLN gate row in an explicitly enabled research run."""

    raw_path = os.environ.get("H3_NATIVE_CAPTURE_MLP_GATES_PATH", "").strip()
    if not raw_path or layer is None:
        return
    from .kernels import current_attention_step

    step = current_attention_step()
    if step is None:
        return
    global _GATE_CAPTURE_REGISTERED
    if not _GATE_CAPTURE_REGISTERED:
        atexit.register(_write_gate_capture)
        _GATE_CAPTURE_REGISTERED = True
    values = gate[int(row)].detach().float()
    _GATE_ROWS.append(
        {
            "step_index": int(step[0]),
            "step_count": int(step[1]),
            "layer": int(layer),
            "row": int(row),
            "rms": float(values.square().mean().sqrt()),
            "mean_absolute": float(values.abs().mean()),
            "maximum_absolute": float(values.abs().max()),
        }
    )


def capture_target(layer: int | None) -> tuple[Path, int, int] | None:
    raw_path = os.environ.get("H3_NATIVE_CAPTURE_MLP_PATH", "").strip()
    if not raw_path or layer is None:
        return None
    from .kernels import current_attention_step

    step = current_attention_step()
    if step is None:
        return None
    target_step = int(os.environ.get("H3_NATIVE_CAPTURE_MLP_STEP", "0"))
    target_layer = int(os.environ.get("H3_NATIVE_CAPTURE_MLP_LAYER", "20"))
    path = Path(raw_path).expanduser().resolve()
    if path.exists() or step[0] != target_step or layer != target_layer:
        return None
    return path, int(step[0]), int(step[1])


def capture_kind() -> str:
    kind = os.environ.get("H3_NATIVE_CAPTURE_MLP_KIND", "delta").strip().lower()
    if kind not in ("delta", "quantized_fc2"):
        raise ValueError("H3_NATIVE_CAPTURE_MLP_KIND must be delta or quantized_fc2")
    return kind


def capture_quantized_fc2_chunk(
    target: tuple[Path, int, int],
    *,
    qx: torch.Tensor,
    x_scale: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    chunk_start: int,
    chunk_stop: int,
    protected_tokens: int,
) -> None:
    """Collect only generated-video FC2 operands for an offline oracle.

    This deliberately duplicates the fused SwiGLU/ConvRot quantizer only in a
    research capture process.  Production execution never calls this helper.
    """

    if capture_kind() != "quantized_fc2":
        return
    path = target[0]
    video_start = max(int(chunk_start), int(protected_tokens))
    if video_start >= chunk_stop:
        return
    local_start = video_start - int(chunk_start)
    _QUANTIZED_CHUNKS.setdefault(path, []).append(
        (
            qx[local_start:].detach().to("cpu", non_blocking=False),
            x_scale[local_start:].detach().to("cpu", non_blocking=False),
        )
    )
    if path not in _QUANTIZED_WEIGHTS:
        _QUANTIZED_WEIGHTS[path] = (
            qweight.detach().to("cpu", non_blocking=False),
            weight_scale.detach().to("cpu", dtype=torch.float32, non_blocking=False),
        )


def persist_mlp_capture(
    target: tuple[Path, int, int],
    *,
    hidden_video: torch.Tensor | None,
    delta_video: torch.Tensor | None,
    protected_tokens: int,
) -> None:
    from .kernels import current_attention_video_layout

    path, step_index, step_count = target
    layout = current_attention_video_layout()
    if layout is None:
        raise RuntimeError("MLP capture requires the generated-video layout")
    latent_frames, frame_tokens = layout
    expected = latent_frames * frame_tokens
    kind = capture_kind()
    document = {
        "schema_version": 1,
        "kind": kind,
        "step_index": step_index,
        "step_count": step_count,
        "layer": int(os.environ.get("H3_NATIVE_CAPTURE_MLP_LAYER", "20")),
        "protected_tokens": int(protected_tokens),
        "latent_frames": int(latent_frames),
        "frame_tokens": int(frame_tokens),
    }
    if kind == "delta":
        if hidden_video is None or delta_video is None:
            raise RuntimeError("delta MLP capture requires hidden and residual tensors")
        if hidden_video.shape[0] != expected or delta_video.shape != hidden_video.shape:
            raise RuntimeError("captured MLP tensors do not match the video layout")
        document["hidden_video"] = hidden_video.detach().to(
            "cpu", non_blocking=False
        )
        document["delta_video"] = delta_video.detach().to(
            "cpu", non_blocking=False
        )
    else:
        chunks = _QUANTIZED_CHUNKS.pop(path, [])
        weights = _QUANTIZED_WEIGHTS.pop(path, None)
        if not chunks or weights is None:
            raise RuntimeError("quantized FC2 capture did not observe any video chunks")
        qx = torch.cat(tuple(chunk[0] for chunk in chunks), dim=0)
        x_scale = torch.cat(tuple(chunk[1] for chunk in chunks), dim=0)
        if qx.shape[0] != expected or x_scale.shape[0] != expected:
            raise RuntimeError("quantized FC2 capture does not match the video layout")
        document.update(
            {
                "qx_video": qx,
                "x_scale_video": x_scale,
                "qweight": weights[0],
                "weight_scale": weights[1],
                "convrot_group_size": 256,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(document, temporary)
    temporary.replace(path)
    if os.environ.get("H3_NATIVE_CAPTURE_MLP_STOP", "1") == "1":
        raise ResearchCaptureComplete(f"MLP research capture written to {path}")


__all__ = [
    "ResearchCaptureComplete",
    "capture_kind",
    "capture_quantized_fc2_chunk",
    "capture_target",
    "persist_mlp_capture",
    "record_video_mlp_gate",
]
