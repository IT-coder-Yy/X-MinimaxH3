"""Research-only AdaLN-gated video MLP budget.

Unlike spatial token interpolation, this route never copies or blends one
token into another.  At explicitly selected solver-step/layer pairs it keeps
the complete Attention path and every conditioning/audio MLP row, while the
generated-video rows receive an identity MLP residual.  The selection is
derived from H3's own timestep-conditioned AdaLN gate trajectory.

The feature is default-off and is not part of the published planner until a
complete Human video review accepts it.
"""

from __future__ import annotations

import atexit
import json
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path

import torch


_ADAPTIVE_DECISIONS: Counter[tuple[int, int]] = Counter()
_REPORT_REGISTERED = False


def _integers(name: str) -> frozenset[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return frozenset()
    try:
        values = frozenset(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated integer list") from error
    if any(value < 0 for value in values):
        raise ValueError(f"{name} cannot contain negative values")
    return values


@lru_cache(maxsize=1)
def configured_budget() -> tuple[frozenset[int], frozenset[int]]:
    layers = _integers("H3_NATIVE_EXPERIMENTAL_VIDEO_MLP_SKIP_LAYERS")
    steps = _integers("H3_NATIVE_EXPERIMENTAL_VIDEO_MLP_SKIP_STEPS")
    if bool(layers) != bool(steps):
        raise ValueError("video MLP skip requires both layer and step lists")
    return layers, steps


@lru_cache(maxsize=1)
def adaptive_gate_config() -> tuple[float | None, int, int]:
    """Return the model-native gate threshold and protected solver margins.

    The threshold is deliberately opt-in.  A positive value enables a
    request-shape-independent policy: retain complete MLPs near both ends of
    the sigma trajectory and omit generated-video MLP residuals only when
    H3's own timestep-conditioned gate has sufficiently low RMS energy.
    """

    raw = os.environ.get(
        "H3_NATIVE_EXPERIMENTAL_VIDEO_MLP_GATE_MAX_RMS", ""
    ).strip()
    if not raw:
        return None, 1, 3
    threshold = float(raw)
    head = int(os.environ.get("H3_NATIVE_EXPERIMENTAL_VIDEO_MLP_GATE_HEAD_STEPS", "1"))
    tail = int(os.environ.get("H3_NATIVE_EXPERIMENTAL_VIDEO_MLP_GATE_TAIL_STEPS", "3"))
    if threshold <= 0.0:
        raise ValueError("video MLP gate threshold must be positive")
    if head < 0 or tail < 0:
        raise ValueError("video MLP gate protected step counts cannot be negative")
    return threshold, head, tail


def _write_report() -> None:
    raw = os.environ.get("H3_NATIVE_EXPERIMENTAL_VIDEO_MLP_GATE_REPORT", "").strip()
    if not raw or not _ADAPTIVE_DECISIONS:
        return
    path = Path(raw).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"step_index": step, "layer": layer, "count": count}
        for (step, layer), count in sorted(_ADAPTIVE_DECISIONS.items())
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": "h3_adaln_gate_budget",
                "skipped_evaluations": sum(_ADAPTIVE_DECISIONS.values()),
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    temporary.replace(path)


def skip_video_mlp(
    *,
    layer: int | None,
    step: int | None,
    step_count: int | None = None,
    gate: torch.Tensor | None = None,
    row: int | None = None,
) -> bool:
    if layer is None or step is None:
        return False
    layers, steps = configured_budget()
    if layers:
        return int(layer) in layers and int(step) in steps

    threshold, protected_head, protected_tail = adaptive_gate_config()
    if threshold is None or step_count is None or gate is None or row is None:
        return False
    if step < protected_head or step >= max(protected_head, step_count - protected_tail):
        return False

    # The video modulation is one AdaLN row, so this reduction is tiny.  It
    # intentionally observes the model's current sigma-conditioned gate
    # instead of routing by a hand-authored layer list or spatial proxy.
    gate_rms = float(gate[int(row)].detach().float().square().mean().sqrt())
    if gate_rms > threshold:
        return False

    global _REPORT_REGISTERED
    _ADAPTIVE_DECISIONS[(int(step), int(layer))] += 1
    if not _REPORT_REGISTERED:
        atexit.register(_write_report)
        _REPORT_REGISTERED = True
    return True


__all__ = ["adaptive_gate_config", "configured_budget", "skip_video_mlp"]
