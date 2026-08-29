#!/usr/bin/env python3
"""Measure final-latent response to single H3 approximation impulses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch


SERVE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVE_ROOT))

from h3serve.native_engine.planner.mechanistic_control import (  # noqa: E402
    H3MechanisticControlModel,
    H3MechanisticWorkload,
)


SCHEMA_VERSION = "h3_mechanistic_downstream_impulse_response_v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"latent shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
    reference = reference.detach().to(dtype=torch.float64, device="cpu")
    candidate = candidate.detach().to(dtype=torch.float64, device="cpu")
    delta = candidate - reference
    count = reference.numel()
    reference_energy = float(torch.sum(reference * reference).item())
    candidate_energy = float(torch.sum(candidate * candidate).item())
    delta_energy = float(torch.sum(delta * delta).item())
    reference_rms = math.sqrt(reference_energy / max(1, count))
    candidate_rms = math.sqrt(candidate_energy / max(1, count))
    delta_rms = math.sqrt(delta_energy / max(1, count))
    denominator = max(reference_rms, 1.0e-12)
    dot = float(torch.sum(reference * candidate).item())
    cosine = dot / max(math.sqrt(reference_energy * candidate_energy), 1.0e-24)
    return {
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "reference_rms": reference_rms,
        "candidate_rms": candidate_rms,
        "delta_rms": delta_rms,
        "relative_rms": delta_rms / denominator,
        "relative_l1": float(torch.mean(torch.abs(delta)).item()) / denominator,
        "relative_max_abs": float(torch.max(torch.abs(delta)).item()) / denominator,
        "cosine_distance": max(0.0, 1.0 - min(1.0, cosine)),
    }


def _load_latents(path: Path) -> dict[str, Any]:
    try:
        document = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        document = torch.load(path, map_location="cpu")
    if not isinstance(document, dict):
        raise ValueError(f"latent artifact is not a mapping: {path}")
    for modality in ("video", "audio"):
        if not isinstance(document.get(modality), torch.Tensor):
            raise ValueError(f"latent artifact lacks {modality} tensor: {path}")
    return document


def _find_one_latent(root: Path, probe_name: str) -> Path:
    matches = tuple(sorted(root.glob(f"{probe_name}_*.pt")))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one latent for {probe_name}, found {len(matches)}"
        )
    return matches[0]


def _request_profile(report: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    matches = []
    for request in report.get("requests", ()):
        profile = request.get("execution_profile", {})
        summary = profile.get("joint_acceleration")
        if not isinstance(summary, dict):
            summary = profile.get("acceleration_plan_summary")
        if not isinstance(summary, dict):
            summary = request.get("acceleration_plan_summary")
        if isinstance(summary, dict) and summary.get("candidate_id") == candidate_id:
            matches.append(request)
        elif str(request.get("scene", "")).startswith(candidate_id + "_"):
            matches.append(request)
    if len(matches) != 1:
        raise ValueError(
            f"expected one hot-session request for {candidate_id}, found {len(matches)}"
        )
    return matches[0]


def _execution_profile(request: dict[str, Any]) -> dict[str, Any]:
    profile = request.get("execution_profile")
    if not isinstance(profile, dict):
        raise ValueError("hot-session request lacks execution_profile")
    return profile


def _fit_log_gain(
    rows: list[dict[str, Any]],
    *,
    probe_type: str,
    modality: str,
    degree: int,
) -> dict[str, Any]:
    selected = [row for row in rows if row["probe_type"] == probe_type]
    if len(selected) <= degree:
        raise ValueError(f"insufficient {probe_type} probes for degree-{degree} fit")
    progress = torch.tensor(
        [float(row["normalized_phase"]) for row in selected],
        dtype=torch.float64,
    )
    gains = torch.tensor(
        [float(row["modalities"][modality]["downstream_gain"]) for row in selected],
        dtype=torch.float64,
    )
    if torch.any(gains <= 0.0):
        raise ValueError("downstream gains must be positive for log fitting")
    design = torch.stack(
        [progress ** power for power in range(degree + 1)], dim=1
    )
    coefficients = torch.linalg.lstsq(design, torch.log(gains)).solution
    prediction = design @ coefficients
    residual = torch.log(gains) - prediction
    centered = torch.log(gains) - torch.mean(torch.log(gains))
    residual_energy = float(torch.sum(residual * residual).item())
    total_energy = float(torch.sum(centered * centered).item())
    grid = torch.linspace(0.0, 1.0, 1001, dtype=torch.float64)
    if degree == 1:
        derivative = torch.full_like(grid, coefficients[1])
    else:
        derivative = coefficients[1] + 2.0 * coefficients[2] * grid
    return {
        "theoretical_form": (
            "log G(p) is polynomial because Gronwall propagation gives "
            "G(p)=exp(integral_p^1 L(s) ds), with a low-order phase model "
            "for the local trajectory Lipschitz rate L"
        ),
        "probe_type": probe_type,
        "modality": modality,
        "degree": degree,
        "feature_order": [f"p^{power}" for power in range(degree + 1)],
        "coefficients": [float(value) for value in coefficients],
        "sample_count": len(selected),
        "phase_steps": [int(row["phase_step"]) for row in selected],
        "log_residuals": [float(value) for value in residual],
        "absolute_log_residual_max": float(torch.max(torch.abs(residual)).item()),
        "log_rmse": math.sqrt(residual_energy / len(selected)),
        "r_squared": (
            1.0 if total_energy <= 1.0e-24 else 1.0 - residual_energy / total_energy
        ),
        "monotone_nonincreasing_on_unit_interval": bool(
            torch.all(derivative <= 1.0e-12).item()
        ),
        "maximum_log_derivative": float(torch.max(derivative).item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-set", type=Path, required=True)
    parser.add_argument("--latents-dir", type=Path, required=True)
    parser.add_argument("--hot-session-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    probe_path = args.probe_set.resolve()
    latent_root = args.latents_dir.resolve()
    report_path = args.hot_session_report.resolve()
    probe_set = json.loads(probe_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if probe_set.get("schema_version") != "h3_mechanistic_impulse_probe_set_v1":
        raise ValueError("unsupported impulse probe-set schema")
    probes = probe_set.get("probes")
    if not isinstance(probes, list) or not probes:
        raise ValueError("probe set is empty")
    dense_rows = [row for row in probes if row.get("probe_type") == "dense_control"]
    if len(dense_rows) != 1:
        raise ValueError("probe set requires one Dense control")
    dense_row = dense_rows[0]
    dense_name = str(dense_row["name"])
    dense_path = _find_one_latent(latent_root, dense_name)
    dense = _load_latents(dense_path)
    dense_request = _request_profile(report, dense_name)
    dense_profile = _execution_profile(dense_request)
    packed_tokens = int(dense_profile["packed_tokens"])
    spatial_tokens = int(dense_profile["spatial_tokens"])
    latent_frames = int(dense_profile["latent_frames"])
    condition_count = int(dense_profile.get("condition_count", 0))
    total_steps = int(probe_set["total_steps"])
    workload = H3MechanisticWorkload(
        total_steps=total_steps,
        packed_tokens=packed_tokens,
        video_tokens=spatial_tokens * latent_frames,
        condition_count=condition_count,
        service_family=(
            "reference"
            if str(dense.get("engine", "")) in ("reference", "reference_lora")
            else "first_last"
        ),
    )
    model = H3MechanisticControlModel(workload)
    rows: list[dict[str, Any]] = []
    metadata_keys = ("frames", "fps", "width", "height", "engine", "seed")
    for probe in probes:
        name = str(probe["name"])
        latent_path = _find_one_latent(latent_root, name)
        latents = _load_latents(latent_path)
        mismatch = {
            key: [dense.get(key), latents.get(key)]
            for key in metadata_keys
            if dense.get(key) != latents.get(key)
        }
        if mismatch:
            raise ValueError(f"counterfactual metadata mismatch for {name}: {mismatch}")
        request = _request_profile(report, name)
        profile = _execution_profile(request)
        if (
            int(profile["packed_tokens"]) != packed_tokens
            or int(profile["spatial_tokens"]) != spatial_tokens
            or int(profile["latent_frames"]) != latent_frames
        ):
            raise ValueError(f"token layout changed inside counterfactual {name}")
        phase_step = probe.get("phase_step")
        local_scale: dict[str, float | None] = {"audio": None, "video": None}
        if probe["probe_type"] == "single_forecast":
            step = int(phase_step)
            weight = model._integration_weights()[step]
            for modality in ("audio", "video"):
                response = model._forecast_response(
                    modality=modality,
                    step=step,
                    horizon=1,
                )
                local_scale[modality] = weight * response.upper
        elif probe["probe_type"] == "single_attention":
            step = int(phase_step)
            start, stop = (int(value) for value in probe_set["attention_layers"])
            action = str(probe_set["attention_action"])
            norm = math.sqrt(sum(
                model._attention_error(step=step, layer=layer, action=action).upper ** 2
                for layer in range(start, stop)
            ))
            local_scale = {"audio": norm, "video": norm}

        modalities: dict[str, Any] = {}
        for modality in ("audio", "video"):
            metrics = _tensor_metrics(dense[modality], latents[modality])
            scale = local_scale[modality]
            metrics["declared_local_ucb_amplitude"] = scale
            metrics["downstream_gain"] = (
                None
                if scale is None or scale <= 0.0
                else metrics["relative_rms"] / scale
            )
            modalities[modality] = metrics
        rows.append({
            **probe,
            "latent_path": str(latent_path),
            "latent_sha256": _file_sha256(latent_path),
            "denoise_seconds": float(request["phases"]["denoise"]),
            "wall_seconds": float(request["total_seconds"]),
            "actual_evaluations": int(profile["actual_evaluations"]),
            "forecast_evaluations": int(profile["forecast_evaluations"]),
            "modalities": modalities,
        })

    fits = {
        "single_forecast": {
            modality: _fit_log_gain(
                rows,
                probe_type="single_forecast",
                modality=modality,
                degree=2,
            )
            for modality in ("audio", "video")
        },
        "single_attention": {
            modality: _fit_log_gain(
                rows,
                probe_type="single_attention",
                modality=modality,
                degree=1,
            )
            for modality in ("audio", "video")
        },
    }
    output = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": (
            "phase-conditioned final-latent response to isolated approximations; "
            "not a Human quality threshold and not a release schedule"
        ),
        "identification_contract": probe_set["identification_contract"],
        "workload": {
            "total_steps": workload.total_steps,
            "packed_tokens": workload.packed_tokens,
            "video_tokens": workload.video_tokens,
            "condition_count": workload.condition_count,
            "frames": dense["frames"],
            "fps": dense["fps"],
            "width": dense["width"],
            "height": dense["height"],
            "engine": dense["engine"],
            "seed": dense["seed"],
        },
        "mechanism_model_digest": model.model_digest,
        "provenance": {
            "probe_set_path": str(probe_path),
            "probe_set_sha256": _file_sha256(probe_path),
            "hot_session_report_path": str(report_path),
            "hot_session_report_sha256": _file_sha256(report_path),
            "dense_latent_path": str(dense_path),
            "dense_latent_sha256": _file_sha256(dense_path),
        },
        "continuous_phase_gain_fits": fits,
        "fit_limitations": [
            "one 720p5 prompt/seed identifies phase shape but not production UCB",
            "the Forecast local denominator comes from the separate secant-tail model",
            "the Attention local denominator comes from Round218 local output probes",
            "a long-video and held-out-seed replication is required before admission",
        ],
        "responses": rows,
    }
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(target),
        "response_count": len(rows),
        "forecast_gains": [
            {
                "step": row["phase_step"],
                "audio": row["modalities"]["audio"]["downstream_gain"],
                "video": row["modalities"]["video"]["downstream_gain"],
            }
            for row in rows
            if row["probe_type"] == "single_forecast"
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
