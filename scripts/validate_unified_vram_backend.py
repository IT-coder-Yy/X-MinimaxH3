#!/usr/bin/env python3
"""Validate the three isolated resource-backend envelopes cheaply.

The gate deliberately models the release smoke trajectory (five solver steps,
acceleration 95, one modeled actual DiT evaluation).  The production scheduler
can retain additional opening/terminal anchors; evaluation count changes total
latency, not the packed working set.  Both clean FL2VA and a 15-item static
Ref2VA-style context are required; a much larger reference-video context is
reported separately as a diagnostic rather than hidden inside the claim.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from h3serve.contract import ASPECT_RATIOS, resolve_frames, resolve_geometry
from h3serve.native_engine.planner import (
    H3WorkloadAnalyzer,
    select_memory_execution,
)
from h3serve.native_engine.ultimate_upscale import plan_ultimate_upscale


GIB = 1024**3


@dataclass(frozen=True, slots=True)
class ProductProfile:
    name: str
    weight_tier: str
    provisioned_gib: float
    planner_budget_gib: float
    first_generation: tuple[str, ...]
    second_sampling: tuple[str, ...]


PROFILES = (
    ProductProfile(
        name="int8_16gb",
        weight_tier="int8",
        provisioned_gib=16.0,
        planner_budget_gib=15.25,
        first_generation=("360p", "480p", "720p", "1080p"),
        second_sampling=("720p", "1080p"),
    ),
    ProductProfile(
        name="int8_24gb",
        weight_tier="int8",
        provisioned_gib=24.0,
        planner_budget_gib=23.25,
        first_generation=("360p", "480p", "720p", "1080p"),
        second_sampling=("720p", "1080p", "2k"),
    ),
    ProductProfile(
        name="w4a8_8gb",
        weight_tier="w4a8",
        provisioned_gib=8.0,
        planner_budget_gib=7.25,
        first_generation=("360p", "480p", "720p"),
        second_sampling=("720p", "1080p"),
    ),
)

# Images and standalone audio are the INT8 context stress case.  12k encoded
# condition tokens is deliberately above the common 9x720p-image + short-audio
# pack.  At the 8-GiB 720p15 boundary, the physically gated W4 contract is one
# reference item; the public runtime still measures exact packed tokens and
# fails closed for larger packs rather than borrowing another backend's budget.
INT8_REQUIRED_CONTEXTS = (
    ("fl2va_clean", "original", 0, None),
    ("ref2va_static_15_items", "reference", 15, 12_000),
)
W4_REQUIRED_CONTEXTS = (
    ("fl2va_clean", "original", 0, None),
    ("ref2va_single_item", "reference", 1, None),
)
COMMON_DIAGNOSTIC_CONTEXTS = (
    ("ref2va_video_heavy", "reference", 15, 50_000),
)
W4_DIAGNOSTIC_CONTEXTS = (
    ("ref2va_static_15_items", "reference", 15, 12_000),
    *COMMON_DIAGNOSTIC_CONTEXTS,
)
DURATIONS = (1.0, 5.0, 10.0, 15.0)


def _case(
    analyzer: H3WorkloadAnalyzer,
    *,
    profile: ProductProfile,
    workload: str,
    resolution: str,
    aspect_ratio: str,
    duration_seconds: float,
    context: tuple[str, str, int, int | None],
    required: bool,
) -> dict[str, Any]:
    width, height = resolve_geometry(resolution, aspect_ratio)
    frames, actual_duration = resolve_frames(duration_seconds)
    context_name, engine, condition_count, condition_tokens = context
    features = analyzer.analyze(
        width=width,
        height=height,
        frames=frames,
        text_tokens=1024,
        condition_count=condition_count,
        condition_tokens_override=condition_tokens,
        engine=engine,
        actual_evaluations=1,
        forecast_evaluations=4 if workload == "first_generation" else 0,
    )
    if workload == "second_sampling":
        upscale = plan_ultimate_upscale(
            target_width=width,
            target_height=height,
            frames=frames,
            device_budget_bytes=round(profile.planner_budget_gib * GIB),
            text_tokens=1024,
            condition_count=condition_count,
            engine=engine,
            actual_evaluations=1,
            requested_mode="auto",
            resource_profile=profile.name,
            weight_tier=profile.weight_tier,
            allow_spatial_tiles=False,
        )
        memory = upscale.memory_execution
        temporal_pieces = len(upscale.temporal)
        redundancy_ratio = upscale.redundancy_ratio
    else:
        decision = select_memory_execution(
            features,
            requested_mode="auto",
            device_budget_bytes=round(profile.planner_budget_gib * GIB),
            resource_profile=profile.name,
            weight_tier=profile.weight_tier,
        )
        memory = decision.telemetry()
        temporal_pieces = 1
        redundancy_ratio = 1.0
    return {
        "profile": profile.name,
        "workload": workload,
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
        "requested_duration_seconds": duration_seconds,
        "actual_duration_seconds": actual_duration,
        "width": width,
        "height": height,
        "frames": frames,
        "context": context_name,
        "condition_count": condition_count,
        "condition_tokens": features.condition_tokens,
        "packed_tokens": features.packed_tokens,
        "required": required,
        "fits_budget": bool(memory["fits_budget"]),
        "selected_scheme": memory["selected_scheme"],
        "query_chunk_tokens": memory["query_chunk_tokens"],
        "compact_kv": memory["compact_kv"],
        "block_buffer_count": memory["block_buffer_count"],
        "mlp_chunk_tokens": memory["mlp_chunk_tokens"],
        "vae_spatial_tile": memory["vae_spatial_tile"],
        "vae_temporal_tile": memory["vae_temporal_tile"],
        "vae_output_strategy": memory["vae_output_strategy"],
        "temporal_pieces": temporal_pieces,
        "redundancy_ratio": redundancy_ratio,
        "predicted_dit_peak_gib": round(memory["estimated_dit_peak_gib"], 4),
        "predicted_vae_peak_gib": round(memory["estimated_vae_selected_peak_gib"], 4),
        "predicted_peak_gib": round(memory["estimated_selected_peak_gib"], 4),
        "budget_gib": profile.planner_budget_gib,
    }


def build_report() -> dict[str, Any]:
    analyzer = H3WorkloadAnalyzer()
    rows: list[dict[str, Any]] = []
    for profile in PROFILES:
        required_contexts = (
            W4_REQUIRED_CONTEXTS
            if profile.weight_tier == "w4a8"
            else INT8_REQUIRED_CONTEXTS
        )
        diagnostic_contexts = (
            W4_DIAGNOSTIC_CONTEXTS
            if profile.weight_tier == "w4a8"
            else COMMON_DIAGNOSTIC_CONTEXTS
        )
        for workload, resolutions in (
            ("first_generation", profile.first_generation),
            ("second_sampling", profile.second_sampling),
        ):
            for resolution in resolutions:
                for aspect_ratio in ASPECT_RATIOS:
                    for duration in DURATIONS:
                        for context in required_contexts:
                            rows.append(_case(
                                analyzer,
                                profile=profile,
                                workload=workload,
                                resolution=resolution,
                                aspect_ratio=aspect_ratio,
                                duration_seconds=duration,
                                context=context,
                                required=True,
                            ))
                        # The expensive reference-video diagnostic matters only
                        # at the maximum duration where it can alter admission.
                        if duration == max(DURATIONS):
                            for context in diagnostic_contexts:
                                rows.append(_case(
                                    analyzer,
                                    profile=profile,
                                    workload=workload,
                                    resolution=resolution,
                                    aspect_ratio=aspect_ratio,
                                    duration_seconds=duration,
                                    context=context,
                                    required=False,
                                ))
    required = [row for row in rows if row["required"]]
    diagnostics = [row for row in rows if not row["required"]]
    return {
        "schema_version": "h3_isolated_resource_backend_envelope_v4",
        "execution_policy": "minimum_predicted_latency_under_vram_budget",
        "weight_tiers": ["int8_convrot", "grouped_w4a8_convrot"],
        "physical_smoke_contract": {
            "sampling_steps": 5,
            "acceleration": 95,
            "modeled_actual_dit_evaluations": 1,
            "physical_smoke_actual_dit_evaluations": 3,
            "reason": "capacity is shape/context dominated; keep validation fast",
        },
        "profiles": [asdict(profile) for profile in PROFILES],
        "summary": {
            "required_cases": len(required),
            "required_passed": sum(row["fits_budget"] for row in required),
            "required_failed": sum(not row["fits_budget"] for row in required),
            "diagnostic_cases": len(diagnostics),
            "diagnostic_failed": sum(not row["fits_budget"] for row in diagnostics),
        },
        "rows": rows,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Isolated H3 resource-backend product-envelope validation",
        "",
        "Required gate: all aspect ratios and 1/5/10/15-second grids. INT8 "
        "requires FL2VA plus a 15-item static Ref2VA stress context; the W4A8 "
        "8GB boundary requires FL2VA plus one reference item.",
        "",
        f"- Required: {report['summary']['required_passed']}/"
        f"{report['summary']['required_cases']} passed",
        f"- Video-heavy diagnostics failing: "
        f"{report['summary']['diagnostic_failed']}/"
        f"{report['summary']['diagnostic_cases']}",
        "- Physical smoke: 5 steps, acceleration 95; V24 retained 3 Actual + 2 Forecast",
        "",
        "| Profile | Workload | Resolution | Worst required peak | Route | Status |",
        "|---|---|---:|---:|---|---|",
    ]
    required_rows = [row for row in report["rows"] if row["required"]]
    for profile in (item.name for item in PROFILES):
        workloads = sorted({
            (row["workload"], row["resolution"])
            for row in required_rows if row["profile"] == profile
        })
        for workload, resolution in workloads:
            group = [
                row for row in required_rows
                if row["profile"] == profile
                and row["workload"] == workload
                and row["resolution"] == resolution
            ]
            worst = max(group, key=lambda row: row["predicted_peak_gib"])
            status = "PASS" if all(row["fits_budget"] for row in group) else "FAIL"
            lines.append(
                f"| {profile} | {workload} | {resolution} | "
                f"{worst['predicted_peak_gib']:.4f} GiB | "
                f"{worst['selected_scheme']} | {status} |"
            )
    lines.extend([
        "",
        "Each profile is admitted independently; no row may borrow another "
        "profile's device budget. The video-heavy Ref2VA diagnostic is not "
        "included in the release claim; "
        "it identifies where a future tiled executor is still required.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runtime/calibration/unified_vram_backend_20260827",
    )
    args = parser.parse_args()
    report = build_report()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "product_envelope.json"
    md_path = args.output_dir / "PRODUCT_ENVELOPE.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "summary": report["summary"],
        "json": str(json_path),
        "markdown": str(md_path),
    }, ensure_ascii=False, indent=2))
    return 1 if report["summary"]["required_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
