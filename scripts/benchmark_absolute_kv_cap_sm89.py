#!/usr/bin/env python3
"""Gate a long-horizon absolute KV-block cap on real H3 SM89 kernels.

This is an isolated prepared-KV Attention experiment, not a video-quality
claim and not a production action registration.  Every baseline/candidate pair
uses the same synthetic BF16 Q/K/V tensors, H3 head budgets, protected prefix,
MTCR and rotating global anchors.  The 100K case proves the cap is inert at the
accepted reference horizon; the 220K case measures whether stopping only the
discretionary global selection from growing is worth a real-block experiment.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import torch


GIB = 1024**3


CASES = (
    {
        "name": "accepted_720p15_100k",
        "sequence": 100_141,
        "protected_tokens": 1_701,
        "latent_frames": 107,
        "frame_tokens": 920,
        "grid_height": 23,
        "grid_width": 40,
    },
    {
        "name": "target_1080p15_220k",
        "sequence": 219_890,
        "protected_tokens": 1_610,
        "latent_frames": 107,
        "frame_tokens": 2_040,
        "grid_height": 34,
        "grid_width": 60,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--query-chunk-tokens", type=int, default=32_768)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=82_351)
    parser.add_argument(
        "--cap-multiplier",
        type=float,
        default=1.0,
        help=(
            "multiply the accepted 100K selected-block count before using it "
            "as the long-horizon absolute cap; values above one remain inert "
            "at the reference shape"
        ),
    )
    parser.add_argument(
        "--selector",
        choices=("absolute", "mass_guarded", "mass_probe"),
        default="absolute",
    )
    parser.add_argument("--minimum-retained-topk-mass", type=float, default=0.95)
    parser.add_argument("--probe-cap-ladder", default="")
    parser.add_argument(
        "--profiles",
        default="ordinary,causal",
        help="comma-separated subset of ordinary and causal",
    )
    parser.add_argument(
        "--sparge-build-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "H3_NATIVE_SPARGE_BUILD_DIR",
                "runtime/extensions/sparge-sm89-py310-torch213-cu133",
            )
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runtime/calibration/v19_long_video_20260825/"
            "absolute_kv_cap_sm89_gate.json"
        ),
    )
    return parser.parse_args()


def _contexts(case: dict[str, int], layer: int):
    from h3serve.native_engine.model.kernels import (
        attention_layer,
        attention_protected_prefix,
        attention_step,
        attention_video_layout,
    )

    stack = ExitStack()
    stack.enter_context(attention_protected_prefix(case["protected_tokens"]))
    stack.enter_context(
        attention_video_layout(
            case["latent_frames"],
            case["frame_tokens"],
            grid_height=case["grid_height"],
            grid_width=case["grid_width"],
        )
    )
    stack.enter_context(attention_step(10, 20))
    stack.enter_context(attention_layer(layer))
    return stack


def _output_metrics(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, object]:
    if baseline.shape != candidate.shape:
        raise ValueError("paired outputs must have the same shape")
    exact = bool(torch.equal(baseline, candidate))
    difference_squared = 0.0
    baseline_squared = 0.0
    candidate_squared = 0.0
    dot = 0.0
    max_abs = 0.0
    for start in range(0, baseline.shape[0], 2_048):
        stop = min(baseline.shape[0], start + 2_048)
        reference = baseline[start:stop].float()
        trial = candidate[start:stop].float()
        delta = trial - reference
        difference_squared += float(delta.square().sum())
        baseline_squared += float(reference.square().sum())
        candidate_squared += float(trial.square().sum())
        dot += float((reference * trial).sum())
        max_abs = max(max_abs, float(delta.abs().max()))
        del reference, trial, delta
    nrmse = math.sqrt(difference_squared / max(baseline_squared, 1e-20))
    cosine = dot / math.sqrt(
        max(baseline_squared, 1e-20) * max(candidate_squared, 1e-20)
    )

    sample_indices = torch.linspace(
        0,
        baseline.shape[0] - 1,
        min(257, baseline.shape[0]),
        device=baseline.device,
    ).round().to(torch.long).unique()
    reference = baseline.index_select(0, sample_indices).float()
    trial = candidate.index_select(0, sample_indices).float()
    per_head_relative_rms = (
        (trial - reference).square().mean(dim=-1).sqrt()
        / reference.square().mean(dim=-1).sqrt().clamp_min(1e-6)
    )
    result = {
        "bit_exact": exact,
        "global_relative_rms": nrmse,
        "global_cosine": cosine,
        "maximum_absolute_difference": max_abs,
        "sampled_token_count": int(sample_indices.numel()),
        "sampled_head_relative_rms_mean": float(per_head_relative_rms.mean()),
        "sampled_head_relative_rms_p95": float(
            torch.quantile(per_head_relative_rms.flatten(), 0.95)
        ),
        "sampled_head_relative_rms_max": float(per_head_relative_rms.max()),
    }
    del sample_indices, reference, trial, per_head_relative_rms
    return result


def _cap_value(counts: torch.Tensor) -> int | tuple[int, ...]:
    values = tuple(int(value) for value in counts.tolist())
    return values[0] if len(set(values)) == 1 else values


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    profiles = tuple(item.strip() for item in args.profiles.split(",") if item.strip())
    if not profiles or any(item not in ("ordinary", "causal") for item in profiles):
        raise SystemExit("--profiles must contain ordinary and/or causal")
    if args.repeats < 2:
        raise SystemExit("--repeats must be at least two")
    if not 1.0 <= args.cap_multiplier <= 4.0:
        raise SystemExit("--cap-multiplier must lie inside [1, 4]")
    if not 0.0 < args.minimum_retained_topk_mass <= 1.0:
        raise SystemExit("--minimum-retained-topk-mass must lie inside (0, 1]")
    try:
        probe_cap_ladder = tuple(
            int(value.strip())
            for value in args.probe_cap_ladder.split(",")
            if value.strip()
        )
    except ValueError as error:
        raise SystemExit("--probe-cap-ladder must be comma-separated integers") from error
    if (
        tuple(sorted(set(probe_cap_ladder))) != probe_cap_ladder
        or any(value <= 0 for value in probe_cap_ladder)
        or (probe_cap_ladder and args.selector != "mass_probe")
    ):
        raise SystemExit(
            "--probe-cap-ladder must be sorted/unique/positive and requires mass_probe"
        )
    if args.query_chunk_tokens < 128 or args.query_chunk_tokens % 128:
        raise SystemExit("--query-chunk-tokens must be a positive multiple of 128")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this benchmark requires one RTX 4090 / SM89 GPU")
    build_dir = args.sparge_build_dir.resolve()
    if not build_dir.is_dir():
        raise SystemExit(f"missing Sparge build: {build_dir}")
    sys.path.insert(0, str(build_dir))

    from h3serve.native_engine.model.kernels import (
        SplitModalityProtectedSpargeAttentionBackend,
        make_joint_physical_action_backends_sm89,
    )

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    actions = make_joint_physical_action_backends_sm89()
    profile_actions = {
        "ordinary": ("fastfrontier:sparse_topk_0.0625", 20),
        "causal": ("fastfrontier:sparse_topk_0.1", 35),
    }
    reference_key_blocks = math.ceil(CASES[0]["sequence"] / 64)
    report: dict[str, object] = {
        "schema_version": "h3_absolute_kv_cap_sm89_gate_v1",
        "status": "running",
        "warning": (
            "Synthetic prepared-KV Attention evidence only. Passing permits a "
            "real H3 block probe, not V19 registration or a video-quality claim."
        ),
        "hypothesis": (
            "Keep accepted fractional Top-K through the 100K-token horizon; "
            "above it cap only discretionary selected KV blocks while retaining "
            "dense prefix, MTCR and rotating global anchors."
        ),
        "predeclared_gate": {
            "reference_100k_bit_exact": True,
            "reference_100k_max_slowdown_fraction": 0.05,
            "target_220k_minimum_kernel_speedup": 1.10,
            "target_220k_maximum_global_relative_rms": 0.30,
            "target_220k_minimum_global_cosine": 0.95,
            "target_220k_maximum_incremental_peak_growth_fraction": 0.05,
        },
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "driver": torch.cuda.get_device_properties(device).name,
            "total_memory_gib": torch.cuda.get_device_properties(device).total_memory
            / GIB,
            "sparge_build_dir": str(build_dir),
        },
        "reference_key_blocks": reference_key_blocks,
        "cap_multiplier": args.cap_multiplier,
        "selector": args.selector,
        "minimum_retained_topk_mass": args.minimum_retained_topk_mass,
        "probe_cap_ladder": probe_cap_ladder,
        "query_chunk_tokens": args.query_chunk_tokens,
        "repeats": args.repeats,
        "profiles": {},
    }

    started = time.perf_counter()
    try:
        for case_index, case_source in enumerate(CASES):
            case = dict(case_source)
            if (
                case["protected_tokens"]
                + case["latent_frames"] * case["frame_tokens"]
                != case["sequence"]
            ):
                raise RuntimeError(f"invalid embedded H3 shape: {case['name']}")
            video_tokens = case["sequence"] - case["protected_tokens"]
            chunk_tokens = min(args.query_chunk_tokens, video_tokens)
            chunk_tokens = max(128, chunk_tokens // 128 * 128)
            local_start = max(0, (video_tokens - chunk_tokens) // 2)
            local_start = local_start // 128 * 128
            local_stop = local_start + chunk_tokens
            if local_stop > video_tokens:
                local_stop = video_tokens
                local_start = local_stop - chunk_tokens

            torch.manual_seed(args.seed + case_index * 10_000)
            key = torch.empty(
                (1, args.heads, case["sequence"], args.head_dim),
                device=device,
                dtype=torch.bfloat16,
            ).normal_().clamp_(-4, 4)
            value = torch.empty_like(key).normal_().clamp_(-4, 4)
            preparation_backend = actions[profile_actions[profiles[0]][0]]
            value_fp8, value_scale, heads, tokens, head_dim = (
                preparation_backend.prepare_long_sequence_values(value)
            )
            del value
            gc.collect()
            torch.cuda.empty_cache()
            prepared = preparation_backend.prepare_long_sequence_keys(
                key, value_fp8, value_scale
            )
            query = torch.empty(
                (chunk_tokens, args.heads, args.head_dim),
                device=device,
                dtype=torch.bfloat16,
            ).normal_().clamp_(-4, 4)
            query_indices = torch.arange(
                local_start, local_stop, device=device, dtype=torch.int64
            )
            torch.cuda.synchronize()

            case_report: dict[str, object] = {
                "shape": case,
                "key_blocks": math.ceil(case["sequence"] / 64),
                "query_chunk": {
                    "tokens": chunk_tokens,
                    "local_start": local_start,
                    "local_stop": local_stop,
                },
                "prepared_resident_gib": torch.cuda.memory_allocated(device) / GIB,
                "profiles": {},
            }
            for profile in profiles:
                action_name, layer = profile_actions[profile]
                baseline = actions[action_name]
                reference_counts = baseline._selected_key_block_counts(
                    args.heads, reference_key_blocks, torch.device("cpu")
                )
                cap_counts = torch.ceil(
                    reference_counts.float() * args.cap_multiplier
                ).to(torch.int64)
                cap = _cap_value(cap_counts)
                candidate_mode = {
                    "absolute": "fixed_topk_absolute_cap",
                    "mass_guarded": "fixed_topk_mass_guarded_cap",
                    "mass_probe": "fixed_topk_mass_probe",
                }[args.selector]
                candidate = SplitModalityProtectedSpargeAttentionBackend(
                    baseline.topk,
                    experimental_minimum_topk=baseline.experimental_minimum_topk,
                    temporal_correspondence_radius=(
                        baseline.temporal_correspondence_radius
                    ),
                    temporal_spatial_block_radius=(
                        baseline.temporal_spatial_block_radius
                    ),
                    temporal_global_anchor_stride=(
                        baseline.temporal_global_anchor_stride
                    ),
                    temporal_global_spatial_block_radius=(
                        baseline.temporal_global_spatial_block_radius
                    ),
                    selection_mode=candidate_mode,
                    maximum_selected_key_blocks=cap,
                    minimum_retained_topk_mass=args.minimum_retained_topk_mass,
                    mass_probe_selected_key_blocks=probe_cap_ladder,
                )

                def run(backend) -> torch.Tensor:
                    with _contexts(case, layer):
                        return backend.long_sequence_video_queries(
                            query,
                            prepared,
                            protected_tokens=case["protected_tokens"],
                            query_token_indices=query_indices,
                        )

                # Compile/warm both identities before timing.  These outputs
                # are retained for the paired numerical comparison.
                baseline_output = run(baseline)
                candidate_output = run(candidate)
                torch.cuda.synchronize()
                metrics = _output_metrics(baseline_output, candidate_output)
                del baseline_output, candidate_output
                gc.collect()
                torch.cuda.empty_cache()

                timings: dict[str, list[float]] = {"baseline": [], "candidate": []}
                incremental_peaks: dict[str, list[float]] = {
                    "baseline": [],
                    "candidate": [],
                }
                for repeat in range(args.repeats):
                    order = (
                        (("baseline", baseline), ("candidate", candidate))
                        if repeat % 2 == 0
                        else (("candidate", candidate), ("baseline", baseline))
                    )
                    for name, backend in order:
                        torch.cuda.synchronize()
                        allocated_before = torch.cuda.memory_allocated(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        begin = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        begin.record()
                        output = run(backend)
                        end.record()
                        end.synchronize()
                        timings[name].append(float(begin.elapsed_time(end)) / 1_000.0)
                        incremental_peaks[name].append(
                            (
                                torch.cuda.max_memory_allocated(device)
                                - allocated_before
                            )
                            / GIB
                        )
                        del output, begin, end

                baseline_median = statistics.median(timings["baseline"])
                candidate_median = statistics.median(timings["candidate"])
                baseline_peak = statistics.median(incremental_peaks["baseline"])
                candidate_peak = statistics.median(incremental_peaks["candidate"])
                current_key_blocks = math.ceil(case["sequence"] / 64)
                nominal_counts = baseline._selected_key_block_counts(
                    args.heads, current_key_blocks, torch.device("cpu")
                )
                capped_counts = candidate._selected_key_block_counts(
                    args.heads, current_key_blocks, torch.device("cpu")
                )
                profile_report = {
                    "baseline_action": action_name,
                    "candidate_selection_mode": candidate.selection_mode,
                    "reference_absolute_cap": cap,
                    "nominal_selected_per_head": nominal_counts.tolist(),
                    "capped_selected_per_head": capped_counts.tolist(),
                    "discretionary_block_reduction_fraction": 1.0
                    - float(capped_counts.sum()) / float(nominal_counts.sum()),
                    "timing_seconds": {
                        "baseline_samples": timings["baseline"],
                        "candidate_samples": timings["candidate"],
                        "baseline_median": baseline_median,
                        "candidate_median": candidate_median,
                        "speedup": baseline_median / candidate_median,
                    },
                    "incremental_peak_allocated_gib": {
                        "baseline_samples": incremental_peaks["baseline"],
                        "candidate_samples": incremental_peaks["candidate"],
                        "baseline_median": baseline_peak,
                        "candidate_median": candidate_peak,
                        "growth_fraction": (
                            candidate_peak / baseline_peak - 1.0
                            if baseline_peak > 0
                            else 0.0
                        ),
                    },
                    "paired_output": metrics,
                    "candidate_telemetry": candidate.telemetry(),
                }
                case_report["profiles"][profile] = profile_report
            report["profiles"][case["name"]] = case_report

            del prepared, key, value_fp8, value_scale, query, query_indices
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        reference = report["profiles"]["accepted_720p15_100k"]["profiles"]
        target = report["profiles"]["target_1080p15_220k"]["profiles"]
        gate_rows: dict[str, object] = {}
        for profile in profiles:
            ref = reference[profile]
            trial = target[profile]
            reference_pass = (
                ref["paired_output"]["bit_exact"]
                and ref["timing_seconds"]["candidate_median"]
                <= ref["timing_seconds"]["baseline_median"] * 1.05
            )
            target_pass = (
                trial["timing_seconds"]["speedup"] >= 1.10
                and trial["paired_output"]["global_relative_rms"] <= 0.30
                and trial["paired_output"]["global_cosine"] >= 0.95
                and trial["incremental_peak_allocated_gib"]["growth_fraction"]
                <= 0.05
            )
            gate_rows[profile] = {
                "reference_pass": reference_pass,
                "target_pass": target_pass,
                "eligible_for_real_block_probe": reference_pass and target_pass,
            }
        report["gate"] = gate_rows
        report["eligible_for_real_block_probe"] = all(
            row["eligible_for_real_block_probe"] for row in gate_rows.values()
        )
        report["status"] = "complete"
    except torch.OutOfMemoryError as error:
        report["status"] = "oom"
        report["error"] = str(error)
        torch.cuda.empty_cache()
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
