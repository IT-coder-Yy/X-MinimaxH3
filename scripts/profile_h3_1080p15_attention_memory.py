#!/usr/bin/env python3
"""Profile the attention-memory boundary of a 1080p/15s H3 packed shape.

This is an isolated execution probe, not a quality candidate.  It uses H3's
real SM89 Dense/Sparge kernels with synthetic Q/K/V so we can distinguish:

* exact Dense kernel residency;
* the current whole-query Sparge selector/LUT peak; and
* a query-chunked Sparge proof of feasibility which keeps full K/V context.

OOM is evidence and is recorded in the JSON report instead of aborting the
remaining stages.  The chunked stage intentionally calls the existing private
prepared-K/V seam; production integration must expose a typed public contract
and cache K quantization before it can be considered a speed path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import torch


GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=220_003)
    parser.add_argument("--protected-tokens", type=int, default=1_723)
    parser.add_argument("--latent-frames", type=int, default=107)
    parser.add_argument("--frame-tokens", type=int, default=2_040)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--topk", type=float, default=0.25)
    parser.add_argument("--query-chunk-tokens", type=int, default=32_768)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument(
        "--input-storage",
        choices=("separate", "qkv_interleaved"),
        default="separate",
        help=(
            "separate contiguous Q/K/V or the strided views produced by H3's "
            "fused QKV projection"
        ),
    )
    parser.add_argument(
        "--stages",
        default="dense,chunked_dense,full_sparse,chunked_sparse",
        help=(
            "comma-separated subset of dense,chunked_dense,full_sparse,"
            "chunked_sparse"
        ),
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
            "runtime/calibration/long_1080p15_20260824/"
            "attention_memory_probe.json"
        ),
    )
    return parser.parse_args()


class NvidiaSmiSampler:
    """Low-rate external telemetry that does not synchronize CUDA work."""

    def __init__(self, interval_seconds: float = 0.20) -> None:
        self.interval_seconds = float(interval_seconds)
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _sample() -> dict[str, float] | None:
        import subprocess

        command = (
            "nvidia-smi",
            "--query-gpu=power.draw,utilization.gpu,memory.used,clocks.sm",
            "--format=csv,noheader,nounits",
        )
        try:
            raw = subprocess.check_output(
                command, text=True, timeout=2.0
            ).strip().splitlines()[0]
            power, utilization, memory, clock = (
                float(item.strip()) for item in raw.split(",")
            )
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None
        return {
            "power_w": power,
            "gpu_util_percent": utilization,
            "nvml_memory_mib": memory,
            "sm_clock_mhz": clock,
        }

    def start(self) -> None:
        def loop() -> None:
            while not self._stop.is_set():
                sample = self._sample()
                if sample is not None:
                    sample["elapsed_seconds"] = time.perf_counter() - started
                    self.samples.append(sample)
                self._stop.wait(self.interval_seconds)

        started = time.perf_counter()
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def summary(self) -> dict[str, object]:
        if not self.samples:
            return {"sample_count": 0}
        keys = ("power_w", "gpu_util_percent", "nvml_memory_mib", "sm_clock_mhz")
        result: dict[str, object] = {"sample_count": len(self.samples)}
        for key in keys:
            values = [sample[key] for sample in self.samples]
            result[f"{key}_mean"] = statistics.fmean(values)
            result[f"{key}_max"] = max(values)
        return result


def analytical_selector_bytes(
    *, video_tokens: int, sequence: int, heads: int, query_chunk_tokens: int
) -> dict[str, object]:
    """Conservative live-tensor accounting for the current Top-K selector."""

    key_blocks = math.ceil(sequence / 64)
    full_query_blocks = math.ceil(video_tokens / 128)
    chunk_query_blocks = math.ceil(min(video_tokens, query_chunk_tokens) / 128)
    # Current fixed-TopK construction can overlap pooled_score, sorted values,
    # int64 sorted indices, CDF, bool block map and int32 LUT.  This excludes
    # Q/K/V, quantized tensors, outputs and allocator fragmentation.
    bytes_per_cell = 2 + 2 + 8 + 2 + 1 + 4

    def estimate(query_blocks: int) -> dict[str, object]:
        cells = heads * query_blocks * key_blocks
        return {
            "query_blocks": query_blocks,
            "key_blocks": key_blocks,
            "head_block_cells": cells,
            "selector_live_tensor_lower_bound_gib": cells * bytes_per_cell / GIB,
        }

    return {
        "assumed_bytes_per_head_block_cell": bytes_per_cell,
        "whole_query": estimate(full_query_blocks),
        "one_query_chunk": estimate(chunk_query_blocks),
    }


def main() -> int:
    args = parse_args()
    stages = tuple(item.strip() for item in args.stages.split(",") if item.strip())
    allowed = {"dense", "chunked_dense", "full_sparse", "chunked_sparse"}
    if not stages or any(stage not in allowed for stage in stages):
        raise SystemExit(f"--stages must be a subset of {sorted(allowed)}")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this probe requires one RTX 4090 / SM89 GPU")
    if args.protected_tokens + args.latent_frames * args.frame_tokens != args.sequence:
        raise SystemExit("protected + latent_frames*frame_tokens must equal sequence")
    if args.query_chunk_tokens < 128 or args.query_chunk_tokens % 128:
        raise SystemExit("--query-chunk-tokens must be a positive multiple of 128")
    if not 0.0625 <= args.topk <= 1.0:
        raise SystemExit("--topk must lie inside [0.0625, 1]")

    build_dir = args.sparge_build_dir.resolve()
    if not build_dir.is_dir():
        raise SystemExit(f"missing Sparge build: {build_dir}")
    sys.path.insert(0, str(build_dir))

    from h3serve.native_engine.model.kernels import (
        DenseLongSequenceAttentionBackend,
        SplitModalityProtectedSpargeAttentionBackend,
        attention_layer,
        attention_protected_prefix,
        attention_step,
        attention_video_layout,
        dense_qk_quantization,
        sage_attention_sm89,
    )

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    shape = (args.sequence, args.heads, args.head_dim)
    qkv_storage = None
    if args.input_storage == "separate":
        q = torch.empty(shape, device=device, dtype=torch.bfloat16).normal_().clamp_(-4, 4)
        k = torch.empty(shape, device=device, dtype=torch.bfloat16).normal_().clamp_(-4, 4)
        v = torch.empty(shape, device=device, dtype=torch.bfloat16).normal_().clamp_(-4, 4)
    else:
        inner_width = args.heads * args.head_dim
        qkv_storage = torch.empty(
            args.sequence,
            3 * inner_width,
            device=device,
            dtype=torch.bfloat16,
        ).normal_().clamp_(-4, 4)
        q_rows, k_rows, v_rows = qkv_storage.split(inner_width, dim=-1)
        q = q_rows.view(shape)
        k = k_rows.view(shape)
        v = v_rows.view(shape)
    torch.cuda.synchronize()

    sparse = SplitModalityProtectedSpargeAttentionBackend(
        args.topk,
        experimental_minimum_topk=0.0625,
        temporal_correspondence_radius=1,
        temporal_spatial_block_radius=1,
        temporal_global_anchor_stride=8,
    )
    streamed_dense = DenseLongSequenceAttentionBackend()

    def context():
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(attention_protected_prefix(args.protected_tokens))
        stack.enter_context(
            attention_video_layout(args.latent_frames, args.frame_tokens)
        )
        stack.enter_context(attention_step(10, 20))
        stack.enter_context(attention_layer(20))
        return stack

    def dense() -> torch.Tensor:
        with dense_qk_quantization("per_warp"):
            return sage_attention_sm89(q, k, v)

    def chunked_dense() -> torch.Tensor:
        key_hnd = k.unsqueeze(0).contiguous()
        value_hnd = v.unsqueeze(0).contiguous()
        value_fp8, value_scale, _heads, _tokens, _head_dim = (
            streamed_dense.prepare_long_sequence_values(value_hnd)
        )
        del value_hnd
        prepared = streamed_dense.prepare_long_sequence_keys(
            key_hnd, value_fp8, value_scale
        )
        del key_hnd
        output = torch.empty_like(q)
        with dense_qk_quantization("per_warp"):
            for start in range(0, args.sequence, args.query_chunk_tokens):
                stop = min(args.sequence, start + args.query_chunk_tokens)
                output[start:stop].copy_(
                    streamed_dense.long_sequence_all_queries(
                        q[start:stop], prepared
                    )
                )
        return output

    def full_sparse() -> torch.Tensor:
        with context():
            return sparse(q, k, v)

    def chunked_sparse() -> torch.Tensor:
        # Proof of the memory geometry only.  K pooling/quantization is still
        # repeated by the existing helper for every Query chunk, so this is
        # expected to be slower than a production prepared-KV implementation.
        with context():
            prepared_k, v_fp8, v_scale, heads, _kv_len, head_dim = (
                sparse._prepare_kv(k, v)
            )
            output = torch.empty_like(q)
            prefix = sparse._dense_prefix(
                q[: args.protected_tokens],
                prepared_k,
                v_fp8,
                v_scale,
                head_dim=head_dim,
            )
            output[: args.protected_tokens].copy_(prefix)
            del prefix
            video_tokens = args.sequence - args.protected_tokens
            for local_start in range(0, video_tokens, args.query_chunk_tokens):
                local_stop = min(video_tokens, local_start + args.query_chunk_tokens)
                global_start = args.protected_tokens + local_start
                global_stop = args.protected_tokens + local_stop
                query_indices = torch.arange(
                    local_start, local_stop, device=device, dtype=torch.int64
                )
                chunk = sparse._sparse_video_queries(
                    q[global_start:global_stop],
                    prepared_k,
                    v_fp8,
                    v_scale,
                    protected_tokens=args.protected_tokens,
                    heads=heads,
                    head_dim=head_dim,
                    query_token_indices=query_indices,
                )
                output[global_start:global_stop].copy_(chunk)
                del query_indices, chunk
            return output

    operations: dict[str, Callable[[], torch.Tensor]] = {
        "dense": dense,
        "chunked_dense": chunked_dense,
        "full_sparse": full_sparse,
        "chunked_sparse": chunked_sparse,
    }
    report: dict[str, object] = {
        "schema_version": "h3_1080p15_attention_memory_probe_v1",
        "warning": (
            "Synthetic isolated-kernel evidence only; it neither certifies "
            "end-to-end feasibility nor video quality."
        ),
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_total_gib": torch.cuda.get_device_properties(device).total_memory / GIB,
            "sparge_build_dir": str(build_dir),
        },
        "shape": {
            "sequence": args.sequence,
            "protected_tokens": args.protected_tokens,
            "video_tokens": args.sequence - args.protected_tokens,
            "latent_frames": args.latent_frames,
            "frame_tokens": args.frame_tokens,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "topk": args.topk,
            "query_chunk_tokens": args.query_chunk_tokens,
            "input_storage": args.input_storage,
            "qkv_row_stride": int(q.stride(0)),
            "input_qkv_gib": 3 * args.sequence * args.heads * args.head_dim * 2 / GIB,
        },
        "analytical_selector_memory": analytical_selector_bytes(
            video_tokens=args.sequence - args.protected_tokens,
            sequence=args.sequence,
            heads=args.heads,
            query_chunk_tokens=args.query_chunk_tokens,
        ),
        "stages": {},
        "comparisons": {},
    }

    baseline_allocated = int(torch.cuda.memory_allocated(device))
    sample_points = set(
        torch.linspace(0, args.sequence - 1, 257).round().to(torch.int64).tolist()
    )
    sample_points.update((args.protected_tokens - 1, args.protected_tokens))
    for local_boundary in range(
        args.query_chunk_tokens,
        args.sequence - args.protected_tokens,
        args.query_chunk_tokens,
    ):
        global_boundary = args.protected_tokens + local_boundary
        sample_points.update((global_boundary - 1, global_boundary))
    sample_indices = torch.tensor(
        sorted(point for point in sample_points if 0 <= point < args.sequence),
        device=device,
        dtype=torch.int64,
    )
    sampled_outputs: dict[str, torch.Tensor] = {}
    for name in stages:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        sampler = NvidiaSmiSampler()
        sampler.start()
        started = time.perf_counter()
        result = None
        stage: dict[str, object]
        try:
            start_event = torch.cuda.Event(enable_timing=True)
            stop_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            result = operations[name]()
            stop_event.record()
            stop_event.synchronize()
            checksum = float(
                result[:: max(1, args.sequence // 32)].float().mean().cpu()
            )
            sampled_outputs[name] = result.index_select(
                0, sample_indices
            ).detach().cpu()
            stage = {
                "status": "ok",
                "cuda_ms": float(start_event.elapsed_time(stop_event)),
                "wall_seconds": time.perf_counter() - started,
                "checksum_mean_sample": checksum,
            }
        except torch.OutOfMemoryError as error:
            stage = {
                "status": "oom",
                "wall_seconds": time.perf_counter() - started,
                "error": str(error),
            }
        except Exception as error:  # Preserve unexpected failures as evidence.
            stage = {
                "status": "error",
                "wall_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        finally:
            sampler.stop()
            del result
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass
            stage["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / GIB
            stage["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / GIB
            stage["incremental_peak_allocated_gib"] = (
                torch.cuda.max_memory_allocated(device) - baseline_allocated
            ) / GIB
            stage["nvml"] = sampler.summary()
            report["stages"][name] = stage
            torch.cuda.empty_cache()

    if "full_sparse" in sampled_outputs and "chunked_sparse" in sampled_outputs:
        reference = sampled_outputs["full_sparse"]
        candidate = sampled_outputs["chunked_sparse"]
        delta = candidate.float() - reference.float()
        report["comparisons"]["chunked_sparse_vs_full_sparse_sample"] = {
            "sampled_rows": int(sample_indices.numel()),
            "includes_protected_boundary": True,
            "includes_every_query_chunk_boundary": True,
            "exact_equal": bool(torch.equal(reference, candidate)),
            "mean_abs": float(delta.abs().mean()),
            "max_abs": float(delta.abs().max()),
            "rmse": float(delta.square().mean().sqrt()),
        }
    if "dense" in sampled_outputs and "chunked_dense" in sampled_outputs:
        reference = sampled_outputs["dense"]
        candidate = sampled_outputs["chunked_dense"]
        delta = candidate.float() - reference.float()
        report["comparisons"]["chunked_dense_vs_dense_sample"] = {
            "sampled_rows": int(sample_indices.numel()),
            "exact_equal": bool(torch.equal(reference, candidate)),
            "mean_abs": float(delta.abs().mean()),
            "max_abs": float(delta.abs().max()),
            "rmse": float(delta.square().mean().sqrt()),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
