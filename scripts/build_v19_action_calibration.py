#!/usr/bin/env python3
"""Seal repeated physical action probes into a strict V19 calibration.

Each ``--action-cost`` must be paired with the ``--session-record`` produced by
the same invocation of ``benchmark_native_hot_session.py``.  Historical files
without V19 source/runtime fingerprints are intentionally rejected.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys


SERVE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVE_ROOT))

from h3serve.native_engine.planner import (  # noqa: E402
    FIXED_TOPK_ACTION_IMPLEMENTATION,
    ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ROUND215_ACTION_IMPLEMENTATION,
    ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
    V19CalibrationError,
    V19CalibrationWorkload,
    V19NumericalErrorSample,
    V19RuntimeFingerprint,
    V19SourceRecord,
    V19TimingMeasurement,
    build_v19_bootstrap_registry,
    create_v19_action_calibration,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=("dense", "round215", "round188", "round228", "round229"),
        required=True,
        help="physical action family contained in the repeated probes",
    )
    parser.add_argument("--action-cost", type=Path, action="append", required=True)
    parser.add_argument("--session-record", type=Path, action="append", required=True)
    parser.add_argument(
        "--service-family", choices=("first_last", "reference"), required=True
    )
    parser.add_argument("--condition-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-warm-samples", type=int, default=3)
    return parser.parse_args()


def _implementation_ids() -> dict[str, str]:
    return {
        "fixed_topk": FIXED_TOPK_ACTION_IMPLEMENTATION,
        "round215": ROUND215_ACTION_IMPLEMENTATION,
        "round188": ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
        "round228": ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
        "round229": ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
    }


def _relative_source(path: Path, *, serve_root: Path, source_id: str) -> V19SourceRecord:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(serve_root.resolve()).as_posix()
    except ValueError as error:
        raise V19CalibrationError(
            f"calibration provenance is outside the release: {resolved}"
        ) from error
    return V19SourceRecord(
        source_id=source_id,
        relative_path=relative,
        sha256=sha256_file(resolved),
    )


def _model_variant(engine: str) -> str:
    if engine in ("original", "reference"):
        return "base"
    if engine in ("lora", "reference-lora"):
        return "lora"
    raise V19CalibrationError(f"unknown benchmark engine: {engine}")


def main() -> int:
    args = parse_args()
    if len(args.action_cost) != len(args.session_record):
        raise SystemExit("each --action-cost requires one paired --session-record")
    if args.minimum_warm_samples < 3:
        raise SystemExit("--minimum-warm-samples must be at least three")
    serve_root = SERVE_ROOT
    registry = build_v19_bootstrap_registry(implementation_ids=_implementation_ids())
    action_id, calibration_id, implementation_id = {
        "dense": (
            "h3.attention.dense.sage_per_warp.sm89.v1",
            "v19_dense_full_head_cost_v1",
            "sage_dense_per_warp_sm89_v1",
        ),
        "round215": (
            "h3.attention.interaction_hybrid.round215.v1",
            "v19_round215_full_head_cost_error_v1",
            ROUND215_ACTION_IMPLEMENTATION,
        ),
        "round188": (
            "h3.attention.mtcr_head_rail.round188.v1",
            "v19_round188_full_head_cost_error_v1",
            ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
        ),
        "round228": (
            "h3.attention.mtcr_head_rail.round228.v1",
            "v19_round228_full_head_cost_error_v1",
            ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
        ),
        "round229": (
            "h3.attention.mtcr_head_rail.round229.v1",
            "v19_round229_attention_cost_error_v1",
            ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
        ),
    }[args.action]

    samples: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    initialization_samples: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    peak_samples: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    error_samples: dict[
        tuple[str, int, int], list[V19NumericalErrorSample]
    ] = defaultdict(list)
    workload: V19CalibrationWorkload | None = None
    runtime: V19RuntimeFingerprint | None = None
    sources: list[V19SourceRecord] = []
    complete = True
    for repeat_index, (cost_path, session_path) in enumerate(
        zip(args.action_cost, args.session_record), start=1
    ):
        cost = json.loads(cost_path.read_text(encoding="utf-8"))
        session = json.loads(session_path.read_text(encoding="utf-8"))
        contract = cost.get("contract", {})
        session_contract = session.get("contract", {})
        fingerprint_document = session.get("v19_runtime_fingerprint")
        if not isinstance(fingerprint_document, dict):
            raise V19CalibrationError(
                f"session lacks V19 physical fingerprint: {session_path}"
            )
        observed_implementation = session_contract.get("physical_action_implementation")
        if args.action != "dense" and observed_implementation != implementation_id:
            raise V19CalibrationError(
                "session did not execute the requested registered implementation"
            )
        if (
            args.action != "dense"
            and contract.get("physical_action_implementation") != implementation_id
        ):
            raise V19CalibrationError(
                "action-cost payload and requested implementation disagree"
            )
        if session_contract.get("attention_backend") != "layer-calibration":
            raise V19CalibrationError("session is not a full-head layer calibration")
        if args.action == "round215" and contract.get("sparse_selection_mode") != "interaction_hybrid":
            raise V19CalibrationError("action-cost probe is not interaction_hybrid")
        steps = cost.get("steps")
        if not isinstance(steps, list) or not steps:
            raise V19CalibrationError("action-cost probe contains no step measurements")
        requests = session.get("requests")
        if not isinstance(requests, list) or len(requests) != 1:
            raise V19CalibrationError(
                "layer calibration must bind exactly one executed request geometry"
            )
        executed_request = requests[0]
        if executed_request.get("error") != (
            "layer calibration completed; generation intentionally stopped"
        ):
            raise V19CalibrationError(
                "layer calibration session did not stop at the expected boundary"
            )
        current_workload = V19CalibrationWorkload(
            model_variant=_model_variant(str(session["engine"])),
            service_family=args.service_family,
            # Root session fields are CLI defaults.  Scenario manifests may
            # override them, so the only valid physical geometry is the
            # request that actually reached the DiT.
            width=int(executed_request["width"]),
            height=int(executed_request["height"]),
            frames=int(executed_request["frames"]),
            packed_tokens=int(contract["sequence_tokens"]),
            condition_count=args.condition_count,
            steps=int(session_contract["steps"]),
            actual_step_indices=tuple(int(value) for value in session_contract["actual_step_indices"]),
            device_arch=str(fingerprint_document["device_arch"]),
            sampler=str(session_contract["sampler"]),
            scheduler=str(session_contract["scheduler"]),
        )
        current_runtime = V19RuntimeFingerprint(**fingerprint_document)
        if workload is None:
            workload = current_workload
            runtime = current_runtime
        elif current_workload != workload:
            raise V19CalibrationError("repeat probes do not share one exact workload bucket")
        elif current_runtime != runtime:
            raise V19CalibrationError("repeat probes do not share one runtime/build fingerprint")

        for step in steps:
            step_index = int(step["step_index"])
            layers = step.get("layers", ())
            if len(layers) != 50:
                complete = False
            for layer in layers:
                layer_index = int(layer["layer"])
                if args.action == "dense":
                    key = ("dense", step_index, layer_index)
                    warm = layer.get("dense_warm_ms") or (layer["dense_ms"],)
                    samples[key].extend(float(value) for value in warm)
                    if "dense_initialization_ms" in layer:
                        initialization_samples[key].append(
                            float(layer["dense_initialization_ms"])
                        )
                    peak_samples[key].extend(
                        float(value)
                        for value in layer.get("dense_warm_peak_gib", ())
                    )
                    error_samples[key].append(V19NumericalErrorSample(
                        mean_cosine=1.0,
                        min_cosine=1.0,
                        global_relative_rms=0.0,
                        mean_head_relative_rms=0.0,
                        max_head_relative_rms=0.0,
                        max_relative_l1=0.0,
                    ))
                else:
                    for candidate in layer.get("candidates", ()):
                        key = (str(candidate["name"]), step_index, layer_index)
                        warm = candidate.get("full_head_warm_ms") or (
                            candidate["full_head_ms"],
                        )
                        samples[key].extend(float(value) for value in warm)
                        if "full_head_initialization_ms" in candidate:
                            initialization_samples[key].append(
                                float(candidate["full_head_initialization_ms"])
                            )
                        peak_samples[key].extend(
                            float(value)
                            for value in candidate.get(
                                "full_head_warm_peak_gib", ()
                            )
                        )
                        head_relative_rms = tuple(
                            float(value)
                            for value in candidate.get("head_relative_rms", ())
                        )
                        if not head_relative_rms:
                            raise V19CalibrationError(
                                "sparse calibration lacks per-Head numerical error"
                            )
                        error_samples[key].append(V19NumericalErrorSample(
                            mean_cosine=float(candidate["mean_cosine"]),
                            min_cosine=float(candidate["min_cosine"]),
                            global_relative_rms=float(candidate["global_relative_rms"]),
                            mean_head_relative_rms=float(
                                candidate["mean_head_relative_rms"]
                            ),
                            max_head_relative_rms=max(head_relative_rms),
                            max_relative_l1=float(candidate["max_relative_l1"]),
                        ))
        sources.extend((
            _relative_source(
                cost_path,
                serve_root=serve_root,
                source_id=f"repeat{repeat_index}_action_cost",
            ),
            _relative_source(
                session_path,
                serve_root=serve_root,
                source_id=f"repeat{repeat_index}_session",
            ),
        ))

    assert workload is not None and runtime is not None
    measurements = tuple(
        V19TimingMeasurement(
            canonical_action=canonical_action,
            step_index=step_index,
            layer_start=layer_index,
            layer_stop=layer_index + 1,
            warm_samples_ms=tuple(values),
            initialization_samples_ms=tuple(
                initialization_samples[
                    (canonical_action, step_index, layer_index)
                ]
            ),
            peak_vram_gib_samples=tuple(
                peak_samples[(canonical_action, step_index, layer_index)]
            ),
            numerical_error_samples=tuple(
                error_samples[(canonical_action, step_index, layer_index)]
            ),
        )
        for (canonical_action, step_index, layer_index), values in sorted(samples.items())
    )
    artifact = create_v19_action_calibration(
        registry=registry,
        action_id=action_id,
        calibration_id=calibration_id,
        workload=workload,
        runtime=runtime,
        measurements=measurements,
        sources=tuple(sources),
        timing_scope="attention_layer_call",
        complete=complete,
        minimum_warm_samples=args.minimum_warm_samples,
    )
    artifact.require_planner_ready()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "action_id": action_id,
        "implementation_id": implementation_id,
        "workload_digest": workload.digest,
        "runtime_digest": runtime.digest,
        "payload_sha256": artifact.payload_sha256,
        "measurements": len(measurements),
        "samples_per_measurement": min(len(row.warm_samples_ms) for row in measurements),
        "planner_ready": artifact.planner_ready,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
