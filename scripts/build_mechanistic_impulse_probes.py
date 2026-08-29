#!/usr/bin/env python3
"""Build counterfactual H3 impulse probes for trajectory identification.

The generated candidates are not acceleration schedules.  Each probe changes
exactly one mechanism at one denoising phase and keeps the remainder of the
trajectory Dense.  Comparing its final AV latents with the Dense control
therefore measures the downstream amplification of that local approximation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


SERVE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVE_ROOT))

from h3serve.native_engine.planner import (  # noqa: E402
    blueprint_from_runtime_schedule,
    save_v19_candidate_blueprint,
    v19_blueprint_execution_digest,
)
from h3serve.native_engine.planner.v19_runtime_bridge import (  # noqa: E402
    ROUND229_FORECAST_ANCHOR,
)


SCHEMA_VERSION = "h3_mechanistic_impulse_probe_set_v1"


def _parse_indices(raw: str, *, role: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as error:
        raise ValueError(f"{role} must be comma-separated integers") from error
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{role} must be sorted and unique")
    return values


def _parse_layer_range(raw: str) -> tuple[int, int]:
    try:
        start_text, stop_text = raw.split(":", 1)
        start, stop = int(start_text), int(stop_text)
    except ValueError as error:
        raise ValueError("attention layers must use START:STOP") from error
    if not 0 <= start < stop <= 50:
        raise ValueError("attention layers must lie inside [0, 50]")
    return start, stop


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_schedule(
    *,
    total_steps: int,
    actual_steps: tuple[int, ...],
    forecast_step: int | None = None,
    attention_step: int | None = None,
    attention_action: str = "sparse_topk_0.25",
    attention_layers: tuple[int, int] = (0, 50),
) -> tuple[tuple[int, int, str], ...]:
    actual = frozenset(actual_steps)
    rows: list[tuple[int, int, str]] = []
    for step in range(total_steps):
        if step not in actual:
            if step != forecast_step:
                raise AssertionError("probe schedule contains an undeclared Forecast")
            rows.extend(
                (step, layer, ROUND229_FORECAST_ANCHOR) for layer in range(3)
            )
            continue
        for layer in range(50):
            action = "dense"
            if (
                step == attention_step
                and attention_layers[0] <= layer < attention_layers[1]
            ):
                action = f"round215:{attention_action}"
            rows.append((step, layer, action))
    return tuple(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenario-manifest", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=20)
    parser.add_argument(
        "--forecast-steps",
        default="2,5,9,13,17",
        help="single-Forecast probe phases; step 0/1/final are structurally invalid",
    )
    parser.add_argument(
        "--attention-steps",
        default="",
        help="optional single sparse-Attention impulse phases",
    )
    parser.add_argument(
        "--attention-action",
        choices=(
            "sparse_topk_0.5",
            "sparse_topk_0.25",
            "sparse_topk_0.1",
            "sparse_topk_0.0625",
        ),
        default="sparse_topk_0.25",
    )
    parser.add_argument("--attention-layers", default="0:50")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.total_steps < 4:
        raise ValueError("total steps must be at least four")
    scenario = args.scenario_manifest.resolve()
    if not scenario.is_file():
        raise ValueError(f"scenario manifest does not exist: {scenario}")
    forecast_steps = _parse_indices(args.forecast_steps, role="forecast steps")
    attention_steps = _parse_indices(args.attention_steps, role="attention steps")
    invalid_forecasts = tuple(
        step
        for step in forecast_steps
        if step < 2 or step >= args.total_steps - 1
    )
    invalid_attention = tuple(
        step for step in attention_steps if not 0 <= step < args.total_steps
    )
    if invalid_forecasts:
        raise ValueError(f"invalid single-Forecast phases: {invalid_forecasts}")
    if invalid_attention:
        raise ValueError(f"invalid Attention phases: {invalid_attention}")
    attention_layers = _parse_layer_range(args.attention_layers)

    root = args.output_root.resolve()
    blueprint_root = root / "blueprints"
    blueprint_root.mkdir(parents=True, exist_ok=True)
    all_steps = tuple(range(args.total_steps))
    probe_rows: list[dict[str, object]] = []

    def add_probe(
        *,
        name: str,
        probe_type: str,
        actual_steps: tuple[int, ...],
        schedule: tuple[tuple[int, int, str], ...],
        phase_step: int | None,
    ) -> None:
        blueprint = blueprint_from_runtime_schedule(
            candidate_id=name,
            total_steps=args.total_steps,
            actual_step_indices=actual_steps,
            attention_action_schedule=schedule,
            source="mechanistic_single_impulse_identification_v1",
        )
        path = blueprint_root / f"{name}.json"
        save_v19_candidate_blueprint(path, blueprint)
        probe_rows.append({
            "name": name,
            "probe_type": probe_type,
            "phase_step": phase_step,
            "normalized_phase": (
                None
                if phase_step is None
                else phase_step / max(1, args.total_steps - 1)
            ),
            "blueprint": str(path.relative_to(root)),
            "execution_digest": v19_blueprint_execution_digest(blueprint),
        })

    add_probe(
        name="dense_control",
        probe_type="dense_control",
        actual_steps=all_steps,
        schedule=_runtime_schedule(
            total_steps=args.total_steps,
            actual_steps=all_steps,
        ),
        phase_step=None,
    )
    for step in forecast_steps:
        actual_steps = tuple(value for value in all_steps if value != step)
        add_probe(
            name=f"forecast_impulse_step{step:02d}",
            probe_type="single_forecast",
            actual_steps=actual_steps,
            schedule=_runtime_schedule(
                total_steps=args.total_steps,
                actual_steps=actual_steps,
                forecast_step=step,
            ),
            phase_step=step,
        )
    for step in attention_steps:
        add_probe(
            name=f"attention_impulse_step{step:02d}",
            probe_type="single_attention",
            actual_steps=all_steps,
            schedule=_runtime_schedule(
                total_steps=args.total_steps,
                actual_steps=all_steps,
                attention_step=step,
                attention_action=args.attention_action,
                attention_layers=attention_layers,
            ),
            phase_step=step,
        )

    batch_manifest = {
        "schema_version": 1,
        "purpose": (
            "Mechanistic H3 counterfactual probes: one approximation impulse, "
            "Dense elsewhere; never a release schedule or Human-quality fit."
        ),
        "candidates": [
            {"name": row["name"], "blueprint": row["blueprint"]}
            for row in probe_rows
        ],
    }
    (root / "blueprint_manifest.json").write_text(
        json.dumps(batch_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "identification_contract": {
            "intervention": "one approximation mechanism at one phase",
            "counterfactual_control": "same seed/prompt/geometry, all Dense",
            "downstream_response": "final AV latent delta to Dense control",
            "historical_schedule_used": False,
            "human_acceptance_used": False,
        },
        "total_steps": args.total_steps,
        "forecast_steps": list(forecast_steps),
        "attention_steps": list(attention_steps),
        "attention_action": args.attention_action,
        "attention_layers": list(attention_layers),
        "scenario_manifest": str(scenario),
        "scenario_manifest_sha256": _file_sha256(scenario),
        "probes": probe_rows,
    }
    (root / "probe_set.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "probe_set": str(root / "probe_set.json"),
        "blueprint_manifest": str(root / "blueprint_manifest.json"),
        "candidate_count": len(probe_rows),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
