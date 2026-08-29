from __future__ import annotations

from dataclasses import replace
import unittest
from pathlib import Path
import tempfile

from h3serve.native_engine.planner import (
    ActionKind,
    ActionRegistry,
    EvidenceStatus,
    RegisteredAction,
    V19ActionUse,
    V19CandidatePlan,
    V19CertifiedEnvelopeEntry,
    V19CertifiedFrontierEntry,
    V19ForecastCompositeKey,
    V19ForecastUse,
    V19HumanRiskVector,
    V19MaterializedCandidate,
    V19PlanCertificate,
    V19PlanningError,
    V19PlanningRequest,
    V19ReleaseFrontierCatalog,
    V19ReferenceProfile,
    V19RuntimeFingerprint,
    V19TrajectoryDebt,
    V19WorkloadContext,
    V19WorkloadEnvelope,
    build_v19_certified_envelope_entry,
    load_v19_release_bundle,
    save_v19_release_bundle,
)


ACTION_ID = "h3.attention.synthetic.v19"
FORECAST_ID = "h3.forecast.synthetic.v19"


def _registry() -> ActionRegistry:
    return ActionRegistry((
        RegisteredAction(
            action_id=ACTION_ID,
            implementation_id="synthetic_v19",
            kind=ActionKind.SPARSE_ATTENTION,
            executor_id="synthetic",
            canonical_actions=("sparse_topk_0.5",),
            exact=False,
            evidence_status=EvidenceStatus.HUMAN_REVIEWED,
            calibration_ids=("synthetic_cost",),
            risk_model_ids=("synthetic_risk",),
            planner_eligible=True,
        ),
        RegisteredAction(
            action_id=FORECAST_ID,
            implementation_id="synthetic_forecast_v19",
            kind=ActionKind.FORECAST_COMPOSITE,
            executor_id="synthetic_forecast",
            canonical_actions=("forecast",),
            exact=False,
            evidence_status=EvidenceStatus.HUMAN_REVIEWED,
            calibration_ids=("synthetic_forecast_cost",),
            risk_model_ids=("synthetic_risk",),
            planner_eligible=True,
        ),
    ))


def _runtime() -> V19RuntimeFingerprint:
    return V19RuntimeFingerprint(
        gpu_name="NVIDIA GeForce RTX 4090",
        device_arch="sm89",
        torch_version="2.8.0+cu126",
        cuda_runtime="12.6",
        driver_version="test",
        quant_backend="cuda",
        comfy_kitchen_cuda_sha256="1" * 64,
        sageattention_sm89_sha256="2" * 64,
        action_source_sha256="3" * 64,
        planner_source_sha256="4" * 64,
    )


def _workload(
    actual: tuple[int, ...],
    *,
    frames: int = 124,
    packed_tokens: int = 34_871,
) -> V19WorkloadContext:
    return V19WorkloadContext(
        model_variant="base",
        service_family="first_last",
        packed_tokens=packed_tokens,
        condition_count=0,
        width=1280,
        height=736,
        frames=frames,
        steps=4,
        actual_step_indices=actual,
    )


def _risk(value: float, *, first: float | None = None) -> V19HumanRiskVector:
    values = [value] * 7
    if first is not None:
        values[0] = first
    return V19HumanRiskVector(*values)


def _entry(
    registry: ActionRegistry,
    *,
    actual: tuple[int, ...],
    cost: float,
    risk: V19HumanRiskVector,
    peak_vram_gib: float = 9.0,
    terminal_debt: V19TrajectoryDebt = V19TrajectoryDebt(),
    maximum_debt: V19TrajectoryDebt = V19TrajectoryDebt(),
) -> V19CertifiedFrontierEntry:
    workload = _workload(actual)
    uses = [V19ActionUse(
        action_id=ACTION_ID,
        canonical_action="sparse_topk_0.5",
        step_indices=actual,
    )]
    missing = sorted(set(range(4)) - set(actual))
    while missing:
        start = missing[0]
        stop = start
        while stop + 1 in missing:
            stop += 1
        run = tuple(range(start, stop + 1))
        uses.append(V19ForecastUse(
            action_id=FORECAST_ID,
            composite_key=V19ForecastCompositeKey(
                forecast_step_indices=run,
                preceding_actual_step=start - 1,
                following_actual_step=stop + 1,
                anchor_depth=1,
                anchor_action_id=ACTION_ID,
                anchor_canonical_action="sparse_topk_0.5",
                extrapolator_id="synthetic_directional",
                correction_id="synthetic_actual",
            ),
        ))
        missing = [value for value in missing if value > stop]
    candidate = V19CandidatePlan(
        candidate_id=f"candidate_{cost}",
        action_uses=tuple(uses),
        predicted_cost_p50_ms=cost - 1.0,
        predicted_cost_p90_ms=cost,
        predicted_peak_vram_gib=peak_vram_gib,
        risk_ucb=risk,
        terminal_debt=terminal_debt,
        maximum_debt=maximum_debt,
        evidence_ids=(
            "synthetic_cost",
            "synthetic_forecast_cost",
            "synthetic_risk",
        ),
    )
    request = V19PlanningRequest(
        workload=workload,
        runtime=_runtime(),
        maximum_cost_p90_ms=cost,
        maximum_peak_vram_gib=24.0,
        risk_limits=risk,
        debt_limits=maximum_debt,
    )
    materialized = V19MaterializedCandidate(
        candidate=candidate,
        end_to_end_cost_calibrated=True,
        human_risk_calibrated=True,
    )
    return V19CertifiedFrontierEntry(
        planning_request=request,
        materialized=materialized,
        certificate=V19PlanCertificate.issue(registry, request, candidate),
    )


def _envelope(
    *sources: V19CertifiedFrontierEntry,
    envelope_id: str = "synthetic_720p5",
) -> V19CertifiedEnvelopeEntry:
    tokens = tuple(
        source.planning_request.workload.packed_tokens for source in sources
    )
    scope = V19WorkloadEnvelope(
        envelope_id=envelope_id,
        model_variant="base",
        service_family="first_last",
        device_arch="sm89",
        width=1280,
        height=736,
        frames=124,
        steps=4,
        sampler="res_multistep",
        scheduler="simple",
        min_packed_tokens=min(tokens),
        max_packed_tokens=max(tokens),
        reference_profiles=(V19ReferenceProfile(condition_count=0),),
    )
    return build_v19_certified_envelope_entry(
        _registry(),
        envelope=scope,
        sources=tuple(sources),
    )


class V19ReleaseFrontierTests(unittest.TestCase):
    def test_one_dial_selects_complete_trajectory_by_certified_cost(self) -> None:
        registry = _registry()
        catalog = V19ReleaseFrontierCatalog(registry)
        for row in (
            _entry(registry, actual=(0, 1, 2, 3), cost=100.0, risk=_risk(0.10)),
            _entry(registry, actual=(0, 1, 3), cost=80.0, risk=_risk(0.15)),
            _entry(registry, actual=(0, 3), cost=50.0, risk=_risk(0.20)),
        ):
            catalog.add(_envelope(row))
        runtime_digest = _runtime().digest
        dense = catalog.select(
            workload=_workload((0, 1, 2, 3)),
            runtime_digest=runtime_digest,
            acceleration=0.0,
        )
        self.assertFalse(dense.accelerated)
        selected = catalog.select(
            workload=_workload((0, 1, 2, 3)),
            runtime_digest=runtime_digest,
            acceleration=25.0,
        )
        self.assertTrue(selected.accelerated)
        self.assertEqual(selected.candidate.predicted_cost_p90_ms, 80.0)
        fastest = catalog.select(
            workload=_workload((0, 1, 2, 3)),
            runtime_digest=runtime_digest,
            acceleration=100.0,
        )
        self.assertEqual(fastest.candidate.predicted_cost_p90_ms, 50.0)

    def test_actual_step_schedule_is_not_part_of_generation_request(self) -> None:
        registry = _registry()
        catalog = V19ReleaseFrontierCatalog(registry)
        catalog.add(_envelope(_entry(
            registry,
            actual=(0, 1, 3),
            cost=80.0,
            risk=_risk(0.15),
        )))
        selected = catalog.select(
            workload=_workload((0, 2, 3)),
            runtime_digest=_runtime().digest,
            acceleration=100.0,
        )
        self.assertTrue(selected.accelerated)
        self.assertEqual(selected.candidate.action_uses[0].step_indices, (0, 1, 3))

    def test_preview_constraint_selects_a_certified_slower_trajectory(self) -> None:
        registry = _registry()
        catalog = V19ReleaseFrontierCatalog(registry)
        for row in (
            _entry(registry, actual=(0, 1, 2, 3), cost=100.0, risk=_risk(0.10)),
            _entry(registry, actual=(0, 1, 3), cost=80.0, risk=_risk(0.15)),
        ):
            catalog.add(_envelope(row))
        selected = catalog.select(
            workload=_workload((0, 1, 3)),
            runtime_digest=_runtime().digest,
            acceleration=100.0,
            required_actual_step_indices=(2,),
        )
        self.assertTrue(selected.accelerated)
        self.assertEqual(selected.candidate.predicted_cost_p90_ms, 100.0)

    def test_ood_request_falls_back_without_reducing_input_capability(self) -> None:
        registry = _registry()
        catalog = V19ReleaseFrontierCatalog(registry)
        catalog.add(_envelope(_entry(
            registry,
            actual=(0, 1, 3),
            cost=80.0,
            risk=_risk(0.15),
        )))
        decision = catalog.select(
            workload=_workload((0, 1, 3), frames=362),
            runtime_digest=_runtime().digest,
            acceleration=100.0,
        )
        self.assertFalse(decision.accelerated)
        self.assertEqual(decision.reason, "uncalibrated_or_ood_dense_fallback")

    def test_incomparable_human_risks_cannot_be_hidden_by_scalar_dial(self) -> None:
        registry = _registry()
        catalog = V19ReleaseFrontierCatalog(registry)
        catalog.add(_envelope(_entry(
            registry,
            actual=(0, 1, 2, 3),
            cost=100.0,
            risk=_risk(0.10),
        )))
        with self.assertRaises(V19PlanningError):
            catalog.add(_envelope(_entry(
                registry,
                actual=(0, 1, 3),
                cost=80.0,
                risk=_risk(0.20, first=0.05),
            )))

    def test_incomparable_trajectory_debt_cannot_be_hidden_by_scalar_dial(self) -> None:
        registry = _registry()
        catalog = V19ReleaseFrontierCatalog(registry)
        catalog.add(_envelope(_entry(
            registry,
            actual=(0, 1, 2, 3),
            cost=100.0,
            risk=_risk(0.10),
            terminal_debt=V19TrajectoryDebt(
                forecast_debt=0.10,
                audio_debt=0.20,
            ),
            maximum_debt=V19TrajectoryDebt(
                consecutive_forecasts=1,
                forecast_debt=0.20,
                audio_debt=0.20,
            ),
        )))
        with self.assertRaisesRegex(V19PlanningError, "trajectory-debt"):
            catalog.add(_envelope(_entry(
                registry,
                actual=(0, 1, 3),
                cost=80.0,
                risk=_risk(0.15),
                # Faster is worse in Forecast debt but better in audio debt:
                # a one-dimensional dial must not conceal this exchange.
                terminal_debt=V19TrajectoryDebt(
                    forecast_debt=0.30,
                    audio_debt=0.10,
                ),
                maximum_debt=V19TrajectoryDebt(
                    consecutive_forecasts=2,
                    forecast_debt=0.40,
                    audio_debt=0.10,
                ),
            )))

    def test_uncalibrated_candidate_cannot_enter_release_frontier(self) -> None:
        registry = _registry()
        entry = _entry(
            registry,
            actual=(0, 1, 3),
            cost=80.0,
            risk=_risk(0.15),
        )
        unsafe = V19CertifiedFrontierEntry(
            planning_request=entry.planning_request,
            materialized=V19MaterializedCandidate(
                candidate=entry.candidate,
                end_to_end_cost_calibrated=True,
                human_risk_calibrated=False,
            ),
            certificate=entry.certificate,
        )
        with self.assertRaises(V19PlanningError):
            _envelope(unsafe)

    def test_self_consistent_certificate_cannot_hide_an_exceeded_budget(self) -> None:
        registry = _registry()
        entry = _entry(
            registry,
            actual=(0, 1, 3),
            cost=80.0,
            risk=_risk(0.15),
        )
        request = replace(entry.planning_request, maximum_cost_p90_ms=79.0)
        unsafe = V19CertifiedFrontierEntry(
            planning_request=request,
            materialized=entry.materialized,
            certificate=V19PlanCertificate.issue(
                registry, request, entry.candidate
            ),
        )
        with self.assertRaises(V19PlanningError):
            _envelope(unsafe)

    def test_release_bundle_round_trip_replays_admission_and_digest(self) -> None:
        registry = _registry()
        entry = _entry(
            registry,
            actual=(0, 1, 3),
            cost=80.0,
            risk=_risk(0.15),
        )
        envelope_entry = _envelope(entry)
        evidence = {
            "synthetic_cost": "5" * 64,
            "synthetic_forecast_cost": "7" * 64,
            "synthetic_risk": "6" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v19_release.json"
            save_v19_release_bundle(
                path,
                registry=registry,
                entries=(envelope_entry,),
                evidence_sha256=evidence,
            )
            loaded = load_v19_release_bundle(path, registry=registry)
            selected = loaded.catalog.select(
                workload=_workload((0, 2, 3)),
                runtime_digest=_runtime().digest,
                acceleration=100.0,
            )
        self.assertTrue(selected.accelerated)
        self.assertEqual(selected.candidate.execution_digest, entry.candidate.execution_digest)

    def test_release_bundle_requires_every_candidate_evidence_digest(self) -> None:
        registry = _registry()
        entry = _envelope(_entry(
            registry,
            actual=(0, 1, 3),
            cost=80.0,
            risk=_risk(0.15),
        ))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(V19PlanningError):
                save_v19_release_bundle(
                    Path(directory) / "v19_release.json",
                    registry=registry,
                    entries=(entry,),
                    evidence_sha256={"synthetic_cost": "5" * 64},
                )

    def test_closed_token_interval_accepts_only_observed_endpoints(self) -> None:
        registry = _registry()
        low = _entry(
            registry,
            actual=(0, 1, 3),
            cost=80.0,
            risk=_risk(0.12),
        )
        high_workload = _workload((0, 1, 3), packed_tokens=35_127)
        high_candidate = V19CandidatePlan(
            candidate_id="high_token_source",
            action_uses=low.candidate.action_uses,
            predicted_cost_p50_ms=84.0,
            predicted_cost_p90_ms=85.0,
            predicted_peak_vram_gib=9.5,
            risk_ucb=_risk(0.16),
            evidence_ids=low.candidate.evidence_ids,
        )
        high_request = V19PlanningRequest(
            workload=high_workload,
            runtime=_runtime(),
            maximum_cost_p90_ms=85.0,
            maximum_peak_vram_gib=24.0,
            risk_limits=_risk(0.16),
        )
        high = V19CertifiedFrontierEntry(
            planning_request=high_request,
            materialized=V19MaterializedCandidate(
                candidate=high_candidate,
                end_to_end_cost_calibrated=True,
                human_risk_calibrated=True,
            ),
            certificate=V19PlanCertificate.issue(
                registry, high_request, high_candidate
            ),
        )
        envelope = _envelope(low, high)
        self.assertEqual(envelope.candidate.predicted_cost_p90_ms, 85.0)
        self.assertEqual(envelope.candidate.risk_ucb, _risk(0.16))
        catalog = V19ReleaseFrontierCatalog(registry)
        catalog.add(envelope)
        inside = catalog.select(
            workload=_workload((0, 2, 3), packed_tokens=35_000),
            runtime_digest=_runtime().digest,
            acceleration=100.0,
        )
        outside = catalog.select(
            workload=_workload((0, 2, 3), packed_tokens=35_128),
            runtime_digest=_runtime().digest,
            acceleration=100.0,
        )
        self.assertTrue(inside.accelerated)
        self.assertFalse(outside.accelerated)

    def test_unobserved_token_endpoint_cannot_be_certified(self) -> None:
        registry = _registry()
        source = _entry(
            registry,
            actual=(0, 1, 3),
            cost=80.0,
            risk=_risk(0.12),
        )
        scope = V19WorkloadEnvelope(
            envelope_id="unsafe_gap",
            model_variant="base",
            service_family="first_last",
            device_arch="sm89",
            width=1280,
            height=736,
            frames=124,
            steps=4,
            sampler="res_multistep",
            scheduler="simple",
            min_packed_tokens=34_800,
            max_packed_tokens=34_900,
            reference_profiles=(V19ReferenceProfile(condition_count=0),),
        )
        with self.assertRaises(V19PlanningError):
            build_v19_certified_envelope_entry(
                registry,
                envelope=scope,
                sources=(source,),
            )

    def test_reference_profiles_do_not_create_unobserved_cross_products(self) -> None:
        scope = V19WorkloadEnvelope(
            envelope_id="reference_profile",
            model_variant="base",
            service_family="reference",
            device_arch="sm89",
            width=1280,
            height=736,
            frames=124,
            steps=4,
            sampler="res_multistep",
            scheduler="simple",
            min_packed_tokens=35_000,
            max_packed_tokens=36_000,
            reference_profiles=(
                V19ReferenceProfile(condition_count=2, images=1, audio=1),
            ),
        )
        calibrated = V19WorkloadContext(
            model_variant="base",
            service_family="reference",
            packed_tokens=35_500,
            condition_count=2,
            reference_images=1,
            reference_audio=1,
            width=1280,
            height=736,
            frames=124,
            steps=4,
            actual_step_indices=(0, 1, 3),
        )
        unobserved_mix = V19WorkloadContext(
            model_variant="base",
            service_family="reference",
            packed_tokens=35_500,
            condition_count=2,
            reference_images=2,
            width=1280,
            height=736,
            frames=124,
            steps=4,
            actual_step_indices=(0, 1, 3),
        )
        self.assertTrue(scope.contains(calibrated))
        self.assertFalse(scope.contains(unobserved_mix))

    def test_duplicate_exact_source_cannot_inflate_envelope_coverage(self) -> None:
        registry = _registry()
        source = _entry(
            registry,
            actual=(0, 1, 3),
            cost=80.0,
            risk=_risk(0.12),
        )
        with self.assertRaises(V19PlanningError):
            _envelope(source, source)


if __name__ == "__main__":
    unittest.main()
