from __future__ import annotations

import unittest

from h3serve.native_engine.planner import (
    ActionKind,
    ActionRegistry,
    DENSE_ACTION_ID,
    EvidenceStatus,
    FORECAST_ACTION_ID,
    ROUND229_ACTION_ID,
    RegisteredAction,
    V19ActionUse,
    V19CandidatePlan,
    V19CertifiedFrontierEntry,
    V19ForecastCompositeKey,
    V19ForecastUse,
    V19HumanRiskVector,
    V19MaterializedCandidate,
    V19PlanCertificate,
    V19PlanningRequest,
    V19ReferenceProfile,
    V19ReleaseFrontierCatalog,
    V19RuntimeFingerprint,
    V19RuntimeSelector,
    V19WorkloadContext,
    V19WorkloadEnvelope,
    build_v19_certified_envelope_entry,
)


def _registry() -> ActionRegistry:
    def action(
        action_id: str,
        kind: ActionKind,
        canonical: tuple[str, ...],
    ) -> RegisteredAction:
        return RegisteredAction(
            action_id=action_id,
            implementation_id=action_id + ".impl",
            kind=kind,
            executor_id=action_id + ".executor",
            canonical_actions=canonical,
            exact=False,
            evidence_status=EvidenceStatus.HUMAN_REVIEWED,
            calibration_ids=(action_id + ".cost",),
            risk_model_ids=("synthetic_risk",),
            planner_eligible=True,
        )

    return ActionRegistry((
        action(DENSE_ACTION_ID, ActionKind.DENSE_ATTENTION, ("dense",)),
        action(
            ROUND229_ACTION_ID,
            ActionKind.SPARSE_ATTENTION,
            ("sparse_topk_0.0625",),
        ),
        action(FORECAST_ACTION_ID, ActionKind.FORECAST_COMPOSITE, ("forecast",)),
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


def _workload(*, frames: int = 124) -> V19WorkloadContext:
    return V19WorkloadContext(
        model_variant="base",
        service_family="first_last",
        packed_tokens=34_871,
        condition_count=0,
        width=1280,
        height=736,
        frames=frames,
        steps=4,
        actual_step_indices=(0, 2, 3),
    )


def _catalog() -> tuple[V19ReleaseFrontierCatalog, V19RuntimeFingerprint]:
    registry = _registry()
    runtime = _runtime()
    workload = _workload()
    candidate = V19CandidatePlan(
        candidate_id="runtime_candidate",
        action_uses=(
            V19ActionUse(
                action_id=DENSE_ACTION_ID,
                canonical_action="dense",
                step_indices=(0, 2, 3),
            ),
            V19ForecastUse(
                action_id=FORECAST_ACTION_ID,
                composite_key=V19ForecastCompositeKey(
                    forecast_step_indices=(1,),
                    preceding_actual_step=0,
                    following_actual_step=2,
                    anchor_depth=3,
                    anchor_action_id=ROUND229_ACTION_ID,
                    anchor_canonical_action="sparse_topk_0.0625",
                    extrapolator_id="native_depth3_local_directional_v1",
                    correction_id="next_actual_full_stack_v1",
                ),
            ),
        ),
        predicted_cost_p50_ms=79.0,
        predicted_cost_p90_ms=80.0,
        predicted_peak_vram_gib=9.0,
        risk_ucb=V19HumanRiskVector(*(0.1 for _ in range(7))),
        evidence_ids=("e2e", "human"),
    )
    request = V19PlanningRequest(
        workload=workload,
        runtime=runtime,
        maximum_cost_p90_ms=80.0,
        maximum_peak_vram_gib=24.0,
        risk_limits=candidate.risk_ucb,
    )
    source = V19CertifiedFrontierEntry(
        planning_request=request,
        materialized=V19MaterializedCandidate(
            candidate=candidate,
            end_to_end_cost_calibrated=True,
            human_risk_calibrated=True,
        ),
        certificate=V19PlanCertificate.issue(registry, request, candidate),
    )
    envelope = build_v19_certified_envelope_entry(
        registry,
        envelope=V19WorkloadEnvelope(
            envelope_id="runtime_720p5",
            model_variant="base",
            service_family="first_last",
            device_arch="sm89",
            width=1280,
            height=736,
            frames=124,
            steps=4,
            sampler="res_multistep",
            scheduler="simple",
            min_packed_tokens=34_871,
            max_packed_tokens=34_871,
            reference_profiles=(V19ReferenceProfile(condition_count=0),),
        ),
        sources=(source,),
    )
    catalog = V19ReleaseFrontierCatalog(registry)
    catalog.add(envelope)
    return catalog, runtime


class V19RuntimeSelectorTests(unittest.TestCase):
    def test_certified_candidate_compiles_to_complete_hot_session_schedule(self) -> None:
        catalog, runtime = _catalog()
        selection = V19RuntimeSelector(
            catalog, runtime_digest=runtime.digest
        ).select(workload=_workload(), acceleration=100.0)
        self.assertTrue(selection.decision.accelerated)
        self.assertEqual(selection.actual_step_indices, (0, 2, 3))
        self.assertEqual(len(selection.attention_action_schedule), 153)
        self.assertEqual(selection.summary["forecast_steps"], 1)
        self.assertIn("certificate_digest", selection.summary)
        self.assertEqual(
            selection.summary["technique_mix"],
            {
                "actual_dit_evaluations": 3,
                "forecast_evaluations": 1,
                "actual_attention_cells": {"dense": 150},
                "forecast_anchor_attention_cells": {
                    "sparse_topk_0.0625": 3
                },
                "coupled_techniques": [
                    "exact_runtime",
                    "directional_forecast",
                    "block_sparse_attention",
                ],
            },
        )
        self.assertIn("maximum_debt", selection.summary)

    def test_ood_workload_compiles_to_dense_without_rejecting_request(self) -> None:
        catalog, runtime = _catalog()
        selection = V19RuntimeSelector(
            catalog, runtime_digest=runtime.digest
        ).select(workload=_workload(frames=362), acceleration=100.0)
        self.assertFalse(selection.decision.accelerated)
        self.assertEqual(selection.actual_step_indices, (0, 1, 2, 3))
        self.assertEqual(selection.attention_action_schedule, ())
        self.assertEqual(
            selection.summary["technique_mix"]["actual_attention_cells"],
            {"dense": 200},
        )
        self.assertEqual(
            selection.summary["reason"],
            "uncalibrated_or_ood_dense_fallback",
        )

    def test_uncalibrated_preview_anchor_falls_back_to_dense(self) -> None:
        catalog, runtime = _catalog()
        selection = V19RuntimeSelector(
            catalog, runtime_digest=runtime.digest
        ).select(
            workload=_workload(),
            acceleration=100.0,
            required_actual_step_indices=(1,),
        )
        self.assertFalse(selection.decision.accelerated)
        self.assertEqual(selection.actual_step_indices, (0, 1, 2, 3))
        self.assertEqual(
            selection.summary["reason"],
            "preview_anchor_uncalibrated_dense_fallback",
        )


if __name__ == "__main__":
    unittest.main()
