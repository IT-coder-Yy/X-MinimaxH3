from __future__ import annotations

from dataclasses import replace
import unittest
from pathlib import Path
import tempfile

from h3serve.native_engine.planner import (
    ActionKind,
    ActionRegistry,
    DENSE_ACTION_ID,
    EvidenceStatus,
    FORECAST_ACTION_ID,
    ROUND229_ACTION_ID,
    RegisteredAction,
    V19ActionUse,
    V19BudgetedCellOptimizer,
    V19BudgetedProposalRequest,
    V19CalibrationCatalog,
    V19CalibrationWorkload,
    V19CandidateBlueprint,
    V19CellImportanceProfile,
    V19CellAction,
    V19CoupledProposal,
    V19ForecastCalibrationCatalog,
    V19ForecastCompositeKey,
    V19ForecastCompositeMeasurement,
    V19ForecastUse,
    V19NumericalErrorSample,
    V19PlanningError,
    V19RuntimeFingerprint,
    V19SourceRecord,
    V19TimingMeasurement,
    V19TrajectoryDebt,
    couple_v19_proposal,
    create_v19_action_calibration,
    create_v19_forecast_calibration,
    v19_coupled_numerical_frontier,
    load_v19_candidate_blueprint,
    save_v19_candidate_blueprint,
    v19_av_clarity_importance_profile,
    v19_numerical_proposal_frontier,
)


DENSE_CALIBRATION = "v19_dense_full_head_cost_v1"
SPARSE_CALIBRATION = "v19_round229_attention_cost_error_v1"
FORECAST_CALIBRATION = "v19_round229_forecast_composite_cost_v1"


def _registry() -> ActionRegistry:
    return ActionRegistry((
        RegisteredAction(
            action_id=DENSE_ACTION_ID,
            implementation_id="sage_dense_per_warp_sm89_v1",
            kind=ActionKind.DENSE_ATTENTION,
            executor_id="dense",
            canonical_actions=("dense",),
            exact=False,
            evidence_status=EvidenceStatus.CALIBRATED,
            calibration_ids=(DENSE_CALIBRATION,),
            planner_eligible=True,
        ),
        RegisteredAction(
            action_id=FORECAST_ACTION_ID,
            implementation_id="forecast_test",
            kind=ActionKind.FORECAST_COMPOSITE,
            executor_id="directional",
            canonical_actions=("forecast",),
            exact=False,
            evidence_status=EvidenceStatus.CALIBRATED,
            calibration_ids=(FORECAST_CALIBRATION,),
            planner_eligible=True,
        ),
        RegisteredAction(
            action_id=ROUND229_ACTION_ID,
            implementation_id="round229_test",
            kind=ActionKind.SPARSE_ATTENTION,
            executor_id="forecastfrontier",
            canonical_actions=("sparse_topk_0.1", "sparse_topk_0.5"),
            exact=False,
            evidence_status=EvidenceStatus.CALIBRATED,
            calibration_ids=(SPARSE_CALIBRATION,),
            planner_eligible=True,
        ),
    ))


def _workload() -> V19CalibrationWorkload:
    return V19CalibrationWorkload(
        model_variant="base",
        service_family="first_last",
        width=1280,
        height=736,
        frames=124,
        packed_tokens=34_871,
        condition_count=0,
        steps=1,
        actual_step_indices=(0,),
    )


def _runtime() -> V19RuntimeFingerprint:
    return V19RuntimeFingerprint(
        gpu_name="NVIDIA GeForce RTX 4090",
        device_arch="sm89",
        torch_version="2.8.0+cu126",
        cuda_runtime="12.6",
        driver_version="560.94",
        quant_backend="cuda",
        comfy_kitchen_cuda_sha256="1" * 64,
        sageattention_sm89_sha256="2" * 64,
        action_source_sha256="3" * 64,
        planner_source_sha256="4" * 64,
    )


def _error(value: float) -> V19NumericalErrorSample:
    return V19NumericalErrorSample(
        mean_cosine=1.0 - value / 4.0,
        min_cosine=1.0 - value / 2.0,
        global_relative_rms=value,
        mean_head_relative_rms=value,
        max_head_relative_rms=value,
        max_relative_l1=value,
    )


def _catalog() -> V19CalibrationCatalog:
    registry = _registry()
    source = (V19SourceRecord(
        source_id="synthetic",
        relative_path="runtime/calibration/synthetic.json",
        sha256="5" * 64,
    ),)
    dense = create_v19_action_calibration(
        registry=registry,
        action_id=DENSE_ACTION_ID,
        calibration_id=DENSE_CALIBRATION,
        workload=_workload(),
        runtime=_runtime(),
        measurements=tuple(
            V19TimingMeasurement(
                canonical_action="dense",
                step_index=0,
                layer_start=layer,
                layer_stop=layer + 1,
                warm_samples_ms=(10.0, 10.0, 10.0),
                peak_vram_gib_samples=(9.0, 9.0, 9.0),
                numerical_error_samples=(_error(0.0),),
            )
            for layer in range(50)
        ),
        sources=source,
        timing_scope="attention_layer_call",
        complete=True,
    )
    sparse = create_v19_action_calibration(
        registry=registry,
        action_id=ROUND229_ACTION_ID,
        calibration_id=SPARSE_CALIBRATION,
        workload=_workload(),
        runtime=_runtime(),
        measurements=tuple(
            V19TimingMeasurement(
                canonical_action=canonical,
                step_index=0,
                layer_start=layer,
                layer_stop=layer + 1,
                warm_samples_ms=(cost, cost, cost),
                peak_vram_gib_samples=(8.0, 8.0, 8.0),
                numerical_error_samples=(_error(error),),
            )
            for layer in range(50)
            for canonical, cost, error in (
                ("sparse_topk_0.1", 3.0, 0.4),
                ("sparse_topk_0.5", 6.0, 0.1),
            )
        ),
        sources=source,
        timing_scope="attention_layer_call",
        complete=True,
    )
    catalog = V19CalibrationCatalog(registry)
    catalog.add(dense)
    catalog.add(sparse)
    return catalog


def _comparator() -> V19CandidateBlueprint:
    return V19CandidateBlueprint(
        candidate_id="reviewed_comparator",
        action_uses=(
            V19ActionUse(
                action_id=DENSE_ACTION_ID,
                canonical_action="dense",
                step_indices=(0,),
                layer_start=0,
                layer_stop=1,
            ),
            V19ActionUse(
                action_id=ROUND229_ACTION_ID,
                canonical_action="sparse_topk_0.5",
                step_indices=(0,),
                layer_start=1,
                layer_stop=50,
            ),
        ),
        source="reviewed",
    )


class V19OptimizerTests(unittest.TestCase):
    def _request(self, budget: float) -> V19BudgetedProposalRequest:
        return V19BudgetedProposalRequest(
            candidate_id="budgeted",
            comparator=_comparator(),
            workload=_workload(),
            runtime=_runtime(),
            maximum_attention_p90_ms=budget,
            actions=(
                V19CellAction(DENSE_ACTION_ID, "dense"),
                V19CellAction(ROUND229_ACTION_ID, "sparse_topk_0.5"),
                V19CellAction(ROUND229_ACTION_ID, "sparse_topk_0.1"),
            ),
            cost_quantum_ms=1.0,
        )

    def test_budgeted_dp_preserves_dense_rail_and_spends_budget_on_quality(self) -> None:
        proposal = V19BudgetedCellOptimizer(_catalog()).optimize(
            self._request(200.0)
        )
        self.assertEqual(proposal.protected_cell_count, 1)
        self.assertLessEqual(proposal.calibrated_attention_p90_ms, 200.0)
        self.assertEqual(proposal.conservative_quantized_p90_ms, 199.0)
        counts = {
            (action_id, canonical): count
            for action_id, canonical, count in proposal.action_cell_counts
        }
        self.assertEqual(counts[(DENSE_ACTION_ID, "dense")], 1)
        self.assertEqual(counts[(ROUND229_ACTION_ID, "sparse_topk_0.5")], 14)
        self.assertEqual(counts[(ROUND229_ACTION_ID, "sparse_topk_0.1")], 35)
        self.assertFalse(proposal.numerical_proxy_is_human_risk)
        self.assertEqual(len(proposal.proxy_component_sums), 6)
        self.assertEqual(len(proposal.proxy_component_maxima), 6)
        self.assertEqual(len(proposal.numerical_pareto_objective), 22)
        self.assertEqual(proposal.calibrated_attention_peak_vram_gib, 9.0)
        self.assertEqual(
            proposal.attention_evidence_ids,
            (DENSE_CALIBRATION, SPARSE_CALIBRATION),
        )
        self.assertTrue(all(
            total >= maximum
            for total, maximum in zip(
                proposal.proxy_component_sums,
                proposal.proxy_component_maxima,
            )
        ))

    def test_budget_below_physical_minimum_fails_closed(self) -> None:
        with self.assertRaises(V19PlanningError):
            V19BudgetedCellOptimizer(_catalog()).optimize(self._request(156.0))

    def test_numerical_frontier_prunes_only_strictly_dominated_proposal(self) -> None:
        base = V19BudgetedCellOptimizer(_catalog()).optimize(self._request(200.0))
        dominated = replace(
            base,
            blueprint=replace(base.blueprint, candidate_id="dominated"),
            calibrated_attention_p90_ms=base.calibrated_attention_p90_ms + 1.0,
            proxy_component_sums=tuple(
                value + 0.1 for value in base.proxy_component_sums
            ),
            proxy_component_maxima=tuple(
                value + 0.01 for value in base.proxy_component_maxima
            ),
        )
        tradeoff = replace(
            base,
            blueprint=replace(base.blueprint, candidate_id="tradeoff"),
            calibrated_attention_p50_ms=base.calibrated_attention_p50_ms - 1.0,
            calibrated_attention_p90_ms=base.calibrated_attention_p90_ms - 1.0,
            proxy_component_sums=tuple(
                value + 0.1 for value in base.proxy_component_sums
            ),
            proxy_component_maxima=tuple(
                value + 0.01 for value in base.proxy_component_maxima
            ),
        )
        frontier = v19_numerical_proposal_frontier(
            (dominated, tradeoff, base)
        )
        self.assertEqual(
            {proposal.blueprint.candidate_id for proposal in frontier},
            {"budgeted", "tradeoff"},
        )

    def test_coupled_frontier_prices_vram_and_trajectory_debt(self) -> None:
        base = V19BudgetedCellOptimizer(_catalog()).optimize(self._request(200.0))
        coupled = V19CoupledProposal(
            attention=base,
            workload_digest=_workload().digest,
            runtime_digest=_runtime().digest,
            forecast_p50_ms=0.0,
            forecast_p90_ms=0.0,
            physical_p50_ms=base.calibrated_attention_p50_ms,
            physical_p90_ms=base.calibrated_attention_p90_ms,
            peak_vram_gib=base.calibrated_attention_peak_vram_gib,
            evidence_ids=base.attention_evidence_ids,
        )
        dominated = replace(
            coupled,
            attention=replace(
                base,
                blueprint=replace(base.blueprint, candidate_id="dominated"),
            ),
            physical_p90_ms=coupled.physical_p90_ms + 1.0,
            peak_vram_gib=coupled.peak_vram_gib + 1.0,
        )
        frontier = v19_coupled_numerical_frontier((dominated, coupled))
        self.assertEqual(
            tuple(row.blueprint.candidate_id for row in frontier),
            ("budgeted",),
        )

    def test_coupling_adds_exact_forecast_composite_cost(self) -> None:
        base = V19BudgetedCellOptimizer(_catalog()).optimize(self._request(200.0))
        workload = replace(
            _workload(),
            steps=5,
            actual_step_indices=(0, 4),
        )
        key = V19ForecastCompositeKey(
            forecast_step_indices=(1, 2, 3),
            preceding_actual_step=0,
            following_actual_step=4,
            anchor_depth=3,
            anchor_action_id=ROUND229_ACTION_ID,
            anchor_canonical_action="sparse_topk_0.0625",
            extrapolator_id="native_depth3_local_directional_v1",
            correction_id="next_actual_full_stack_v1",
        )
        blueprint = V19CandidateBlueprint(
            candidate_id="coupled",
            action_uses=(
                V19ActionUse(
                    action_id=DENSE_ACTION_ID,
                    canonical_action="dense",
                    step_indices=(0, 4),
                    layer_start=0,
                    layer_stop=1,
                ),
                V19ActionUse(
                    action_id=ROUND229_ACTION_ID,
                    canonical_action="sparse_topk_0.5",
                    step_indices=(0, 4),
                    layer_start=1,
                    layer_stop=50,
                ),
                V19ForecastUse(action_id=FORECAST_ACTION_ID, composite_key=key),
            ),
            terminal_debt=V19TrajectoryDebt(
                consecutive_forecasts=3,
                forecast_debt=3.0,
                audio_debt=3.0,
                last_refresh_step=4,
            ),
            maximum_debt=V19TrajectoryDebt(
                consecutive_forecasts=3,
                forecast_debt=3.0,
                audio_debt=3.0,
                last_refresh_step=4,
            ),
        )
        proposal = replace(base, blueprint=blueprint)
        forecast_artifact = create_v19_forecast_calibration(
            registry=_registry(),
            action_id=FORECAST_ACTION_ID,
            calibration_id=FORECAST_CALIBRATION,
            workload=workload,
            runtime=_runtime(),
            measurements=(V19ForecastCompositeMeasurement(
                key=key,
                warm_samples_ms=(30.0, 31.0, 32.0),
                peak_vram_gib_samples=(10.0, 10.5, 10.25),
            ),),
            sources=(V19SourceRecord(
                source_id="forecast",
                relative_path="runtime/calibration/forecast.json",
                sha256="6" * 64,
            ),),
            complete=True,
        )
        catalog = V19ForecastCalibrationCatalog(_registry())
        catalog.add(forecast_artifact)
        coupled = couple_v19_proposal(
            proposal,
            workload=workload,
            runtime=_runtime(),
            forecast_catalog=catalog,
        )
        self.assertEqual(coupled.forecast_p90_ms, 32.0)
        self.assertEqual(
            coupled.physical_p90_ms,
            proposal.calibrated_attention_p90_ms + 32.0,
        )
        self.assertEqual(coupled.peak_vram_gib, 10.5)
        self.assertIn(FORECAST_CALIBRATION, coupled.evidence_ids)

    def test_cell_proxy_limit_is_noncompensating(self) -> None:
        request = self._request(304.0)
        request = V19BudgetedProposalRequest(
            candidate_id=request.candidate_id,
            comparator=request.comparator,
            workload=request.workload,
            runtime=request.runtime,
            maximum_attention_p90_ms=request.maximum_attention_p90_ms,
            actions=request.actions,
            cost_quantum_ms=request.cost_quantum_ms,
            maximum_cell_proxy=0.2,
        )
        proposal = V19BudgetedCellOptimizer(_catalog()).optimize(request)
        self.assertLessEqual(proposal.proxy_max, 0.2)
        self.assertEqual(proposal.calibrated_attention_p90_ms, 304.0)

    def test_importance_profile_spends_the_last_upgrade_on_important_cell(self) -> None:
        request = self._request(160.0)
        request = V19BudgetedProposalRequest(
            candidate_id=request.candidate_id,
            comparator=request.comparator,
            workload=request.workload,
            runtime=request.runtime,
            maximum_attention_p90_ms=request.maximum_attention_p90_ms,
            actions=request.actions,
            cost_quantum_ms=request.cost_quantum_ms,
            cell_importance=V19CellImportanceProfile(
                profile_id="synthetic_importance",
                step_weights=(1.0,),
                layer_weights=(1.0,) * 49 + (10.0,),
            ),
        )
        proposal = V19BudgetedCellOptimizer(_catalog()).optimize(request)
        high_fidelity_layers = {
            layer
            for use in proposal.blueprint.action_uses
            if isinstance(use, V19ActionUse)
            and use.canonical_action == "sparse_topk_0.5"
            for layer in range(use.layer_start, use.layer_stop)
        }
        self.assertEqual(high_fidelity_layers, {49})
        self.assertEqual(proposal.importance_profile_id, "synthetic_importance")
        self.assertGreater(
            proposal.importance_weighted_proxy_sum,
            proposal.proxy_sum,
        )

    def test_importance_profile_enforces_layer_keep_floors(self) -> None:
        request = self._request(304.0)
        request = V19BudgetedProposalRequest(
            candidate_id=request.candidate_id,
            comparator=request.comparator,
            workload=request.workload,
            runtime=request.runtime,
            maximum_attention_p90_ms=request.maximum_attention_p90_ms,
            actions=request.actions,
            cost_quantum_ms=request.cost_quantum_ms,
            cell_importance=V19CellImportanceProfile(
                profile_id="synthetic_floor",
                step_weights=(1.0,),
                layer_weights=(1.0,) * 50,
                minimum_layer_keep_ratios=(0.5,) * 50,
            ),
        )
        proposal = V19BudgetedCellOptimizer(_catalog()).optimize(request)
        counts = {
            canonical: count
            for _action_id, canonical, count in proposal.action_cell_counts
        }
        self.assertNotIn("sparse_topk_0.1", counts)
        self.assertEqual(counts["sparse_topk_0.5"], 49)

    def test_av_clarity_profile_keeps_critical_causal_layers_exact(self) -> None:
        profile = v19_av_clarity_importance_profile(20)
        self.assertEqual(profile.minimum_layer_keep_ratios[:30], (0.25,) * 30)
        self.assertEqual(profile.minimum_layer_keep_ratios[30:40], (0.5,) * 10)
        self.assertEqual(profile.minimum_layer_keep_ratios[40:44], (1.0,) * 4)
        self.assertEqual(profile.minimum_layer_keep_ratios[45], 1.0)
        self.assertEqual(profile.step_weights[18:], (2.0, 2.0))

    def test_blueprint_round_trip_verifies_execution_digest(self) -> None:
        blueprint = _comparator()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blueprint.json"
            save_v19_candidate_blueprint(path, blueprint)
            loaded = load_v19_candidate_blueprint(path)
        self.assertEqual(loaded, blueprint)


if __name__ == "__main__":
    unittest.main()
