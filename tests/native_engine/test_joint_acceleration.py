import unittest
from dataclasses import replace

import torch
from pathlib import Path

from h3serve.native_engine.planner.joint_acceleration import (
    H3JointAccelerationScheduler,
    JOINT_POLICY_V1_HEURISTIC,
    JOINT_POLICY_V2_EXACT_ATTENTION,
    JOINT_POLICY_V3_GLOBAL_DP,
    JOINT_POLICY_V4_EVIDENCE_GLOBAL_DP,
    JOINT_POLICY_V5_CALIBRATION_MATCHED_GLOBAL_DP,
    JOINT_POLICY_V6_CAUSAL_ISLAND_GLOBAL_DP,
    JOINT_POLICY_V7_LAYER_RISK_GLOBAL_DP,
    JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP,
    JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP,
    JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP,
    JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
    JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
    JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP,
    JOINT_POLICY_V14_TRAJECTORY_CORRECTION_GLOBAL_DP,
    JOINT_POLICY_V15_OPENING_ANCHORED_MTCR_GLOBAL_DP,
    JOINT_POLICY_V16_FRONTIER_DOMINANCE_GLOBAL_DP,
    JOINT_POLICY_V17_ZERO_TAX_FRONTIER_GLOBAL_DP,
    JOINT_POLICY_V18_FORECAST_AWARE_FRONTIER_GLOBAL_DP,
    FIXED_TOPK_ACTION_IMPLEMENTATION,
    ROUND215_ACTION_IMPLEMENTATION,
    ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ROUND216_CAUSAL_ISLAND_CONSTRAINT,
    ROUND224_ADAPTIVE_LATENCY_CONSTRAINT,
    ROUND225_TRAJECTORY_CORRECTION_CONSTRAINT,
    ROUND226_OPENING_ANCHORED_MTCR_CONSTRAINT,
    ROUND227_FRONTIER_DOMINANCE_CONSTRAINT,
    ROUND215_LAYER_RISK_MODEL,
    ROUND218_PHASE_LAYER_RISK_MODEL,
    ROUND219_BOUNDED_ONLINE_GUARD,
    ROUND220_PHASE_SENTINEL_GUARD,
    ROUND221_CALIBRATED_GROWTH_GUARD,
    ROUND223_RESERVE_REBATE_GUARD,
    JointWorkloadContext,
    JointAccelerationError,
    clear_joint_plan_cache,
    verify_joint_plan_certificate,
)
from h3serve.native_engine.planner.online_guard import (
    CalibratedPhaseGrowthGuard,
    ROUND221_PROBE_GROWTH_CALIBRATION,
    ROUND221_RUNTIME_GROWTH_THRESHOLD,
    allocate_phase_sentinels,
)
from h3serve.native_engine.model.kernels import (
    AttentionOnlineBudget,
    CausalCheckpointVerifierAttentionBackend,
    RequestActionScheduledAttentionBackend,
    SplitModalityProtectedSpargeAttentionBackend,
    _can_fallback_to_unstreamed_exact_attention,
    _resolve_long_sequence_physical_backend,
    attention_action_schedule,
    attention_actual_steps,
    attention_online_budget,
    attention_layer,
    attention_step,
    dense_qk_quantization,
    sage_attention_sm89,
)
from h3serve.native_engine.hot_session import HotSessionRequest, NativeT2AVHotSession


_RANK = {
    "sparse_topk_0.0625": 0,
    "sparse_topk_0.1": 1,
    "sparse_topk_0.25": 2,
    "sparse_topk_0.5": 3,
    "dense": 4,
}


class JointAccelerationSchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduler = H3JointAccelerationScheduler()

    def test_zero_acceleration_is_exact_dense_endpoint(self) -> None:
        plan = self.scheduler.plan(15, 0)
        self.assertEqual(plan.actual_step_indices, tuple(range(15)))
        self.assertEqual(plan.forecast_step_indices, ())
        self.assertEqual(
            set(plan.physical_action_schedule().values()), {"dense"}
        )
        self.assertEqual(plan.estimated_compute_ratio, 1.0)
        self.assertEqual(plan.estimated_risk_debt, 0.0)

    def test_fast_twenty_step_endpoint_preserves_reviewed_twelve_eight_spine(self) -> None:
        plan = H3JointAccelerationScheduler(
            policy_id=JOINT_POLICY_V2_EXACT_ATTENTION
        ).plan(20, 100)
        self.assertEqual(
            plan.actual_step_indices,
            (0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 18, 19),
        )
        self.assertEqual(plan.forecast_evaluations, 8)
        self.assertLess(plan.estimated_compute_ratio, 0.40)
        self.assertLessEqual(plan.estimated_compute_units, plan.target_compute_units)
        self.assertEqual(plan.policy_id, JOINT_POLICY_V2_EXACT_ATTENTION)
        self.assertIsNotNone(plan.attention_optimality_certificate)
        self.assertTrue(verify_joint_plan_certificate(plan).valid)

    def test_v3_globally_plans_shape_aware_trajectory_and_attention(self) -> None:
        plan = H3JointAccelerationScheduler(
            policy_id=JOINT_POLICY_V3_GLOBAL_DP
        ).plan(
            20,
            100,
            workload=JointWorkloadContext(packed_tokens=34_871),
        )
        self.assertEqual(plan.policy_id, JOINT_POLICY_V3_GLOBAL_DP)
        self.assertEqual(plan.actual_evaluations, 12)
        self.assertEqual(plan.forecast_evaluations, 8)
        self.assertIsNotNone(plan.global_optimality_certificate)
        self.assertIsNone(plan.attention_optimality_certificate)
        self.assertEqual(plan.workload_context.packed_tokens, 34_871)
        self.assertFalse(plan.workload_extrapolated)
        self.assertTrue(verify_joint_plan_certificate(plan).valid)

    def test_v3_global_certificate_rejects_tampering(self) -> None:
        plan = H3JointAccelerationScheduler(
            policy_id=JOINT_POLICY_V3_GLOBAL_DP
        ).plan(
            20,
            75,
            workload=JointWorkloadContext(packed_tokens=34_871),
        )
        corrupted = replace(
            plan,
            global_optimality_certificate=replace(
                plan.global_optimality_certificate,
                choice_sha256="0" * 64,
            ),
        )
        verification = verify_joint_plan_certificate(corrupted)
        self.assertFalse(verification.valid)
        self.assertIn("global certificate mismatch", verification.reasons)

    def test_v3_cost_model_distinguishes_short_and_long_shapes(self) -> None:
        scheduler = H3JointAccelerationScheduler(policy_id=JOINT_POLICY_V3_GLOBAL_DP)
        short = scheduler.plan(
            20, 75, workload=JointWorkloadContext(packed_tokens=34_871)
        )
        long = scheduler.plan(
            20, 75, workload=JointWorkloadContext(packed_tokens=100_163)
        )
        self.assertEqual(short.workload_calibration_mix, 0.0)
        self.assertEqual(long.workload_calibration_mix, 1.0)
        self.assertLess(short.predicted_compute_ms, long.predicted_compute_ms)
        self.assertNotEqual(
            short.global_optimality_certificate.model_sha256,
            long.global_optimality_certificate.model_sha256,
        )

    def test_v4_fast_endpoint_prefers_human_supported_trajectory(self) -> None:
        plan = H3JointAccelerationScheduler(
            policy_id=JOINT_POLICY_V4_EVIDENCE_GLOBAL_DP
        ).plan(
            20,
            100,
            workload=JointWorkloadContext(packed_tokens=34_871),
        )
        self.assertEqual(plan.policy_id, JOINT_POLICY_V4_EVIDENCE_GLOBAL_DP)
        self.assertEqual(
            plan.attention_implementation_id,
            FIXED_TOPK_ACTION_IMPLEMENTATION,
        )
        self.assertEqual(
            plan.actual_step_indices,
            (0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 18, 19),
        )
        self.assertEqual(
            plan.trajectory_prior_id,
            "round143_round216_human_positive_20step_12a8f_v1",
        )
        self.assertTrue(verify_joint_plan_certificate(plan).valid)

    def test_v4_retains_global_search_instead_of_freezing_one_path(self) -> None:
        scheduler = H3JointAccelerationScheduler(
            policy_id=JOINT_POLICY_V4_EVIDENCE_GLOBAL_DP
        )
        moderate = scheduler.plan(
            20,
            50,
            workload=JointWorkloadContext(packed_tokens=34_871),
        )
        fast = scheduler.plan(
            20,
            100,
            workload=JointWorkloadContext(packed_tokens=34_871),
        )
        self.assertGreater(moderate.actual_evaluations, fast.actual_evaluations)
        self.assertNotEqual(moderate.actual_step_indices, fast.actual_step_indices)
        self.assertEqual(moderate.trajectory_prior_id, fast.trajectory_prior_id)

    def test_v4_does_not_transfer_base_forecast_prior_to_lora(self) -> None:
        plan = H3JointAccelerationScheduler(
            policy_id=JOINT_POLICY_V4_EVIDENCE_GLOBAL_DP
        ).plan(
            8,
            100,
            allow_forecast=False,
            workload=JointWorkloadContext(
                packed_tokens=34_871,
                model_variant="lora",
            ),
        )
        self.assertIsNone(plan.trajectory_prior_id)
        self.assertEqual(plan.actual_step_indices, tuple(range(8)))
        self.assertTrue(verify_joint_plan_certificate(plan).valid)

    def test_v5_binds_round215_calibration_to_runtime_actions(self) -> None:
        plan = H3JointAccelerationScheduler(
            policy_id=JOINT_POLICY_V5_CALIBRATION_MATCHED_GLOBAL_DP
        ).plan(
            20,
            100,
            workload=JointWorkloadContext(packed_tokens=34_871),
        )
        self.assertEqual(
            plan.policy_id,
            JOINT_POLICY_V5_CALIBRATION_MATCHED_GLOBAL_DP,
        )
        self.assertEqual(
            plan.attention_implementation_id,
            ROUND215_ACTION_IMPLEMENTATION,
        )
        self.assertEqual(
            plan.global_optimality_certificate.action_implementation_id,
            ROUND215_ACTION_IMPLEMENTATION,
        )
        canonical = plan.physical_action_schedule()
        runtime = plan.runtime_action_schedule()
        self.assertEqual(canonical.keys(), runtime.keys())
        self.assertTrue(
            all(
                runtime[cell] == action
                if action == "dense"
                else runtime[cell] == f"round215:{action}"
                for cell, action in canonical.items()
            )
        )
        self.assertTrue(verify_joint_plan_certificate(plan).valid)

    def test_v6_binds_hybrid_actions_and_exact_causal_island(self) -> None:
        plan = H3JointAccelerationScheduler(
            policy_id=JOINT_POLICY_V6_CAUSAL_ISLAND_GLOBAL_DP
        ).plan(
            20,
            100,
            workload=JointWorkloadContext(packed_tokens=34_871),
        )
        self.assertEqual(plan.policy_id, JOINT_POLICY_V6_CAUSAL_ISLAND_GLOBAL_DP)
        self.assertEqual(plan.attention_implementation_id, ROUND215_ACTION_IMPLEMENTATION)
        self.assertEqual(plan.quality_constraint_id, ROUND216_CAUSAL_ISLAND_CONSTRAINT)
        self.assertEqual(
            plan.global_optimality_certificate.quality_constraint_id,
            ROUND216_CAUSAL_ISLAND_CONSTRAINT,
        )
        schedule = plan.physical_action_schedule()
        causal_layers = set((*range(30, 44), 45))
        for (step, layer), action in schedule.items():
            if step == 0 or layer in causal_layers:
                self.assertEqual(action, "dense")
        self.assertTrue(verify_joint_plan_certificate(plan).valid)

    def test_v7_allocates_physical_layers_under_same_hard_island(self) -> None:
        plan = self.scheduler.plan(
            20,
            100,
            workload=JointWorkloadContext(packed_tokens=34_871),
        )
        self.assertEqual(plan.policy_id, JOINT_POLICY_V7_LAYER_RISK_GLOBAL_DP)
        self.assertEqual(plan.risk_model_id, ROUND215_LAYER_RISK_MODEL)
        self.assertEqual(plan.global_optimality_certificate.risk_model_id, ROUND215_LAYER_RISK_MODEL)
        self.assertTrue(all(
            decision.layer_stop == decision.layer_start + 1
            for decision in plan.attention_decisions
        ))
        self.assertEqual(len(plan.attention_decisions), plan.actual_evaluations * 50)
        self.assertTrue(verify_joint_plan_certificate(plan).valid)

    def test_v8_allocates_layers_from_measured_trajectory_anchors(self) -> None:
        plan = H3JointAccelerationScheduler(
            policy_id=JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP
        ).plan(
            20,
            100,
            workload=JointWorkloadContext(packed_tokens=100_163),
        )
        self.assertEqual(
            plan.policy_id, JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP
        )
        self.assertEqual(plan.risk_model_id, ROUND218_PHASE_LAYER_RISK_MODEL)
        self.assertEqual(
            plan.global_optimality_certificate.risk_model_id,
            ROUND218_PHASE_LAYER_RISK_MODEL,
        )
        self.assertEqual(len(plan.attention_decisions), plan.actual_evaluations * 50)
        self.assertTrue(all(
            decision.layer_stop == decision.layer_start + 1
            for decision in plan.attention_decisions
        ))
        self.assertTrue(verify_joint_plan_certificate(plan).valid)

    def test_v9_binds_guard_without_changing_the_offline_optimum(self) -> None:
        workload = JointWorkloadContext(packed_tokens=34_871)
        v8 = H3JointAccelerationScheduler(
            policy_id=JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP
        ).plan(20, 75, workload=workload)
        v9 = H3JointAccelerationScheduler(
            policy_id=JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP
        ).plan(20, 75, workload=workload)
        self.assertEqual(v9.online_guard_id, ROUND219_BOUNDED_ONLINE_GUARD)
        self.assertGreater(v9.online_recovery_reserve_units, 0.0)
        self.assertEqual(v9.actual_step_indices, v8.actual_step_indices)
        self.assertEqual(v9.physical_action_schedule(), v8.physical_action_schedule())
        self.assertTrue(verify_joint_plan_certificate(v9).valid)

    def test_v10_preserves_v8_offline_optimum_and_binds_phase_guard(self) -> None:
        workload = JointWorkloadContext(packed_tokens=34_871)
        v8 = H3JointAccelerationScheduler(
            policy_id=JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP
        ).plan(20, 100, workload=workload)
        v10 = H3JointAccelerationScheduler(
            policy_id=JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP
        ).plan(20, 100, workload=workload)
        self.assertEqual(v10.online_guard_id, ROUND220_PHASE_SENTINEL_GUARD)
        self.assertEqual(v10.actual_step_indices, v8.actual_step_indices)
        self.assertEqual(v10.physical_action_schedule(), v8.physical_action_schedule())
        self.assertTrue(verify_joint_plan_certificate(v10).valid)

    def test_v10_phase_probe_slots_cover_phases_and_preserve_repair_budget(self) -> None:
        actual = (0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 18, 19)
        low = CausalCheckpointVerifierAttentionBackend._phase_probe_slots(
            actual, 5.75
        )
        high = CausalCheckpointVerifierAttentionBackend._phase_probe_slots(
            actual, 21.15
        )
        self.assertEqual(low, ((8, 24),))
        self.assertEqual(len(high), 9)
        self.assertEqual(
            {step for step, layer in high if layer == 24}, {3, 8, 14}
        )
        self.assertGreaterEqual(21.15 - len(high), 5.0)

        allocation = allocate_phase_sentinels(actual, 21.15)
        self.assertEqual(allocation.slots, high)
        self.assertTrue(allocation.budget_respected)
        self.assertEqual(allocation.observation_dense_layers, 9.0)
        self.assertAlmostEqual(allocation.remaining_dense_layers, 12.15)
        self.assertEqual(allocation.required_remaining_dense_layers, 5.0)

    def test_v11_preserves_v10_plan_and_binds_calibrated_growth_guard(self) -> None:
        workload = JointWorkloadContext(packed_tokens=34_871)
        v10 = H3JointAccelerationScheduler(
            policy_id=JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP
        ).plan(20, 100, workload=workload)
        v11 = H3JointAccelerationScheduler(
            policy_id=JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP
        ).plan(20, 100, workload=workload)
        self.assertEqual(v11.online_guard_id, ROUND221_CALIBRATED_GROWTH_GUARD)
        self.assertEqual(v11.actual_step_indices, v10.actual_step_indices)
        self.assertEqual(v11.physical_action_schedule(), v10.physical_action_schedule())
        self.assertTrue(verify_joint_plan_certificate(v11).valid)

    def test_v11_calibration_and_request_relative_growth_are_fail_closed(self) -> None:
        calibration = ROUND221_PROBE_GROWTH_CALIBRATION
        self.assertEqual(calibration.task_count, 14)
        self.assertEqual(calibration.probe_domain, "per_frame_spatial_anchors")
        self.assertAlmostEqual(
            calibration.runtime_growth_threshold,
            calibration.observed_max_task_score * calibration.safety_margin,
        )
        guard = CalibratedPhaseGrowthGuard(ROUND221_RUNTIME_GROWTH_THRESHOLD)
        first_request = object()
        self.assertTrue(guard.begin_request(first_request))
        baseline = guard.observe(4, 0.20)
        safe = guard.observe(4, 0.24)
        unsafe = guard.observe(4, 0.26)
        independent_layer = guard.observe(24, 0.50)
        self.assertFalse(baseline.triggered)
        self.assertFalse(safe.triggered)
        self.assertTrue(unsafe.triggered)
        self.assertAlmostEqual(unsafe.growth_ratio, 1.30)
        self.assertEqual(independent_layer.growth_ratio, 1.0)
        self.assertFalse(guard.begin_request(first_request))
        second_request = object()
        self.assertTrue(guard.begin_request(second_request))
        reset = guard.observe(4, 0.75)
        self.assertEqual(reset.growth_ratio, 1.0)
        self.assertFalse(reset.triggered)

    def test_v12_rebates_unused_reserve_with_replayable_topk_certificate(self) -> None:
        workload = JointWorkloadContext(
            packed_tokens=100_163,
            service_family="first_last",
            model_variant="base",
        )
        v11 = H3JointAccelerationScheduler(
            policy_id=JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP
        ).plan(20, 100, workload=workload)
        v12 = H3JointAccelerationScheduler(
            policy_id=JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP
        ).plan(20, 100, workload=workload)
        self.assertEqual(v12.online_guard_id, ROUND223_RESERVE_REBATE_GUARD)
        self.assertEqual(v12.actual_step_indices, v11.actual_step_indices)
        self.assertEqual(v12.physical_action_schedule(), v11.physical_action_schedule())
        self.assertEqual(len(v12.online_rebate_schedule), 11)
        self.assertTrue(all(step > 14 for step, _ in v12.online_rebate_schedule))
        self.assertEqual(
            v12.online_rebate_certificate.solver,
            "exact equal-charge top-k conditional allocation",
        )
        self.assertEqual(
            v12.online_rebate_certificate.selected_count,
            len(v12.online_rebate_schedule),
        )
        self.assertTrue(verify_joint_plan_certificate(v12).valid)

        corrupted = replace(
            v12,
            online_rebate_schedule=v12.online_rebate_schedule[:-1],
        )
        verification = verify_joint_plan_certificate(corrupted)
        self.assertFalse(verification.valid)
        self.assertIn("online rebate schedule mismatch", verification.reasons)

    def test_v13_independently_solves_the_adaptive_latency_frontier(self) -> None:
        scheduler = H3JointAccelerationScheduler(
            policy_id=JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP
        )
        workload = JointWorkloadContext(
            packed_tokens=100_163,
            service_family="first_last",
            model_variant="base",
        )
        moderate = scheduler.plan(20, 75, workload=workload)
        fast = scheduler.plan(20, 100, workload=workload)

        self.assertIsNone(fast.trajectory_prior_id)
        self.assertEqual(
            fast.quality_constraint_id,
            ROUND224_ADAPTIVE_LATENCY_CONSTRAINT,
        )
        self.assertEqual(fast.online_guard_id, ROUND221_CALIBRATED_GROWTH_GUARD)
        self.assertEqual(fast.actual_evaluations + fast.forecast_evaluations, 20)
        self.assertGreater(moderate.actual_evaluations, fast.actual_evaluations)
        self.assertNotEqual(moderate.actual_step_indices, fast.actual_step_indices)
        self.assertLessEqual(fast.target_compute_ms, 177_000.0)
        self.assertLessEqual(fast.predicted_compute_ms, fast.target_compute_ms)
        self.assertLessEqual(fast.estimated_compute_ratio, 0.22)
        self.assertTrue(verify_joint_plan_certificate(fast).valid)

    def test_v14_couples_forecast_debt_to_exact_causal_correction(self) -> None:
        scheduler = H3JointAccelerationScheduler(
            policy_id=JOINT_POLICY_V14_TRAJECTORY_CORRECTION_GLOBAL_DP
        )
        workload = JointWorkloadContext(
            packed_tokens=100_163,
            service_family="first_last",
            model_variant="base",
        )
        plan = scheduler.plan(20, 100, workload=workload)

        self.assertIsNone(plan.trajectory_prior_id)
        self.assertEqual(
            plan.quality_constraint_id,
            ROUND225_TRAJECTORY_CORRECTION_CONSTRAINT,
        )
        self.assertGreaterEqual(plan.actual_evaluations, 8)
        run = 0
        longest = 0
        actions = {
            (decision.step_index, layer): decision.action
            for decision in plan.attention_decisions
            for layer in range(decision.layer_start, decision.layer_stop)
        }
        for step in range(plan.total_steps):
            if step in plan.forecast_step_indices:
                run += 1
                longest = max(longest, run)
                continue
            if run == 2:
                for layer in (*range(30, 44), 45):
                    self.assertEqual(actions[(step, layer)], "dense")
            run = 0
        self.assertLessEqual(longest, 2)
        self.assertGreater(plan.predicted_compute_ms, 200_000.0)
        self.assertLessEqual(plan.predicted_compute_ms, plan.target_compute_ms)
        self.assertTrue(verify_joint_plan_certificate(plan).valid)

    def test_v15_spends_real_compute_on_opening_and_mtcr_on_later_debt(self) -> None:
        scheduler = H3JointAccelerationScheduler(
            policy_id=JOINT_POLICY_V15_OPENING_ANCHORED_MTCR_GLOBAL_DP
        )
        workload = JointWorkloadContext(
            packed_tokens=100_163,
            service_family="first_last",
            model_variant="base",
        )
        plan = scheduler.plan(20, 100, workload=workload)

        self.assertEqual(
            plan.quality_constraint_id,
            ROUND226_OPENING_ANCHORED_MTCR_CONSTRAINT,
        )
        self.assertEqual(plan.actual_step_indices[:5], (0, 1, 2, 3, 4))
        self.assertGreaterEqual(plan.actual_evaluations, 9)
        run = 0
        longest = 0
        for step in range(plan.total_steps):
            if step in plan.forecast_step_indices:
                run += 1
                longest = max(longest, run)
            else:
                run = 0
        self.assertLessEqual(longest, 4)
        self.assertLess(plan.predicted_compute_ms, 230_000.0)
        self.assertLess(plan.predicted_compute_ms, 0.85 * 269_351.1)
        self.assertTrue(verify_joint_plan_certificate(plan).valid)

    def test_v16_searches_the_human_frontier_head_rail_with_a_ten_step_floor(self) -> None:
        scheduler = H3JointAccelerationScheduler(
            policy_id=JOINT_POLICY_V16_FRONTIER_DOMINANCE_GLOBAL_DP
        )
        workload = JointWorkloadContext(
            packed_tokens=100_163,
            service_family="first_last",
            model_variant="base",
        )
        plan = scheduler.plan(20, 100, workload=workload)

        self.assertEqual(
            plan.quality_constraint_id,
            ROUND227_FRONTIER_DOMINANCE_CONSTRAINT,
        )
        self.assertEqual(
            plan.attention_implementation_id,
            ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
        )
        self.assertEqual(plan.actual_step_indices[:5], (0, 1, 2, 3, 4))
        self.assertGreaterEqual(plan.actual_evaluations, 10)
        run = 0
        longest = 0
        for step in range(plan.total_steps):
            if step in plan.forecast_step_indices:
                run += 1
                longest = max(longest, run)
            else:
                run = 0
        self.assertLessEqual(longest, 3)
        runtime_actions = set(plan.runtime_action_schedule().values())
        self.assertTrue(runtime_actions)
        self.assertTrue(
            all(action == "dense" or action.startswith("frontier:") for action in runtime_actions)
        )
        self.assertLess(plan.predicted_compute_ms, 240_000.0)
        self.assertTrue(verify_joint_plan_certificate(plan).valid)

    def test_v17_removes_teacher_tax_but_preserves_the_v16_search_space(self) -> None:
        scheduler = H3JointAccelerationScheduler(
            policy_id=JOINT_POLICY_V17_ZERO_TAX_FRONTIER_GLOBAL_DP
        )
        workload = JointWorkloadContext(
            packed_tokens=100_163,
            service_family="first_last",
            model_variant="base",
        )
        plan = scheduler.plan(20, 100, workload=workload)

        self.assertEqual(
            plan.quality_constraint_id,
            ROUND227_FRONTIER_DOMINANCE_CONSTRAINT,
        )
        self.assertEqual(
            plan.attention_implementation_id,
            ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
        )
        self.assertIsNone(plan.online_guard_id)
        self.assertEqual(plan.actual_step_indices[:5], (0, 1, 2, 3, 4))
        self.assertGreaterEqual(plan.actual_evaluations, 10)
        runtime_actions = set(plan.runtime_action_schedule().values())
        self.assertTrue(runtime_actions)
        self.assertTrue(
            all(
                action == "dense" or action.startswith("fastfrontier:")
                for action in runtime_actions
            )
        )
        self.assertEqual(len(plan.runtime_action_schedule()), 10 * 50)
        self.assertTrue(verify_joint_plan_certificate(plan).valid)

    def test_v18_routes_the_three_forecast_anchor_blocks_on_the_reviewed_rail(self) -> None:
        scheduler = H3JointAccelerationScheduler(
            policy_id=JOINT_POLICY_V18_FORECAST_AWARE_FRONTIER_GLOBAL_DP
        )
        workload = JointWorkloadContext(
            packed_tokens=100_163,
            service_family="first_last",
            model_variant="base",
        )
        plan = scheduler.plan(20, 100, workload=workload)

        self.assertEqual(
            plan.attention_implementation_id,
            ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
        )
        self.assertIsNone(plan.online_guard_id)
        runtime = plan.runtime_action_schedule()
        self.assertEqual(len(runtime), 10 * 50 + 10 * 3)
        self.assertTrue(
            all(
                runtime[(step, layer)]
                == "forecastfrontier:sparse_topk_0.0625"
                for step in plan.forecast_step_indices
                for layer in range(3)
            )
        )
        self.assertTrue(verify_joint_plan_certificate(plan).valid)

    def test_v10_phase_allocation_is_total_and_budget_safe(self) -> None:
        actual = (0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 18, 19)
        for quarter in range(0, 161):
            limit = quarter / 4.0
            allocation = allocate_phase_sentinels(actual, limit)
            self.assertTrue(allocation.budget_respected, limit)
            self.assertLessEqual(len(allocation.slots), 9)
            self.assertEqual(len(allocation.slots), len(set(allocation.slots)))
            self.assertTrue(all(step in actual for step, _ in allocation.slots))
            self.assertTrue(
                all(layer in (4, 24, 44) for _, layer in allocation.slots)
            )

    def test_online_ledger_is_upgrade_only_and_never_overspends(self) -> None:
        ledger = AttentionOnlineBudget("test_guard", 2.0)
        self.assertTrue(ledger.try_spend(1.0, kind="probe", step=3, layer=4))
        self.assertTrue(
            ledger.try_spend(1.0, kind="trigger_upgrade", step=3, layer=4)
        )
        self.assertFalse(
            ledger.try_spend(1.0, kind="recovery", step=3, layer=10)
        )
        telemetry = ledger.telemetry()
        self.assertTrue(telemetry["budget_respected"])
        self.assertTrue(telemetry["upgrade_only"])
        self.assertEqual(telemetry["spent_dense_layers"], 2.0)
        self.assertEqual(telemetry["denied_count"], 1)

        checkpoint = ledger.checkpoint_state()
        restored = AttentionOnlineBudget("test_guard", 2.0)
        restored.restore_checkpoint_state(checkpoint)
        self.assertEqual(restored.telemetry(), telemetry)
        corrupted = dict(checkpoint)
        corrupted["spent_dense_layers"] = 1.0
        with self.assertRaisesRegex(ValueError, "spent total mismatch"):
            AttentionOnlineBudget("test_guard", 2.0).restore_checkpoint_state(
                corrupted
            )

    def test_v6_complete_bundle_frontier_retains_true_cheapest_bundle(self) -> None:
        from h3serve.native_engine.planner import joint_global_dp as global_dp

        profile = global_dp._profile(JointWorkloadContext(packed_tokens=34_871))
        frontier = global_dp._bundle_frontier(
            profile,
            "ordinary",
            False,
            ROUND216_CAUSAL_ISLAND_CONSTRAINT,
        )
        cheapest_actions = (
            "sparse_topk_0.0625",
            "sparse_topk_0.0625",
            "dense",
            "dense",
            "sparse_topk_0.0625",
            "dense",
            "sparse_topk_0.0625",
        )
        cheapest_cost = sum(
            global_dp._action(profile, band, stop - start, action, "ordinary").cost_ms
            for action, (band, start, stop) in zip(cheapest_actions, global_dp.LAYER_BANDS)
        )
        cheapest_units = global_dp._ceil_units(cheapest_cost, profile.quantum_ms)
        # The quantised solver may choose a lower-risk combination inside the
        # same unit, but it must not lose the true cheapest unit itself.
        self.assertEqual(frontier[0].conservative_units, cheapest_units)

    def test_v1_heuristic_is_retained_as_a_reproducible_comparator(self) -> None:
        plan = H3JointAccelerationScheduler(
            policy_id=JOINT_POLICY_V1_HEURISTIC
        ).plan(20, 73)
        self.assertEqual(plan.policy_id, JOINT_POLICY_V1_HEURISTIC)
        self.assertIsNone(plan.attention_optimality_certificate)
        self.assertIn("retained v1", plan.formal_optimality_scope)

    def test_identical_request_reuses_immutable_cached_plan(self) -> None:
        clear_joint_plan_cache()
        first = H3JointAccelerationScheduler().plan(15, 63.4)
        second = H3JointAccelerationScheduler().plan(15, 63.4)
        self.assertIs(first, second)

    def test_acceleration_budget_is_monotone(self) -> None:
        plans = [self.scheduler.plan(20, value) for value in range(0, 101, 10)]
        targets = [plan.target_compute_units for plan in plans]
        estimates = [plan.estimated_compute_units for plan in plans]
        actuals = [plan.actual_evaluations for plan in plans]
        self.assertEqual(targets, sorted(targets, reverse=True))
        self.assertEqual(estimates, sorted(estimates, reverse=True))
        self.assertEqual(actuals, sorted(actuals, reverse=True))

    def test_internal_quality_floors_cannot_be_disabled(self) -> None:
        plan = self.scheduler.plan(20, 100)
        schedule = plan.physical_action_schedule()
        terminal = frozenset(plan.actual_step_indices[-3:])
        causal = frozenset((*range(30, 44), 45))
        for (step, layer), action in schedule.items():
            rank = _RANK[action]
            if step == plan.actual_step_indices[0]:
                self.assertGreaterEqual(rank, 3 if layer in causal else 2)
            if step in terminal:
                self.assertGreaterEqual(rank, 2)
            if layer in causal:
                self.assertGreaterEqual(rank, 1)
        for previous, current in zip(
            plan.actual_step_indices, plan.actual_step_indices[1:]
        ):
            if current - previous >= 3:
                for layer in causal:
                    self.assertGreaterEqual(_RANK[schedule[(current, layer)]], 2)

    def test_every_actual_step_has_all_fifty_attention_layers(self) -> None:
        plan = self.scheduler.plan(15, 67)
        schedule = plan.physical_action_schedule()
        self.assertEqual(len(schedule), plan.actual_evaluations * 50)
        for step in plan.actual_step_indices:
            self.assertEqual(
                {layer for candidate_step, layer in schedule if candidate_step == step},
                set(range(50)),
            )

    def test_lora_policy_can_disable_forecast_without_adding_a_third_knob(self) -> None:
        plan = self.scheduler.plan(8, 100, allow_forecast=False)
        self.assertEqual(plan.actual_step_indices, tuple(range(8)))
        self.assertEqual(plan.forecast_step_indices, ())
        self.assertTrue(plan.uses_sparse_attention)

    def test_invalid_controls_fail_before_runtime(self) -> None:
        for steps, acceleration in ((3, 50), (31, 50), (20, -1), (20, 101)):
            with self.assertRaises(JointAccelerationError):
                self.scheduler.plan(steps, acceleration)


class JointAccelerationRuntimeRoutingTests(unittest.TestCase):
    @staticmethod
    def _record(name, calls):
        def backend(query, key, value):
            del key, value
            calls.append(name)
            return query

        return backend

    def test_hot_backend_reads_request_local_schedule(self) -> None:
        calls: list[str] = []
        backend = RequestActionScheduledAttentionBackend(
            {
                "dense": self._record("dense", calls),
                "sparse_topk_0.1": self._record("sparse", calls),
            }
        )
        tensor = torch.zeros(256, 2, 4)
        with (
            attention_action_schedule({(3, 17): "sparse_topk_0.1"}),
            attention_step(3, 20),
            attention_layer(17),
        ):
            backend(tensor, tensor, tensor)
        with (
            attention_action_schedule({(3, 17): "sparse_topk_0.1"}),
            attention_step(3, 20),
            attention_layer(18),
        ):
            backend(tensor, tensor, tensor)
        self.assertEqual(calls, ["sparse", "dense"])

    def test_hot_backend_exposes_only_opted_in_physical_telemetry(self) -> None:
        class TelemetryBackend:
            def __init__(self, enabled: bool) -> None:
                self.enabled = enabled

            def __call__(self, query, key, value):
                del key, value
                return query

            def telemetry(self):
                return {
                    "route_probe_enabled": self.enabled,
                    "route_probe_records": [],
                }

        backend = RequestActionScheduledAttentionBackend(
            {
                "dense": TelemetryBackend(False),
                "sparse_topk_0.1": TelemetryBackend(True),
            }
        )
        self.assertEqual(
            set(backend.telemetry()["physical_action_telemetry"]),
            {"sparse_topk_0.1"},
        )

    def test_route_probe_observes_actual_steps_and_resets_at_request_boundary(self) -> None:
        backend = SplitModalityProtectedSpargeAttentionBackend(
            0.1,
            experimental_minimum_topk=0.0625,
            route_probe=True,
        )
        block_map = torch.ones((1, 2, 4, 5), dtype=torch.bool)
        with attention_actual_steps((0, 2)):
            with attention_step(0, 4), attention_layer(7):
                backend._record_route_probe(block_map)
            with attention_step(1, 4), attention_layer(7):
                backend._record_route_probe(~block_map)
            with attention_step(2, 4), attention_layer(7):
                backend._record_route_probe(block_map)
        report = backend.telemetry()
        self.assertEqual(report["route_probe_record_count"], 1)
        self.assertEqual(report["route_probe_records"][0]["step_gap"], 2)
        self.assertEqual(report["route_probe_records"][0]["global_jaccard"], 1.0)

        with attention_actual_steps((0, 2)):
            with attention_step(0, 4), attention_layer(7):
                backend._record_route_probe(~block_map)
        self.assertEqual(backend.telemetry()["route_probe_record_count"], 1)

    def test_route_probe_rejects_non_fixed_selector(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed_topk"):
            SplitModalityProtectedSpargeAttentionBackend(
                0.5,
                selection_mode="budget_adaptive",
                route_probe=True,
            )

    def test_long_sequence_resolver_uses_the_current_v19_physical_action(self) -> None:
        class PreparedPhysicalBackend:
            def __init__(self, name: str) -> None:
                self.name = name

            def resolve_long_sequence_backend(self, query_tokens: int):
                return self if query_tokens >= 128 else None

        dense = PreparedPhysicalBackend("dense")
        sparse = PreparedPhysicalBackend("sparse")
        backend = RequestActionScheduledAttentionBackend(
            {"dense": dense, "sparse_topk_0.1": sparse}
        )
        with (
            attention_action_schedule({(3, 17): "sparse_topk_0.1"}),
            attention_step(3, 20),
            attention_layer(17),
        ):
            self.assertIs(backend.resolve_long_sequence_backend(220_003), sparse)
        with (
            attention_action_schedule({(3, 17): "sparse_topk_0.1"}),
            attention_step(3, 20),
            attention_layer(18),
        ):
            self.assertIs(backend.resolve_long_sequence_backend(220_003), dense)

    def test_dense_long_sequence_resolver_preserves_reviewed_per_thread_path(self) -> None:
        with dense_qk_quantization("per_thread"):
            self.assertIsNone(
                _resolve_long_sequence_physical_backend(
                    sage_attention_sm89, 67_535
                )
            )
        with dense_qk_quantization("per_warp"):
            self.assertIsNotNone(
                _resolve_long_sequence_physical_backend(
                    sage_attention_sm89, 100_163
                )
            )

    def test_only_exact_actions_may_fallback_from_streaming(self) -> None:
        dense = lambda query, key, value: query
        sparse = lambda query, key, value: query
        backend = RequestActionScheduledAttentionBackend(
            {"dense": dense, "sparse_topk_0.1": sparse}
        )
        with (
            attention_action_schedule({(3, 17): "sparse_topk_0.1"}),
            attention_step(3, 20),
            attention_layer(17),
        ):
            self.assertFalse(
                _can_fallback_to_unstreamed_exact_attention(backend, 67_535)
            )
        with (
            attention_action_schedule({(3, 17): "sparse_topk_0.1"}),
            attention_step(3, 20),
            attention_layer(18),
        ):
            self.assertTrue(
                _can_fallback_to_unstreamed_exact_attention(backend, 67_535)
            )

        guarded = CausalCheckpointVerifierAttentionBackend(
            sage_attention_sm89,
            backend,
            probe_layers=(),
            recovery_layers=(),
        )
        with (
            attention_action_schedule({(3, 17): "sparse_topk_0.1"}),
            attention_step(3, 20),
            attention_layer(17),
        ):
            self.assertFalse(
                _can_fallback_to_unstreamed_exact_attention(guarded, 67_535)
            )
        with (
            attention_action_schedule({(3, 17): "sparse_topk_0.1"}),
            attention_step(3, 20),
            attention_layer(18),
        ):
            self.assertTrue(
                _can_fallback_to_unstreamed_exact_attention(guarded, 67_535)
            )
    def test_long_sequence_resolver_rejects_stateful_sparse_selectors(self) -> None:
        fixed = SplitModalityProtectedSpargeAttentionBackend(
            0.25,
            experimental_minimum_topk=0.0625,
            selection_mode="fixed_topk",
        )
        adaptive = SplitModalityProtectedSpargeAttentionBackend(
            0.25,
            experimental_minimum_topk=0.0625,
            selection_mode="budget_adaptive",
        )
        self.assertIs(fixed.resolve_long_sequence_backend(220_003), fixed)
        self.assertIsNone(adaptive.resolve_long_sequence_backend(220_003))

    def test_long_sequence_resolver_fails_closed_for_active_online_verifier(self) -> None:
        class PreparedPhysicalBackend:
            def resolve_long_sequence_backend(self, query_tokens: int):
                return self if query_tokens >= 128 else None

        dense = PreparedPhysicalBackend()
        sparse = PreparedPhysicalBackend()
        scheduled = RequestActionScheduledAttentionBackend(
            {"dense": dense, "sparse_topk_0.1": sparse}
        )
        guarded = CausalCheckpointVerifierAttentionBackend(
            dense,
            scheduled,
            probe_layers=(4,),
            recovery_layers=(),
            detail_step_indices=(),
            detail_layers=(),
            recovery_horizon=0,
            online_guard_id="long-sequence-test-guard",
        )
        ledger = AttentionOnlineBudget("long-sequence-test-guard", 1.0)
        with (
            attention_online_budget(ledger),
            attention_action_schedule({(3, 4): "sparse_topk_0.1"}),
            attention_step(3, 20),
            attention_layer(4),
        ):
            self.assertIsNone(guarded.resolve_long_sequence_backend(220_003))

    def test_hot_request_accepts_implementation_qualified_v5_actions(self) -> None:
        request = HotSessionRequest(
            prompt="contract",
            seed=1,
            width=64,
            height=64,
            frames=22,
            fps=24,
            steps=4,
            output_path=Path("/tmp/contract.mp4"),
            actual_step_indices=(0, 1, 2, 3),
            attention_action_schedule=(
                (0, 0, "round215:sparse_topk_0.25"),
            ),
        )
        request.validate()

    def test_runtime_telemetry_delta_preserves_configuration(self) -> None:
        delta = NativeT2AVHotSession._telemetry_delta(
            {
                "relative_rms_threshold": 0.31,
                "dense_recovery_calls": 8,
                "draft": {"action_calls": {"round215:sparse_topk_0.25": 100}},
            },
            {
                "relative_rms_threshold": 0.31,
                "dense_recovery_calls": 11,
                "draft": {"action_calls": {"round215:sparse_topk_0.25": 151}},
            },
        )
        self.assertEqual(delta["relative_rms_threshold"], 0.31)
        self.assertEqual(delta["dense_recovery_calls"], 3)
        self.assertEqual(
            delta["draft"]["action_calls"]["round215:sparse_topk_0.25"],
            51,
        )

    def test_online_guard_does_not_double_compute_a_dense_cell(self) -> None:
        calls: list[str] = []
        dense = self._record("dense", calls)
        scheduled = RequestActionScheduledAttentionBackend(
            {"dense": dense, "sparse_topk_0.1": self._record("sparse", calls)}
        )
        guarded = CausalCheckpointVerifierAttentionBackend(
            dense,
            scheduled,
            probe_layers=(30,),
            recovery_layers=(),
            detail_step_indices=(),
            detail_layers=(),
            recovery_horizon=0,
            verification_query_blocks=1,
        )
        tensor = torch.zeros(256, 2, 4)
        with (
            attention_action_schedule({(3, 30): "dense"}),
            attention_step(3, 20),
            attention_layer(30),
        ):
            guarded(tensor, tensor, tensor)
        self.assertEqual(calls, ["dense"])

    def test_v9_runtime_probe_and_recovery_are_ledger_bounded(self) -> None:
        calls: list[str] = []

        def dense(query, key, value):
            del key, value
            calls.append("dense")
            return torch.ones_like(query)

        def sparse(query, key, value):
            del key, value
            calls.append("sparse")
            return torch.zeros_like(query)

        scheduled = RequestActionScheduledAttentionBackend(
            {"dense": dense, "sparse_topk_0.1": sparse}
        )
        guarded = CausalCheckpointVerifierAttentionBackend(
            dense,
            scheduled,
            probe_layers=(4,),
            recovery_layers=(10,),
            detail_step_indices=(),
            detail_layers=(),
            recovery_horizon=0,
            verification_query_blocks=1,
            relative_rms_threshold=0.01,
            online_guard_id=ROUND219_BOUNDED_ONLINE_GUARD,
        )
        ledger = AttentionOnlineBudget(ROUND219_BOUNDED_ONLINE_GUARD, 3.0)
        tensor = torch.zeros(256, 2, 4)
        schedule = {(3, 4): "sparse_topk_0.1", (3, 10): "sparse_topk_0.1"}
        with (
            attention_online_budget(ledger),
            attention_action_schedule(schedule),
            attention_step(3, 20),
            attention_layer(4),
        ):
            guarded(tensor, tensor, tensor)
        with (
            attention_online_budget(ledger),
            attention_action_schedule(schedule),
            attention_step(3, 20),
            attention_layer(10),
        ):
            guarded(tensor, tensor, tensor)
        telemetry = guarded.telemetry()
        self.assertEqual(len(telemetry["probe_records"]), 1)
        self.assertTrue(telemetry["probe_records"][0]["upgrade_applied"])
        self.assertEqual(ledger.spent_dense_layers, 3.0)
        self.assertTrue(ledger.telemetry()["budget_respected"])
        self.assertEqual(calls, ["sparse", "dense", "dense", "dense"])

    def test_v11_growth_trigger_upgrades_without_exceeding_reserve(self) -> None:
        state = {"sparse_value": 0.90}

        def dense(query, key, value):
            del key, value
            return torch.ones_like(query)

        def sparse(query, key, value):
            del key, value
            return torch.full_like(query, state["sparse_value"])

        scheduled = RequestActionScheduledAttentionBackend(
            {"dense": dense, "sparse_topk_0.1": sparse}
        )
        guarded = CausalCheckpointVerifierAttentionBackend(
            dense,
            scheduled,
            probe_layers=(24,),
            recovery_layers=(),
            detail_step_indices=(),
            detail_layers=(),
            recovery_horizon=0,
            verification_query_blocks=1,
            relative_rms_threshold=0.50,
            online_guard_id=ROUND221_CALIBRATED_GROWTH_GUARD,
            phase_probe_guard_ids=(ROUND221_CALIBRATED_GROWTH_GUARD,),
            phase_growth_guard_ids=(ROUND221_CALIBRATED_GROWTH_GUARD,),
        )
        actual = (0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 18, 19)
        schedule = {
            (8, 24): "sparse_topk_0.1",
            (14, 24): "sparse_topk_0.1",
        }
        ledger = AttentionOnlineBudget(ROUND221_CALIBRATED_GROWTH_GUARD, 7.0)
        tensor = torch.zeros(256, 2, 4)
        with (
            attention_online_budget(ledger),
            attention_actual_steps(actual),
            attention_action_schedule(schedule),
            attention_step(8, 20),
            attention_layer(24),
        ):
            first = guarded(tensor, tensor, tensor)
        state["sparse_value"] = 0.70
        with (
            attention_online_budget(ledger),
            attention_actual_steps(actual),
            attention_action_schedule(schedule),
            attention_step(14, 20),
            attention_layer(24),
        ):
            second = guarded(tensor, tensor, tensor)
        records = guarded.telemetry()["probe_records"]
        self.assertTrue(torch.allclose(first, torch.full_like(tensor, 0.90)))
        self.assertTrue(torch.allclose(second, torch.ones_like(tensor)))
        self.assertFalse(records[0]["phase_growth_trigger"])
        self.assertTrue(records[1]["phase_growth_trigger"])
        self.assertTrue(records[1]["upgrade_applied"])
        self.assertEqual(ledger.spent_dense_layers, 3.0)
        self.assertGreaterEqual(ledger.remaining_dense_layers, 4.0)
        self.assertTrue(ledger.telemetry()["budget_respected"])

    def test_v12_no_trigger_rebate_is_dense_upgrade_and_budget_bounded(self) -> None:
        calls: list[str] = []

        def dense(query, key, value):
            del key, value
            calls.append("dense")
            return torch.ones_like(query)

        def sparse(query, key, value):
            del key, value
            calls.append("sparse")
            return torch.zeros_like(query)

        scheduled = RequestActionScheduledAttentionBackend(
            {"dense": dense, "sparse_topk_0.1": sparse}
        )
        guarded = CausalCheckpointVerifierAttentionBackend(
            dense,
            scheduled,
            probe_layers=(24,),
            recovery_layers=(),
            detail_step_indices=(),
            detail_layers=(),
            recovery_horizon=0,
            verification_query_blocks=1,
            relative_rms_threshold=0.50,
            online_guard_id=ROUND223_RESERVE_REBATE_GUARD,
            phase_probe_guard_ids=(ROUND223_RESERVE_REBATE_GUARD,),
            phase_growth_guard_ids=(ROUND223_RESERVE_REBATE_GUARD,),
            reserve_rebate_guard_ids=(ROUND223_RESERVE_REBATE_GUARD,),
        )
        actual = (0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 18, 19)
        ledger = AttentionOnlineBudget(
            ROUND223_RESERVE_REBATE_GUARD,
            5.0,
            rebate_schedule=((17, 10),),
        )
        tensor = torch.zeros(256, 2, 4)
        with (
            attention_online_budget(ledger),
            attention_actual_steps(actual),
            attention_action_schedule({(17, 10): "sparse_topk_0.1"}),
            attention_step(17, 20),
            attention_layer(10),
        ):
            output = guarded(tensor, tensor, tensor)
        self.assertTrue(torch.allclose(output, torch.ones_like(tensor)))
        self.assertEqual(calls, ["dense"])
        self.assertEqual(ledger.spent_dense_layers, 1.0)
        self.assertEqual(ledger.events[0]["kind"], "reserve_rebate")
        self.assertEqual(guarded.telemetry()["reserve_rebate_calls"], 1)
        self.assertTrue(ledger.telemetry()["budget_respected"])

    def test_v12_online_checkpoint_preserves_budget_and_growth_baseline(self) -> None:
        state = {"sparse_value": 0.90}

        def dense(query, key, value):
            del key, value
            return torch.ones_like(query)

        def sparse(query, key, value):
            del key, value
            return torch.full_like(query, state["sparse_value"])

        def make_backend():
            scheduled = RequestActionScheduledAttentionBackend(
                {"dense": dense, "sparse_topk_0.1": sparse}
            )
            return CausalCheckpointVerifierAttentionBackend(
                dense,
                scheduled,
                probe_layers=(24,),
                recovery_layers=(),
                detail_step_indices=(),
                detail_layers=(),
                recovery_horizon=0,
                verification_query_blocks=1,
                relative_rms_threshold=0.50,
                online_guard_id=ROUND223_RESERVE_REBATE_GUARD,
                phase_probe_guard_ids=(ROUND223_RESERVE_REBATE_GUARD,),
                phase_growth_guard_ids=(ROUND223_RESERVE_REBATE_GUARD,),
                reserve_rebate_guard_ids=(ROUND223_RESERVE_REBATE_GUARD,),
            )

        actual = (0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 18, 19)
        schedule = {
            (8, 24): "sparse_topk_0.1",
            (14, 24): "sparse_topk_0.1",
            (17, 10): "sparse_topk_0.1",
        }
        source_ledger = AttentionOnlineBudget(
            ROUND223_RESERVE_REBATE_GUARD,
            7.0,
            rebate_schedule=((17, 10),),
        )
        source_backend = make_backend()
        tensor = torch.zeros(256, 2, 4)
        with (
            attention_online_budget(source_ledger),
            attention_actual_steps(actual),
            attention_action_schedule(schedule),
            attention_step(8, 20),
            attention_layer(24),
        ):
            baseline_output = source_backend(tensor, tensor, tensor)
        self.assertTrue(torch.allclose(baseline_output, torch.full_like(tensor, 0.90)))
        self.assertEqual(source_ledger.spent_dense_layers, 1.0)

        budget_state = source_ledger.checkpoint_state()
        verifier_state = source_backend.online_checkpoint_state(source_ledger)
        restored_ledger = AttentionOnlineBudget(
            ROUND223_RESERVE_REBATE_GUARD,
            7.0,
            rebate_schedule=((17, 10),),
        )
        restored_ledger.restore_checkpoint_state(budget_state)
        restored_backend = make_backend()
        restored_backend.restore_online_checkpoint_state(
            restored_ledger, verifier_state
        )

        state["sparse_value"] = 0.70
        with (
            attention_online_budget(restored_ledger),
            attention_actual_steps(actual),
            attention_action_schedule(schedule),
            attention_step(14, 20),
            attention_layer(24),
        ):
            triggered_output = restored_backend(tensor, tensor, tensor)
        self.assertTrue(torch.allclose(triggered_output, torch.ones_like(tensor)))
        records = restored_backend.telemetry()["probe_records"]
        self.assertEqual(len(records), 2)
        self.assertGreater(records[-1]["phase_growth_ratio"], 2.9)
        self.assertTrue(records[-1]["phase_growth_trigger"])
        self.assertTrue(restored_backend.telemetry()["request_had_trigger"])
        self.assertEqual(restored_ledger.spent_dense_layers, 3.0)

        with (
            attention_online_budget(restored_ledger),
            attention_actual_steps(actual),
            attention_action_schedule(schedule),
            attention_step(17, 20),
            attention_layer(10),
        ):
            rebate_output = restored_backend(tensor, tensor, tensor)
        self.assertTrue(torch.allclose(rebate_output, torch.full_like(tensor, 0.70)))
        self.assertEqual(restored_backend.telemetry()["reserve_rebate_calls"], 0)
        self.assertEqual(restored_ledger.spent_dense_layers, 3.0)


if __name__ == "__main__":
    unittest.main()
