import unittest

import torch

from h3serve.native_engine.model.kernels import (
    LayerSensitivityRoutedSplitSpargeAttentionBackend,
    LayerHeadBudgetOverrideBackend,
    ModalityProtectedSpargeAttentionBackend,
    BudgetConstrainedAdaptiveSpargeAttentionBackend,
    QualityConstrainedAdaptiveSpargeAttentionBackend,
    SplitModalityProtectedSpargeAttentionBackend,
    StepScheduledAttentionBackend,
    CausalCheckpointVerifierAttentionBackend,
    TrajectoryLayerModalityRoutedSpargeAttentionBackend,
    TrajectoryLayerModalityRoutedSolAttentionBackend,
    _ATTENTION_LAYER,
    _ATTENTION_STEP,
    _ATTENTION_FORCE_DENSE,
    _ATTENTION_PROTECTED_PREFIX,
    current_attention_video_grid,
    attention_protected_prefix,
    attention_video_layout,
    attention_layer,
    attention_step,
    attention_force_dense,
)


class ModalityAttentionPolicyTests(unittest.TestCase):
    def test_causal_verifier_reports_local_error_tail_separately(self):
        reference = torch.ones(10, 2, 2)
        candidate = reference.clone()
        candidate[-1].zero_()
        statistics = CausalCheckpointVerifierAttentionBackend._relative_rms_statistics(
            reference, candidate
        )
        self.assertLess(statistics["global"], statistics["query_max"])
        self.assertAlmostEqual(statistics["query_max"], 1.0)

    def test_causal_verifier_spreads_exact_anchors_across_every_frame(self):
        backend = CausalCheckpointVerifierAttentionBackend(
            lambda query, key, value: query,
            lambda query, key, value: query,
            verification_query_blocks=1,
        )
        with attention_protected_prefix(4), attention_video_layout(4, 64):
            indices, strategy = backend._verification_indices(260, torch.device("cpu"))
        self.assertEqual(strategy, "per_frame_spatial_anchors")
        for frame in range(4):
            lower = 4 + frame * 64
            upper = lower + 64
            self.assertTrue(bool(((indices >= lower) & (indices < upper)).any()))

    def test_causal_verifier_can_inject_non_rejected_exact_query_rows(self):
        def dense(query, key, value):
            return torch.ones_like(query)

        def draft(query, key, value):
            return torch.zeros_like(query)

        backend = CausalCheckpointVerifierAttentionBackend(
            dense,
            draft,
            probe_layers=(1,),
            recovery_layers=(),
            detail_step_indices=(),
            detail_layers=(),
            recovery_horizon=0,
            inject_verified_queries=True,
            verification_query_blocks=1,
            relative_rms_threshold=2.0,
            minimum_sparse_tokens=8,
        )
        tensor = torch.zeros(260, 2, 2)
        with attention_step(1, 20), attention_layer(1):
            with attention_protected_prefix(4), attention_video_layout(4, 64):
                output = backend(tensor, tensor, tensor)
        self.assertGreater(int((output[:, 0, 0] == 1).sum()), 4)
        self.assertGreater(int((output[:, 0, 0] == 0).sum()), 4)
        self.assertEqual(backend.telemetry()["verified_query_injection_calls"], 1)

    def test_causal_verifier_repairs_complete_high_error_heads(self):
        def dense(query, key, value):
            return torch.ones_like(query)

        def draft(query, key, value):
            return torch.zeros_like(query)

        backend = CausalCheckpointVerifierAttentionBackend(
            dense,
            draft,
            probe_layers=(1,),
            recovery_layers=(),
            detail_step_indices=(),
            detail_layers=(),
            recovery_horizon=0,
            repair_high_error_heads=True,
            head_error_mass_coverage=0.50,
            head_repair_activation_ratio=0.40,
            verification_query_blocks=1,
            relative_rms_threshold=2.0,
            minimum_sparse_tokens=8,
        )
        tensor = torch.zeros(260, 4, 2)
        with attention_step(1, 20), attention_layer(1):
            output = backend(tensor, tensor, tensor)
        repaired_heads = (output[0, :, 0] == 1)
        self.assertGreaterEqual(int(repaired_heads.sum()), 1)
        self.assertLess(int(repaired_heads.sum()), 4)
        self.assertTrue(bool((output[:, repaired_heads] == 1).all()))
        self.assertEqual(backend.telemetry()["head_repair_calls"], 1)

    def test_causal_verifier_fails_closed_without_context(self):
        calls = []

        def dense(query, key, value):
            calls.append("dense")
            return query

        def draft(query, key, value):
            calls.append("draft")
            return query

        backend = CausalCheckpointVerifierAttentionBackend(
            dense, draft, minimum_sparse_tokens=8
        )
        tensor = torch.zeros(16, 2, 2)
        backend(tensor, tensor, tensor)
        self.assertEqual(calls, ["dense"])

    def test_causal_verifier_rejection_recovers_later_causal_layers(self):
        calls = []

        def dense(query, key, value):
            calls.append(("dense", _ATTENTION_LAYER.get()))
            return torch.ones_like(query)

        def draft(query, key, value):
            calls.append(("draft", _ATTENTION_LAYER.get()))
            return torch.zeros_like(query)

        backend = CausalCheckpointVerifierAttentionBackend(
            dense,
            draft,
            probe_layers=(1,),
            recovery_layers=(2, 3),
            verification_query_blocks=1,
            relative_rms_threshold=0.1,
            recovery_horizon=0,
            minimum_sparse_tokens=8,
        )
        tensor = torch.zeros(16, 2, 2)
        with attention_step(4, 20):
            with attention_layer(1):
                output = backend(tensor, tensor, tensor)
            with attention_layer(2):
                backend(tensor, tensor, tensor)
            with attention_layer(4):
                backend(tensor, tensor, tensor)
        self.assertTrue(torch.equal(output, torch.ones_like(tensor)))
        self.assertEqual(
            calls,
            [
                ("draft", 1),
                ("dense", 1),
                ("dense", 1),
                ("dense", 2),
                ("draft", 4),
            ],
        )
        report = backend.telemetry()
        self.assertEqual(report["dense_recovery_calls"], 2)
        self.assertTrue(report["probe_records"][0]["triggered"])

    def test_causal_verifier_hysteresis_spans_next_full_solver_step(self):
        calls = []

        def dense(query, key, value):
            calls.append(("dense", _ATTENTION_STEP.get()[0], _ATTENTION_LAYER.get()))
            return torch.ones_like(query)

        def draft(query, key, value):
            calls.append(("draft", _ATTENTION_STEP.get()[0], _ATTENTION_LAYER.get()))
            return torch.zeros_like(query)

        backend = CausalCheckpointVerifierAttentionBackend(
            dense,
            draft,
            probe_layers=(1,),
            recovery_layers=(2,),
            detail_step_indices=(),
            detail_layers=(),
            verification_query_blocks=1,
            relative_rms_threshold=0.1,
            recovery_horizon=1,
            minimum_sparse_tokens=8,
        )
        tensor = torch.zeros(16, 2, 2)
        with attention_step(4, 20), attention_layer(1):
            backend(tensor, tensor, tensor)
        # A shallow forecast never reaches the causal band and cannot consume
        # the pending recovery hold.
        with attention_step(5, 20), attention_layer(0):
            backend(tensor, tensor, tensor)
        with attention_step(6, 20), attention_layer(1):
            backend(tensor, tensor, tensor)
        with attention_step(6, 20), attention_layer(2):
            backend(tensor, tensor, tensor)
        self.assertEqual(calls[-2:], [("dense", 6, 1), ("dense", 6, 2)])
        self.assertEqual(backend.telemetry()["dense_hysteresis_calls"], 2)

    def test_causal_verifier_uses_only_teacher_accepted_graded_recovery(self):
        def dense(query, key, value):
            return torch.ones_like(query)

        def draft(query, key, value):
            return torch.zeros_like(query)

        def graded(query, key, value):
            return torch.ones_like(query) * 0.8

        backend = CausalCheckpointVerifierAttentionBackend(
            dense,
            draft,
            recovery_backend=graded,
            probe_layers=(1,),
            recovery_layers=(2,),
            detail_step_indices=(),
            detail_layers=(),
            recovery_horizon=1,
            verification_query_blocks=1,
            relative_rms_threshold=0.25,
            minimum_sparse_tokens=8,
        )
        tensor = torch.zeros(16, 2, 2)
        with attention_step(1, 20):
            with attention_layer(1):
                probe_output = backend(tensor, tensor, tensor)
            with attention_layer(2):
                recovery_output = backend(tensor, tensor, tensor)
        with attention_step(2, 20), attention_layer(1):
            held_output = backend(tensor, tensor, tensor)
        self.assertTrue(torch.equal(probe_output, torch.ones_like(tensor) * 0.8))
        self.assertTrue(torch.equal(recovery_output, torch.ones_like(tensor) * 0.8))
        self.assertTrue(torch.equal(held_output, torch.ones_like(tensor) * 0.8))
        report = backend.telemetry()
        self.assertEqual(report["graded_accept_count"], 1)
        self.assertEqual(report["graded_recovery_calls"], 2)
        self.assertEqual(report["graded_hysteresis_calls"], 1)
        self.assertEqual(report["dense_recovery_calls"], 0)

    def test_causal_verifier_can_limit_cross_step_hold_to_early_layers(self):
        calls = []

        def dense(query, key, value):
            calls.append(("dense", _ATTENTION_STEP.get()[0], _ATTENTION_LAYER.get()))
            return torch.ones_like(query)

        def draft(query, key, value):
            calls.append(("draft", _ATTENTION_STEP.get()[0], _ATTENTION_LAYER.get()))
            return torch.zeros_like(query)

        backend = CausalCheckpointVerifierAttentionBackend(
            dense,
            draft,
            probe_layers=(1,),
            recovery_layers=(2, 3),
            hysteresis_layers=(1, 2),
            detail_step_indices=(),
            detail_layers=(),
            recovery_horizon=1,
            verification_query_blocks=1,
            relative_rms_threshold=0.1,
            minimum_sparse_tokens=8,
        )
        tensor = torch.zeros(16, 2, 2)
        with attention_step(4, 20), attention_layer(1):
            backend(tensor, tensor, tensor)
        with attention_step(6, 20):
            for layer in (1, 2, 3):
                with attention_layer(layer):
                    backend(tensor, tensor, tensor)
        self.assertEqual(
            calls[-3:],
            [("dense", 6, 1), ("dense", 6, 2), ("draft", 6, 3)],
        )

    def test_causal_verifier_learns_probe_growth_and_recovers_early(self):
        layer_scale = {1: 0.20, 2: 0.30}
        calls = []

        def dense(query, key, value):
            calls.append(("dense", _ATTENTION_STEP.get()[0], _ATTENTION_LAYER.get()))
            return torch.ones_like(query)

        def draft(query, key, value):
            calls.append(("draft", _ATTENTION_STEP.get()[0], _ATTENTION_LAYER.get()))
            return torch.ones_like(query) * (1.0 - layer_scale.get(_ATTENTION_LAYER.get(), 0.0))

        backend = CausalCheckpointVerifierAttentionBackend(
            dense,
            draft,
            probe_layers=(1, 2),
            recovery_layers=(3,),
            detail_step_indices=(),
            detail_layers=(),
            recovery_horizon=0,
            verification_query_blocks=1,
            relative_rms_threshold=0.35,
            minimum_sparse_tokens=8,
        )
        tensor = torch.zeros(16, 2, 2)
        # The first complete pair teaches a 1.5x early-to-late growth ratio.
        with attention_step(0, 20):
            with attention_layer(1):
                backend(tensor, tensor, tensor)
            with attention_layer(2):
                backend(tensor, tensor, tensor)
        self.assertAlmostEqual(
            backend.telemetry()["online_probe_growth_upper"], 1.5, places=5
        )

        # A later 0.24 early error predicts 0.36 at the late probe and starts
        # exact recovery before layers 2/3 are allowed onto the draft path.
        layer_scale[1] = 0.24
        with attention_step(1, 20):
            with attention_layer(1):
                backend(tensor, tensor, tensor)
            with attention_layer(2):
                backend(tensor, tensor, tensor)
            with attention_layer(3):
                backend(tensor, tensor, tensor)
        report = backend.telemetry()
        self.assertEqual(report["preemptive_trigger_count"], 1)
        self.assertTrue(report["probe_records"][-1]["preemptive_trigger"])
        self.assertEqual(calls[-2:], [("dense", 1, 2), ("dense", 1, 3)])

    def test_causal_verifier_effort_components_are_monotonic(self):
        previous_blocks = 0
        previous_threshold = float("inf")
        for effort in (0.0, 0.25, 0.5, 0.75, 1.0):
            blocks = round(8 + 24 * effort)
            threshold = 0.34 * (1.0 - effort) ** 0.5
            self.assertGreaterEqual(blocks, previous_blocks)
            self.assertLessEqual(threshold, previous_threshold)
            previous_blocks = blocks
            previous_threshold = threshold
        self.assertEqual(previous_threshold, 0.0)

    def test_causal_verifier_probe_first_skips_rejected_full_draft(self):
        calls = []

        def dense(query, key, value):
            calls.append(("dense", int(query.shape[0])))
            return torch.ones_like(query)

        class Draft:
            def __call__(self, query, key, value):
                calls.append(("draft-full", int(query.shape[0])))
                return torch.zeros_like(query)

            def selected_video_queries(
                self,
                video_query,
                key,
                value,
                *,
                protected_tokens,
                video_query_indices,
            ):
                calls.append(("draft-probe", int(video_query.shape[0])))
                self.assert_aligned(video_query_indices)
                return torch.zeros_like(video_query)

            @staticmethod
            def assert_aligned(indices):
                # Every selected run starts at an original 128-row boundary.
                starts = indices[torch.cat((torch.tensor([True]), indices[1:] != indices[:-1] + 1))]
                assert bool((starts % 128 == 0).all())

        backend = CausalCheckpointVerifierAttentionBackend(
            dense,
            Draft(),
            probe_layers=(1,),
            recovery_layers=(),
            detail_step_indices=(),
            detail_layers=(),
            recovery_horizon=0,
            probe_first_short_circuit=True,
            verification_query_blocks=1,
            relative_rms_threshold=0.1,
            minimum_sparse_tokens=8,
        )
        tensor = torch.zeros(260, 2, 2)
        with attention_step(1, 20), attention_layer(1):
            with attention_protected_prefix(4):
                output = backend(tensor, tensor, tensor)
        self.assertTrue(torch.equal(output, torch.ones_like(tensor)))
        self.assertFalse(any(name == "draft-full" for name, _ in calls))
        report = backend.telemetry()
        self.assertEqual(report["probe_first_calls"], 1)
        self.assertEqual(report["probe_first_rejected_full_drafts"], 1)

    def test_causal_verifier_probe_first_materializes_accepted_full_draft(self):
        calls = []

        def dense(query, key, value):
            calls.append(("dense", int(query.shape[0])))
            return torch.zeros_like(query)

        class Draft:
            def __call__(self, query, key, value):
                calls.append(("draft-full", int(query.shape[0])))
                return torch.zeros_like(query)

            def selected_video_queries(
                self,
                video_query,
                key,
                value,
                *,
                protected_tokens,
                video_query_indices,
            ):
                calls.append(("draft-probe", int(video_query.shape[0])))
                return torch.zeros_like(video_query)

        backend = CausalCheckpointVerifierAttentionBackend(
            dense,
            Draft(),
            probe_layers=(1,),
            recovery_layers=(),
            detail_step_indices=(),
            detail_layers=(),
            recovery_horizon=0,
            probe_first_short_circuit=True,
            verification_query_blocks=1,
            relative_rms_threshold=0.1,
            minimum_sparse_tokens=8,
        )
        tensor = torch.zeros(260, 2, 2)
        with attention_step(1, 20), attention_layer(1):
            with attention_protected_prefix(4):
                backend(tensor, tensor, tensor)
        self.assertEqual(sum(name == "draft-full" for name, _ in calls), 1)
        self.assertEqual(backend.telemetry()["probe_first_rejected_full_drafts"], 0)

    def test_causal_verifier_shared_kv_preserves_original_sample_contract(self):
        calls = []

        def dense(query, key, value):
            calls.append(("dense", int(query.shape[0])))
            return torch.ones_like(query)

        class Draft:
            def __call__(self, query, key, value):
                raise AssertionError("ordinary draft path should not be called")

            def full_with_exact_sample(
                self, query, key, value, *, sample_indices
            ):
                calls.append(("shared", int(sample_indices.numel())))
                return torch.zeros_like(query), torch.ones_like(
                    query.index_select(0, sample_indices)
                )

        backend = CausalCheckpointVerifierAttentionBackend(
            dense,
            Draft(),
            probe_layers=(1,),
            recovery_layers=(),
            detail_step_indices=(),
            detail_layers=(),
            recovery_horizon=0,
            shared_kv_exact_probe=True,
            verification_query_blocks=1,
            relative_rms_threshold=0.1,
            minimum_sparse_tokens=8,
        )
        tensor = torch.zeros(260, 2, 2)
        with attention_step(1, 20), attention_layer(1):
            with attention_protected_prefix(4), attention_video_layout(4, 64):
                output = backend(tensor, tensor, tensor)
        self.assertTrue(torch.equal(output, torch.ones_like(tensor)))
        self.assertEqual(sum(name == "shared" for name, _ in calls), 1)
        self.assertEqual(backend.telemetry()["shared_kv_exact_probe_calls"], 1)

    def test_causal_verifier_latches_minimal_complete_head_island(self):
        def dense(query, key, value):
            return torch.ones_like(query)

        def draft(query, key, value):
            output = torch.ones_like(query)
            output[:, :2] = 0
            return output

        backend = CausalCheckpointVerifierAttentionBackend(
            dense,
            draft,
            probe_layers=(1,),
            recovery_layers=(2,),
            detail_step_indices=(),
            detail_layers=(),
            recovery_horizon=1,
            causal_head_island=True,
            verification_query_blocks=1,
            relative_rms_threshold=0.1,
            minimum_sparse_tokens=8,
        )
        tensor = torch.zeros(16, 4, 2)
        with attention_step(1, 20):
            with attention_layer(1):
                probe = backend(tensor, tensor, tensor)
            with attention_layer(2):
                recovery = backend(tensor, tensor, tensor)
        with attention_step(2, 20), attention_layer(1):
            held = backend(tensor, tensor, tensor)

        self.assertTrue(torch.equal(probe, torch.ones_like(tensor)))
        self.assertTrue(torch.equal(recovery, torch.ones_like(tensor)))
        self.assertTrue(torch.equal(held, torch.ones_like(tensor)))
        report = backend.telemetry()
        self.assertEqual(report["head_island_trigger_count"], 1)
        self.assertEqual(report["head_island_calls"], 3)
        self.assertEqual(report["head_island_heads"], 6)
        self.assertEqual(report["head_island_total_heads"], 12)
        self.assertEqual(report["dense_recovery_calls"], 0)
        self.assertEqual(
            report["probe_records"][0]["head_island_selected_heads"], 2
        )

    def test_causal_verifier_terminal_detail_floor_is_effort_invariant(self):
        calls = []

        def dense(query, key, value):
            calls.append("dense")
            return query

        def draft(query, key, value):
            calls.append("draft")
            return query

        backend = CausalCheckpointVerifierAttentionBackend(
            dense,
            draft,
            probe_layers=(1,),
            recovery_layers=(2,),
            detail_step_indices=(17, 18, 19),
            detail_layers=(4,),
            minimum_sparse_tokens=8,
        )
        tensor = torch.zeros(16, 2, 2)
        with attention_step(18, 20), attention_layer(4):
            backend(tensor, tensor, tensor)
        with attention_step(14, 20), attention_layer(4):
            backend(tensor, tensor, tensor)
        self.assertEqual(calls, ["dense", "draft"])
        self.assertEqual(backend.telemetry()["dense_detail_calls"], 1)

    def test_force_dense_scope_is_restored(self):
        self.assertFalse(_ATTENTION_FORCE_DENSE.get())
        with attention_force_dense():
            self.assertTrue(_ATTENTION_FORCE_DENSE.get())
        self.assertFalse(_ATTENTION_FORCE_DENSE.get())

    def test_quality_constrained_solver_selects_only_no_worse_budgets(self):
        backend = QualityConstrainedAdaptiveSpargeAttentionBackend()
        accepted = (0.20, 0.20)
        accepted_l1 = torch.tensor([0.10, 0.30])
        accepted_cosine = torch.tensor([0.90, 0.70])
        candidates = (
            (
                0.10,
                torch.tensor([0.25, 0.40]),
                torch.tensor([0.75, 0.60]),
            ),
            (
                0.15,
                torch.tensor([0.20, 0.20]),
                torch.tensor([0.80, 0.80]),
            ),
        )
        self.assertEqual(
            backend.solve_head_budgets(
                accepted,
                accepted_l1,
                accepted_cosine,
                candidates,
            ),
            (0.15, 0.15),
        )

    def test_quality_constrained_reference_envelope_matches_tlhb_phases(self):
        backend = QualityConstrainedAdaptiveSpargeAttentionBackend()
        self.assertEqual(set(backend._accepted_budget("cruise", 10)), {0.125})
        self.assertEqual(
            set(backend._accepted_budget("cruise", 35)), {0.15, 0.175}
        )
        self.assertEqual(
            set(backend._accepted_budget("anchor", 10)), {0.35, 0.40, 0.45}
        )
        self.assertEqual(
            set(backend._accepted_budget("recovery", 35)), {0.30, 0.35}
        )

    def test_budget_projection_preserves_exact_global_quota(self):
        desired = torch.tensor(
            [[[1.0, 7.0, 2.0, 5.0], [6.0, 1.0, 4.0, 3.0]]]
        )
        projected = SplitModalityProtectedSpargeAttentionBackend._project_counts_to_exact_budget(
            desired,
            torch.tensor([32]),
            minimum=1,
            maximum=8,
        )
        self.assertEqual(int(projected.sum()), 32)
        self.assertGreaterEqual(int(projected.min()), 1)
        self.assertLessEqual(int(projected.max()), 8)
        self.assertGreater(int(projected[0, 0, 1]), int(projected[0, 0, 0]))

    def test_budget_adaptive_zero_safety_is_fixed_topk(self):
        base = torch.tensor([4, 3])
        mass = torch.tensor([[[1, 8, 2], [7, 1, 6]]])
        selected = SplitModalityProtectedSpargeAttentionBackend._budget_adaptive_counts(
            base_count=base,
            mass_count=mass,
            high_risk=torch.tensor(
                [[[False, True, False], [False, False, True]]]
            ),
            safety_margin=0.0,
            minimum=1,
            maximum=8,
        )
        expected = base.view(1, 2, 1).expand_as(mass)
        self.assertTrue(torch.equal(selected, expected))

    def test_budget_adaptive_reallocates_without_changing_cost(self):
        base = torch.tensor([4, 4])
        mass = torch.full((1, 2, 4), 4)
        risk = torch.zeros_like(mass, dtype=torch.bool)
        risk[0, 0, 1] = True
        selected = SplitModalityProtectedSpargeAttentionBackend._budget_adaptive_counts(
            base_count=base,
            mass_count=mass,
            high_risk=risk,
            safety_margin=1.0,
            minimum=1,
            maximum=8,
        )
        self.assertEqual(int(selected.sum()), 32)
        self.assertGreater(int(selected[0, 0, 1]), int(selected[0, 0, 0]))

    def test_budget_adaptive_public_knobs_are_bounded(self):
        backend = BudgetConstrainedAdaptiveSpargeAttentionBackend(
            0.35, safety_margin=0.75
        )
        self.assertEqual(backend.telemetry()["compute_budget"], 0.35)
        self.assertEqual(backend.telemetry()["safety_margin"], 0.75)
        with self.assertRaises(ValueError):
            BudgetConstrainedAdaptiveSpargeAttentionBackend(0.05)
        with self.assertRaises(ValueError):
            BudgetConstrainedAdaptiveSpargeAttentionBackend(
                0.35, safety_margin=1.1
            )

    def test_budget_adaptive_uses_interaction_hybrid_route(self):
        backend = BudgetConstrainedAdaptiveSpargeAttentionBackend(
            0.407, safety_margin=1.0
        )
        routed = backend._backend((0.25,) * 56)
        self.assertEqual(routed.selection_mode, "interaction_hybrid")

    def test_budget_adaptive_layer_route_preserves_exact_pass_quota(self):
        counts = BudgetConstrainedAdaptiveSpargeAttentionBackend.solve_layer_counts(
            0.35, 1.0, key_blocks=100
        )
        self.assertEqual(len(counts), 50)
        self.assertEqual(sum(counts), 35 * 50)
        self.assertEqual(min(counts), 6)
        self.assertTrue(
            BudgetConstrainedAdaptiveSpargeAttentionBackend._CAUSAL_LAYERS.issubset(
                {index for index, count in enumerate(counts) if count > 6}
            )
        )

    def test_budget_adaptive_zero_safety_keeps_uniform_layer_quota(self):
        counts = BudgetConstrainedAdaptiveSpargeAttentionBackend.solve_layer_counts(
            0.35, 0.0, key_blocks=100
        )
        self.assertEqual(counts, (35,) * 50)

    def test_budget_adaptive_task_signal_cannot_demote_causal_stratum(self):
        task_error = tuple(
            100.0 if layer == 0 else 0.01 for layer in range(50)
        )
        order = BudgetConstrainedAdaptiveSpargeAttentionBackend._priority_order(
            task_error
        )
        causal = BudgetConstrainedAdaptiveSpargeAttentionBackend._CAUSAL_LAYERS
        self.assertTrue(all(layer in causal for layer in order[: len(causal)]))
        self.assertEqual(order[len(causal)], 0)

    def test_budget_adaptive_trajectory_quota_is_exact(self):
        actual = (0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 18, 19)
        route = BudgetConstrainedAdaptiveSpargeAttentionBackend.solve_trajectory_counts(
            0.41, 1.0, key_blocks=100, actual_steps=actual
        )
        self.assertEqual(sum(sum(route[step]) for step in actual), 41 * 50 * 12)
        causal = BudgetConstrainedAdaptiveSpargeAttentionBackend._CAUSAL_LAYERS
        self.assertTrue(
            all(route[step][layer] == 100 for step in actual for layer in causal)
        )
        self.assertTrue(
            all(route[0][layer] == 100 for layer in range(50) if layer not in causal)
        )
        noncausal = tuple(layer for layer in range(50) if layer not in causal)
        self.assertGreater(
            sum(route[17][layer] for layer in noncausal),
            sum(route[1][layer] for layer in noncausal),
        )

    def test_budget_adaptive_trajectory_zero_safety_is_uniform(self):
        actual = (0, 2, 4)
        route = BudgetConstrainedAdaptiveSpargeAttentionBackend.solve_trajectory_counts(
            0.35, 0.0, key_blocks=100, actual_steps=actual
        )
        self.assertTrue(all(route[step] == (35,) * 50 for step in actual))

    def test_budget_adaptive_avoids_near_dense_sparse_dead_zone(self):
        actual = (0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 18, 19)
        route = BudgetConstrainedAdaptiveSpargeAttentionBackend.solve_trajectory_counts(
            0.34, 1.0, key_blocks=1566, actual_steps=actual
        )
        sparse_maximum = int(
            BudgetConstrainedAdaptiveSpargeAttentionBackend._MAX_EFFECTIVE_SPARSE_FRACTION
            * 1566
        )
        self.assertTrue(
            all(
                count <= sparse_maximum or count == 1566
                for step in actual
                for count in route[step]
            )
        )

    def test_budget_adaptive_head_trajectory_has_one_shape_invariant_quota(self):
        actual = (0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 18, 19)
        for key_blocks in (545, 1566):
            route = BudgetConstrainedAdaptiveSpargeAttentionBackend.solve_trajectory_head_counts(
                0.407,
                1.0,
                key_blocks=key_blocks,
                heads=56,
                actual_steps=actual,
            )
            flattened = [
                count
                for step in actual
                for layer_counts in route[step]
                for count in layer_counts
            ]
            self.assertEqual(
                sum(flattened),
                round(0.407 * key_blocks * len(actual) * 50 * 56),
            )
            self.assertTrue(
                all(
                    count == key_blocks
                    for layer_counts in route[0]
                    for count in layer_counts
                )
            )
            self.assertGreater(route[17][0][3], route[17][0][0])

    def test_budget_adaptive_integer_floor_roundtrips_on_long_shape(self):
        ratio = BudgetConstrainedAdaptiveSpargeAttentionBackend._count_to_budget(
            97, 1565
        )
        self.assertGreaterEqual(ratio, 0.0625)
        self.assertEqual(int(ratio * 1565), 97)

    def test_budget_adaptive_stops_calibration_at_quality_topology(self):
        actual = (0, 1, 2, 17, 18, 19)
        route = BudgetConstrainedAdaptiveSpargeAttentionBackend.solve_trajectory_counts(
            0.51, 1.0, key_blocks=100, actual_steps=actual
        )
        self.assertTrue(
            BudgetConstrainedAdaptiveSpargeAttentionBackend._quality_topology_saturated(
                route, actual_steps=actual, key_blocks=100
            )
        )
        layers = list(route[1])
        layers[39] = 99
        route[1] = tuple(layers)
        self.assertFalse(
            BudgetConstrainedAdaptiveSpargeAttentionBackend._quality_topology_saturated(
                route, actual_steps=actual, key_blocks=100
            )
        )

    def test_protected_prefix_is_request_scoped(self):
        self.assertEqual(_ATTENTION_PROTECTED_PREFIX.get(), 0)
        with attention_protected_prefix(1560):
            self.assertEqual(_ATTENTION_PROTECTED_PREFIX.get(), 1560)
        self.assertEqual(_ATTENTION_PROTECTED_PREFIX.get(), 0)

    def test_short_or_unscoped_calls_fail_safe_to_dense(self):
        backend = ModalityProtectedSpargeAttentionBackend(0.5, minimum_sparse_tokens=8)
        calls = []

        def dense(query, key, value):
            calls.append(query.shape[0])
            return value

        import h3serve.native_engine.model.kernels as kernels

        original = kernels.sage_attention_sm89
        kernels.sage_attention_sm89 = dense
        try:
            tensor = torch.zeros(4, 2, 2)
            self.assertIs(backend(tensor, tensor, tensor), tensor)
            long_tensor = torch.zeros(16, 2, 2)
            self.assertIs(backend(long_tensor, long_tensor, long_tensor), long_tensor)
        finally:
            kernels.sage_attention_sm89 = original
        self.assertEqual(calls, [4, 16])

    def test_invalid_prefix_rejected(self):
        with self.assertRaises(ValueError):
            with attention_protected_prefix(-1):
                pass

    def test_temporal_correspondence_rail_tracks_nearby_frames(self):
        backend = SplitModalityProtectedSpargeAttentionBackend(
            0.25,
            experimental_minimum_topk=0.125,
            temporal_correspondence_radius=1,
            temporal_spatial_block_radius=0,
        )
        # Three frame-major video grids, 256 tokens per frame. Q blocks are
        # 128 rows; K blocks are 64 rows and begin after a 100-token prefix.
        block_map = torch.zeros((1, 2, 6, 14), dtype=torch.bool)
        with attention_video_layout(3, 256):
            backend._protect_temporal_correspondence(
                block_map,
                query_tokens=768,
                key_tokens=868,
                protected_tokens=100,
            )

        # Query block 0 sees the aligned spatial region in frame 0 and frame 1.
        self.assertTrue(block_map[0, 0, 0, 2])
        self.assertTrue(block_map[0, 1, 0, 6])
        # Frame 2 is outside the one-frame temporal radius; a distant region
        # in frame 0 is also not forced into the exact rail.
        self.assertFalse(block_map[0, 0, 0, 10])
        self.assertFalse(block_map[0, 0, 0, 5])

    def test_video_grid_scope_is_validated_and_restored(self):
        self.assertIsNone(current_attention_video_grid())
        with attention_video_layout(3, 256, grid_height=16, grid_width=16):
            self.assertEqual(current_attention_video_grid(), (16, 16))
        self.assertIsNone(current_attention_video_grid())
        with self.assertRaises(ValueError):
            with attention_video_layout(3, 256, grid_height=8, grid_width=16):
                pass

    def test_interaction_guard_selects_second_order_motion_outlier(self):
        backend = SplitModalityProtectedSpargeAttentionBackend(
            0.25,
            experimental_minimum_topk=0.125,
            selection_mode="interaction_guard",
        )
        # Five frames, two query blocks per frame. Block zero changes temporal
        # direction abruptly in the middle while block one remains coherent.
        pooled = torch.zeros((1, 1, 10, 4), dtype=torch.float32)
        for frame in range(5):
            pooled[0, 0, frame * 2 + 1] = torch.tensor([1.0, frame * 0.1, 0.0, 0.0])
            pooled[0, 0, frame * 2] = torch.tensor(
                [1.0, float(frame if frame < 2 else 4 - frame), 0.0, 0.0]
            )
        with attention_video_layout(5, 256):
            selected = backend._interaction_risk_guard(pooled, query_tokens=1280)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertTrue(selected.any())
        self.assertFalse(selected.all())

    def test_interaction_dense_propagates_only_matching_temporal_blocks(self):
        backend = SplitModalityProtectedSpargeAttentionBackend(
            0.25,
            experimental_minimum_topk=0.125,
            selection_mode="interaction_dense",
        )
        # Block zero changes direction while block one remains coherent in
        # each of five frames. Dense recovery must follow block zero through
        # time without spreading to the unrelated interleaved block.
        pooled = torch.zeros((1, 1, 10, 4), dtype=torch.float32)
        for frame in range(5):
            pooled[0, 0, frame * 2 + 1] = torch.tensor(
                [1.0, frame * 0.1, 0.0, 0.0]
            )
            pooled[0, 0, frame * 2] = torch.tensor(
                [1.0, float(frame if frame < 2 else 4 - frame), 0.0, 0.0]
            )
        with attention_video_layout(5, 256):
            selected = backend._interaction_risk_guard(
                pooled, query_tokens=1280
            )
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertTrue(selected[..., ::2].all())
        self.assertFalse(selected[..., 1::2].any())

    def test_row_coherent_rail_keeps_complete_horizontal_contact_band(self):
        import os
        from unittest.mock import patch

        backend = SplitModalityProtectedSpargeAttentionBackend(
            0.25,
            experimental_minimum_topk=0.125,
            temporal_correspondence_radius=0,
            temporal_spatial_block_radius=0,
        )
        # One 16x16 frame. Query block zero is centred at flattened token 64,
        # row four. The row-coherent rail retains the far-right side of the
        # same row band even though its flattened distance exceeds 96.
        block_map = torch.zeros((1, 1, 2, 6), dtype=torch.bool)
        with (
            patch.dict(
                os.environ,
                {"H3_NATIVE_EXPERIMENTAL_MTCR_ROW_COHERENCE": "1"},
                clear=False,
            ),
            attention_video_layout(1, 256, grid_height=16, grid_width=16),
        ):
            backend._protect_temporal_correspondence(
                block_map,
                query_tokens=256,
                key_tokens=356,
                protected_tokens=100,
            )
        self.assertTrue(block_map[0, 0, 0, 3])

    def test_interaction_rail_expands_only_marked_query_neighbourhood(self):
        backend = SplitModalityProtectedSpargeAttentionBackend(
            0.25,
            experimental_minimum_topk=0.125,
            temporal_correspondence_radius=0,
            temporal_spatial_block_radius=0,
            selection_mode="interaction_rail",
        )
        block_map = torch.zeros((1, 1, 2, 6), dtype=torch.bool)
        risk = torch.tensor([[[True, False]]])
        with attention_video_layout(1, 256):
            backend._protect_temporal_correspondence(
                block_map,
                query_tokens=256,
                key_tokens=356,
                protected_tokens=100,
                query_protection_mask=risk,
            )
        # Query zero gains a distant-but-local key that the ordinary 96-token
        # rail does not include. Query one receives no symmetric expansion.
        self.assertTrue(block_map[0, 0, 0, 5])
        self.assertFalse(block_map[0, 0, 1, 2])

    def test_temporal_correspondence_fails_closed_for_selected_rows(self):
        backend = SplitModalityProtectedSpargeAttentionBackend(
            0.25,
            experimental_minimum_topk=0.125,
            temporal_correspondence_radius=1,
        )
        block_map = torch.zeros((1, 1, 2, 14), dtype=torch.bool)
        with attention_video_layout(3, 256):
            backend._protect_temporal_correspondence(
                block_map,
                query_tokens=256,
                key_tokens=868,
                protected_tokens=100,
            )
        self.assertFalse(block_map.any())

    def test_mtcr_adds_rotating_remote_same_spatial_anchors(self):
        backend = SplitModalityProtectedSpargeAttentionBackend(
            0.25,
            experimental_minimum_topk=0.125,
            temporal_correspondence_radius=0,
            temporal_spatial_block_radius=0,
            temporal_global_anchor_stride=2,
            temporal_global_spatial_block_radius=0,
        )
        block_map = torch.zeros((1, 1, 8, 18), dtype=torch.bool)
        with (
            attention_video_layout(4, 256),
            attention_step(0, 20),
            attention_layer(0),
        ):
            backend._protect_temporal_correspondence(
                block_map,
                query_tokens=1024,
                key_tokens=1124,
                protected_tokens=100,
            )

        # Query block zero is local to frame zero.  MTCR additionally keeps
        # aligned evidence from the phase-zero frame 2 and terminal frame 3,
        # but not an unrelated spatial block in either remote frame.
        self.assertTrue(block_map[0, 0, 0, 10])
        self.assertTrue(block_map[0, 0, 0, 14])
        self.assertFalse(block_map[0, 0, 0, 13])

    def test_temporal_correspondence_uses_true_selected_query_coordinates(self):
        backend = SplitModalityProtectedSpargeAttentionBackend(
            0.25,
            experimental_minimum_topk=0.125,
            temporal_correspondence_radius=0,
        )
        block_map = torch.zeros((1, 1, 1, 14), dtype=torch.bool)
        # The selected rows are the first 128 tokens of frame 2, not frame 0.
        selected = torch.arange(512, 640, dtype=torch.long)
        with attention_video_layout(3, 256):
            backend._protect_temporal_correspondence(
                block_map,
                query_tokens=128,
                key_tokens=868,
                protected_tokens=100,
                query_token_indices=selected,
            )
        self.assertTrue(block_map[0, 0, 0, 10])
        self.assertFalse(block_map[0, 0, 0, 2])

    def test_attention_layer_is_request_scoped(self):
        self.assertIsNone(_ATTENTION_LAYER.get())
        with attention_layer(17):
            self.assertEqual(_ATTENTION_LAYER.get(), 17)
        self.assertIsNone(_ATTENTION_LAYER.get())
        with self.assertRaises(ValueError):
            with attention_layer(-1):
                pass

    def test_scheduled_backend_keeps_selected_layer_dense(self):
        calls = []

        def dense(query, key, value):
            calls.append("dense")
            return value

        def sparse(query, key, value):
            calls.append("sparse")
            return value

        backend = StepScheduledAttentionBackend(
            dense, sparse, dense_layer_indices=(0, 49), minimum_sparse_tokens=8
        )
        tensor = torch.zeros(16, 2, 2)
        # Fails safe when the caller did not expose a real layer index.
        backend(tensor, tensor, tensor)
        with attention_layer(0):
            backend(tensor, tensor, tensor)
        with attention_layer(25):
            backend(tensor, tensor, tensor)
        self.assertEqual(calls, ["dense", "dense", "sparse"])

    def test_scheduled_backend_keeps_only_selected_step_layer_pair_dense(self):
        calls = []

        def dense(query, key, value):
            calls.append("dense")
            return value

        def sparse(query, key, value):
            calls.append("sparse")
            return value

        backend = StepScheduledAttentionBackend(
            dense,
            sparse,
            dense_step_layer_pairs=((4, 39),),
            minimum_sparse_tokens=8,
        )
        tensor = torch.zeros(16, 2, 2)
        with attention_step(4, 20), attention_layer(39):
            backend(tensor, tensor, tensor)
        with attention_step(4, 20), attention_layer(38):
            backend(tensor, tensor, tensor)
        with attention_step(3, 20), attention_layer(39):
            backend(tensor, tensor, tensor)
        self.assertEqual(calls, ["dense", "sparse", "sparse"])

    def test_split_backend_short_calls_fail_safe_to_dense(self):
        backend = SplitModalityProtectedSpargeAttentionBackend(
            0.5, minimum_sparse_tokens=8
        )
        calls = []

        def dense(query, key, value):
            calls.append(query.shape[0])
            return value

        import h3serve.native_engine.model.kernels as kernels

        original = kernels.sage_attention_sm89
        kernels.sage_attention_sm89 = dense
        try:
            tensor = torch.zeros(4, 2, 2)
            self.assertIs(backend(tensor, tensor, tensor), tensor)
        finally:
            kernels.sage_attention_sm89 = original
        self.assertEqual(calls, [4])

    def test_split_backend_accepts_and_validates_headwise_budgets(self):
        backend = SplitModalityProtectedSpargeAttentionBackend((0.5, 0.65, 0.75))
        self.assertEqual(backend.topk, (0.5, 0.65, 0.75))
        selected = backend._head_topk(3, torch.device("cpu"))
        self.assertTrue(torch.equal(selected, torch.tensor([0.5, 0.65, 0.75])))
        with self.assertRaisesRegex(ValueError, "2 values for 3 heads"):
            SplitModalityProtectedSpargeAttentionBackend((0.5, 0.75))._head_topk(
                3, torch.device("cpu")
            )

    def test_split_backend_rejects_invalid_headwise_budget(self):
        with self.assertRaises(ValueError):
            SplitModalityProtectedSpargeAttentionBackend((0.5, 0.49))

    def test_absolute_cap_is_an_explicit_distinct_physical_contract(self):
        with self.assertRaisesRegex(ValueError, "requires maximum selected"):
            SplitModalityProtectedSpargeAttentionBackend(
                0.25,
                experimental_minimum_topk=0.0625,
                selection_mode="fixed_topk_absolute_cap",
            )
        with self.assertRaisesRegex(ValueError, "require an absolute-cap selector"):
            SplitModalityProtectedSpargeAttentionBackend(
                0.25,
                experimental_minimum_topk=0.0625,
                maximum_selected_key_blocks=97,
            )
        with self.assertRaisesRegex(ValueError, "positive integers"):
            SplitModalityProtectedSpargeAttentionBackend(
                0.25,
                experimental_minimum_topk=0.0625,
                selection_mode="fixed_topk_absolute_cap",
                maximum_selected_key_blocks=(97, 0),
            )

    def test_absolute_cap_preserves_reference_horizon_then_clamps_per_head(self):
        backend = SplitModalityProtectedSpargeAttentionBackend(
            (0.0625, 0.08, 0.10),
            experimental_minimum_topk=0.0625,
            selection_mode="fixed_topk_absolute_cap",
            maximum_selected_key_blocks=(97, 125, 156),
        )
        reference = backend._selected_key_block_counts(
            3, 1_565, torch.device("cpu")
        )
        self.assertTrue(torch.equal(reference, torch.tensor([97, 125, 156])))
        long = backend._selected_key_block_counts(
            3, 3_436, torch.device("cpu")
        )
        self.assertTrue(torch.equal(long, torch.tensor([97, 125, 156])))
        nominal = backend._selected_key_block_counts(
            3,
            3_436,
            torch.device("cpu"),
            apply_absolute_cap=False,
        )
        self.assertTrue(torch.equal(nominal, torch.tensor([214, 274, 343])))
        self.assertIs(backend.resolve_long_sequence_backend(220_003), backend)

    def test_absolute_cap_validates_head_count_at_selection_time(self):
        backend = SplitModalityProtectedSpargeAttentionBackend(
            (0.0625, 0.08, 0.10),
            experimental_minimum_topk=0.0625,
            selection_mode="fixed_topk_absolute_cap",
            maximum_selected_key_blocks=(97, 125),
        )
        with self.assertRaisesRegex(ValueError, "2 values for 3 heads"):
            backend._selected_key_block_counts(3, 3_436, torch.device("cpu"))

    def test_mass_guarded_cap_is_stateless_long_streaming_only(self):
        backend = SplitModalityProtectedSpargeAttentionBackend(
            0.0625,
            experimental_minimum_topk=0.0625,
            selection_mode="fixed_topk_mass_guarded_cap",
            maximum_selected_key_blocks=170,
            minimum_retained_topk_mass=0.95,
        )
        self.assertIs(backend.resolve_long_sequence_backend(220_003), backend)
        self.assertEqual(backend.minimum_retained_topk_mass, 0.95)
        with self.assertRaisesRegex(ValueError, r"inside \(0, 1\]"):
            SplitModalityProtectedSpargeAttentionBackend(
                0.0625,
                experimental_minimum_topk=0.0625,
                selection_mode="fixed_topk_mass_guarded_cap",
                maximum_selected_key_blocks=170,
                minimum_retained_topk_mass=0.0,
            )

    def test_mass_probe_is_side_effect_free_prepared_streaming_contract(self):
        backend = SplitModalityProtectedSpargeAttentionBackend(
            0.0625,
            experimental_minimum_topk=0.0625,
            selection_mode="fixed_topk_mass_probe",
            maximum_selected_key_blocks=170,
            minimum_retained_topk_mass=0.95,
            mass_probe_selected_key_blocks=(122, 146, 170, 194),
        )
        self.assertIs(backend.resolve_long_sequence_backend(220_003), backend)
        self.assertTrue(backend.telemetry()["mass_guard_probe_only"])
        with self.assertRaisesRegex(ValueError, "sorted unique"):
            SplitModalityProtectedSpargeAttentionBackend(
                0.0625,
                experimental_minimum_topk=0.0625,
                selection_mode="fixed_topk_mass_probe",
                maximum_selected_key_blocks=170,
                mass_probe_selected_key_blocks=(170, 146),
            )

    def test_causal_head_guard_selects_diffuse_heads_without_fixed_ratio(self):
        # Four heads over two query blocks and four key blocks. The final head
        # is uniformly diffuse while the others are increasingly concentrated.
        probability = torch.tensor(
            [[
                [[0.97, 0.01, 0.01, 0.01], [0.96, 0.02, 0.01, 0.01]],
                [[0.90, 0.04, 0.03, 0.03], [0.89, 0.05, 0.03, 0.03]],
                [[0.80, 0.08, 0.06, 0.06], [0.79, 0.09, 0.06, 0.06]],
                [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]],
            ]],
            dtype=torch.float32,
        )
        selected = (
            SplitModalityProtectedSpargeAttentionBackend._causal_head_dense_mask(
                probability
            )
        )
        self.assertFalse(bool(selected[0]))
        self.assertTrue(bool(selected[-1]))
        self.assertGreaterEqual(int(selected.sum()), 1)
        self.assertLess(int(selected.sum()), 4)

    def test_causal_head_guard_only_runs_on_sensitive_layer_backend(self):
        backend = LayerSensitivityRoutedSplitSpargeAttentionBackend(
            aggressive_topk=0.25,
            safe_topk=0.50,
            sensitive_layers=(30, 31, 45),
            experimental_minimum_topk=0.125,
            selection_mode="causal_head_guard",
        )
        self.assertEqual(backend.aggressive.selection_mode, "fixed_topk")
        self.assertEqual(backend.safe.selection_mode, "causal_head_guard")

    def test_layer_routed_backend_fails_safe_and_routes_by_layer(self):
        backend = LayerSensitivityRoutedSplitSpargeAttentionBackend(
            aggressive_topk=0.35,
            sensitive_layers=(30, 31, 45),
        )
        calls = []

        class FakeBackend:
            def __init__(self, name):
                self.name = name

            def __call__(self, query, key, value):
                calls.append((self.name, "full"))
                return query

            def protected_queries(self, query, key, value):
                calls.append((self.name, "protected"))
                return query

            def selected_queries(
                self,
                prefix_query,
                video_query,
                key,
                value,
                *,
                protected_tokens,
            ):
                calls.append((self.name, f"selected:{protected_tokens}"))
                return torch.cat((prefix_query, video_query))

        backend.safe = FakeBackend("safe")
        backend.aggressive = FakeBackend("aggressive")
        tensor = torch.zeros(2, 1, 1)

        backend(tensor, tensor, tensor)
        with attention_layer(12):
            backend.protected_queries(tensor, tensor, tensor)
        with attention_layer(31):
            backend.selected_queries(
                tensor[:1], tensor[1:], tensor, tensor, protected_tokens=1
            )

        self.assertEqual(
            calls,
            [
                ("safe", "full"),
                ("aggressive", "protected"),
                ("safe", "selected:1"),
            ],
        )

    def test_layer_routed_backend_validates_policy(self):
        with self.assertRaises(ValueError):
            LayerSensitivityRoutedSplitSpargeAttentionBackend(
                aggressive_topk=0.50,
                sensitive_layers=(30,),
            )

    def test_teacher_head_budget_routes_only_configured_phase_and_layer(self):
        calls = []

        class FakeBackend:
            def __call__(self, query, key, value):
                calls.append("fallback")
                return query

        backend = LayerHeadBudgetOverrideBackend(
            FakeBackend(),
            default_layer_topks={39: (0.25,) * 56},
            recovery_layer_topks={42: (0.35,) * 56},
            default_step_indices=(4,),
            recovery_step_indices=(17, 18, 19),
        )
        backend.default_layers[39] = lambda query, key, value: (
            calls.append("default39") or query
        )
        backend.recovery_layers[42] = lambda query, key, value: (
            calls.append("recovery42") or query
        )
        tensor = torch.zeros(16, 2, 2)
        with attention_step(4, 20), attention_layer(39):
            backend(tensor, tensor, tensor)
        with attention_step(17, 20), attention_layer(42):
            backend(tensor, tensor, tensor)
        with attention_step(17, 20), attention_layer(39):
            backend(tensor, tensor, tensor)
        with attention_step(8, 20), attention_layer(39):
            backend(tensor, tensor, tensor)
        self.assertEqual(
            calls, ["default39", "recovery42", "fallback", "fallback"]
        )
        with self.assertRaises(ValueError):
            LayerSensitivityRoutedSplitSpargeAttentionBackend(
                aggressive_topk=0.35,
                sensitive_layers=(31, 30),
            )

    def test_trajectory_layer_modality_policy_fails_closed_and_routes(self):
        backend = TrajectoryLayerModalityRoutedSpargeAttentionBackend(
            aggressive_topk=0.40,
            sensitive_layers=(30, 31),
            dense_step_indices=(0, 19),
            minimum_sparse_tokens=8,
        )
        calls = []

        def dense(query, key, value):
            calls.append("dense")
            return query

        class FakeLayerPolicy:
            def __call__(self, query, key, value):
                calls.append("layer")
                return query

        import h3serve.native_engine.model.kernels as kernels

        original = kernels.sage_attention_sm89
        kernels.sage_attention_sm89 = dense
        backend.layer_policy = FakeLayerPolicy()
        tensor = torch.zeros(16, 2, 2)
        try:
            backend(tensor, tensor, tensor)
            with attention_step(0, 20):
                backend(tensor, tensor, tensor)
            with attention_step(8, 20):
                backend(tensor, tensor, tensor)
        finally:
            kernels.sage_attention_sm89 = original
        self.assertEqual(calls, ["dense", "dense", "layer"])

    def test_trajectory_layer_modality_policy_validates_dense_steps(self):
        with self.assertRaisesRegex(ValueError, "dense solver steps"):
            TrajectoryLayerModalityRoutedSpargeAttentionBackend(
                aggressive_topk=0.40,
                sensitive_layers=(30,),
                dense_step_indices=(19, 0),
            )

    def test_trajectory_policy_accepts_distinct_safe_budget(self):
        backend = TrajectoryLayerModalityRoutedSpargeAttentionBackend(
            aggressive_topk=0.50,
            safe_topk=0.65,
            sensitive_layers=(30, 31),
            dense_step_indices=(0, 17, 18, 19),
        )
        self.assertEqual(backend.layer_policy.aggressive.topk, 0.50)
        self.assertEqual(backend.layer_policy.safe.topk, 0.65)

    def test_trajectory_policy_accepts_ordered_headwise_budgets(self):
        backend = TrajectoryLayerModalityRoutedSpargeAttentionBackend(
            aggressive_topk=(0.25, 0.35, 0.40),
            safe_topk=(0.35, 0.45, 0.50),
            sensitive_layers=(30, 31),
            dense_step_indices=(0, 19),
        )
        self.assertEqual(backend.layer_policy.aggressive.topk, (0.25, 0.35, 0.40))
        self.assertEqual(backend.layer_policy.safe.topk, (0.35, 0.45, 0.50))

    def test_trajectory_policy_rejects_unordered_headwise_budgets(self):
        with self.assertRaisesRegex(ValueError, "not ordered"):
            TrajectoryLayerModalityRoutedSpargeAttentionBackend(
                aggressive_topk=(0.25, 0.50),
                safe_topk=(0.35, 0.45),
                sensitive_layers=(30,),
                dense_step_indices=(0, 19),
            )

    def test_trajectory_policy_routes_conservative_sparse_anchor(self):
        backend = TrajectoryLayerModalityRoutedSpargeAttentionBackend(
            aggressive_topk=(0.25, 0.30),
            safe_topk=(0.30, 0.35),
            sensitive_layers=(30,),
            dense_step_indices=(),
            anchor_step_indices=(0,),
            anchor_aggressive_topk=(0.40, 0.45),
            anchor_safe_topk=(0.50, 0.55),
            minimum_sparse_tokens=8,
        )
        calls = []

        class FakePolicy:
            def __init__(self, name):
                self.name = name

            def __call__(self, query, key, value):
                calls.append(self.name)
                return query

        backend.layer_policy = FakePolicy("normal")
        backend.anchor_policy = FakePolicy("anchor")
        tensor = torch.zeros(16, 2, 2)
        with attention_step(0, 20):
            backend(tensor, tensor, tensor)
        with attention_step(1, 20):
            backend(tensor, tensor, tensor)
        self.assertEqual(calls, ["anchor", "normal"])

    def test_trajectory_policy_rejects_invalid_anchor_contract(self):
        with self.assertRaisesRegex(ValueError, "both aggressive and safe"):
            TrajectoryLayerModalityRoutedSpargeAttentionBackend(
                aggressive_topk=0.25,
                safe_topk=0.35,
                sensitive_layers=(30,),
                dense_step_indices=(),
                anchor_step_indices=(0,),
                anchor_aggressive_topk=0.40,
            )
        with self.assertRaisesRegex(ValueError, "disjoint"):
            TrajectoryLayerModalityRoutedSpargeAttentionBackend(
                aggressive_topk=0.25,
                safe_topk=0.35,
                sensitive_layers=(30,),
                dense_step_indices=(0,),
                anchor_step_indices=(0,),
                anchor_aggressive_topk=0.40,
                anchor_safe_topk=0.50,
            )

    def test_trajectory_policy_allows_explicit_research_budget_below_quarter(self):
        backend = TrajectoryLayerModalityRoutedSpargeAttentionBackend(
            aggressive_topk=(0.15, 0.20),
            sensitive_layers=(38, 39, 40, 41, 42),
            dense_step_indices=(),
            safe_topk=(0.20, 0.25),
            anchor_step_indices=(0,),
            anchor_aggressive_topk=(0.35, 0.40),
            anchor_safe_topk=(0.45, 0.50),
            recovery_step_indices=(17, 18, 19),
            recovery_aggressive_topk=(0.25, 0.30),
            recovery_safe_topk=(0.30, 0.35),
            experimental_minimum_topk=0.125,
        )

        self.assertEqual(backend.anchor_step_indices, frozenset((0,)))
        self.assertEqual(backend.recovery_step_indices, frozenset((17, 18, 19)))

    def test_temporal_rail_allows_sub_eighth_far_field_research_budget(self):
        backend = TrajectoryLayerModalityRoutedSpargeAttentionBackend(
            aggressive_topk=0.10,
            sensitive_layers=(38,),
            dense_step_indices=(),
            safe_topk=0.125,
            experimental_minimum_topk=0.0625,
            temporal_correspondence_radius=1,
            temporal_spatial_block_radius=1,
            temporal_global_anchor_stride=8,
            temporal_global_spatial_block_radius=0,
        )

        self.assertEqual(backend.layer_policy.aggressive.topk, 0.10)
        self.assertEqual(backend.layer_policy.safe.topk, 0.125)
        self.assertEqual(
            backend.layer_policy.aggressive.temporal_correspondence_radius, 1
        )
        self.assertEqual(
            backend.layer_policy.aggressive.temporal_global_anchor_stride, 8
        )

    def test_trajectory_policy_keeps_quarter_as_default_floor(self):
        with self.assertRaisesRegex(ValueError, "budgets are not ordered"):
            TrajectoryLayerModalityRoutedSpargeAttentionBackend(
                aggressive_topk=0.20,
                sensitive_layers=(38,),
                dense_step_indices=(),
                safe_topk=0.25,
            )

    def test_sol_policy_routes_by_step_layer_and_protects_prefix(self):
        backend = TrajectoryLayerModalityRoutedSolAttentionBackend(
            source="/tmp/unused-sol-source",
            tau=1.0,
            sensitive_tau=0.8,
            sensitive_layers=(30,),
            anchor_step_indices=(0,),
            anchor_tau=0.6,
            recovery_step_indices=(17, 18, 19),
            recovery_tau=0.8,
            minimum_sparse_tokens=8,
        )
        calls = []

        def fake_kernel(query, key, value, **kwargs):
            calls.append(kwargs)
            return query

        import h3serve.native_engine.model.kernels as kernels

        original = kernels._load_experimental_sol_attn
        kernels._load_experimental_sol_attn = lambda _source: fake_kernel
        tensor = torch.zeros(16, 2, 2)
        try:
            with attention_protected_prefix(5), attention_step(0, 20), attention_layer(5):
                backend(tensor, tensor, tensor)
            with attention_protected_prefix(5), attention_step(8, 20), attention_layer(30):
                backend(tensor, tensor, tensor)
            with attention_protected_prefix(5), attention_step(17, 20), attention_layer(5):
                backend(tensor, tensor, tensor)
        finally:
            kernels._load_experimental_sol_attn = original

        self.assertEqual([item["tau"] for item in calls], [0.6, 0.8, 0.8])
        self.assertTrue(all(item["sink_blocks"] == (0, 1) for item in calls))
        self.assertTrue(all(item["sink_q"] == (0, 1) for item in calls))

    def test_sol_policy_fails_closed_without_request_context(self):
        backend = TrajectoryLayerModalityRoutedSolAttentionBackend(
            source="/tmp/unused-sol-source",
            tau=1.0,
            sensitive_tau=0.8,
            sensitive_layers=(30,),
            minimum_sparse_tokens=8,
        )
        calls = []

        def dense(query, key, value):
            calls.append("dense")
            return value

        import h3serve.native_engine.model.kernels as kernels

        original = kernels.sage_attention_sm89
        kernels.sage_attention_sm89 = dense
        tensor = torch.zeros(16, 2, 2)
        try:
            self.assertIs(backend(tensor, tensor, tensor), tensor)
        finally:
            kernels.sage_attention_sm89 = original
        self.assertEqual(calls, ["dense"])


if __name__ == "__main__":
    unittest.main()
