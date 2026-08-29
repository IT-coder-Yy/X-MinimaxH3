from __future__ import annotations

from dataclasses import replace
import itertools
import unittest

from h3serve.native_engine.planner import (
    H3AttentionCellKey,
    H3BudgetedRiskScheduler,
    H3LayerBand,
    HumanRiskVector,
    SparseActionEstimate,
    SparseBudgetInfeasible,
    SparseBudgetLedger,
    SparseDecisionCell,
    apply_h3_checkpoint_recovery,
    build_h3_sparse_cells,
    update_action_risk,
    verify_sparse_optimality_certificate,
)


def _key(step: int, band: H3LayerBand = H3LayerBand.EARLY) -> H3AttentionCellKey:
    return H3AttentionCellKey(step, band, 0, 15, "ordinary")


def _action(
    name: str,
    cost: float,
    risk: float,
    rank: int,
    *,
    motion: float = 0.0,
) -> SparseActionEstimate:
    return SparseActionEstimate(
        name=name,
        measured_cost_ms=cost,
        reject_risk_ucb=risk,
        fidelity_rank=rank,
        components=HumanRiskVector(motion=motion),
    )


class SparseBudgetSchedulerTests(unittest.TestCase):
    def test_exact_knapsack_prefers_lower_risk_at_same_measured_cost(self) -> None:
        actions = (
            _action("aggressive", 10.0, 10.0, 0),
            _action("protected", 25.0, 2.0, 1),
            _action("dense", 40.0, 0.0, 2),
        )
        cells = (
            SparseDecisionCell(_key(0), actions),
            SparseDecisionCell(_key(1), actions),
        )
        schedule = H3BudgetedRiskScheduler().solve(
            cells,
            ledger=SparseBudgetLedger(50.0),
        )
        self.assertEqual(
            tuple(choice.action.name for choice in schedule.choices),
            ("protected", "protected"),
        )
        self.assertEqual(schedule.estimated_cost_ms, 50.0)
        self.assertEqual(schedule.estimated_reject_risk_ucb, 4.0)
        self.assertTrue(
            verify_sparse_optimality_certificate(cells, schedule).valid
        )

    def test_certificate_matches_independent_bruteforce_and_rejects_tamper(self) -> None:
        cells = (
            SparseDecisionCell(
                _key(0),
                (
                    _action("draft", 9.0, 5.0, 0),
                    _action("safe", 16.0, 1.5, 1),
                    _action("dense", 25.0, 0.0, 2),
                ),
            ),
            SparseDecisionCell(
                _key(1),
                (
                    _action("draft", 8.0, 4.0, 0),
                    _action("safe", 17.0, 1.0, 1),
                    _action("dense", 26.0, 0.0, 2),
                ),
            ),
            SparseDecisionCell(
                _key(2),
                (
                    _action("draft", 7.0, 3.0, 0),
                    _action("safe", 18.0, 0.8, 1),
                    _action("dense", 27.0, 0.0, 2),
                ),
            ),
        )
        budget = 52.0
        quantum = 1.0
        feasible = []
        for actions in itertools.product(*(cell.actions for cell in cells)):
            conservative_units = sum(
                int((action.measured_cost_ms + quantum - 1e-12) // quantum)
                for action in actions
            )
            if conservative_units <= int(budget // quantum):
                feasible.append(
                    (
                        sum(action.reject_risk_ucb for action in actions),
                        max(
                            sum(action.components.motion for action in actions),
                            sum(action.components.clarity for action in actions),
                            sum(action.components.identity for action in actions),
                            sum(action.components.audio for action in actions),
                        ),
                        sum(action.measured_cost_ms for action in actions),
                        tuple(action.name for action in actions),
                    )
                )
        expected = min(feasible)
        schedule = H3BudgetedRiskScheduler(cost_quantum_ms=quantum).solve(
            cells, ledger=SparseBudgetLedger(budget)
        )
        observed = (
            schedule.estimated_reject_risk_ucb,
            schedule.estimated_components.worst_component,
            schedule.estimated_cost_ms,
            tuple(choice.action.name for choice in schedule.choices),
        )
        self.assertEqual(observed, expected)
        self.assertTrue(
            verify_sparse_optimality_certificate(cells, schedule).valid
        )
        certificate = schedule.optimality_certificate
        self.assertIsNotNone(certificate)
        tampered = replace(
            schedule,
            optimality_certificate=replace(
                certificate, choice_sha256="0" * 64
            ),
        )
        verification = verify_sparse_optimality_certificate(cells, tampered)
        self.assertFalse(verification.valid)
        self.assertIn("certificate choice_sha256 mismatch", verification.reasons)

    def test_cost_quantization_is_conservative(self) -> None:
        cells = (
            SparseDecisionCell(
                _key(0),
                (_action("only", 10.01, 0.0, 0),),
            ),
        )
        scheduler = H3BudgetedRiskScheduler(cost_quantum_ms=1.0)
        with self.assertRaises(SparseBudgetInfeasible):
            scheduler.solve(cells, ledger=SparseBudgetLedger(10.99))
        schedule = scheduler.solve(cells, ledger=SparseBudgetLedger(11.0))
        self.assertLessEqual(schedule.estimated_cost_ms, schedule.budget_limit_ms)

    def test_recovery_reserve_is_held_then_opened(self) -> None:
        cell = SparseDecisionCell(
            _key(0),
            (
                _action("draft", 20.0, 5.0, 0),
                _action("dense", 50.0, 0.0, 2),
            ),
        )
        ledger = SparseBudgetLedger(total_budget_ms=60.0, recovery_reserve_ms=20.0)
        scheduler = H3BudgetedRiskScheduler()
        initial = scheduler.solve((cell,), ledger=ledger)
        recovered = scheduler.solve(
            (cell,), ledger=ledger, open_recovery_reserve=True
        )
        self.assertEqual(initial.choices[0].action.name, "draft")
        self.assertEqual(recovered.choices[0].action.name, "dense")
        self.assertFalse(initial.used_recovery_reserve)
        self.assertTrue(recovered.used_recovery_reserve)

    def test_quality_first_reports_budget_overrun(self) -> None:
        cell = SparseDecisionCell(
            _key(0),
            (_action("dense", 40.0, 0.0, 2),),
            minimum_fidelity_rank=2,
        )
        scheduler = H3BudgetedRiskScheduler()
        with self.assertRaises(SparseBudgetInfeasible):
            scheduler.solve((cell,), ledger=SparseBudgetLedger(20.0))
        schedule = scheduler.solve(
            (cell,),
            ledger=SparseBudgetLedger(20.0),
            overflow_policy="quality_first",
        )
        self.assertEqual(schedule.choices[0].action.name, "dense")
        self.assertEqual(schedule.budget_overrun_ms, 20.0)

    def test_h3_builder_encodes_only_reusable_structural_floors(self) -> None:
        def factory(_key):
            return (
                _action("aggressive", 1.0, 2.0, 0),
                _action("protected", 2.0, 1.0, 1),
                _action("dense", 3.0, 0.0, 2),
            )

        cells = build_h3_sparse_cells((0, 1, 2, 3, 4, 5), factory)
        self.assertEqual(len(cells), 42)
        opening = [cell for cell in cells if cell.key.actual_step == 0]
        ordinary_early = next(
            cell
            for cell in cells
            if cell.key.actual_step == 1
            and cell.key.layer_band is H3LayerBand.EARLY
        )
        ordinary_causal = next(
            cell
            for cell in cells
            if cell.key.actual_step == 1
            and cell.key.layer_band is H3LayerBand.CAUSAL
        )
        terminal = [cell for cell in cells if cell.key.actual_step == 5]
        self.assertTrue(all(cell.minimum_fidelity_rank == 1 for cell in opening))
        self.assertEqual(ordinary_early.minimum_fidelity_rank, 0)
        self.assertEqual(ordinary_causal.minimum_fidelity_rank, 1)
        self.assertTrue(all(cell.minimum_fidelity_rank == 1 for cell in terminal))

    def test_checkpoint_trigger_forces_complete_current_causal_band(self) -> None:
        def factory(_key):
            return (
                _action("aggressive", 1.0, 2.0, 0),
                _action("protected", 2.0, 1.0, 1),
                _action("dense", 3.0, 0.0, 2),
            )

        cells = build_h3_sparse_cells((0, 2, 4, 6, 8, 10), factory)
        recovered = apply_h3_checkpoint_recovery(cells, trigger_step=4)
        current = {
            cell.key.layer_band: cell.minimum_fidelity_rank
            for cell in recovered
            if cell.key.actual_step == 4
        }
        following = {
            cell.key.layer_band: cell.minimum_fidelity_rank
            for cell in recovered
            if cell.key.actual_step == 6
        }
        self.assertEqual(current[H3LayerBand.CAUSAL], 2)
        self.assertEqual(current[H3LayerBand.CAUSAL_DETAIL], 2)
        self.assertEqual(current[H3LayerBand.CAUSAL_TERMINAL], 2)
        self.assertGreaterEqual(following[H3LayerBand.CAUSAL], 1)

    def test_request_local_risk_update_changes_future_allocation(self) -> None:
        actions = (
            _action("draft", 10.0, 1.0, 0, motion=1.0),
            _action("dense", 20.0, 0.0, 2),
        )
        cells = (
            SparseDecisionCell(_key(0), actions),
            SparseDecisionCell(_key(1), actions),
        )
        scheduler = H3BudgetedRiskScheduler()
        before = scheduler.solve(cells, ledger=SparseBudgetLedger(30.0))
        updated = update_action_risk(
            cells,
            {
                (cells[0].key.cell_id, "draft"): (
                    100.0,
                    HumanRiskVector(motion=100.0),
                )
            },
        )
        after = scheduler.solve(updated, ledger=SparseBudgetLedger(30.0))
        self.assertEqual(before.action_for(cells[0].key.cell_id).name, "draft")
        self.assertEqual(before.action_for(cells[1].key.cell_id).name, "dense")
        self.assertEqual(after.action_for(cells[0].key.cell_id).name, "dense")
        self.assertEqual(after.action_for(cells[1].key.cell_id).name, "draft")


if __name__ == "__main__":
    unittest.main()
