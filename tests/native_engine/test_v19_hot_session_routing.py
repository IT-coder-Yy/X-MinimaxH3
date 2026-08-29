from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

from h3serve.native_engine.hot_session import HotSessionRequest, NativeT2AVHotSession


class _RecordingSelector:
    def __init__(self) -> None:
        self.workload = None
        self.acceleration = None
        self.runtime_controller = object()

    def select(self, *, workload, acceleration, required_actual_step_indices=()):
        self.workload = workload
        self.acceleration = acceleration
        self.required_actual_step_indices = required_actual_step_indices
        return SimpleNamespace(
            actual_step_indices=(0, 1, 2, 3),
            attention_action_schedule=(),
            summary={
                "policy_id": "test_v19",
                "accelerated": False,
                "reason": "test_dense",
            },
            runtime_controller=self.runtime_controller,
        )


def _request() -> HotSessionRequest:
    return HotSessionRequest(
        prompt="test",
        seed=1,
        width=1280,
        height=736,
        frames=124,
        fps=24,
        steps=4,
        output_path=Path("/tmp/v19-routing-test.mp4"),
        actual_step_indices=(0, 3),
        v19_acceleration=100.0,
    )


def _session(selector):
    session = object.__new__(NativeT2AVHotSession)
    session.engine = "original"
    session.v19_selector = selector
    return session


class V19HotSessionRoutingTests(unittest.TestCase):
    def test_selection_uses_exact_qwen_token_count_not_prompt_length(self) -> None:
        selector = _RecordingSelector()
        selected = _session(selector)._apply_v19_selection(
            _request(),
            text_tokens=500,
        )
        # 37 video latent frames * 920 spatial rows + 414 audio + 500 text.
        self.assertEqual(selector.workload.packed_tokens, 34_954)
        self.assertEqual(selector.acceleration, 100.0)
        self.assertEqual(selector.required_actual_step_indices, ())
        self.assertEqual(selected.actual_step_indices, (0, 1, 2, 3))
        self.assertEqual(selected.acceleration_plan_summary["reason"], "test_dense")
        self.assertIs(
            selected.mechanistic_runtime_controller,
            selector.runtime_controller,
        )

    def test_missing_release_bundle_fails_closed_to_complete_dense(self) -> None:
        selected = _session(None)._apply_v19_selection(
            _request(),
            text_tokens=500,
        )
        self.assertEqual(selected.actual_step_indices, (0, 1, 2, 3))
        self.assertEqual(selected.attention_action_schedule, ())
        self.assertEqual(
            selected.acceleration_plan_summary["reason"],
            "v19_release_bundle_unavailable_dense_fallback",
        )

    def test_preview_step_is_a_hard_actual_compute_constraint(self) -> None:
        selector = _RecordingSelector()
        request = replace(
            _request(),
            preview_step_index=2,
            preview_output_path=Path("/tmp/v19-routing-preview.mp4"),
        )
        _session(selector)._apply_v19_selection(request, text_tokens=500)
        self.assertEqual(selector.required_actual_step_indices, (2,))


if __name__ == "__main__":
    unittest.main()
