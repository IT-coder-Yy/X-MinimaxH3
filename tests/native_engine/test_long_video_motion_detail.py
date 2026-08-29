from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch

from h3serve.contract import GenerationSpec
from h3serve.native_engine.long_video_motion_detail import (
    ENVIRONMENT_SWITCH,
    make_attention_backend,
    select_candidate,
)
from h3serve.native_engine.model import attention_step, long_video_attention


def _spec(**overrides) -> GenerationSpec:
    values = {
        "prompt": "long motion detail routing contract",
        "engine": "original",
        "quality": "quality",
        "resolution": "720p",
        "aspect_ratio": "16:9",
        "duration_seconds": 15,
        "seed": 82303,
    }
    values.update(overrides)
    return GenerationSpec.from_mapping(values)


def _select(spec: GenerationSpec, **overrides):
    values = {
        "first_frame": None,
        "last_frame": None,
        "reference_images": (),
        "reference_videos": (),
        "reference_audios": (),
    }
    values.update(overrides)
    return select_candidate(spec, **values)


class LongVideoMotionDetailContractTest(unittest.TestCase):
    def test_selector_is_default_off_and_matches_only_measured_envelope(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENVIRONMENT_SWITCH, None)
            self.assertIsNone(_select(_spec()))

        with patch.dict(os.environ, {ENVIRONMENT_SWITCH: "1"}):
            candidate = _select(_spec())
            self.assertIsNotNone(candidate)
            self.assertEqual(
                (
                    candidate.initial_width,
                    candidate.initial_height,
                    candidate.refinement_steps,
                    candidate.refinement_denoise,
                    candidate.dense_tail_steps,
                ),
                (864, 480, 2, 0.025, 1),
            )
            self.assertIsNone(_select(_spec(duration_seconds=5)))
            self.assertIsNone(_select(_spec(quality="balanced")))
            self.assertIsNone(_select(_spec(engine="lora", quality="quality")))
            self.assertIsNone(_select(_spec(engine="reference")))
            self.assertIsNone(_select(_spec(), first_frame=object()))
            self.assertIsNone(_select(_spec(), reference_images=(object(),)))

    def test_measured_policy_reproduces_round98_and_fails_closed(self):
        router = make_attention_backend()
        measured = router.measured_backend
        self.assertTrue(measured.request_guarded)
        self.assertEqual(measured.anchor_step_indices, frozenset((0,)))
        self.assertEqual(measured.recovery_step_indices, frozenset((17, 18, 19)))
        self.assertEqual(measured.layer_policy.aggressive.topk, (0.0625,) * 56)
        self.assertEqual(len(measured.layer_policy.safe.topk), 56)
        self.assertEqual(
            measured.layer_policy.aggressive.temporal_correspondence_radius, 1
        )
        self.assertEqual(
            measured.layer_policy.aggressive.temporal_spatial_block_radius, 1
        )
        self.assertEqual(
            measured.layer_policy.aggressive.temporal_global_anchor_stride, 8
        )

        tensor = torch.zeros(256, 2, 2)
        self.assertTrue(measured._use_dense(tensor))
        with long_video_attention(True), attention_step(1, 20):
            self.assertFalse(measured._use_dense(tensor))
        self.assertTrue(measured._use_dense(tensor))


if __name__ == "__main__":
    unittest.main()
