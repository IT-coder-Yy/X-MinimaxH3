from __future__ import annotations

import unittest

from h3serve.contract import (
    ContractError,
    GenerationSpec,
    default_quality,
    public_options,
    resolve_frames,
    resolve_geometry,
    resolve_upscale_geometry,
)


class ContractTest(unittest.TestCase):
    def test_prompt_is_validated_but_not_rewritten(self) -> None:
        prompt = "  subject_definitions:\n<Subject 1> from <Picture 1>.\n  "
        self.assertEqual(GenerationSpec.from_mapping({"prompt": prompt}).prompt, prompt)

    def test_reference_media_resolution_policy_round_trips(self) -> None:
        defaults = GenerationSpec.from_mapping({"prompt": "reference defaults"})
        self.assertEqual(defaults.reference_image_resolution, "720p")
        self.assertEqual(defaults.reference_video_resolution, "360p")

        configured = GenerationSpec.from_mapping({
            "prompt": "reference policies",
            "reference_image_resolution": "original",
            "reference_video_resolution": "480p",
        })
        self.assertEqual(configured.reference_image_resolution, "original")
        self.assertEqual(configured.reference_video_resolution, "480p")
        self.assertEqual(GenerationSpec.from_mapping(configured.to_dict()), configured)

        processing = public_options()["reference_media_processing"]
        self.assertEqual(processing["image_default"], "720p")
        self.assertEqual(processing["video_default"], "360p")
        self.assertTrue(processing["preserve_aspect_ratio"])
        self.assertFalse(processing["upscale_small_inputs"])

        with self.assertRaisesRegex(ContractError, "reference_image_resolution"):
            GenerationSpec.from_mapping({
                "prompt": "bad image policy",
                "reference_image_resolution": "1080p",
            })

    def test_two_control_joint_acceleration_contract_round_trips(self) -> None:
        spec = GenerationSpec.from_mapping({
            "prompt": "joint planner",
            "engine": "original",
            "mode": "advanced",
            "width": 1280,
            "height": 736,
            "duration_seconds": 15,
            "sampling_steps": 15,
            "acceleration": 72.5,
            "seed": 23,
        })
        self.assertTrue(spec.joint_acceleration_enabled)
        self.assertEqual(spec.sampling_steps, 15)
        self.assertEqual(spec.acceleration, 72.5)

    def test_sampling_steps_accept_each_variant_published_range(self) -> None:
        for variant, bounds in (("base", (5, 30)), ("lora", (4, 10))):
            for steps in bounds:
                spec = GenerationSpec.from_mapping({
                    "prompt": "range contract",
                    "service_family": "first_last",
                    "model_variant": variant,
                    "sampling_steps": steps,
                    "acceleration": 0,
                })
                self.assertEqual(spec.sampling_steps, steps)
        self.assertNotIn("actual_steps", spec.preset)
        self.assertNotIn("attention_keep_ratio", spec.preset)
        self.assertEqual(GenerationSpec.from_mapping(spec.to_dict()), spec)
        limits = public_options()["advanced_limits"]
        self.assertIn("sampling_steps", limits)
        self.assertIn("acceleration", limits)
        self.assertEqual(limits["quality_protection"], "internal_non_disableable")

        with self.assertRaisesRegex(ContractError, "cannot be mixed"):
            GenerationSpec.from_mapping({
                "prompt": "mixed controls", "mode": "advanced",
                "width": 864, "height": 480,
                "sampling_steps": 20, "acceleration": 50,
                "actual_steps": 12,
            })
        with self.assertRaisesRegex(ContractError, "provided together"):
            GenerationSpec.from_mapping({
                "prompt": "missing dial", "mode": "advanced",
                "width": 864, "height": 480,
                "sampling_steps": 20,
            })
        preset_controls = GenerationSpec.from_mapping({
            "prompt": "preset with direct execution controls",
            "resolution": "480p", "sampling_steps": 20, "acceleration": 0,
        })
        self.assertFalse(preset_controls.advanced)
        self.assertTrue(preset_controls.joint_acceleration_enabled)

    def test_checkpoint_task_contract_round_trips(self) -> None:
        spec = GenerationSpec.from_mapping({
            "prompt": "checkpoint",
            "execution_mode": "checkpoint",
            "checkpoint_step": 6,
            "checkpoint_retain": True,
            "checkpoint_preview": True,
            "checkpoint_preview_steps": 4,
            "checkpoint_preview_resolution": "360p",
        })
        self.assertEqual(spec.execution_mode, "checkpoint")
        self.assertEqual(spec.checkpoint_step, 6)
        self.assertTrue(spec.checkpoint_retain)
        self.assertTrue(spec.checkpoint_preview)
        self.assertEqual(
            GenerationSpec.from_mapping(spec.to_dict()).checkpoint_step, 6
        )
        with self.assertRaises(ContractError):
            GenerationSpec.from_mapping({
                "prompt": "invalid final checkpoint",
                "execution_mode": "checkpoint",
                "checkpoint_step": 20,
            })

    def test_lora_joint_checkpoint_keeps_user_trajectory_and_dial(self) -> None:
        spec = GenerationSpec.from_mapping({
            "prompt": "scheduled LoRA breakpoint",
            "service_family": "reference",
            "model_variant": "lora",
            "mode": "advanced",
            "width": 864,
            "height": 480,
            "duration_seconds": 5,
            "sampling_steps": 8,
            "acceleration": 50,
            "execution_mode": "checkpoint",
            "checkpoint_step": 3,
            "checkpoint_retain": True,
            "checkpoint_preview": True,
            "checkpoint_preview_steps": 4,
        })
        self.assertEqual(spec.engine, "reference_lora")
        self.assertEqual(spec.sampling_steps, 8)
        self.assertEqual(spec.acceleration, 50.0)
        self.assertEqual(spec.checkpoint_step, 3)
        self.assertTrue(spec.joint_acceleration_enabled)
        self.assertEqual(GenerationSpec.from_mapping(spec.to_dict()), spec)

        with self.assertRaisesRegex(ContractError, "before the final step"):
            GenerationSpec.from_mapping({
                **spec.to_dict(),
                "checkpoint_step": 8,
            })

    def test_public_mode_alias_and_advanced_duration(self) -> None:
        preset = GenerationSpec.from_mapping({
            "mode": "preset", "prompt": "test", "duration_seconds": 5,
        })
        self.assertFalse(preset.advanced)
        self.assertEqual(preset.to_dict()["mode"], "preset")

        advanced = GenerationSpec.from_mapping({
            "mode": "advanced", "prompt": "test", "width": 864,
            "height": 480, "duration_seconds": 5, "actual_steps": 9,
        })
        self.assertTrue(advanced.advanced)
        self.assertEqual(advanced.frames, 124)
        self.assertEqual(advanced.to_dict()["mode"], "advanced")

        with self.assertRaises(ContractError):
            GenerationSpec.from_mapping({
                "mode": "preset", "advanced": True, "prompt": "test",
            })

    def test_common_geometries_follow_h3_grid(self) -> None:
        self.assertEqual(resolve_geometry("480p", "16:9"), (864, 480))
        self.assertEqual(resolve_geometry("720p", "16:9"), (1280, 736))
        self.assertEqual(resolve_geometry("1080p", "16:9"), (1920, 1088))
        self.assertEqual(resolve_geometry("360p", "9:16"), (352, 640))
        for resolution in ("360p", "480p", "720p", "1080p"):
            for ratio in ("1:1", "4:3", "3:4", "16:9", "9:16"):
                width, height = resolve_geometry(resolution, ratio)
                self.assertEqual(width % 32, 0)
                self.assertEqual(height % 32, 0)

    def test_duration_is_aligned_to_h3_frame_grid(self) -> None:
        self.assertEqual(resolve_frames(5), (124, 124 / 24))
        self.assertEqual(resolve_frames(15), (362, 362 / 24))
        for duration in (1, 2.5, 10, 15):
            frames, _ = resolve_frames(duration)
            self.assertEqual((frames - 5) % 17, 0)

    def test_native_pixel_frame_budget_is_enforced_for_presets_and_custom(self) -> None:
        preset = GenerationSpec.from_mapping({
            "prompt": "validated 1080p", "resolution": "1080p",
            "aspect_ratio": "16:9", "duration_seconds": 8,
        })
        self.assertEqual((preset.width, preset.height, preset.frames), (1920, 1088, 192))
        with self.assertRaisesRegex(ContractError, "at most 8.000 seconds"):
            GenerationSpec.from_mapping({
                "prompt": "too long", "resolution": "1080p",
                "aspect_ratio": "16:9", "duration_seconds": 8.5,
            })

        four_three = GenerationSpec.from_mapping({
            "prompt": "longer four by three", "resolution": "1080p",
            "aspect_ratio": "4:3", "duration_seconds": 10,
        })
        self.assertEqual((four_three.width, four_three.height, four_three.frames),
                         (1440, 1088, 243))
        square = GenerationSpec.from_mapping({
            "prompt": "longer square", "resolution": "1080p",
            "aspect_ratio": "1:1", "duration_seconds": 13.5,
        })
        self.assertEqual((square.width, square.height, square.frames),
                         (1088, 1088, 328))
        with self.assertRaisesRegex(ContractError, "at most 10.125 seconds"):
            GenerationSpec.from_mapping({
                "prompt": "four by three too long", "resolution": "1080p",
                "aspect_ratio": "4:3", "duration_seconds": 10.5,
            })

        custom = GenerationSpec.from_mapping({
            "prompt": "custom 1080p", "mode": "advanced",
            "width": 1920, "height": 1088, "frames": 192,
            "actual_steps": 9,
        })
        self.assertEqual(custom.frames, 192)
        with self.assertRaisesRegex(ContractError, r"width\*height\*frames"):
            GenerationSpec.from_mapping({
                "prompt": "custom too long", "mode": "advanced",
                "width": 1920, "height": 1088, "frames": 209,
                "actual_steps": 9,
            })

        options = public_options()
        self.assertEqual(options["duration"]["max_by_resolution"]["1080p"], 8)
        self.assertEqual(options["duration"]["max_by_preset"]["1080p"]["16:9"], 8)
        self.assertEqual(options["duration"]["max_by_preset"]["1080p"]["4:3"], 243 / 24)
        self.assertEqual(options["duration"]["max_by_preset"]["1080p"]["1:1"], 328 / 24)
        self.assertEqual(options["duration"]["max_native_pixel_frames"], 1920 * 1088 * 192)
        self.assertEqual(options["advanced_limits"]["dimension_max"], 1920)

    def test_default_is_balanced_9_actual_11_forecast(self) -> None:
        spec = GenerationSpec.from_mapping({"prompt": "test", "seed": 1})
        self.assertEqual(spec.engine, "original")
        self.assertEqual(spec.quality, "balanced")
        self.assertEqual(spec.preset["actual_steps"], 9)
        self.assertEqual(spec.preset["forecast_steps"], 11)

    def test_two_family_contract_resolves_request_local_model_variant(self) -> None:
        base = GenerationSpec.from_mapping({
            "prompt": "base", "service_family": "reference",
            "model_variant": "base", "seed": 11,
        })
        turbo = GenerationSpec.from_mapping({
            "prompt": "turbo", "service_family": "reference",
            "model_variant": "lora", "quality": "quality", "seed": 12,
        })
        self.assertEqual((base.engine, base.service_family, base.model_variant),
                         ("reference", "reference", "base"))
        self.assertEqual((turbo.engine, turbo.service_family, turbo.model_variant),
                         ("reference_lora", "reference", "lora"))

    def test_fork_preview_contract_is_bounded(self) -> None:
        spec = GenerationSpec.from_mapping({
            "prompt": "preview", "preview_mode": "pause",
            "preview_branch_steps": 3, "seed": 13,
        })
        self.assertEqual(spec.preview_mode, "pause")
        self.assertEqual(spec.preview_branch_steps, 3)
        self.assertFalse(spec.preview_fast_finish)
        fast = GenerationSpec.from_mapping({
            "prompt": "fast preview", "preview_mode": "auto",
            "preview_fast_finish": True,
        })
        self.assertTrue(fast.preview_fast_finish)
        with self.assertRaisesRegex(Exception, "between 1 and 3"):
            GenerationSpec.from_mapping({
                "prompt": "invalid", "preview_mode": "auto",
                "preview_branch_steps": 4,
            })

    def test_public_options_hide_execution_configuration(self) -> None:
        options = public_options()
        balanced = options["engines"]["original"]["presets"]["balanced"]
        self.assertNotIn("actual_steps", balanced)
        self.assertNotIn("backend_preset", balanced)
        self.assertIn("description", balanced)

    def test_fixed_engine_options_publish_only_current_product(self) -> None:
        fidelity = public_options("original")
        self.assertEqual(fidelity["deployment_mode"], "fixed_engine")
        self.assertEqual(fidelity["current_engine"], "first_last")
        self.assertEqual(set(fidelity["engines"]), {"original", "lora"})
        self.assertEqual(fidelity["defaults"]["quality"], "balanced")

        turbo = public_options("lora")
        self.assertEqual(turbo["current_engine"], "first_last")
        self.assertEqual(set(turbo["engines"]), {"original", "lora"})
        # public_options describes a family. Launcher compatibility chooses
        # its initial variant in create_app rather than changing this schema.
        self.assertEqual(turbo["defaults"]["quality"], "balanced")
        self.assertEqual(default_quality("lora"), "quality")

        reference_turbo = public_options("reference_lora")
        self.assertEqual(reference_turbo["current_engine"], "reference")
        self.assertEqual(set(reference_turbo["engines"]), {"reference", "reference_lora"})
        self.assertEqual(reference_turbo["defaults"]["quality"], "balanced")

    def test_reference_turbo_uses_lora_steps_and_reference_product(self) -> None:
        spec = GenerationSpec.from_mapping({
            "prompt": "reference turbo", "engine": "reference_lora",
            "quality": "quality", "seed": 82416,
        })
        self.assertEqual(spec.preset["steps"], 6)
        self.assertNotIn("actual_steps", spec.preset)

    def test_rejects_invalid_request(self) -> None:
        with self.assertRaises(ContractError):
            GenerationSpec.from_mapping({"prompt": ""})
        with self.assertRaises(ContractError):
            GenerationSpec.from_mapping({"prompt": "x", "duration_seconds": 16})
        with self.assertRaises(ContractError):
            GenerationSpec.from_mapping({"prompt": "x", "aspect_ratio": "2:1"})

    def test_upscale_contract_preserves_ratio_and_round_trips(self) -> None:
        self.assertEqual(resolve_upscale_geometry(864, 480, "1080p"), (1944, 1080))
        self.assertEqual(resolve_upscale_geometry(1280, 720, "2k"), (2560, 1440))
        spec = GenerationSpec.from_mapping({
            "prompt": "upscale", "seed": 1, "upscale_enabled": True,
            "upscale_mode": "basic", "upscale_resolution": "1080p",
        })
        self.assertTrue(spec.upscale_enabled)
        self.assertEqual(
            (spec.upscale_target_width, spec.upscale_target_height),
            (1944, 1080),
        )
        self.assertEqual(
            GenerationSpec.from_mapping(spec.to_dict(include_execution=True)), spec
        )
        two_k = GenerationSpec.from_mapping({
            "prompt": "2K delivery", "seed": 2,
            "upscale_enabled": True, "upscale_mode": "basic",
            "upscale_resolution": "2k",
        })
        self.assertEqual(two_k.upscale_resolution, "2k")
        self.assertEqual(
            (two_k.upscale_target_width, two_k.upscale_target_height),
            (2592, 1440),
        )
        with self.assertRaises(ContractError):
            GenerationSpec.from_mapping({
                "prompt": "bad ratio", "upscale_enabled": True,
                "upscale_mode": "advanced", "upscale_target_width": 1920,
                "upscale_target_height": 1200,
            })

    def test_advanced_original_contract_and_round_trip(self) -> None:
        spec = GenerationSpec.from_mapping({
            "prompt": "advanced original",
            "engine": "original",
            "advanced": True,
            "width": 960,
            "height": 544,
            "frames": 243,
            "actual_steps": 10,
            "attention_keep_ratio": 0.75,
            "sparse_scope": "guarded",
            "seed": 7,
        })
        self.assertTrue(spec.advanced)
        self.assertEqual((spec.width, spec.height, spec.frames), (960, 544, 243))
        self.assertEqual(spec.preset["actual_steps"], 10)
        self.assertEqual(spec.preset["forecast_steps"], 10)
        self.assertEqual(spec.preset["attention_keep_ratio"], 0.75)
        self.assertEqual(spec.preset["sparse_scope"], "guarded")
        restored = GenerationSpec.from_mapping(spec.to_dict(include_execution=True))
        self.assertEqual(restored, spec)

    def test_advanced_lora_and_4090_safety_bounds(self) -> None:
        spec = GenerationSpec.from_mapping({
            "prompt": "advanced turbo", "engine": "lora", "advanced": "true",
            "width": 864, "height": 480, "frames": 124, "lora_steps": 7,
        })
        self.assertEqual(spec.preset["steps"], 7)
        self.assertEqual(spec.attention_keep_ratio, 1.0)
        self.assertEqual(spec.sparse_scope, "full")
        with self.assertRaises(ContractError):
            GenerationSpec.from_mapping({
                "prompt": "bad grid", "advanced": True, "width": 850,
                "height": 480, "frames": 124, "actual_steps": 9,
            })
        with self.assertRaises(ContractError):
            GenerationSpec.from_mapping({
                "prompt": "too many pixels", "advanced": True, "width": 1280,
                "height": 1280, "frames": 124, "actual_steps": 9,
            })
        with self.assertRaises(ContractError):
            GenerationSpec.from_mapping({
                "prompt": "unsafe sparse ratio", "advanced": True, "width": 864,
                "height": 480, "frames": 124, "actual_steps": 9,
                "attention_keep_ratio": 0.45,
            })
        with self.assertRaises(ContractError):
            GenerationSpec.from_mapping({
                "prompt": "unknown scope", "advanced": True, "width": 864,
                "height": 480, "frames": 124, "actual_steps": 9,
                "sparse_scope": "everywhere_except_the_bad_parts",
            })


if __name__ == "__main__":
    unittest.main()
