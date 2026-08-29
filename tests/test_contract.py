from __future__ import annotations

import unittest

from h3serve.contract import (
    ContractError,
    GenerationSpec,
    MODEL_LAUNCHERS,
    SecondSamplingSpec,
    default_quality,
    public_options,
    resolve_frames,
    resolve_geometry,
    resolve_upscale_geometry,
)


class ContractTest(unittest.TestCase):
    def test_six_resource_launchers_are_orthogonal_to_base_lora_variant(self) -> None:
        options = public_options()
        self.assertEqual(set(options["model_launchers"]), set(MODEL_LAUNCHERS))
        self.assertEqual(len(MODEL_LAUNCHERS), 6)
        self.assertEqual(
            options["model_launchers"]["fl2va_w4a8_8gb"]["variants"],
            ["base", "lora"],
        )
        self.assertTrue(
            options["model_launchers"]["ref2va_w4a8_8gb"]["second_sampling"]
        )

        base = GenerationSpec.from_mapping({
            "prompt": "8GB base",
            "runtime_launcher": "fl2va_w4a8_8gb",
            "model_variant": "base",
            "resolution": "720p",
            "duration_seconds": 15,
        })
        lora = GenerationSpec.from_mapping({
            "prompt": "8GB LoRA",
            "runtime_launcher": "fl2va_w4a8_8gb",
            "model_variant": "lora",
            "resolution": "720p",
            "duration_seconds": 15,
        })
        self.assertEqual(base.runtime_launcher, "fl2va_w4a8_8gb")
        self.assertEqual(lora.runtime_launcher, "fl2va_w4a8_8gb")
        self.assertEqual((base.vram_profile, lora.vram_profile), ("8gb", "8gb"))
        self.assertEqual((base.weight_tier, lora.weight_tier), ("w4a8", "w4a8"))
        self.assertEqual((base.engine, lora.engine), ("original", "lora"))
        self.assertEqual(GenerationSpec.from_mapping(lora.to_dict()), lora)

        with self.assertRaisesRegex(ContractError, "up to 720p"):
            GenerationSpec.from_mapping({
                "prompt": "unsupported 8GB geometry",
                "runtime_launcher": "fl2va_w4a8_8gb",
                "resolution": "1080p",
                "duration_seconds": 5,
            })
        with self.assertRaisesRegex(ContractError, "disagrees"):
            GenerationSpec.from_mapping({
                "prompt": "mismatched family",
                "runtime_launcher": "ref2va_w4a8_8gb",
                "service_family": "first_last",
            })

    def test_resource_backend_identity_round_trips_and_caps_geometry(self) -> None:
        sixteen = GenerationSpec.from_mapping({
            "prompt": "16GB 720p",
            "runtime_launcher": "fl2va_int8_16gb",
            "resolution": "720p",
            "duration_seconds": 15,
        })
        self.assertEqual(sixteen.vram_profile, "16gb")
        self.assertEqual(sixteen.runtime_launcher, "fl2va_int8_16gb")
        self.assertEqual(GenerationSpec.from_mapping(sixteen.to_dict()), sixteen)
        sixteen_1080p = GenerationSpec.from_mapping({
            "prompt": "16GB experimental native 1080p",
            "runtime_launcher": "fl2va_int8_16gb",
            "resolution": "1080p",
            "duration_seconds": 15,
        })
        self.assertEqual((sixteen_1080p.width, sixteen_1080p.height), (1920, 1088))
        self.assertEqual(sixteen_1080p.frames, 362)

        with self.assertRaisesRegex(ContractError, "24gb.*1080p"):
            GenerationSpec.from_mapping({
                "prompt": "24GB first-pass 2K is not released",
                "runtime_launcher": "fl2va_int8_24gb",
                "resolution": "2k",
                "duration_seconds": 5,
            })
        # Legacy identifiers remain readable but canonicalize immediately.
        legacy = GenerationSpec.from_mapping({
            "prompt": "persisted launcher", "runtime_launcher": "fl2va_int8"
        })
        self.assertEqual(legacy.runtime_launcher, "fl2va_int8_24gb")

    def test_second_sampling_contract_uses_h3_geometry_and_own_solver_range(self) -> None:
        source = GenerationSpec.from_mapping({
            "prompt": "source card", "resolution": "480p",
            "aspect_ratio": "16:9", "duration_seconds": 15,
        })
        second = SecondSamplingSpec.from_mapping({
            "resolution": "1080p", "steps": 1, "acceleration": 80,
            "denoise": 0.2, "memory_mode": "low_vram",
            "temporal_window_frames": 119,
        }, source=source)
        self.assertEqual((second.width, second.height), (1920, 1088))
        self.assertEqual(second.steps, 1)
        self.assertEqual(second.strength, "standard")
        self.assertEqual(second.model_variant, "base")
        self.assertEqual(second.denoise, 0.2)
        self.assertEqual(second.memory_mode, "auto")
        self.assertEqual(second.temporal_window_frames, 119)
        self.assertEqual(SecondSamplingSpec(**second.to_dict()), second)
        second_2k = SecondSamplingSpec.from_mapping({
            "resolution": "2k", "steps": 1, "acceleration": 75,
            "denoise": 0.2,
        }, source=source)
        self.assertEqual((second_2k.width, second_2k.height), (2560, 1440))
        second_1440p = SecondSamplingSpec.from_mapping({
            "resolution": "1440p", "steps": 1, "acceleration": 75,
            "denoise": 0.2,
        }, source=source)
        self.assertEqual(second_1440p.resolution, "2k")
        self.assertEqual((second_1440p.width, second_1440p.height), (2560, 1440))

        strong = SecondSamplingSpec.from_mapping({
            "resolution": "1080p", "steps": 8,
            "strength": "strong",
        }, source=source)
        self.assertEqual(strong.model_variant, "base")
        self.assertEqual(strong.strength, "strong")
        self.assertEqual(strong.denoise, 0.30)
        with self.assertRaisesRegex(ContractError, "Base weights only"):
            SecondSamplingSpec.from_mapping({
                "resolution": "1080p", "model_variant": "lora",
            }, source=source)
        with self.assertRaisesRegex(ContractError, "between 68 and 362"):
            SecondSamplingSpec.from_mapping({
                "resolution": "1080p", "temporal_window_frames": 51,
            }, source=source)

        same_size = GenerationSpec.from_mapping({
            "prompt": "already 720p", "resolution": "720p",
        })
        with self.assertRaisesRegex(ContractError, "larger"):
            SecondSamplingSpec.from_mapping(
                {"resolution": "720p"}, source=same_size
            )

    def test_legacy_memory_modes_normalize_and_are_not_public(self) -> None:
        for mode in ("auto", "performance", "low_vram"):
            spec = GenerationSpec.from_mapping({
                "prompt": "memory route",
                "memory_mode": mode,
            })
            self.assertEqual(spec.memory_mode, "auto")
            self.assertEqual(GenerationSpec.from_mapping(spec.to_dict()), spec)
        options = public_options()
        self.assertNotIn("memory_modes", options)
        self.assertFalse(options["device_memory_backend"]["user_execution_modes"])
        self.assertEqual(
            options["device_memory_backend"]["policy"],
            "startup_fixed_profile_then_minimum_latency_graph",
        )
        self.assertFalse(
            options["device_memory_backend"]["cross_profile_routing"]
        )
        with self.assertRaisesRegex(ContractError, "memory_mode"):
            GenerationSpec.from_mapping({
                "prompt": "bad memory route", "memory_mode": "magic"
            })

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
        self.assertIn("75=Human审阅质量拐点", limits["acceleration"]["meaning"])
        self.assertIn("75–100", limits["acceleration"]["meaning"])
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
        aligned_360 = GenerationSpec.from_mapping({
            "prompt": "360p fixed checkpoint preview",
            "resolution": "360p",
            "sampling_steps": 8,
            "acceleration": 0,
            "execution_mode": "checkpoint",
            "checkpoint_step": 4,
            "checkpoint_retain": True,
            "checkpoint_preview": True,
            "checkpoint_preview_steps": 4,
            "checkpoint_preview_resolution": "360p",
        })
        self.assertEqual((aligned_360.width, aligned_360.height), (640, 352))

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
        self.assertEqual(resolve_geometry("2k", "16:9"), (2560, 1440))
        self.assertEqual(resolve_geometry("360p", "9:16"), (352, 640))
        for resolution in ("360p", "480p", "720p", "1080p", "2k"):
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
            "aspect_ratio": "16:9", "duration_seconds": 15,
        })
        self.assertEqual((preset.width, preset.height, preset.frames), (1920, 1088, 362))

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
        GenerationSpec.from_mapping({
            "prompt": "full-length four by three", "resolution": "1080p",
            "aspect_ratio": "4:3", "duration_seconds": 15,
        })

        custom = GenerationSpec.from_mapping({
            "prompt": "custom 1080p", "mode": "advanced",
            "width": 1920, "height": 1088, "frames": 362,
            "actual_steps": 9,
        })
        self.assertEqual(custom.frames, 362)
        with self.assertRaisesRegex(ContractError, "24gb.*1080p"):
            GenerationSpec.from_mapping({
                "prompt": "2k is reserved for second sampling",
                "resolution": "2k", "aspect_ratio": "16:9",
                "duration_seconds": 15, "memory_mode": "auto",
            })
        with self.assertRaisesRegex(ContractError, r"frames must be 5\.\.362"):
            GenerationSpec.from_mapping({
                "prompt": "custom too long", "mode": "advanced",
                "width": 1920, "height": 1088, "frames": 379,
                "actual_steps": 9,
            })

        options = public_options()
        self.assertEqual(options["duration"]["max_by_resolution"]["1080p"], 15)
        self.assertEqual(options["duration"]["max_by_preset"]["1080p"]["16:9"], 15)
        self.assertEqual(options["duration"]["max_by_preset"]["1080p"]["4:3"], 15)
        self.assertEqual(options["duration"]["max_by_preset"]["1080p"]["1:1"], 15)
        self.assertEqual(options["duration"]["max_by_preset"]["2k"]["16:9"], 15)
        self.assertEqual(options["duration"]["max_native_pixel_frames"], 2560 * 1440 * 362)
        self.assertEqual(options["advanced_limits"]["dimension_max"], 2560)

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
        self.assertEqual(resolve_upscale_geometry(1280, 720, "1440p"), (2560, 1440))
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

    def test_advanced_lora_and_native_2k_safety_bounds(self) -> None:
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

    def test_2k_is_internal_second_sampling_geometry_not_public_first_pass(self) -> None:
        payload = {
            "prompt": "persisted H3 second sampling target",
            "engine": "original",
            "mode": "advanced",
            "width": 2560,
            "height": 1440,
            "frames": 362,
            "duration_seconds": 362 / 24,
            "actual_steps": 20,
            "weight_tier": "int8",
            "vram_profile": "24gb",
        }
        with self.assertRaises(ContractError):
            GenerationSpec.from_mapping(payload)
        restored = GenerationSpec.from_mapping(
            payload, allow_second_sampling_target=True
        )
        self.assertEqual((restored.width, restored.height), (2560, 1440))
        with self.assertRaises(ContractError):
            GenerationSpec.from_mapping({
                "prompt": "too many pixels", "advanced": True, "width": 2560,
                "height": 1472, "frames": 124, "actual_steps": 9,
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
