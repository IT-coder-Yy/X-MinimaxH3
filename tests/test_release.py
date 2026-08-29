from __future__ import annotations

import tempfile
import unittest
import socket
from unittest import mock
import importlib.util
import re
from unittest.mock import patch
from pathlib import Path

from h3serve.config import ServicePaths
from h3serve.models import MODEL_FILES, model_status


class ReleaseLayoutTest(unittest.TestCase):
    def test_busy_port_fails_before_application_or_model_construction(self) -> None:
        from h3serve import app as app_module

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen(1)
            port = occupied.getsockname()[1]
            args = mock.Mock(
                host="127.0.0.1", port=port, api_key=None,
                release_root=Path(__file__).resolve().parents[1],
                data_dir=Path(__file__).resolve().parents[1] / "data",
                max_queued_jobs=1, engine="reference", lazy_load=False,
                unified_console=True,
                memory_profile="auto",
            )
            with mock.patch.object(app_module, "parse_args", return_value=args), mock.patch.object(
                app_module, "create_app"
            ) as create_app:
                with self.assertRaisesRegex(SystemExit, "already in use"):
                    app_module.main()
                create_app.assert_not_called()

    def test_fixed_engine_start_scripts_are_present(self) -> None:
        root = Path(__file__).resolve().parents[1]
        fidelity = root / "scripts/start-fidelity.sh"
        turbo = root / "scripts/start-turbo.sh"
        unified = root / "scripts/start.sh"
        stop = root / "scripts/stop.sh"
        resolver = root / "scripts/_runtime.sh"
        self.assertTrue(fidelity.is_file())
        self.assertTrue(turbo.is_file())
        self.assertTrue(unified.is_file())
        self.assertTrue(stop.is_file())
        self.assertTrue(resolver.is_file())
        self.assertIn("--engine original", fidelity.read_text(encoding="utf-8"))
        self.assertIn("--engine lora", turbo.read_text(encoding="utf-8"))
        self.assertIn("h3_configure_runtime", resolver.read_text(encoding="utf-8"))
        self.assertIn("--unified-console", unified.read_text(encoding="utf-8"))
        self.assertIn("kill -INT", stop.read_text(encoding="utf-8"))
        self.assertIn("server.py", stop.read_text(encoding="utf-8"))

    def test_control_panel_has_no_engine_switch_input(self) -> None:
        root = Path(__file__).resolve().parents[1]
        html = (root / "static/index.html").read_text(encoding="utf-8")
        app = (root / "static/app.js").read_text(encoding="utf-8")
        self.assertNotIn('name="engine"', html)
        self.assertIn('id="engineBanner"', html)
        self.assertIn('id="engineLobby"', html)
        self.assertIn('<strong>X-MinimaxH3</strong>', html)
        self.assertIn('id="chooseWorkspace"', html)
        self.assertIn('id="workspaceDialog"', html)
        self.assertIn('id="exitEngine"', html)
        self.assertIn('id="createPage" hidden', html)
        self.assertIn('class="workspace-tabs" aria-label="工作区" hidden', html)
        self.assertIn('id="tasksPage"', html)
        self.assertIn('id="conversationFeed"', html)
        self.assertIn('class="conversation-composer panel"', html)
        self.assertIn('class="submit-button composer-submit"', html)
        self.assertIn('aria-label="发送生成任务" disabled', html)
        self.assertEqual(html.count('title="发送生成任务"'), 1)
        self.assertLess(html.index('id="formMessage"'), html.index('id="storyboardTitle"'))
        self.assertIn('class="conversation-history panel"', html)
        self.assertNotIn('id="geometryText"', html)
        self.assertLess(html.index('id="exitEngine"'), html.index('<main'))
        self.assertIn('id="cpuUsage"', html)
        self.assertIn('id="vramUsage"', html)
        self.assertIn('id="inferenceDrawer"', html)
        self.assertIn('id="videoSettingsDrawer"', html)
        self.assertIn('name="size_mode"', html)
        self.assertIn('id="customSizeFields" hidden', html)
        self.assertIn('id="queuedJobs"', html)
        self.assertIn('id="shotList"', html)
        self.assertIn('data-prompt-mode="structured"', html)
        self.assertIn('data-prompt-mode="freeform"', html)
        self.assertIn('id="structuredPromptEditor"', html)
        self.assertIn('id="freeformPromptEditor"', html)
        self.assertIn('id="freeformPrompt"', html)
        self.assertIn('class="freeform-writing-guide"', html)
        self.assertIn('<b>撰写建议</b>', html)
        self.assertIn('<code>overall_soundscape:</code>', html)
        self.assertIn('<code>non_diegetic_music:</code>', html)
        self.assertIn('id="freeformDurationField" hidden', html)
        self.assertNotIn('id="referenceStyleOpening"', html)
        self.assertNotIn('整体画面与连续性', html)
        self.assertIn('写什么：定义目标视频中需要反复引用', html)
        self.assertIn('怎么写：建议使用英文', html)
        self.assertIn('示例：&lt;Subject 1&gt;', html)
        self.assertIn('id="enhanceReferences"', html)
        self.assertIn('id="enhanceVisuals"', html)
        self.assertIn('id="enhanceSound"', html)
        self.assertIn('id="referenceFileDrop"', html)
        self.assertIn('id="referenceFiles"', html)
        self.assertIn('id="globalReferenceImageResolution"', html)
        self.assertIn('id="globalReferenceVideoResolution"', html)
        self.assertIn('始终按原宽高比等比例缩小', html)
        self.assertIn('原画幅尺寸、完整构图和视频时长不变', html)
        self.assertIn('id="mimoApiKeyInput"', html)
        self.assertNotIn('id="generationLimitEditor"', html)
        self.assertNotIn('id="generationLimitStatus"', html)
        self.assertNotIn('serverGenerationLimitSettings', app)
        self.assertIn('＞64GB 高速模式', app)
        self.assertIn('≤64GB 兼容模式', app)
        self.assertIn("function setPromptEditorMode(mode)", app)
        self.assertIn("function syncStructuredEditorState()", app)
        self.assertIn("if (document.hidden || uiPollPromise)", app)
        self.assertIn("syncStructuredEditorState();", app)
        self.assertIn("服务响应超时；请检查8090端口转发", app)
        self.assertNotIn("reconcileEngineState().catch(() => {}); }, 1500", app)
        self.assertIn("if (promptEditorMode === 'freeform')", app)
        self.assertIn("$('#compiledPrompt').value = $('#freeformPrompt').value;", app)
        self.assertIn("/studio/prompt-enhancements", app)
        self.assertNotIn("/api/v1/prompt-enhancements", app)
        self.assertIn("summary: $('#referenceSummary').value.trim(),", app)
        self.assertIn("subject_definitions: protocolLines($('#referenceDefinitions').value)", app)
        self.assertIn("请填写总体摘要（summary）", app)
        self.assertNotIn("populateReferenceEditors(fallbackReferenceProtocol(referenceMediaPayload()), {overwrite:true})", app)
        self.assertIn('font-size:10px', (root / "static" / "storyboard.css").read_text(encoding="utf-8"))
        self.assertNotIn('id="upscaleEnabled"', html)
        self.assertNotIn('name="upscale_resolution"', html)
        self.assertIn('name="sampling_steps" type="range" min="5" max="30"', html)
        self.assertIn('name="acceleration" type="range" min="0" max="100"', html)
        self.assertNotIn('name="memory_mode"', html)
        self.assertNotIn('显存执行后端', html)
        self.assertIn('id="checkpointEnabled"', html)
        self.assertIn('name="checkpoint_step" type="range"', html)
        self.assertIn('断点任务', html)
        self.assertNotIn('class="upscale-setting"', html)
        self.assertIn('name="execution_mode" type="hidden" value="complete"', html)
        self.assertIn('id="engineLoadProgress"', html)
        self.assertIn('id="engineLoadBar"', html)
        self.assertIn('renderEngineLoadProgress', app)
        self.assertNotIn('name="checkpoint_preview_steps" type="range"', html)
        self.assertIn('id="globalCheckpointPreviewSteps" type="range" min="1" max="8"', html)
        self.assertIn('id="globalCheckpointPreviewResolution"', html)
        self.assertIn("form.set('checkpoint_preview_steps', String(checkpointPreviewPolicy.steps))", app)
        self.assertIn("form.set('checkpoint_preview_resolution', checkpointPreviewPolicy.resolution)", app)
        self.assertIn("20260829-temporal-window-r1", html)
        self.assertIn('id="secondSamplingDialog"', html)
        self.assertIn('id="secondSamplingForm"', html)
        self.assertNotIn('id="secondSamplingMemoryMode"', html)
        self.assertIn('id="secondSamplingSteps" type="range" min="1" max="8"', html)
        self.assertIn('id="secondSamplingAcceleration" type="range" min="0" max="100"', html)
        self.assertNotIn('id="secondSamplingModelVariant"', html)
        self.assertIn('原始 Base 权重与 SA Solver', html)
        self.assertIn('id="secondSamplingStrength"', html)
        self.assertIn('id="globalSecondSamplingWindow" type="range" min="68" max="362"', html)
        self.assertIn('class="compact-second-sampling-dialog"', html)
        self.assertIn('temporal_window_frames:secondSamplingWindowFrames', app)
        self.assertIn('id="clearLatentCache"', html)
        self.assertNotIn('id="secondSamplingDenoise"', html)
        self.assertIn('name="width" type="range" min="192" max="2560"', html)
        self.assertIn('type="range" min="0.5" max="${currentMaxDuration()}" step="0.5"', app)
        self.assertIn("$('.shot-duration output', card).textContent", app)
        self.assertNotIn('id="qualitySlider"', html)
        self.assertNotIn('id="advancedDrawer"', html)
        self.assertNotIn('data-upscale-resolution="720p"', html)
        self.assertNotIn('data-upscale-resolution="2k"', html)
        self.assertIn('<option>1080p</option>', html)
        self.assertIn('<option>720p</option>', html)
        self.assertIn('<option value="720p">720P</option>', html)
        self.assertIn('<option value="1440p">1440P（实验）</option>', html)
        self.assertNotIn('>2K（实验）</option>', html)
        self.assertIn("options.advanced_limits?.second_sampling?.levels", app)
        self.assertIn("!allowedTargets.has(option.value)", app)
        self.assertNotIn('id="upscaleEnabled" name="upscale_enabled" type="checkbox" checked', html)
        self.assertLess(html.index('id="firstDrop"'), html.index('id="shotList"'))

    def test_open_settings_drawer_uses_full_row(self) -> None:
        root = Path(__file__).resolve().parents[1]
        controls_css = (root / "static" / "composer-controls.css").read_text()
        self.assertIn(".config-drawer[open]{grid-column:1/-1}", controls_css)

    def test_default_linux_service_launch_refreshes_runtime_mirror(self) -> None:
        root = Path(__file__).resolve().parents[1]
        launcher = (root / "scripts/linux_runtime_exec.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('"${X_MINIMAXH3_SYNC:-1}" == "1"', launcher)
        self.assertIn('"${canonical_root}/scripts/sync_linux_runtime.sh"', launcher)

    def test_prompt_enhancement_state_is_invalidated_after_user_edits(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("function invalidatePromptEnhancement(scope='visuals')", script)
        self.assertIn("sectionUndoSnapshots[item] = null;", script)
        self.assertIn(
            "event => { activeShotIndex = index; invalidatePromptEnhancement();",
            script,
        )
        self.assertIn(
            "if (previousEngine !== currentEngine || previousLauncher !== currentLauncher) invalidatePromptEnhancement('all');",
            script,
        )

    def test_storyboard_compiler_keeps_h3_prompt_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "static/app.js").read_text(encoding="utf-8")
        for required in (
            "integrated_multimodal_description:",
            "subject_definitions:",
            "retention_analysis:",
            "detailed_description:",
            "overall_soundscape:",
            "non_diegetic_music:",
            "function conditionMode()",
            "function pictureAlignment(",
            "effective_duration_seconds",
            "reference_media",
            "referenceMediaPayload()",
            "function renderReferencePreviews(",
            "function renderConversation()",
            "function bindFileDrop(",
            "function removeReferenceFile(",
            "function insertReferenceToken(",
            "function distributeReferenceFiles(",
            "function updateReferenceMentionMenu(",
            "function refreshResources()",
            "<Picture 1>",
        ):
            self.assertIn(required, script)
        self.assertIn(
            "setInterval(() => { if (!document.hidden) refreshResources(); }, 1000);",
            script,
        )

    def test_default_paths_are_inside_release_root(self) -> None:
        root = Path(__file__).resolve().parents[1]
        paths = ServicePaths.defaults(root)
        self.assertEqual(paths.release_root, root)
        self.assertEqual(paths.model_dir, root / "models")
        self.assertEqual(paths.output_dir, root / "output")
        self.assertEqual(paths.minimax_source_dir, root / "runtime/vendor/MiniMax-H3")
        self.assertEqual(paths.lightx_source_dir, root / "runtime/vendor/LightX2V")
        self.assertEqual(paths.flashvsr_source_dir, root / "third_party/flashvsr")
        self.assertEqual(
            paths.flashvsr_model_dir, root / "models/upscalers/flashvsr-v1.1"
        )
        self.assertEqual(
            paths.flashvsr_python_executable,
            (root / "runtime/flashvsr-venv/bin/python").absolute(),
        )
        self.assertEqual(
            paths.turbo_curve_path,
            root / "backends/turbo/custom_node/h3_silu_temb_grid.safetensors",
        )

    def test_original_engine_does_not_require_lora_weight(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for role, (folder, filename) in MODEL_FILES.items():
                if role == "lora":
                    continue
                path = root / folder / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            status = model_status(root)
            self.assertTrue(status["engines"]["original"]["ready"])
            self.assertFalse(status["engines"]["lora"]["ready"])
            self.assertFalse(status["engines"]["reference_lora"]["ready"])

    def test_virtualenv_python_symlink_is_not_resolved_away(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            base = root / "base-python"
            base.touch()
            venv_python = root / "venv-python"
            venv_python.symlink_to(base)
            with patch.dict("os.environ", {"H3_SERVE_PYTHON": str(venv_python)}):
                paths = ServicePaths.defaults(root)
            self.assertEqual(paths.python_executable, venv_python.absolute())

    def test_model_downloader_prepares_optional_64gb_qwen_cache(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "scripts/download_models.py").read_text(encoding="utf-8")
        self.assertIn("--skip-local-qwen-cache", script)
        self.assertIn("materialize_local_checkpoint", script)

    def test_model_downloader_maps_root_w4_checkpoint_into_dit_folder(self) -> None:
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "h3_download_models_test",
            root / "scripts/download_models.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model_root = root / "models"
        target = (
            model_root
            / "diffusion_models/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors"
        )
        local_dir = module._download_local_dir(
            model_root,
            target,
            {
                "filename": "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
                "install_path": (
                    "diffusion_models/"
                    "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors"
                ),
            },
        )
        self.assertEqual(local_dir, model_root / "diffusion_models")

    def test_preflight_checks_real_host_memory_capacity(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "scripts/preflight.py").read_text(encoding="utf-8")
        self.assertIn("host_memory_supported", script)
        self.assertIn("resolve_host_memory_profile", script)
        self.assertTrue((root / "docs/DEPLOY_64GB_WSL.md").is_file())

    def test_comfyui_api_connector_is_part_of_the_release(self) -> None:
        root = Path(__file__).resolve().parents[1]
        connector = root / "integrations/comfyui"
        self.assertTrue((connector / "install_local.py").is_file())
        self.assertTrue((connector / "h3serve_connector/nodes.py").is_file())
        build_script = (root / "scripts/build_release.sh").read_text(encoding="utf-8")
        self.assertIn("integrations", build_script)

    def test_flashvsr_alignment_box_preserves_the_full_composition(self) -> None:
        root = Path(__file__).resolve().parents[1]
        worker = root / "scripts/flashvsr_worker.py"
        spec = importlib.util.spec_from_file_location("flashvsr_worker_contract", worker)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # A 640x352 H3 frame delivered at short-edge 720 becomes 1310x720.
        # The 128-aligned model canvas is 1408x768, but its explicit content
        # box must preserve the full composition instead of centre-cropping it.
        geometry = module.fit_geometry(640, 352, 1408, 768)
        self.assertEqual(geometry, (1396, 768, 6, 0, 1402, 768))

    def test_public_surfaces_do_not_contain_research_release_names(self) -> None:
        root = Path(__file__).resolve().parents[1]
        files = [
            root / "README.md",
            root / "static/index.html",
            root / "static/app.js",
            root / "h3serve/backend.py",
        ]
        forbidden = (r"\bV[5-8](?:[.\-])", r"Fast\s+Max", r"final[_ ]audit", r"hot\d")
        for path in files:
            text = path.read_text(encoding="utf-8")
            for pattern in forbidden:
                self.assertIsNone(
                    re.search(pattern, text, re.IGNORECASE),
                    f"research name pattern {pattern} leaked through {path.name}",
                )


if __name__ == "__main__":
    unittest.main()
