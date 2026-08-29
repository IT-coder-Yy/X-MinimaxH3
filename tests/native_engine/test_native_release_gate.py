from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path


SERVE_ROOT = Path(__file__).resolve().parents[2]
STRICT = os.environ.get("H3_NATIVE_RELEASE_GATE") == "1"


def imported_modules(path: Path) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                modules.add("." * node.level + (node.module or ""))
            elif node.module:
                modules.add(node.module)
    return modules


def method_args(path: Path, class_name: str, method_name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )
    return [argument.arg for argument in method.args.args]


@unittest.skipUnless(
    STRICT,
    "strict native source isolation runs with H3_NATIVE_RELEASE_GATE=1",
)
class NativeReleaseSourceGateTest(unittest.TestCase):
    """Final release gate; intentionally fails until the compatibility backend is gone."""

    def setUp(self) -> None:
        self.backend = SERVE_ROOT / "h3serve/backend.py"
        self.engine = SERVE_ROOT / "h3serve/native_engine/engine.py"
        self.native_root = SERVE_ROOT / "h3serve/native_engine"

    def test_required_native_entry_points_exist_with_the_service_contract(self) -> None:
        self.assertTrue(self.engine.is_file(), "NativeH3Engine has not been implemented")
        engine_args = method_args(self.engine, "NativeH3Engine", "generate")
        self.assertGreaterEqual(len(engine_args), 2)
        self.assertEqual(engine_args[0], "self")
        self.assertNotIn(
            "job_id", engine_args,
            "service persistence identity must stop at NativeBackendManager",
        )
        manager = "NativeBackendManager"
        try:
            args = method_args(self.backend, manager, "generate")
        except StopIteration:
            manager = "BackendManager"
            args = method_args(self.backend, manager, "generate")
        self.assertEqual(
            args[:9],
            ["self", "spec", "job_id", "first_frame", "last_frame", "reference_images", "reference_videos", "reference_audios", "cancel_event"],
        )

    def test_backend_and_native_engine_have_no_comfy_or_http_subbackend(self) -> None:
        self.assertTrue(self.native_root.is_dir())
        paths = [self.backend, *sorted(self.native_root.rglob("*.py"))]
        forbidden_import_roots = {
            "comfy", "aiohttp", "httpx", "requests", "urllib3",
        }
        forbidden_text = {
            "h3serve.workflows", ".workflows", "class_type", "node_ids",
            "/system_stats", "/prompt", "/history/", "/upload/image",
            "--comfy-dir", "comfy_main", "create_subprocess_exec",
            "subprocess.popen", "runpy.run_path",
        }
        for path in paths:
            with self.subTest(path=path.relative_to(SERVE_ROOT)):
                imports = imported_modules(path)
                roots = {name.lstrip(".").split(".", 1)[0] for name in imports}
                self.assertTrue(
                    forbidden_import_roots.isdisjoint(roots),
                    f"forbidden imports: {sorted(forbidden_import_roots & roots)}",
                )
                lowered = path.read_text(encoding="utf-8").lower()
                present = sorted(token for token in forbidden_text if token in lowered)
                self.assertEqual(present, [], f"forbidden runtime tokens: {present}")

    def test_workflow_module_and_comfy_paths_are_removed(self) -> None:
        self.assertFalse(
            (SERVE_ROOT / "h3serve/workflows.py").exists(),
            "workflow JSON must not ship in the native service",
        )
        config = (SERVE_ROOT / "h3serve/config.py").read_text(encoding="utf-8").lower()
        for token in (
            "original_comfy_dir", "turbo_comfy_dir", "comfy_dir(",
            "model_config_path(",
        ):
            self.assertNotIn(token, config)

    def test_install_and_runtime_launchers_do_not_install_or_start_comfy(self) -> None:
        runtime_files = [
            SERVE_ROOT / "scripts/install.sh",
            SERVE_ROOT / "scripts/preflight.py",
            SERVE_ROOT / "requirements.txt",
            SERVE_ROOT / "requirements.lock",
        ]
        for path in runtime_files:
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8").lower()
                self.assertNotIn("comfyui", source)
                self.assertNotIn("spectrum", source)


if __name__ == "__main__":
    unittest.main()
