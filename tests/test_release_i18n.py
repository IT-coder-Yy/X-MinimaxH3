import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseI18nTests(unittest.TestCase):
    def test_language_switch_and_asset_order(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="languageSwitch"', html)
        self.assertIn('data-locale="zh-CN"', html)
        self.assertIn('data-locale="en"', html)
        self.assertLess(html.index("i18n.js"), html.index("app.js"))

    def test_runtime_translation_contract(self):
        source = (ROOT / "static" / "i18n.js").read_text(encoding="utf-8")
        self.assertIn("h3serve_locale", source)
        self.assertIn("new URLSearchParams", source)
        self.assertIn("new MutationObserver", source)
        self.assertIn("h3serve:locale-changed", source)
        self.assertIn(".job-title", source)
        self.assertIn("textarea, code, pre", source)
        self.assertIn("Checking model files and the CUDA runtime", source)
        self.assertIn("Model components ready:", source)
        self.assertIn("V24 unified Pareto scheduler", source)
        self.assertIn("Controls CPU weight residency", source)

    def test_release_scripts_parse(self):
        for script in ("i18n.js", "app.js"):
            result = subprocess.run(
                ["node", "--check", str(ROOT / "static" / script)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
