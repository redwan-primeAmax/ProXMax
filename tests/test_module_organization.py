"""The domain folders preserve historic imports, object identity and resources."""
from __future__ import annotations

import importlib
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import proxmax
from proxmax._module_groups import GROUPS
from proxmax.hotwords import default_hotwords_path


class TestModuleOrganization(unittest.TestCase):
    def test_every_old_and_domain_import_is_same_module(self):
        for group, names in GROUPS.items():
            for name in names:
                with self.subTest(group=group, module=name):
                    old = importlib.import_module(f"proxmax.{name}")
                    new = importlib.import_module(f"proxmax.{group}.{name}")
                    self.assertIs(old, new)
                    self.assertEqual(Path(old.__file__).parent.name, group)
                    self.assertIs(getattr(importlib.import_module(f"proxmax.{group}"), name), old)

    def test_patches_affect_both_import_paths(self):
        old = importlib.import_module("proxmax.video_render")
        new = importlib.import_module("proxmax.video.video_render")
        with mock.patch("proxmax.video.video_render._frame_count", return_value=42):
            self.assertEqual(old._frame_count(Path("anything")), 42)
        with mock.patch("proxmax.pipeline_process.write_detailed_report") as report:
            self.assertIs(importlib.import_module("proxmax.workflow.pipeline_process").write_detailed_report, report)
        self.assertIs(old, new)

    def test_fresh_interpreter_and_notebook_entry_imports(self):
        command = (
            "from proxmax.speech.stt import ensure_whisper_model; "
            "from proxmax.stt import ensure_whisper_model as legacy; "
            "from proxmax.workflow.pipeline import run_Single_or_batch; "
            "from proxmax.translator import verify_gemini_api; "
            "from proxmax.config import AppConfig; "
            "from proxmax.hotwords import load_hotwords; "
            "assert ensure_whisper_model is legacy; "
            "assert callable(run_Single_or_batch) and callable(verify_gemini_api)"
        )
        subprocess.run([sys.executable, "-c", command], check=True, capture_output=True, timeout=20)
        self.assertTrue(default_hotwords_path().is_file())
        self.assertEqual(default_hotwords_path().parent, Path(proxmax.__file__).resolve().parent.parent)


if __name__ == "__main__":
    unittest.main()
