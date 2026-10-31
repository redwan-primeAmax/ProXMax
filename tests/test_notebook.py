"""Focused regression tests for ProXMax."""





from tests._support import *





class TestNotebookRequirements(unittest.TestCase):
    """নোটবুকের কাঠামো যাচাই (ব্যাজ, GPU ধাপ, ভয়েস ক্লোনিং, প্রি-লোড, API যাচাই)।"""

    def test_notebook_meets_all_user_constraints(self):
        nb_path = REPO_ROOT / "notebook" / "ProXMax_Colab.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        cells = nb["cells"]
        full_source = "\n".join("".join(c.get("source", [])) for c in cells)

        # ১. নোটবুকের ভেতরে উপরে অতিরিক্ত "Open in Colab" বাটন নেই
        header_source = "".join(cells[0].get("source", []))
        self.assertNotIn("colab-badge.svg", header_source)
        self.assertNotIn("Open In Colab", header_source)

        # ২. GPU যাচাই ধাপ (!nvidia-smi) নেই
        self.assertNotIn("nvidia-smi", full_source)
        self.assertNotIn("GPU যাচাই", full_source)

        # ৩. ভয়েস ক্লোনিং / IndicF5 ধাপ নেই
        self.assertNotIn("IndicF5", full_source)
        self.assertNotIn("কণ্ঠ ক্লোন", full_source)

        # ৪. ডিপেন্ডেন্সি ধাপেই Whisper মডেল প্রি-ডাউনলোড হয়
        self.assertIn("ensure_whisper_model", full_source)

        # ৫. Gemini API key সেলেই লাইভ API যাচাই হয়
        self.assertIn("verify_gemini_api", full_source)

        # ৬. Editable Gemini input-token budget has a 5000 default and 1000 minimum
        self.assertIn("GEMINI_MAX_INPUT_TOKENS = 5000", full_source)
        self.assertIn("GEMINI_MAX_INPUT_TOKENS < 1000", full_source)
        self.assertIn("PROXMAX_GEMINI_MAX_INPUT_TOKENS", full_source)

        # TTS concurrency and reprocessing existing outputs can be adjusted.
        self.assertIn("config.tts_concurrency = 5", full_source)
        self.assertIn("config.skip_existing = False", full_source)
        self.assertIn('WHISPER_MODEL = "distil-large-v3"  # @param ["distil-large-v3", "large-v3"]', full_source)
        for index, cell in enumerate(cells):
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), f"notebook cell {index}", "exec")

    def test_pull_failure_stops_and_updated_imports_require_restart(self):
        nb = json.loads((REPO_ROOT / "notebook" / "ProXMax_Colab.ipynb").read_text(encoding="utf-8"))
        setup = "".join(nb["cells"][4]["source"])
        failure = subprocess.CalledProcessError(1, ["git", "pull"])
        with mock.patch("pathlib.Path.exists", return_value=True), \
             mock.patch("subprocess.run", side_effect=[types.SimpleNamespace(stdout="old"), failure]) as run, \
             mock.patch("os.chdir") as chdir, \
             self.assertRaisesRegex(RuntimeError, "clone/pull"):
            exec(setup, {})
        self.assertTrue(run.call_args.kwargs["check"])
        chdir.assert_not_called()

        scope = {}
        results = [types.SimpleNamespace(stdout="old"), types.SimpleNamespace(),
                   types.SimpleNamespace(stdout="new")]
        with mock.patch("pathlib.Path.exists", return_value=True), \
             mock.patch("subprocess.run", side_effect=results) as run, \
             mock.patch("os.chdir") as chdir:
            with self.assertRaisesRegex(RuntimeError, "Restart session"):
                exec(setup, scope)
            self.assertTrue(scope["_PROXMAX_RESTART_REQUIRED"])
            self.assertEqual(run.call_count, 3)
            with self.assertRaisesRegex(RuntimeError, "Restart session"):
                exec(setup, scope)
            self.assertEqual(run.call_count, 3)  # retry cannot bypass the restart
            with self.assertRaisesRegex(RuntimeError, "restart"):
                exec("".join(nb["cells"][10]["source"]), scope)
            chdir.assert_not_called()

    def test_force_option_preserves_yaml_skip_setting_by_default(self):
        nb = json.loads((REPO_ROOT / "notebook" / "ProXMax_Colab.ipynb").read_text(encoding="utf-8"))
        source = "".join(nb["cells"][10]["source"])
        summary = types.SimpleNamespace(succeeded=[], skipped=[], failed=[])
        for configured, force in ((False, False), (True, False), (True, True)):
            with self.subTest(configured=configured, force=force):
                cfg = AppConfig(skip_existing=configured)
                cell = source.replace("FORCE_REPROCESS = False", f"FORCE_REPROCESS = {force}")
                with mock.patch("proxmax.config.AppConfig.from_yaml", return_value=cfg), \
                     mock.patch("proxmax.pipeline.run_Single_or_batch", return_value=summary), \
                     mock.patch("proxmax.utils.setup_logging"):
                    exec(cell, {})
                self.assertEqual(cfg.skip_existing, configured and not force)

    def test_every_python_module_is_at_most_400_lines(self):
        excluded = {".git", ".venv", "__pycache__", "node_modules", "dist", "build"}
        oversized = []
        for path in REPO_ROOT.rglob("*.py"):
            if excluded.intersection(path.parts):
                continue
            line_count = len(path.read_text(encoding="utf-8").splitlines())
            if line_count > 400:
                oversized.append(f"{path.relative_to(REPO_ROOT)}: {line_count}")
        self.assertEqual(oversized, [], "Python files exceed the 400-line limit")
