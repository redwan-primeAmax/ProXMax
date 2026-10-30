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
