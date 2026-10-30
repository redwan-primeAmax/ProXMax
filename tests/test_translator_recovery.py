"""Focused regression tests for ProXMax."""





from tests._support import *





class TestTranslatorRecovery(unittest.TestCase):
    """অনূদিত না হওয়া লাইন ছোট ব্যাচে উদ্ধার — নীরব স্লট যেন না থাকে।"""

    def test_missing_lines_are_recovered_in_small_batches(self):
        def responder(contents, _n):
            lines = _lines_from_contents(contents)
            if len(lines) > 4:      # বড় ব্যাচে মডেল ভুল ফরম্যাট দেয়
                return "sorry, I cannot help with that"
            return json.dumps({"translations": [f"বাংলা: {t}" for t in lines]})

        client = _FakeClient(responder)
        translator = _make_translator(client, batch_size=6, max_retries=1)
        segments = _segments([f"Line {i}." for i in range(6)])
        translator.translate_segments(segments)
        self.assertTrue(all((s.text_bn or "").strip() for s in segments),
                        [s.text_bn for s in segments])
        self.assertTrue(all(s.text_bn.startswith("বাংলা:") for s in segments))

    def test_recovery_failure_is_graceful(self):
        def responder(contents, _n):
            return "still not valid"

        client = _FakeClient(responder)
        translator = _make_translator(client, batch_size=3, max_retries=1)
        segments = _segments(["A.", "B."])
        translator.translate_segments(segments)      # exception নয়
        self.assertEqual([s.text_bn for s in segments], ["", ""])


class TestTranslatorExpandPass(unittest.TestCase):
    """কথার ফাঁক ভরাতে Gemini-কে text বাড়াতে বলা হয় না।"""

    def test_expand_pass_is_disabled_without_a_gemini_request(self):
        translator = _make_translator(_FakeClient(lambda _c, _n: "{}"))
        seg = Segment(index=0, start=0.0, end=4.0, text_en="x", text_bn="খুব ছোট।")
        with mock.patch.object(translator, "_run_batches") as run:
            changed = translator.expand_to_fit([seg], [4.0])
        self.assertEqual(changed, 0)
        self.assertEqual(seg.text_bn, "খুব ছোট।")
        run.assert_not_called()

    def test_measured_shorten_retry_includes_source_and_relative_fit_goal(self):
        translator = _make_translator(_FakeClient(lambda _c, _n: "{}"))
        original = "এটি একটি দীর্ঘ বাংলা অনুবাদের উদাহরণ।"
        concise = "এটি দীর্ঘ বাংলা অনুবাদের উদাহরণ।"
        seg = Segment(index=0, start=0.0, end=1.0,
                      text_en="This is a long Bangla translation example.", text_bn=original)
        seg.stretched_duration = 1.0
        captured = {}

        def batches(items, instruction, task, description):
            captured.update(items=list(items), instruction=instruction, task=task)
            return [concise]

        with mock.patch.object(translator, "_run_batches", side_effect=batches):
            changed = translator.shorten_to_durations([seg], [0.75])
        self.assertEqual(changed, 1)
        payload = json.loads(captured["items"][0][0])
        self.assertEqual(payload["english_source"], seg.text_en)
        self.assertEqual(payload["current_bangla"], original)
        self.assertEqual(payload["reference_bangla"], original)
        self.assertEqual(payload["target_percent"], 75)
        prompt_text = f"{captured['instruction']} {captured['task']} {captured['items']}"
        self.assertNotRegex(prompt_text.lower(), r"duration|seconds|speaking time|timestamp")
        self.assertNotIn("0.75", prompt_text)
        self.assertEqual(seg.text_bn, concise)
