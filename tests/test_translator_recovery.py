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

    def test_measured_shorten_hook_is_a_noop_even_for_long_audio(self):
        client = _FakeClient(lambda _c, _n: "{}")
        translator = _make_translator(client)
        original = "এটি একটি দীর্ঘ বাংলা অনুবাদের উদাহরণ।"
        seg = Segment(index=0, start=0.0, end=1.0,
                      text_en="This is a long example.", text_bn=original)
        seg.stretched_duration = 3.0
        with mock.patch.object(translator, "_run_batches") as batches:
            self.assertEqual(translator.shorten_to_durations([seg], [0.75]), 0)
        batches.assert_not_called()
        self.assertEqual(seg.text_bn, original)
        self.assertEqual(client.calls, [])
