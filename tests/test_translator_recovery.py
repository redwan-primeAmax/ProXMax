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
    """বড় ফাঁকের জন্য লাইন ভরাট করা (expand pass)।"""

    def _translator(self, grow_with: str):
        def responder(contents, _n):
            lines = _lines_from_contents(contents)
            return json.dumps({"translations": [line + grow_with for line in lines]})

        return _make_translator(_FakeClient(responder)), _FakeClient(responder)

    def test_short_line_is_expanded_within_slot(self):
        def responder(contents, _n):
            lines = _lines_from_contents(contents)
            return json.dumps({"translations": [line + " আসলে পুরো সময়টা জুড়ে কথা বলা যায়।" for line in lines]})

        client = _FakeClient(responder)
        translator = _make_translator(client)
        seg = Segment(index=0, start=0.0, end=12.6, text_en="x", text_bn="খুব ছোট।")
        seg.tts_duration = 1.0
        changed = translator.expand_to_fit([seg], [12.5])
        self.assertEqual(changed, 1)
        self.assertGreater(len(seg.text_bn), 20)
        self.assertTrue(seg.text_bn.startswith("খুব ছোট।"))

    def test_absurdly_long_answer_is_rejected(self):
        def responder(_contents, _n):
            return json.dumps({"translations": ["অ" * 5000]})

        translator = _make_translator(_FakeClient(responder))
        seg = Segment(index=0, start=0.0, end=4.0, text_en="x", text_bn="ছোট।")
        self.assertEqual(translator.expand_to_fit([seg], [4.0]), 0)
        self.assertEqual(seg.text_bn, "ছোট।")

    def test_failure_keeps_old_text(self):
        client = _FakeClient(lambda c, n: (_ for _ in ()).throw(RuntimeError("boom")))
        translator = _make_translator(client, max_retries=1)
        seg = Segment(index=0, start=0.0, end=9.0, text_en="x", text_bn="ছোট।")
        with mock.patch("proxmax.translator.time.sleep"):
            self.assertEqual(translator.expand_to_fit([seg], [8.0]), 0)
        self.assertEqual(seg.text_bn, "ছোট।")

    def test_empty_texts_are_skipped(self):
        client = _FakeClient(lambda c, n: json.dumps({"translations": ["কিছু"]}))
        translator = _make_translator(client)
        seg = Segment(index=0, start=0.0, end=5.0, text_en="x", text_bn="  ")
        self.assertEqual(translator.expand_to_fit([seg], [5.0]), 0)
        self.assertEqual(client.calls, [])
