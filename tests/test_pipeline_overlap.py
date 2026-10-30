"""Focused regression tests for ProXMax."""





from tests._support import *





class TestPipelineTranslationOverlap(unittest.TestCase):
    def _pipeline(self, translator, tts):
        from proxmax.pipeline import DubbingPipeline

        pipeline = DubbingPipeline.__new__(DubbingPipeline)
        pipeline.config = AppConfig(
            gemini_batch_size=4, enable_overlap=True, translation_shorten_pass=True,
            unload_stt_before_tts=False,
        ).validate()
        pipeline.translator = translator
        pipeline._tts = tts
        pipeline._profile = None
        pipeline._profile_checked = True
        return pipeline

    def test_groups_are_translated_then_synthesized_in_order(self):
        translator = _RecordingTranslator()
        tts = _RecordingTTS()
        pipeline = self._pipeline(translator, tts)
        segments = [
            Segment(index=i, start=i * 1.5, end=i * 1.5 + 1.2, text_en=f"Line {i}.")
            for i in range(10)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            translate_seconds, tts_seconds = pipeline._translate_and_synthesize(
                segments, Path(tmp)
            )

        self.assertGreaterEqual(translate_seconds, 0.0)
        self.assertGreaterEqual(tts_seconds, 0.0)
        self.assertEqual(len(translator.translated), 3)   # ১০ লাইন / ৪
        self.assertEqual(len(tts.calls), 3)
        self.assertEqual(translator.shorten_calls, 0)
        # কোনো pre-TTS duration predictor অনুবাদ বদলায় না
        self.assertEqual([idx for group in tts.calls for idx, _ in group], list(range(10)))
        for group in tts.calls:
            for _idx, text in group:
                self.assertTrue(text.startswith("টি"), text)
        self.assertTrue(all(s.text_bn == f"টি{s.index}" for s in segments))

    def test_translation_failure_stops_gracefully(self):
        translator = _RecordingTranslator(fail_on_group=2)
        tts = _RecordingTTS()
        pipeline = self._pipeline(translator, tts)
        segments = [
            Segment(index=i, start=i * 1.5, end=i * 1.5 + 1.2, text_en=f"Line {i}.")
            for i in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                pipeline._translate_and_synthesize(segments, Path(tmp))
        # প্রথম গ্রুপের TTS শুরু হলেও বাকিগুলো আর হয়নি — কোনো hang নেই
        self.assertLessEqual(len(tts.calls), 1)

    def test_single_group_uses_sequential_path(self):
        translator = _RecordingTranslator()
        tts = _RecordingTTS()
        pipeline = self._pipeline(translator, tts)
        segments = [Segment(index=0, start=0.0, end=2.0, text_en="Hello.")]
        with tempfile.TemporaryDirectory() as tmp:
            pipeline._translate_and_synthesize(segments, Path(tmp))
        self.assertEqual(len(translator.translated), 1)
        self.assertEqual(len(tts.calls), 1)
        self.assertEqual(translator.shorten_calls, 0)
