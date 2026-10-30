"""Focused regression tests for ProXMax."""

import threading

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
        self.assertGreaterEqual(len(tts.calls), 1)        # ready groups may be combined
        self.assertLessEqual(len(tts.calls), 3)
        self.assertEqual(translator.shorten_calls, 0)
        # কোনো pre-TTS duration predictor অনুবাদ বদলায় না
        self.assertEqual([idx for group in tts.calls for idx, _ in group], list(range(10)))
        for group in tts.calls:
            for _idx, text in group:
                self.assertTrue(text.startswith("টি"), text)
        self.assertTrue(all(s.text_bn == f"টি{s.index}" for s in segments))

    def test_tts_starts_before_the_next_translation_finishes(self):
        tts_started = threading.Event()

        class WaitingTranslator(_RecordingTranslator):
            def translate_segments(self, segments):
                if self._group_no == 1 and not tts_started.wait(2):
                    raise RuntimeError("TTS did not start after the first translated group")
                return super().translate_segments(segments)

        class SignalingTTS(_RecordingTTS):
            def synthesize(self, segments, output_dir):
                tts_started.set()
                return super().synthesize(segments, output_dir)

        translator, tts = WaitingTranslator(), SignalingTTS()
        pipeline = self._pipeline(translator, tts)
        segments = [
            Segment(index=i, start=float(i), end=i + 0.8, text_en=f"Line {i}")
            for i in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            pipeline._translate_and_synthesize(segments, Path(tmp))
        self.assertTrue(tts_started.is_set())
        self.assertEqual([index for group in tts.calls for index, _ in group], list(range(8)))

    def test_ready_groups_are_combined_without_reordering(self):
        first_tts = threading.Event()
        middle_ready = threading.Event()
        second_tts = threading.Event()

        class ControlledTranslator(_RecordingTranslator):
            def translate_segments(self, segments):
                if self._group_no == 1 and not first_tts.wait(2):
                    raise RuntimeError("first TTS batch did not start")
                if self._group_no == 3:
                    # Groups 2 and 3 have already been queued before group 4 starts.
                    middle_ready.set()
                    if not second_tts.wait(2):
                        raise RuntimeError("combined TTS batch did not start")
                return super().translate_segments(segments)

        class ControlledTTS(_RecordingTTS):
            def synthesize(self, segments, output_dir):
                super().synthesize(segments, output_dir)
                if len(self.calls) == 1:
                    first_tts.set()
                    if not middle_ready.wait(2):
                        raise RuntimeError("translation queue did not fill")
                elif len(self.calls) == 2:
                    second_tts.set()
                return segments

        translator, tts = ControlledTranslator(), ControlledTTS()
        pipeline = self._pipeline(translator, tts)
        segments = [
            Segment(index=i, start=float(i), end=i + 0.8, text_en=f"Line {i}")
            for i in range(16)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            pipeline._translate_and_synthesize(segments, Path(tmp))
        self.assertEqual([len(group) for group in tts.calls], [4, 8, 4])
        self.assertEqual([index for group in tts.calls for index, _ in group], list(range(16)))
        self.assertTrue(all(segment.text_bn == f"টি{segment.index}" for segment in segments))

    def test_coalescing_is_bounded_for_large_backlogs(self):
        first_tts = threading.Event()
        backlog_ready = threading.Event()

        class BacklogTranslator(_RecordingTranslator):
            def translate_segments(self, segments):
                if self._group_no == 1 and not first_tts.wait(2):
                    raise RuntimeError("first TTS batch did not start")
                if self._group_no == 19:
                    backlog_ready.set()
                return super().translate_segments(segments)

        class BlockingTTS(_RecordingTTS):
            def synthesize(self, segments, output_dir):
                super().synthesize(segments, output_dir)
                if len(self.calls) == 1:
                    first_tts.set()
                    if not backlog_ready.wait(2):
                        raise RuntimeError("translation backlog did not arrive")
                return segments

        translator, tts = BacklogTranslator(), BlockingTTS()
        pipeline = self._pipeline(translator, tts)
        pipeline.config.tts_concurrency = 2
        segments = [
            Segment(index=i, start=float(i), end=i + 0.8, text_en=f"Line {i}")
            for i in range(80)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            pipeline._translate_and_synthesize(segments, Path(tmp))
        self.assertTrue(all(len(group) <= 8 for group in tts.calls))
        self.assertEqual([index for group in tts.calls for index, _ in group], list(range(80)))

    def test_combined_submission_preserves_each_original_groups_failure_rule(self):
        from proxmax.tts.base import TTSError

        class InlineThread:
            def __init__(self, target, **_kwargs):
                self.target = target

            def start(self):
                self.target()  # Make both translated groups ready deterministically.

            def join(self, timeout=None):
                pass

        class PartlyFailedTTS(_RecordingTTS):
            def __init__(self, successful):
                super().__init__()
                self.successful = successful
                self.submitted = []

            def synthesize(self, segments, output_dir):
                self.submitted.append([s.index for s in segments])
                return super().synthesize(
                    [s for s in segments if s.index in self.successful], output_dir
                )

        class FirstGroupUntranslated(_RecordingTranslator):
            def translate_segments(self, segments):
                result = super().translate_segments(segments)
                if segments[0].index == 0:
                    for segment in segments:
                        segment.text_bn = ""
                return result

        cases = [
            ({4, 5, 6, 7}, True, False),       # Entire original group failed.
            ({3, 4, 5, 6, 7}, False, False),   # Partial failures remain allowed.
            ({4, 5, 6, 7}, False, True),      # No translation to synthesize in group 1.
        ]
        for surviving, expect_error, empty_first in cases:
            with self.subTest(surviving=surviving, empty_first=empty_first):
                tts = PartlyFailedTTS(surviving)
                translator = FirstGroupUntranslated() if empty_first else _RecordingTranslator()
                pipeline = self._pipeline(translator, tts)
                segments = [
                    Segment(index=i, start=float(i), end=i + 0.8, text_en=f"Line {i}")
                    for i in range(8)
                ]
                with tempfile.TemporaryDirectory() as tmp, mock.patch(
                    "proxmax.pipeline_translation.threading.Thread", InlineThread
                ):
                    if expect_error:
                        with self.assertRaisesRegex(TTSError, "segment 0–3"):
                            pipeline._translate_and_synthesize(segments, Path(tmp))
                    else:
                        pipeline._translate_and_synthesize(segments, Path(tmp))
                expected = list(range(4, 8)) if empty_first else list(range(8))
                self.assertEqual(tts.submitted, [expected])

    def test_overlap_can_still_be_disabled(self):
        translator, tts = _RecordingTranslator(), _RecordingTTS()
        pipeline = self._pipeline(translator, tts)
        pipeline.config.enable_overlap = False
        segments = [
            Segment(index=i, start=float(i), end=i + 0.8, text_en=f"Line {i}")
            for i in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            pipeline._translate_and_synthesize(segments, Path(tmp))
        self.assertEqual(len(translator.translated), 1)
        self.assertEqual(len(tts.calls), 1)
        self.assertEqual([index for index, _ in tts.calls[0]], list(range(8)))

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
