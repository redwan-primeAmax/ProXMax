"""Measured fit retries preserve the original audio/text unless proven better."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from proxmax import audio_fx
from proxmax.config import AppConfig
from proxmax.models import Segment
from proxmax.pipeline import DubbingPipeline
from proxmax.sync_engine import SyncEngine
from tests._support import FFMPEG


def make_audio(path, seconds):
    rate = 24000
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    audio_fx.write_wav(path, .1 * np.sin(2 * np.pi * 350 * t), rate)


class Rewriter:
    def __init__(self, text):
        self.text = text
        self.calls = 0
        self.measured_durations = []

    def shorten_to_durations(self, segments, targets):
        self.calls += 1
        self.measured_durations.extend(s.stretched_duration for s in segments)
        for segment in segments:
            segment.text_bn = self.text
        return len(segments)


class LengthTTS:
    def __init__(self, seconds):
        self.seconds = seconds
        self.calls = 0

    def synthesize(self, segments, output_dir):
        self.calls += 1
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for segment in segments:
            path = output_dir / f"seg_{segment.index:04d}.wav"
            make_audio(path, self.seconds)
            segment.tts_audio_path = path
            segment.tts_duration = self.seconds
        return segments


class TestFitRewriteSafety(unittest.TestCase):
    def _setup(self, root, rewrite, seconds):
        pipeline = DubbingPipeline.__new__(DubbingPipeline)
        pipeline.config = AppConfig().validate()
        pipeline.translator = Rewriter(rewrite)
        pipeline._tts = LengthTTS(seconds)
        original = root / "original.wav"
        make_audio(original, 2.0)
        segment = Segment(
            0, 0.0, 1.0, "We have 126 OVR and 5 stars",
            "আমাদের কাছে ১২৬ ওভিআর এবং ৫ স্টার আছে।",
            tts_audio_path=original, tts_duration=2.0, stretched_duration=1.7,
            sync_issue="stretched clip exceeds safe window after tempo cap",
        )
        return pipeline, segment, original

    def test_lost_number_is_rejected_before_synthesis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline, seg, original = self._setup(root, "আমাদের ২৬ ওভিআর ও ৫ স্টার আছে।", .8)
            stats = pipeline._repair_overlong_tts([seg], root / "tts", 2.0)
            self.assertEqual(stats["shortened"], 0)
            self.assertEqual(stats["retry_failed"], [0])
            self.assertEqual(pipeline._tts.calls, 0)
            self.assertEqual(seg.tts_audio_path, original)
            self.assertTrue(any("number" in note for note in seg.translation_notes))
            self.assertEqual(pipeline.translator.measured_durations, [1.7])

    def test_shorter_text_with_longer_spoken_audio_keeps_the_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline, seg, original = self._setup(root, "আমাদের ১২৬ ওভিআর ও ৫ স্টার আছে।", 2.2)
            text = seg.text_bn
            stats = pipeline._repair_overlong_tts([seg], root / "tts", 2.0)
            self.assertEqual(stats["shortened"], 0)
            self.assertEqual(stats["not_shorter_audio"], [0])
            self.assertEqual(seg.text_bn, text)
            self.assertEqual(seg.tts_audio_path, original)
            self.assertIsNone(seg.original_text_bn)
            self.assertTrue(any("not shorter" in note for note in seg.translation_notes))

    def test_silent_retry_audio_is_not_accepted_as_a_shorter_clip(self):
        class SilentTTS:
            def synthesize(self, segments, output_dir):
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                for segment in segments:
                    path = Path(output_dir) / f"seg_{segment.index:04d}.wav"
                    audio_fx.write_wav(path, np.zeros(12000, dtype=np.float32), 24000)
                    segment.tts_audio_path, segment.tts_duration = path, .5

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline, seg, original = self._setup(root, "আমাদের ১২৬ ওভিআর ও ৫ স্টার আছে।", .8)
            pipeline._tts = SilentTTS()
            stats = pipeline._repair_overlong_tts([seg], root / "tts", 2.0)
            self.assertEqual(stats["shortened"], 0)
            self.assertEqual(stats["retry_failed"], [0])
            self.assertEqual(seg.tts_audio_path, original)

    def test_safe_shorter_audio_keeps_original_text_for_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline, seg, original = self._setup(root, "আমাদের ১২৬ ওভিআর ও ৫ স্টার আছে।", .8)
            text = seg.text_bn
            stats = pipeline._repair_overlong_tts([seg], root / "tts", 2.0)
            self.assertEqual(stats["shortened"], 1)
            self.assertEqual(stats["shortened_segments"], [0])
            self.assertEqual(seg.original_text_bn, text)
            self.assertTrue(seg.translation_notes)
            self.assertNotEqual(seg.tts_audio_path, original)
            self.assertTrue(original.exists())
            self.assertAlmostEqual(seg.tts_duration, .8)

    def test_second_round_only_targets_eligible_remaining_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline, seg, original = self._setup(root, "আমাদের ১২৬ ওভিআর ও ৫ স্টার আছে।", .8)
            stats = pipeline._repair_overlong_tts([seg], root / "tts", 2.0, retry_round=2, eligible_indexes={99})
            self.assertEqual(stats["overlong_before"], 0)
            self.assertEqual(pipeline.translator.calls, 0)
            self.assertEqual(seg.tts_audio_path, original)


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class TestFitRetryOrchestration(unittest.TestCase):
    def test_second_measured_round_reuses_unchanged_clip_and_places_rewrite(self):
        class Extractor:
            def extract(self, video, destination, duration_seconds=None):
                make_audio(destination, 3.0)
                return destination

        class STT:
            last_stats = {}

            def transcribe(self, path):
                return [Segment(0, 0.0, 1.0, "We have 126 OVR and 5 stars"),
                        Segment(1, 1.8, 2.6, "Another nice card")]

        class Translator:
            calls = 0

            def translate_segments(self, segments):
                for segment in segments:
                    segment.text_bn = ("আমাদের কাছে এখানে ১২৬ ওভিআর এবং ৫ স্টার আছে।"
                                       if segment.index == 0 else "একটি ভালো কার্ড।")

            def shorten_to_durations(self, segments, targets):
                self.calls += 1
                for segment in segments:
                    segment.text_bn = ("আমাদের কাছে ১২৬ ওভিআর এবং ৫ স্টার আছে।"
                                       if self.calls == 1 else "আমাদের ১২৬ ওভিআর ও ৫ স্টার আছে।")
                return len(segments)

        class TTS:
            def describe(self):
                return "fake-measured-tts"

            def set_budget(self, *_args):
                pass

            def synthesize(self, segments, output_dir):
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                for segment in segments:
                    duration = (.8 if segment.index == 1 else
                                1.65 if output_dir.name == "fit_retry_1" else
                                1.1 if output_dir.name == "fit_retry_2" else 2.0)
                    path = output_dir / f"seg_{segment.index:04d}.wav"
                    make_audio(path, duration)
                    segment.tts_audio_path, segment.tts_duration = path, duration

        class Muxer:
            def mux(self, source, audio, output):
                Path(output).write_bytes(b"synthetic video")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = DubbingPipeline.__new__(DubbingPipeline)
            pipeline.config = AppConfig(enable_overlap=False, unload_stt_before_tts=False).validate()
            pipeline.extractor, pipeline.stt = Extractor(), STT()
            pipeline.translator, pipeline._tts, pipeline.muxer = Translator(), TTS(), Muxer()
            pipeline.sync_engine = SyncEngine(max_workers=1, prefer_rubberband=False)
            with mock.patch("proxmax.pipeline_process.get_media_duration", return_value=3.0), \
                 mock.patch("proxmax.pipeline_process.build_units", return_value=[]):
                segments, timings, stats = pipeline._run_steps(root / "video.mp4", root, root / "out.mp4")
            self.assertEqual(pipeline.translator.calls, 2)
            self.assertEqual(len(stats["tts_fit_retry"]["rounds"]), 2)
            self.assertEqual(stats["sync"]["unplaced"], 0)
            self.assertEqual([item["prepared_reused"] for item in stats["sync_passes"]], [0, 1, 1])
            self.assertEqual(segments[0].original_text_bn, "আমাদের কাছে এখানে ১২৬ ওভিআর এবং ৫ স্টার আছে।")
            self.assertEqual(segments[0].text_bn, "আমাদের ১২৬ ওভিআর ও ৫ স্টার আছে।")
            self.assertTrue(segments[0].tts_audio_path.is_file())
            self.assertGreater(timings.translate_tts_wall, 0)


if __name__ == "__main__":
    unittest.main()
