"""Long/short clips never trigger a Gemini timing rewrite or extra TTS call."""

import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np

from proxmax import audio_fx
from proxmax.config import AppConfig
from proxmax.models import Segment
from proxmax.pipeline import DubbingPipeline
from proxmax.sync_engine import SyncEngine
from tests._support import FFMPEG


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class TestNoFitRewriteOrchestration(unittest.TestCase):
    def test_long_and_short_audio_preserve_text_and_use_video_edits(self):
        class Extractor:
            def extract(self, source, target, **kwargs):
                audio_fx.write_wav(target, np.full(16000 * 3, .1, np.float32), 16000)
                return target
        class STT:
            last_stats = {}
            def transcribe(self, _path):
                return [Segment(0, 0, .8, "126 rating"), Segment(1, 1, 2.5, "Next")]
        class Translator:
            def __init__(self):
                self.translation_calls = self.fit_calls = 0
            def translate_segments(self, segs):
                self.translation_calls += 1
                for s in segs:
                    s.text_bn = "আমাদের ১২৬ রেটিং আছে।" if s.index == 0 else "পরের কার্ড।"
            def shorten_to_durations(self, *_):
                self.fit_calls += 1
                raise AssertionError("Gemini fit rewrite is forbidden")
        class TTS:
            def __init__(self):
                self.calls = 0
            def describe(self):
                return "fake"
            def set_budget(self, *_):
                pass
            def synthesize(self, segs, directory):
                self.calls += 1
                Path(directory).mkdir(parents=True, exist_ok=True)
                for s in segs:
                    duration = 2.0 if s.index == 0 else .3
                    samples = np.arange(round(duration * 8000)) / 8000
                    path = Path(directory) / f"seg_{s.index}.wav"
                    audio_fx.write_wav(path, .2 * np.sin(2 * np.pi * 300 * samples), 8000)
                    s.tts_audio_path, s.tts_duration = path, duration
        class Muxer:
            def video_frame_rate(self, source):
                return Fraction(25)
            def ensure_picture_duration(self, source, required, work, fps):
                return source
            def mux(self, source, audio, output, *, required_video_duration, video_plan):
                self.plan = video_plan
                Path(output).write_bytes(b"synthetic output")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = DubbingPipeline.__new__(DubbingPipeline)
            pipeline.config = AppConfig(enable_overlap=False).validate()
            pipeline.extractor, pipeline.stt, pipeline.translator = Extractor(), STT(), Translator()
            pipeline._tts, pipeline.muxer = TTS(), Muxer()
            pipeline.sync_engine = SyncEngine(sample_rate=8000, max_workers=2, prefer_rubberband=False)
            with mock.patch("proxmax.pipeline_process.get_media_duration", return_value=3), \
                 mock.patch("proxmax.pipeline_process.build_units", return_value=[]):
                segs, _, stats = pipeline._run_steps(root / "source.mp4", root, root / "dub.mp4")
            self.assertEqual(pipeline.translator.translation_calls, 1)
            self.assertEqual(pipeline.translator.fit_calls, 0)
            self.assertEqual(pipeline._tts.calls, 1)
            self.assertEqual(segs[0].text_bn, "আমাদের ১২৬ রেটিং আছে।")
            self.assertTrue(all(s.placed_duration for s in segs))
            self.assertTrue(all(.90 <= s.tempo_used <= 1 for s in segs))
            self.assertGreater(segs[0].video_added_seconds, 0)
            self.assertGreater(segs[1].video_speed, 1)
            self.assertLessEqual(segs[1].video_speed, 1.15)
            self.assertEqual(stats["sync"]["unplaced"], 0)
            self.assertNotIn("tts_fit_retry", stats)
            self.assertTrue(pipeline.muxer.plan.changed)


if __name__ == "__main__":
    unittest.main()
