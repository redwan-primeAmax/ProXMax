"""Translation queue waiting is not reported as active TTS time."""

import tempfile
import time
import unittest
from pathlib import Path

from proxmax.config import AppConfig
from proxmax.models import Segment
from proxmax.pipeline import DubbingPipeline
from tests._support import _RecordingTTS, _RecordingTranslator


class SlowTranslator(_RecordingTranslator):
    def translate_segments(self, segments):
        time.sleep(.18)
        return super().translate_segments(segments)


class FastTTS(_RecordingTTS):
    def synthesize(self, segments, output_dir):
        time.sleep(.015)
        return super().synthesize(segments, output_dir)


class TestTimingDiagnostics(unittest.TestCase):
    def test_tts_calls_exclude_waiting_for_slow_translation_queue(self):
        pipeline = DubbingPipeline.__new__(DubbingPipeline)
        pipeline.config = AppConfig(gemini_batch_size=1, enable_overlap=True).validate()
        pipeline.translator = SlowTranslator()
        pipeline._tts = FastTTS()
        segments = [Segment(i, float(i), i + .8, f"Line {i}") for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            started = time.perf_counter()
            translation, tts = pipeline._translate_and_synthesize(segments, Path(tmp))
            elapsed = time.perf_counter() - started
        self.assertGreaterEqual(translation, .50)
        self.assertGreaterEqual(elapsed, .50)
        self.assertGreater(tts, .03)
        self.assertLess(tts, .30)  # excludes the three translation-queue waits
        self.assertLess(tts, elapsed / 2)


if __name__ == "__main__":
    unittest.main()
