"""Optional source-audio preprocessing stays off by default and preserves time."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmarks.stt_compare import word_errors
from proxmax import audio_fx
from proxmax.audio_extraction import AudioExtractor
from proxmax.config import AppConfig
from tests._support import FFMPEG


class TestSourceAudioConfig(unittest.TestCase):
    def test_filter_is_opt_in_and_validated(self):
        self.assertEqual(AppConfig().validate().stt_audio_filter, "none")
        config = AppConfig.from_yaml(Path(__file__).resolve().parents[1] / "config.yaml").validate()
        self.assertEqual(config.stt_audio_filter, "none")
        config._apply_mapping({"stt": {"audio_filter": "highpass60"}})
        self.assertEqual(config.validate().stt_audio_filter, "highpass60")
        with self.assertRaises(ValueError):
            AppConfig(stt_audio_filter="denoise-magic").validate()
        with self.assertRaises(ValueError):
            AudioExtractor(audio_filter="unknown")

    def test_word_error_rate_counts_real_insertions_deletions_substitutions(self):
        self.assertEqual(word_errors("One, two three", "one two three"), (0, 3))
        self.assertEqual(word_errors("one two three", "one too three four"), (2, 3))
        self.assertEqual(word_errors("one two", ""), (2, 2))
        with self.assertRaises(ValueError):
            word_errors("", "hello")


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class TestActualFiltering(unittest.TestCase):
    def test_highpass_attenuates_low_rumble_without_changing_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rate = 16000
            t = np.arange(rate * 2) / rate
            original = (.2 * np.sin(2 * np.pi * 40 * t) + .15 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
            source = root / "source.wav"
            audio_fx.write_wav(source, original, rate)
            normal = AudioExtractor(audio_filter="none").extract(source, root / "normal.wav", duration_seconds=2)
            filtered = AudioExtractor(audio_filter="highpass60").extract(source, root / "filtered.wav", duration_seconds=2)
            raw, raw_rate = audio_fx.read_wav(normal)
            clean, clean_rate = audio_fx.read_wav(filtered)
            self.assertEqual((raw_rate, clean_rate), (rate, rate))
            self.assertEqual((len(raw), len(clean)), (rate * 2, rate * 2))
            def amplitude(audio, frequency):
                return abs(np.dot(audio, np.exp(-2j * np.pi * frequency * t)))
            self.assertLess(amplitude(clean, 40), amplitude(raw, 40) * .8)
            self.assertGreater(amplitude(clean, 1000), amplitude(raw, 1000) * .9)


if __name__ == "__main__":
    unittest.main()
