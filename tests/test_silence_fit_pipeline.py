"""Silence is removed before any timing decision, for short and long clips."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from proxmax import audio_fx
from proxmax.models import Segment
from proxmax.sync_engine import SyncEngine
from proxmax.tts.base import BaseTTSEngine
from tests._support import FFMPEG


def tone(seconds, rate=8000):
    t = np.arange(round(seconds * rate), dtype=np.float32) / rate
    return (.2 * np.sin(2 * np.pi * 235 * t)).astype(np.float32)


class FakeTTS(BaseTTSEngine):
    sample_rate = 8000
    def synthesize(self, segments, output_dir):
        return segments


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class TestSilenceBeforeTiming(unittest.TestCase):
    def engine(self):
        return SyncEngine(sample_rate=8000, max_workers=1, slot_gap_ms=0,
                          borrow_silence_ms=0, prefer_rubberband=False)

    def test_tts_immediately_trims_all_gaps_and_records_audit_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = np.r_[np.zeros(3200), tone(1), np.zeros(11200), tone(1), np.zeros(9600)]
            seg = Segment(0, 0, 10, "source", "বাংলা")
            engine = FakeTTS(trim_silence=False)
            engine._write_clip(seg, clip, root)
            self.assertTrue(seg.tts_silence_trimmed)
            self.assertAlmostEqual(seg.tts_raw_duration, 5.0)
            self.assertAlmostEqual(seg.tts_trimmed_head_seconds, .375, places=3)
            self.assertAlmostEqual(seg.tts_trimmed_internal_seconds, 1.35, places=3)
            self.assertAlmostEqual(seg.tts_trimmed_tail_seconds, 1.175, places=3)
            self.assertAlmostEqual(seg.tts_duration, 2.10, places=3)
            self.assertEqual(seg.tts_trimmed_internal_cuts, 1)

    def test_same_silence_is_removed_when_audio_is_short_or_long(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = np.r_[np.zeros(1600), tone(.5), np.zeros(2400), tone(.5), np.zeros(2400)]
            audio_fx.write_wav(root / "in.wav", clip, 8000)
            totals = []
            for window in (.5, 3.0):
                seg = Segment(0, 0, window, "line", "বাংলা", tts_audio_path=root / "in.wav")
                engine = self.engine()
                engine.build_final_track([seg], window, root / str(window), root / f"{window}.wav")
                totals.append((seg.sync_trimmed_head_seconds, seg.sync_trimmed_internal_seconds,
                               seg.sync_trimmed_tail_seconds))
                self.assertLessEqual(seg.tempo_used, 1.0)
                self.assertIsNotNone(seg.placed_duration)
            self.assertEqual(totals[0], totals[1])
            self.assertTrue(all(item > 0 for item in totals[0]))

    def test_all_active_samples_survive_before_slowdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = np.r_[np.zeros(800), tone(1), np.zeros(2400), tone(1), np.zeros(800)]
            audio_fx.write_wav(root / "in.wav", clip, 8000)
            seg = Segment(0, 0, 4, "line", "বাংলা", tts_audio_path=root / "in.wav")
            engine = self.engine()
            from proxmax.sync_prepare import run_ffmpeg
            inputs = []
            def inspect(command, **kwargs):
                inputs.append(audio_fx.read_wav(command[command.index("-i") + 1])[0])
                return run_ffmpeg(command, **kwargs)
            with mock.patch("proxmax.sync_prepare.run_ffmpeg", side_effect=inspect):
                engine.build_final_track([seg], 4, root, root / "out.wav")
            raw, _ = audio_fx.read_wav(root / "in.wav")
            trimmed = inputs[0]
            np.testing.assert_allclose(trimmed[trimmed != 0], raw[raw != 0], atol=1 / 32768)
            self.assertAlmostEqual(seg.tempo_used, .9, places=4)

    def test_sub_millisecond_overrun_still_removes_all_safe_silence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_fx.write_wav(root / "in.wav", np.r_[tone(1), np.zeros(800)], 8000)
            seg = Segment(0, 0, 1.0995, "line", "বাংলা", tts_audio_path=root / "in.wav")
            engine = self.engine()
            engine.build_final_track([seg], 1.0995, root, root / "out.wav", video_fps=100)
            self.assertAlmostEqual(seg.sync_trimmed_tail_seconds, .075, places=3)
            self.assertLess(seg.tempo_used, 1)
            self.assertEqual(engine.last_stats["unplaced"], 0)

    def test_invalid_raw_audio_is_not_claimed_as_a_trimmed_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_fx.write_wav(root / "in.wav", np.zeros(8000), 8000)
            seg = Segment(0, 0, 1, "line", "বাংলা", tts_audio_path=root / "in.wav")
            with self.assertRaises(RuntimeError):
                self.engine().build_final_track([seg], 1, root, root / "out.wav")


if __name__ == "__main__":
    unittest.main()
