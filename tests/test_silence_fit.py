"""Conservative silence removal and slowdown safety regressions."""

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
from proxmax.sync_silence import trim_silence_for_overrun
from tests._support import FFMPEG


def tone(seconds, rate=8000):
    t = np.arange(round(seconds * rate), dtype=np.float32) / rate
    return (.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class TestReverseSilenceTrim(unittest.TestCase):
    def setUp(self):
        self.rate = 1000
        self.audio = np.r_[
            np.zeros(300), np.full(200, .1), np.zeros(400), np.full(200, .2),
            np.zeros(500), np.full(300, .3), np.zeros(300),
        ].astype(np.float32)

    def test_tail_then_last_internal_pause_then_earlier_pause_then_head(self):
        result = trim_silence_for_overrun(self.audio, self.rate, .4)
        self.assertAlmostEqual(result.tail_seconds, .275)
        self.assertAlmostEqual(result.internal_seconds, .125)
        self.assertEqual(result.internal_cuts, 1)
        self.assertEqual(result.head_seconds, 0)
        # The first interior gap is untouched; only the later gap was shortened.
        self.assertEqual(np.flatnonzero(result.audio == .2)[0], 900)
        self.assertEqual(np.flatnonzero(result.audio == .3)[0], 1475)
        self.assertEqual(len(result.audio), len(self.audio) - 400)

        result = trim_silence_for_overrun(self.audio, self.rate, .8)
        self.assertAlmostEqual(result.tail_seconds, .275)
        self.assertAlmostEqual(result.internal_seconds, .525)
        self.assertEqual(result.internal_cuts, 2)
        self.assertEqual(result.head_seconds, 0)
        # Only 25 ms on each side remains in the later gap; the earlier is partly trimmed.
        self.assertEqual(np.flatnonzero(result.audio == .3)[0] - np.flatnonzero(result.audio == .2)[-1] - 1, 50)
        self.assertEqual(np.flatnonzero(result.audio == .2)[0] - np.flatnonzero(result.audio == .1)[-1] - 1, 325)

        result = trim_silence_for_overrun(self.audio, self.rate, 10)
        self.assertAlmostEqual(result.head_seconds, .275)
        self.assertAlmostEqual(result.tail_seconds, .275)
        self.assertAlmostEqual(result.internal_seconds, .80)
        self.assertEqual(result.internal_cuts, 2)
        # Every speech sample remains, in the same order; 25 ms edges remain.
        np.testing.assert_array_equal(result.audio[result.audio != 0], self.audio[self.audio != 0])
        self.assertEqual(np.flatnonzero(result.audio)[0], 25)
        self.assertEqual(len(result.audio) - 1 - np.flatnonzero(result.audio)[-1], 25)

    def test_small_overrun_never_touches_interior_or_head(self):
        result = trim_silence_for_overrun(self.audio, self.rate, .2)
        self.assertEqual((result.head_seconds, result.internal_seconds, result.internal_cuts), (0, 0, 0))
        self.assertAlmostEqual(result.tail_seconds, .2)
        np.testing.assert_array_equal(result.audio, self.audio[:-200])

    def test_quiet_but_above_conservative_threshold_is_never_cut(self):
        # -50 dBFS is below the old -45 detector, but not the conservative -70 threshold.
        quiet = np.full(250, 10 ** (-50 / 20), dtype=np.float32)
        clip = np.r_[np.full(200, .1), np.zeros(400), quiet, np.zeros(400), np.full(200, .2)].astype(np.float32)
        result = trim_silence_for_overrun(clip, self.rate, 10, threshold_dbfs=-45)
        self.assertEqual(np.count_nonzero(result.audio == quiet[0]), len(quiet))
        self.assertEqual(result.internal_cuts, 2)
        np.testing.assert_array_equal(result.audio[result.audio != 0], clip[clip != 0])

    def test_short_pauses_and_custom_padding_remain_protected(self):
        clip = np.r_[np.full(200, .1), np.zeros(40), np.full(200, .2)].astype(np.float32)
        result = trim_silence_for_overrun(clip, self.rate, 10)
        np.testing.assert_array_equal(result.audio, clip)
        clip = np.r_[np.full(200, .1), np.zeros(200), np.full(200, .2)].astype(np.float32)
        result = trim_silence_for_overrun(clip, self.rate, 10, pad_ms=80)
        self.assertAlmostEqual(result.internal_seconds, .04)
        self.assertEqual(np.flatnonzero(result.audio == .2)[0] - np.flatnonzero(result.audio == .1)[-1] - 1, 160)
        result = trim_silence_for_overrun(clip, self.rate, 10, min_internal_pause_ms=120)
        self.assertAlmostEqual(result.internal_seconds, .08)
        self.assertEqual(np.flatnonzero(result.audio == .2)[0] - np.flatnonzero(result.audio == .1)[-1] - 1, 120)

    def test_randomized_active_frames_survive_in_original_order(self):
        rng = np.random.default_rng(2026)
        for _ in range(50):
            # Ten-millisecond frames, including a partial final frame.
            activity = rng.integers(0, 2, size=120)
            frames = [rng.uniform(.02, .3, size=10).astype(np.float32) if active
                      else np.zeros(10, np.float32) for active in activity]
            clip = np.concatenate(frames)[:rng.integers(1101, 1201)]
            result = trim_silence_for_overrun(clip, self.rate, rng.uniform(.1, 2.0),
                                              pad_ms=int(rng.integers(0, 60)))
            np.testing.assert_array_equal(result.audio[result.audio != 0], clip[clip != 0])
            self.assertLessEqual(len(result.audio), len(clip))

    def test_entirely_silent_or_no_overrun_is_not_trimmed(self):
        silent = np.zeros(1000, np.float32)
        np.testing.assert_array_equal(trim_silence_for_overrun(silent, self.rate, 1).audio, silent)
        np.testing.assert_array_equal(trim_silence_for_overrun(self.audio, self.rate, 0).audio, self.audio)
        with self.assertRaises(ValueError):
            trim_silence_for_overrun(np.zeros((2, 10)), self.rate, 1)

    def test_stricter_custom_threshold_is_respected(self):
        quiet = np.full(200, 10 ** (-60 / 20), dtype=np.float32)
        clip = np.r_[np.full(100, .1), np.zeros(300), quiet, np.zeros(300), np.full(100, .2)]
        result = trim_silence_for_overrun(clip, self.rate, 10, threshold_dbfs=-70)
        self.assertEqual(np.count_nonzero(result.audio == quiet[0]), len(quiet))


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class TestSafeSlowdown(unittest.TestCase):
    def _setup(self, root, speech=1.0, window=2.0):
        source = root / "source.wav"
        audio_fx.write_wav(source, tone(speech), 8000)
        segment = Segment(0, 0, window, "A line", "একটি বাক্য।", tts_audio_path=source)
        engine = SyncEngine(sample_rate=8000, max_workers=1, slot_gap_ms=0,
                            borrow_silence_ms=0, prefer_rubberband=False)
        return source, segment, engine

    def test_suspicious_short_slowdown_retains_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, seg, engine = self._setup(root)
            with mock.patch("proxmax.sync_prepare.run_ffmpeg", side_effect=lambda cmd, **_: audio_fx.write_wav(cmd[-1], tone(.2), 8000)):
                engine.build_final_track([seg], 2, root, root / "track.wav")
            self.assertEqual(seg.tempo_used, 1.0)
            self.assertEqual(seg.placed_duration, 1.0)
            self.assertIn("unreliable", seg.sync_recovery)

    def test_silent_slowdown_retains_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, seg, engine = self._setup(root)
            with mock.patch("proxmax.sync_prepare.run_ffmpeg", side_effect=lambda cmd, **_: audio_fx.write_wav(cmd[-1], np.zeros(4000), 8000)):
                engine.build_final_track([seg], 2, root, root / "track.wav")
            self.assertEqual(seg.tempo_used, 1.0)
            self.assertAlmostEqual(seg.placed_duration, 1.0)
            self.assertIn("unreliable", seg.sync_recovery)

    def test_overlong_speech_never_invokes_audio_speedup_even_if_old_cap_is_mutated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, seg, engine = self._setup(root, speech=1.4, window=1)
            engine.max_tempo = 8.0
            with mock.patch("proxmax.sync_prepare.run_ffmpeg") as render:
                engine.build_final_track([seg], 1, root, root / "track.wav")
            render.assert_not_called()
            self.assertEqual(seg.tempo_used, 1)
            self.assertEqual(engine.max_tempo, 1)
            self.assertGreater(seg.video_added_seconds, 0)

    def test_measured_slowdown_not_just_filter_ratio_respects_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, seg, engine = self._setup(root)
            lengths = iter((1.20, 1.09))
            def padded_render(cmd, **_):
                audio_fx.write_wav(cmd[-1], tone(next(lengths)), 8000)
            with mock.patch("proxmax.sync_prepare.run_ffmpeg", side_effect=padded_render) as render:
                engine.build_final_track([seg], 2, root, root / "track.wav")
            self.assertEqual(render.call_count, 2)
            self.assertGreater(seg.tempo_used, .90)
            self.assertLessEqual(seg.placed_duration, 1 / .90 + .5 / 8000)
            self.assertIsNone(seg.sync_recovery)

    def test_slowdown_fallback_is_not_cached_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, seg, engine = self._setup(root)
            with mock.patch("proxmax.sync_prepare.run_ffmpeg", side_effect=OSError("temporary")) as render:
                engine.build_final_track([seg], 2, root, root / "first.wav")
                engine.build_final_track([seg], 2, root, root / "second.wav")
            self.assertGreaterEqual(render.call_count, 2)
            self.assertEqual(engine.last_stats["prepared_reused"], 0)
            self.assertEqual(seg.tempo_used, 1)


if __name__ == "__main__":
    unittest.main()
