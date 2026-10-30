"""End-first silence fitting, timing caps, and waveform safety regressions."""

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
        # -50 dBFS is below the old -45 detector, but not below the new -55 threshold.
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
class TestMeasuredCapRetry(unittest.TestCase):
    def _setup(self, root, cap=1.20):
        source = root / "source.wav"
        audio_fx.write_wav(source, tone(1.1), 8000)
        segment = Segment(0, 0.0, 1.0, "A long line", "একটি লম্বা বাক্য।", tts_audio_path=source)
        engine = SyncEngine(max_tempo=cap, sample_rate=8000, max_workers=1,
                            slot_gap_ms=0, borrow_silence_ms=0, prefer_rubberband=False)
        return source, segment, engine

    def test_real_output_overruns_at_calculated_tempo_then_measures_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, segment, engine = self._setup(root)
            calls = []

            def render(command, **_kwargs):
                filter_spec = command[command.index("-filter:a") + 1]
                calls.append((command[command.index("-i") + 1], filter_spec))
                duration = .95 if "atempo=1.2000" in filter_spec else 1.05
                audio_fx.write_wav(command[-1], tone(duration), 8000)

            with mock.patch("proxmax.sync_engine.run_ffmpeg", side_effect=render):
                engine.build_final_track([segment], 1.0, root, root / "track.wav")
            self.assertEqual(len(calls), 2)
            self.assertIn("atempo=1.1000", calls[0][1])
            self.assertIn("atempo=1.2000", calls[1][1])
            self.assertEqual([path for path, _ in calls], [str(source), str(source)])
            self.assertEqual(engine.last_stats["unplaced"], 0)
            self.assertTrue(segment.sync_tempo_cap_measured)
            self.assertEqual(segment.tempo_used, 1.2)
            self.assertAlmostEqual(segment.stretched_duration, .95)
            self.assertLessEqual(segment.placed_duration, 1.0)

    def test_rounding_at_video_end_reaches_cap_instead_of_overhanging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _source, segment, _engine = self._setup(root)
            engine = SyncEngine(sample_rate=8000, max_workers=1, slot_gap_ms=80,
                                borrow_silence_ms=0, prefer_rubberband=False)
            filters = []

            def render(command, **_kwargs):
                spec = command[command.index("-filter:a") + 1]
                filters.append(spec)
                audio_fx.write_wav(command[-1], tone(.95 if "1.2000" in spec else 1.005), 8000)

            with mock.patch("proxmax.sync_engine.run_ffmpeg", side_effect=render):
                engine.build_final_track([segment], 1.0, root, root / "track.wav")
            self.assertEqual(len(filters), 2)
            self.assertIn("atempo=1.2000", filters[-1])
            self.assertLessEqual(segment.placed_duration, 1.0)
            self.assertEqual(engine.last_stats["unplaced"], 0)

    def test_mutating_engine_cap_above_global_limit_is_clamped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _source, segment, engine = self._setup(root)
            engine.max_tempo = 5.0
            filters = []

            def render(command, **_kwargs):
                spec = command[command.index("-filter:a") + 1]
                filters.append(spec)
                audio_fx.write_wav(command[-1], tone(.95 if "1.2000" in spec else 1.05), 8000)

            with mock.patch("proxmax.sync_engine.run_ffmpeg", side_effect=render):
                engine.build_final_track([segment], 1.0, root, root / "track.wav")
            self.assertEqual(engine.max_tempo, 1.20)
            self.assertIn("atempo=1.2000", filters[-1])
            self.assertLessEqual(segment.tempo_used, 1.20)

    def test_explicitly_lower_cap_is_never_exceeded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _source, segment, engine = self._setup(root, cap=1.15)
            filters = []

            def render(command, **_kwargs):
                spec = command[command.index("-filter:a") + 1]
                filters.append(spec)
                audio_fx.write_wav(command[-1], tone(.95 if "1.1500" in spec else 1.05), 8000)

            with mock.patch("proxmax.sync_engine.run_ffmpeg", side_effect=render):
                engine.build_final_track([segment], 1.0, root, root / "track.wav")
            self.assertEqual(len(filters), 2)
            self.assertIn("atempo=1.1500", filters[1])
            self.assertEqual(segment.tempo_used, 1.15)
            self.assertEqual(engine.last_stats["unplaced"], 0)

    def test_cap_retry_replaces_rather_than_double_counts_post_stretch_trim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _source, segment, engine = self._setup(root)

            def render(command, **_kwargs):
                spec = command[command.index("-filter:a") + 1]
                voice = tone(.95 if "1.2000" in spec else 1.0)
                audio_fx.write_wav(command[-1], np.r_[voice, np.zeros(800)], 8000)

            with mock.patch("proxmax.sync_engine.run_ffmpeg", side_effect=render):
                engine.build_final_track([segment], 1.0, root, root / "track.wav")
            self.assertTrue(segment.sync_tempo_cap_measured)
            self.assertEqual(engine.last_stats["unplaced"], 0)
            self.assertAlmostEqual(segment.sync_trimmed_tail_seconds, .05, places=3)
            self.assertAlmostEqual(engine.last_stats["silence_trimmed_seconds"], .05, places=3)
            self.assertAlmostEqual(segment.stretched_duration, 1.0, places=3)

    def test_valid_but_worse_cap_render_keeps_the_shorter_prior_waveform(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _source, segment, engine = self._setup(root)
            calls = []

            def render(command, **_kwargs):
                spec = command[command.index("-filter:a") + 1]
                calls.append(spec)
                audio_fx.write_wav(command[-1], tone(1.08 if "1.2000" in spec else 1.05), 8000)

            with mock.patch("proxmax.sync_engine.run_ffmpeg", side_effect=render):
                engine.build_final_track([segment], 1.0, root, root / "first.wav")
                engine.build_final_track([segment], 1.0, root, root / "second.wav")
            self.assertEqual(len(calls), 2)  # The measured unfit result is safely cached.
            self.assertEqual(engine.last_stats["prepared_reused"], 1)
            self.assertTrue(segment.sync_tempo_cap_measured)
            self.assertAlmostEqual(segment.tempo_used, 1.1)
            self.assertAlmostEqual(segment.stretched_duration, 1.05)
            self.assertEqual(engine.last_stats["unplaced"], 1)
            self.assertIsNone(segment.placed_duration)
            self.assertIn("allowed cap 1.20x was measured", segment.sync_issue)

    def test_unexpectedly_short_stretch_is_rejected_instead_of_losing_speech(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, segment, engine = self._setup(root)
            with mock.patch("proxmax.sync_engine.run_ffmpeg", side_effect=lambda command, **_: audio_fx.write_wav(command[-1], tone(.4), 8000)):
                output = engine._stretch_one(segment, (0.0, 1.0), root / "out.wav")
            self.assertIsNone(output)
            self.assertFalse(segment.sync_tempo_cap_measured)
            self.assertIn("possible speech loss", segment.sync_issue)
            self.assertTrue(source.exists())

    def test_suspiciously_short_cap_render_does_not_trigger_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _source, segment, engine = self._setup(root)

            def render(command, **_kwargs):
                spec = command[command.index("-filter:a") + 1]
                audio_fx.write_wav(command[-1], tone(.4 if "1.2000" in spec else 1.05), 8000)

            with mock.patch("proxmax.sync_engine.run_ffmpeg", side_effect=render):
                engine.build_final_track([segment], 1.0, root, root / "track.wav")
            self.assertEqual(engine.last_stats["unplaced"], 1)
            self.assertFalse(segment.sync_tempo_cap_measured)
            self.assertAlmostEqual(segment.stretched_duration, 1.05)

    def test_silent_cap_render_is_not_accepted_as_a_measured_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _source, segment, engine = self._setup(root)

            def render(command, **_kwargs):
                spec = command[command.index("-filter:a") + 1]
                clip = np.zeros(7000, np.float32) if "1.2000" in spec else tone(1.05)
                audio_fx.write_wav(command[-1], clip, 8000)

            with mock.patch("proxmax.sync_engine.run_ffmpeg", side_effect=render):
                engine.build_final_track([segment], 1.0, root, root / "track.wav")
            self.assertFalse(segment.sync_tempo_cap_measured)
            self.assertAlmostEqual(segment.stretched_duration, 1.05)
            self.assertEqual(engine.last_stats["unplaced"], 1)
            self.assertIn("cap could not be measured", segment.sync_issue)

    def test_cap_failure_blocks_rewrite_and_is_not_cached_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _source, segment, engine = self._setup(root)
            calls = []

            def broken(command, **_kwargs):
                spec = command[command.index("-filter:a") + 1]
                calls.append(spec)
                if "1.2000" in spec:
                    raise RuntimeError("temporary encoder error")
                audio_fx.write_wav(command[-1], tone(1.05), 8000)

            with mock.patch("proxmax.sync_engine.run_ffmpeg", side_effect=broken):
                engine.build_final_track([segment], 1.0, root, root / "first.wav")
            self.assertEqual(len(calls), 2)
            self.assertEqual(engine.last_stats["unplaced"], 1)
            self.assertFalse(segment.sync_tempo_cap_measured)
            self.assertIn("cap could not be measured", segment.sync_issue)
            pipeline = DubbingPipeline.__new__(DubbingPipeline)
            pipeline.config = AppConfig(slot_gap_ms=0, borrow_silence_ms=0).validate()
            pipeline.translator = mock.Mock()
            result = pipeline._repair_overlong_tts([segment], root / "tts", 1.0)
            self.assertEqual(result["awaiting_tempo_cap"], [0])
            pipeline.translator.shorten_to_durations.assert_not_called()

            def recovered(command, **_kwargs):
                spec = command[command.index("-filter:a") + 1]
                audio_fx.write_wav(command[-1], tone(.95 if "1.2000" in spec else 1.05), 8000)

            with mock.patch("proxmax.sync_engine.run_ffmpeg", side_effect=recovered):
                engine.build_final_track([segment], 1.0, root, root / "second.wav")
            self.assertEqual(engine.last_stats["prepared_computed"], 1)
            self.assertEqual(engine.last_stats["unplaced"], 0)
            self.assertTrue(segment.sync_tempo_cap_measured)


if __name__ == "__main__":
    unittest.main()
