"""Synthetic pipeline checks for silence-first, speed, rewrite, and cache order."""

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


def tone(seconds, rate=8000):
    t = np.arange(round(seconds * rate), dtype=np.float32) / rate
    return (.2 * np.sin(2 * np.pi * 235 * t)).astype(np.float32)


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class TestSilenceFitPipeline(unittest.TestCase):
    def engine(self):
        return SyncEngine(sample_rate=8000, max_workers=1, slot_gap_ms=0,
                          borrow_silence_ms=0, prefer_rubberband=False,
                          slow_down_short_clips=False, fill_extension=0)

    def test_ten_second_window_uses_tail_then_internal_before_any_stretch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "long.wav"
            clip = np.r_[np.zeros(3200), tone(4.4), np.zeros(11200), tone(4.0), np.zeros(9600)]
            self.assertAlmostEqual(len(clip) / 8000, 11.4)
            audio_fx.write_wav(source, clip, 8000)
            segment = Segment(0, 0.0, 10.0, "Long line", "একটি দীর্ঘ বাক্য।", tts_audio_path=source)
            engine = self.engine()
            with mock.patch("proxmax.sync_engine.run_ffmpeg") as render:
                engine.build_final_track([segment], 10.0, root, root / "first.wav")
            render.assert_not_called()
            self.assertEqual(engine.last_stats["unplaced"], 0)
            self.assertAlmostEqual(segment.tts_duration, 11.4)
            self.assertAlmostEqual(segment.stretched_duration, 10.0, places=3)
            self.assertEqual(segment.tempo_used, 1.0)
            self.assertAlmostEqual(segment.sync_trimmed_tail_seconds, 1.175, places=3)
            self.assertAlmostEqual(segment.sync_trimmed_internal_seconds, .225, places=3)
            self.assertEqual(segment.sync_trimmed_head_seconds, 0)
            self.assertEqual(segment.sync_trimmed_internal_cuts, 1)
            self.assertEqual(engine.last_stats["silence_trimmed_clips"], 1)
            self.assertAlmostEqual(engine.last_stats["silence_trimmed_seconds"], 1.4, places=3)
            original, _ = audio_fx.read_wav(source)
            fitted, _ = audio_fx.read_wav(segment.stretched_audio_path)
            first_start, first_length = round(.4 * 8000), round(4.4 * 8000)
            second_start, second_length = round(6.2 * 8000), round(4.0 * 8000)
            np.testing.assert_allclose(fitted[first_start:first_start + first_length],
                                       original[first_start:first_start + first_length], atol=1 / 16000)
            shifted = second_start - round(.225 * 8000)
            np.testing.assert_allclose(fitted[shifted:shifted + second_length],
                                       original[second_start:second_start + second_length], atol=1 / 16000)

            first_stats = dict(engine.last_stats)
            engine.build_final_track([segment], 10.0, root, root / "cached.wav")
            self.assertEqual(engine.last_stats["prepared_reused"], 1)
            self.assertEqual(segment.sync_trimmed_internal_cuts, 1)
            self.assertAlmostEqual(segment.sync_trimmed_internal_seconds, .225, places=3)
            for key in ("prepared_reused", "prepared_computed"):
                first_stats.pop(key)
            cached_stats = dict(engine.last_stats)
            for key in ("prepared_reused", "prepared_computed"):
                cached_stats.pop(key)
            self.assertEqual(first_stats, cached_stats)
            a, _ = audio_fx.read_wav(root / "first.wav")
            b, _ = audio_fx.read_wav(root / "cached.wav")
            np.testing.assert_array_equal(a, b)

    def test_all_safe_silence_is_removed_before_speed_and_no_rewrite_if_it_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "clip.wav"
            clip = np.r_[tone(5.5, 24000), np.zeros(9600), tone(6.1, 24000), np.zeros(19200)]
            audio_fx.write_wav(source, clip, 24000)
            segment = Segment(0, 0.0, 10.0, "Long line", "একটি দীর্ঘ বাক্য।", tts_audio_path=source)
            engine = SyncEngine(sample_rate=24000, max_workers=1, slot_gap_ms=0,
                                borrow_silence_ms=0, prefer_rubberband=False)
            engine.build_final_track([segment], 10.0, root, root / "track.wav")
            self.assertAlmostEqual(segment.sync_trimmed_tail_seconds, .775, places=3)
            self.assertAlmostEqual(segment.sync_trimmed_internal_seconds, .35, places=3)
            self.assertEqual(segment.sync_trimmed_head_seconds, 0)
            self.assertGreater(segment.tempo_used, 1.16)
            self.assertLessEqual(segment.tempo_used, 1.20)
            self.assertEqual(engine.last_stats["unplaced"], 0)
            pipeline = DubbingPipeline.__new__(DubbingPipeline)
            pipeline.config = AppConfig(slot_gap_ms=0, borrow_silence_ms=0).validate()
            pipeline.translator = mock.Mock()
            result = pipeline._repair_overlong_tts([segment], root / "tts", 10.0)
            self.assertEqual(result["overlong_before"], 0)
            pipeline.translator.shorten_to_durations.assert_not_called()

    def test_target_is_each_real_window_not_a_ten_second_constant(self):
        for target in (.25, 2.75, 10.0):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "long.wav"
                clip = np.r_[tone(target - .075), np.zeros(round((.5 + .075) * 8000))]
                audio_fx.write_wav(source, clip, 8000)
                segment = Segment(0, 0.0, target, "Line", "বাংলা বাক্য।", tts_audio_path=source)
                engine = self.engine()
                engine.build_final_track([segment], target, root, root / "track.wav")
                self.assertAlmostEqual(segment.placed_duration, target, delta=1 / 8000)
                self.assertAlmostEqual(segment.sync_trimmed_tail_seconds, .5, places=3)
                self.assertEqual(segment.tempo_used, 1.0)
                self.assertEqual(engine.last_stats["unplaced"], 0)

    def test_real_ffmpeg_measures_speed_fit_and_leaves_unfit_speech_whole(self):
        from proxmax.sync_math import available_filters

        for prefer_rubberband in ([False, True] if "rubberband" in available_filters() else [False]):
            for seconds, expected_unplaced in ((11.7, 0), (12.5, 1)):
                with self.subTest(rubberband=prefer_rubberband, seconds=seconds), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    source = root / "source.wav"
                    audio_fx.write_wav(source, tone(seconds, 24000), 24000)
                    segment = Segment(0, 0.0, 10.0, "Long line", "একটি পূর্ণ বাক্য।", tts_audio_path=source)
                    engine = SyncEngine(sample_rate=24000, max_workers=1, slot_gap_ms=0,
                                        borrow_silence_ms=0, prefer_rubberband=prefer_rubberband)
                    engine.build_final_track([segment], 10.0, root, root / "track.wav")
                    self.assertEqual(engine.last_stats["unplaced"], expected_unplaced)
                    self.assertGreaterEqual(segment.tempo_used, 1.15)
                    self.assertLessEqual(segment.tempo_used, 1.20)
                    stretched, rate = audio_fx.read_wav(segment.stretched_audio_path)
                    self.assertGreater(float(np.max(np.abs(stretched[-rate // 5:]))), .05)
                    if expected_unplaced:
                        self.assertTrue(segment.sync_tempo_cap_measured)
                        self.assertIsNone(segment.placed_duration)
                        self.assertGreater(segment.stretched_duration, 10.0)
                        self.assertIn("not truncated", segment.sync_issue)
                    else:
                        self.assertLessEqual(segment.placed_duration, 10.0)

    def test_sub_millisecond_overrun_trims_silence_before_any_speed_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "clip.wav"
            audio_fx.write_wav(source, np.r_[tone(1.0), np.zeros(800)], 8000)
            segment = Segment(0, 0.0, 1.0995, "Line", "বাংলা।", tts_audio_path=source)
            engine = self.engine()
            with mock.patch("proxmax.sync_engine.run_ffmpeg") as render:
                engine.build_final_track([segment], 1.0995, root, root / "track.wav")
            render.assert_not_called()
            self.assertAlmostEqual(segment.sync_trimmed_tail_seconds, .0005)
            self.assertEqual(segment.tempo_used, 1.0)
            self.assertEqual(engine.last_stats["unplaced"], 0)

    def test_very_short_window_is_not_silently_expanded_to_120_ms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "short.wav"
            audio_fx.write_wav(source, tone(.09), 8000)
            segment = Segment(0, 0.0, .06, "Hi", "হ্যাঁ।", tts_audio_path=source)
            engine = self.engine()
            filters = []

            def render(command, **_kwargs):
                filters.append(command[command.index("-filter:a") + 1])
                audio_fx.write_wav(command[-1], tone(.075), 8000)

            with mock.patch("proxmax.sync_engine.run_ffmpeg", side_effect=render):
                engine.build_final_track([segment], .06, root, root / "track.wav")
            self.assertEqual(len(filters), 1)
            self.assertIn("atempo=1.2000", filters[0])
            self.assertEqual(engine.last_stats["unplaced"], 1)
            self.assertIsNone(segment.placed_duration)
            self.assertTrue(segment.sync_tempo_cap_measured)

    def test_ten_second_unfit_after_cap_rewrites_then_measures_regenerated_wav(self):
        events = []

        class Translator:
            def shorten_to_durations(self, segments, targets):
                events.append(("rewrite", targets[0]))
                self.assert_cap = segments[0].sync_tempo_cap_measured
                segments[0].text_bn = "আমাদের ১২৬ ওভিআর ও ৫ স্টার আছে।"
                return 1

        class TTS:
            def synthesize(self, segments, output_dir):
                events.append(("tts", segments[0].text_bn))
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                path = Path(output_dir) / "new.wav"
                audio_fx.write_wav(path, tone(9.5), 8000)
                segments[0].tts_audio_path = path
                segments[0].tts_duration = 4.0  # Deliberately wrong provider metadata.

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "old.wav"
            audio_fx.write_wav(source, tone(12.5), 8000)
            segment = Segment(0, 0.0, 10.0, "We have 126 OVR and 5 stars",
                              "আমাদের কাছে ১২৬ ওভিআর এবং ৫ স্টার আছে।", tts_audio_path=source)
            engine = self.engine()

            def render(command, **_kwargs):
                spec = command[command.index("-filter:a") + 1]
                events.append(("tempo", spec))
                audio_fx.write_wav(command[-1], tone(10.42), 8000)

            pipeline = DubbingPipeline.__new__(DubbingPipeline)
            pipeline.config = AppConfig(slot_gap_ms=0, borrow_silence_ms=0).validate()
            pipeline.sync_engine = engine
            pipeline.translator = Translator()
            pipeline._tts = TTS()
            with mock.patch("proxmax.sync_engine.run_ffmpeg", side_effect=render):
                engine.build_final_track([segment], 10.0, root, root / "first.wav")
                self.assertEqual(engine.last_stats["unplaced"], 1)
                self.assertTrue(segment.sync_tempo_cap_measured)
                self.assertEqual(segment.tempo_used, 1.20)
                self.assertIsNone(segment.placed_duration)
                result = pipeline._repair_overlong_tts([segment], root / "tts", 10.0)
                self.assertEqual(result["shortened"], 1)
                self.assertTrue(pipeline.translator.assert_cap)
                self.assertAlmostEqual(segment.tts_duration, 9.5)
                engine.build_final_track([segment], 10.0, root, root / "second.wav")

            self.assertEqual([event[0] for event in events], ["tempo", "rewrite", "tts"])
            self.assertIn("atempo=1.2000", events[0][1])
            self.assertAlmostEqual(events[1][1], 9.96)
            self.assertEqual(engine.last_stats["unplaced"], 0)
            self.assertLessEqual(segment.placed_duration, 10.0)
            self.assertIn("১২৬", segment.text_bn)
            self.assertIn("৫", segment.text_bn)
            self.assertEqual(segment.original_text_bn, "আমাদের কাছে ১২৬ ওভিআর এবং ৫ স্টার আছে।")
            self.assertTrue(source.exists())


if __name__ == "__main__":
    unittest.main()
