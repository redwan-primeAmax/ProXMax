"""Complete speech placement under the no-speedup video-edit policy."""

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from proxmax import audio_fx
from proxmax.models import Segment
from proxmax.sync_engine import SyncEngine
from tests._support import FFMPEG


def tone(seconds, rate=8000, frequency=330):
    t = np.arange(round(seconds * rate)) / rate
    return (.2 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def make_clip(path, seconds, rate=8000):
    audio_fx.write_wav(path, tone(seconds, rate), rate)
    return path


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class TestCompleteSpeechPlacement(unittest.TestCase):
    def engine(self):
        return SyncEngine(sample_rate=8000, max_workers=2, slot_gap_ms=0,
                          borrow_silence_ms=0, prefer_rubberband=False)

    def test_long_clip_is_trimmed_then_video_tail_repeated_never_sped_up(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = np.r_[np.zeros(800), tone(5.5), np.zeros(4000), tone(5.5), np.zeros(4000)]
            audio_fx.write_wav(root / "tts.wav", raw, 8000)
            seg = Segment(0, 0, 10, "long line", "সম্পূর্ণ বাক্য", tts_audio_path=root / "tts.wav")
            engine = self.engine()
            with mock.patch("proxmax.sync_prepare.run_ffmpeg", side_effect=AssertionError("audio must not speed up")):
                engine.build_final_track([seg], 10, root, root / "out.wav", video_fps=25)
            self.assertEqual(seg.tempo_used, 1.0)
            self.assertEqual(engine.last_stats["unplaced"], 0)
            self.assertGreater(seg.sync_trimmed_head_seconds, 0)
            self.assertGreater(seg.sync_trimmed_internal_seconds, 0)
            self.assertGreater(seg.sync_trimmed_tail_seconds, 0)
            self.assertGreater(seg.placed_duration, 10)
            expected_extra = math.ceil((seg.placed_duration - 10) * 25) / 25
            self.assertAlmostEqual(seg.video_added_seconds, expected_extra, places=3)
            self.assertTrue(any(p.kind == "repeat" for p in engine.video_plan.pieces))
            self.assertAlmostEqual(seg.placed_duration, seg.stretched_duration, places=6)
            prepared, rate = audio_fx.read_wav(seg.stretched_audio_path)
            self.assertGreater(float(np.max(np.abs(prepared[-rate // 10:]))), .05)

    def test_short_clip_trims_every_silence_then_slows_and_caps_video_speed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = np.r_[np.zeros(800), tone(.4), np.zeros(2400), tone(.4), np.zeros(800)]
            audio_fx.write_wav(root / "tts.wav", raw, 8000)
            seg = Segment(0, 0, 3, "short line", "সংক্ষিপ্ত বাক্য", tts_audio_path=root / "tts.wav")
            engine = self.engine()
            engine.build_final_track([seg], 3, root, root / "out.wav", video_fps=25)
            self.assertGreater(seg.sync_trimmed_head_seconds, 0)
            self.assertGreater(seg.sync_trimmed_internal_seconds, 0)
            self.assertGreater(seg.sync_trimmed_tail_seconds, 0)
            self.assertAlmostEqual(seg.tempo_used, .9, places=3)
            self.assertGreater(seg.video_speed, 1)
            self.assertLessEqual(seg.video_speed, 1.15)
            self.assertGreater(seg.video_window_end - (seg.placed_start + seg.placed_duration), .5)
            self.assertEqual(seg.video_added_seconds, 0)

    def test_near_fit_slowdown_does_not_unnecessarily_extend_picture(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_clip(root / "tts.wav", 1.96)
            seg = Segment(0, 0, 2, "line", "বাংলা", tts_audio_path=source)
            engine = self.engine()
            engine.build_final_track([seg], 2, root, root / "out.wav", video_fps=25)
            self.assertGreaterEqual(seg.tempo_used, .9)
            self.assertLess(seg.tempo_used, 1.0)
            self.assertEqual(seg.video_added_seconds, 0)
            self.assertLessEqual(seg.placed_duration, 2 + 1 / 8000)

    def test_conflicting_zero_windows_repeat_picture_and_never_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            segs = [Segment(i, 0, 0, str(i), "বাক্য", tts_audio_path=make_clip(root / f"{i}.wav", .6))
                    for i in range(3)]
            engine = self.engine()
            engine.build_final_track(segs, .5, root, root / "out.wav", video_fps=25)
            self.assertEqual([s.start for s in segs], [0, 0, 0])
            self.assertTrue(all(s.tempo_used == 1.0 for s in segs))
            self.assertTrue(all(s.placed_duration and s.placed_duration >= .599 for s in segs))
            self.assertTrue(all(a.actual_start + a.placed_duration <= b.actual_start + 1e-6
                                for a, b in zip(segs, segs[1:])))
            self.assertGreater(engine.video_plan.duration, 1.8)
            self.assertEqual(engine.last_stats["unplaced"], 0)

    def test_bad_tts_is_explicit_and_cannot_produce_a_success_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "silent.wav"
            audio_fx.write_wav(path, np.zeros(8000, np.float32), 8000)
            seg = Segment(0, 0, 1, "line", "বাংলা", tts_audio_path=path)
            engine = self.engine()
            with self.assertRaisesRegex(RuntimeError, "Cannot prepare all"):
                engine.build_final_track([seg], 1, root, root / "out.wav")
            engine.build_final_track([seg], 1, root, root / "intermediate.wav", allow_empty_intermediate=True)
            self.assertEqual(engine.last_stats["unplaced"], 1)
            self.assertIsNone(engine.video_plan)
            self.assertIsNone(seg.placed_duration)

    def test_negative_source_start_is_placed_at_zero_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seg = Segment(0, -.2, .5, "word", "শব্দ", tts_audio_path=make_clip(root / "a.wav", .6))
            engine = self.engine()
            engine.build_final_track([seg], 1, root, root / "out.wav")
            self.assertEqual(seg.start, -.2)
            self.assertEqual(seg.placed_start, 0)
            self.assertAlmostEqual(seg.placed_duration, .6)

    def test_unreliable_slowdown_retains_original_and_reports_safeguard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seg = Segment(0, 0, 2, "line", "বাংলা", tts_audio_path=make_clip(root / "a.wav", 1))
            engine = self.engine()
            with mock.patch("proxmax.sync_prepare.run_ffmpeg", side_effect=RuntimeError("encoder failed")):
                engine.build_final_track([seg], 2, root, root / "out.wav")
            self.assertEqual(seg.tempo_used, 1.0)
            self.assertAlmostEqual(seg.placed_duration, 1.0)
            self.assertIn("slowdown was unreliable", seg.sync_recovery)
            self.assertEqual(engine.last_stats["unplaced"], 0)

    def test_one_sample_overlap_is_an_error(self):
        engine = self.engine()
        engine.last_stats = {"unplaced": 0, "sync_issues": [], "trimmed": 0}
        one = Segment(0, 0, 1, stretched_audio_path=Path("1.wav"), placed_start=0)
        two = Segment(1, 1, 2, stretched_audio_path=Path("2.wav"), placed_start=(7999 / 8000))
        with mock.patch("proxmax.sync_assembly.audio_fx.read_wav", return_value=(tone(1), 8000)), \
             self.assertRaisesRegex(RuntimeError, "overlapping"):
            engine._assemble([(one, (0, 1)), (two, (7999 / 8000, 2))], 2)


if __name__ == "__main__":
    unittest.main()
