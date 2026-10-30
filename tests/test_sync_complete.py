"""Measured uncapped recovery, impossible windows, and no-overlap guarantees."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from proxmax import audio_fx
from proxmax.models import Segment
from proxmax.sync_engine import SyncEngine
from proxmax.sync_fallback import build_complete_track
from proxmax.sync_math import build_stretch_filter
from tests._support import FFMPEG


def tone(path, seconds, rate=8000, frequency=440):
    samples = np.arange(round(seconds * rate), dtype=np.float32)
    audio = (.24 * np.sin(2 * np.pi * frequency * samples / rate)).astype(np.float32)
    audio_fx.write_wav(path, audio, rate)
    return audio


def engine():
    return SyncEngine(sample_rate=8000, max_workers=1, slot_gap_ms=80,
                      borrow_silence_ms=250, prefer_rubberband=False)


class TestCompleteFallback(unittest.TestCase):
    def test_zero_windows_shift_all_speech_and_extend_timeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "a.wav", root / "b.wav"
            tone(first, .5)
            tone(second, .4, frequency=600)
            segments = [
                Segment(0, 0, 0, "first", "প্রথম", tts_audio_path=first,
                        sync_issue="sync window has no positive duration; clip left unplaced"),
                Segment(1, 0, .2, "second", "দ্বিতীয়", tts_audio_path=second,
                        sync_issue="sync window has no positive duration; clip left unplaced"),
            ]
            sync = engine()
            sync.last_stats = {"unplaced": 2, "sync_issues": [{"segment_index": 0}]}
            output = build_complete_track(sync, segments, .6, root, root / "final.wav")
            self.assertEqual([s.start for s in segments], [0, 0])  # source time preserved
            self.assertEqual([round(s.placed_duration, 3) for s in segments], [.5, .4])
            self.assertAlmostEqual(segments[1].placed_start, .58, places=4)
            self.assertAlmostEqual(sync.last_stats["video_extension_seconds"], .38, places=3)
            self.assertEqual(sync.last_stats["unplaced"], 0)
            self.assertEqual(sync.last_stats["overlaps"], 0)
            audio, rate = audio_fx.read_wav(output)
            self.assertGreater(np.max(np.abs(audio[round(.94 * rate):round(.97 * rate)])), .1)
            self.assertEqual(len(sync.last_stats["fallback_segments"]), 2)

    def test_real_gap_avoids_speedup_and_reuses_already_placed_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "a.wav", root / "b.wav"
            tone(first, 1.5)
            tone(second, .3)
            segments = [
                Segment(0, 0, .5, "first", "প্রথম", tts_audio_path=first, sync_issue="unfit"),
                Segment(1, 3, 3.3, "next", "পরে", tts_audio_path=second,
                        stretched_audio_path=second, placed_duration=.3, placed_start=3),
            ]
            sync = engine()
            sync.last_stats = {"unplaced": 1, "sync_issues": []}
            with mock.patch("proxmax.sync_fallback.run_ffmpeg") as runner:
                build_complete_track(sync, segments, 4.0, root, root / "final.wav")
            runner.assert_not_called()
            self.assertEqual(segments[0].tempo_used, 1.0)
            self.assertIn("used available gap", segments[0].sync_recovery)
            self.assertEqual(segments[1].actual_start, 3)
            self.assertEqual(len(sync.last_stats["fallback_segments"]), 1)
            self.assertEqual(sync.last_stats["video_extension_seconds"], 0)

    def test_bad_or_silent_tts_fails_instead_of_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            silent = root / "silent.wav"
            audio_fx.write_wav(silent, np.zeros(1000, np.float32), 8000)
            for path in (None, root / "missing.wav", silent):
                seg = Segment(4, 0, 1, "word", "শব্দ", tts_audio_path=path, sync_issue="unfit")
                with self.subTest(path=path), self.assertRaisesRegex(RuntimeError, "cannot recover"):
                    build_complete_track(engine(), [seg], 1, root, root / "final.wav")
                self.assertFalse((root / "final.wav").exists())

    def test_failed_renderer_retains_original_instead_of_cutting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "a.wav", root / "b.wav"
            tone(first, 2)
            tone(second, .3)
            segs = [Segment(0, 0, .5, "a", "এক", tts_audio_path=first, sync_issue="unfit"),
                    Segment(1, .8, 1, "b", "দুই", tts_audio_path=second, sync_issue="unfit")]
            sync = engine()
            with mock.patch("proxmax.sync_fallback.run_ffmpeg", side_effect=RuntimeError("failed")):
                build_complete_track(sync, segs, 1.2, root, root / "final.wav")
            self.assertAlmostEqual(segs[0].placed_duration, 2.0)
            self.assertAlmostEqual(segs[1].placed_start, 2.08)
            self.assertGreater(sync.last_stats["video_extension_seconds"], 1)
            self.assertIn("original audio retained", segs[0].sync_recovery)

    def test_all_failed_intermediate_pass_can_reach_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "a.wav"
            tone(source, .4)
            segs = [Segment(0, 0, 0, "a", "এক", tts_audio_path=source),
                    Segment(1, 0, 0, "b", "দুই", tts_audio_path=source)]
            sync = engine()
            sync.build_final_track(segs, .05, root, root / "intermediate.wav", allow_empty_intermediate=True)
            self.assertTrue(all(s.placed_duration is None for s in segs))
            self.assertEqual(sync.last_stats["unplaced"], 2)
            build_complete_track(sync, segs, .05, root, root / "final.wav")
            self.assertTrue(all(s.placed_duration for s in segs))
            self.assertGreater(sync.last_stats["video_extension_seconds"], 0)

    def test_negative_source_start_is_placed_at_zero_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "source.wav"
            tone(path, .3)
            sync = SyncEngine(sample_rate=8000, max_workers=1, slow_down_short_clips=False,
                              fill_extension=0.0, prefer_rubberband=False)
            seg = Segment(0, -.2, .5, "word", "শব্দ", tts_audio_path=path)
            sync.build_final_track([seg], 1.0, root, root / "track.wav")
            self.assertEqual(seg.start, -.2)
            self.assertEqual(seg.placed_start, 0.0)
            self.assertAlmostEqual(seg.placed_duration, .3)
            self.assertEqual(sync.last_stats["unplaced"], 0)

    def test_silent_prepared_artifact_is_not_counted_as_placed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "silent.wav"
            audio_fx.write_wav(path, np.zeros(800, np.float32), 8000)
            sync = engine()
            sync.last_stats = {"unplaced": 0, "sync_issues": []}
            seg = Segment(0, 0, 1, stretched_audio_path=path)
            sync._assemble([(seg, (0, 1))], 1)
            self.assertIsNone(seg.placed_duration)
            self.assertEqual(sync.last_stats["unplaced"], 1)

    def test_one_sample_overlap_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "short.wav"
            tone(path, .01)
            sync = engine()
            sync.last_stats = {"unplaced": 0, "sync_issues": []}
            a = Segment(0, 0, .01, stretched_audio_path=path, placed_start=0)
            b = Segment(1, 79 / 8000, .02, stretched_audio_path=path, placed_start=79 / 8000)
            with self.assertRaisesRegex(RuntimeError, "overlapping"):
                sync._assemble([(a, (0, .01)), (b, (79 / 8000, 159 / 8000))], .2)

    def test_random_conflicting_slots_never_overlap_or_drop_an_original_clip(self):
        rng = np.random.default_rng(42)
        for seed in range(12):
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                segs = []
                for index in range(12):
                    path = root / f"{index}.wav"
                    duration = float(rng.integers(800, 2800)) / 8000
                    tone(path, duration, frequency=250 + index * 15)
                    start = float(rng.integers(0, 20)) / 10  # frequent identical starts
                    segs.append(Segment(index, start, start + .1, "word", "শব্দ",
                                        tts_audio_path=path, sync_issue="unfit"))
                sync = engine()
                with mock.patch("proxmax.sync_fallback.run_ffmpeg", side_effect=RuntimeError("unreliable")):
                    build_complete_track(sync, segs, 2.0, root, root / "final.wav")
                ordered = sorted(segs, key=lambda seg: (seg.start, seg.index))
                self.assertEqual(sync.last_stats["unplaced"], 0)
                for seg in ordered:
                    original, rate = audio_fx.read_wav(seg.tts_audio_path)
                    self.assertAlmostEqual(seg.placed_duration, len(original) / rate)
                for left, right in zip(ordered, ordered[1:]):
                    self.assertGreaterEqual(round(right.actual_start * 8000),
                                            round((left.actual_start + left.placed_duration) * 8000))

    def test_very_large_tempo_uses_chained_filter_instead_of_rubberband_limit(self):
        result = build_stretch_filter(250, 24000, prefer_rubberband=True)
        self.assertIn("atempo=", result)
        self.assertNotIn("rubberband", result)


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class TestActualFFmpegFallback(unittest.TestCase):
    def test_required_tempo_can_exceed_120_percent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "a.wav", root / "b.wav"
            tone(first, 2)
            tone(second, .3)
            segs = [Segment(0, 0, .5, "a", "এক", tts_audio_path=first, sync_issue="unfit"),
                    Segment(1, .8, 1.1, "b", "দুই", tts_audio_path=second,
                            stretched_audio_path=second, placed_start=.8, placed_duration=.3)]
            sync = engine()
            sync.last_stats = {"unplaced": 1, "sync_issues": []}
            build_complete_track(sync, segs, 2.0, root, root / "final.wav")
            self.assertGreater(segs[0].tempo_used, 2.0)
            self.assertLessEqual(segs[0].actual_start + segs[0].placed_duration, segs[1].actual_start - .08 + 1 / 8000)
            self.assertEqual(sync.last_stats["unplaced"], 0)
            self.assertIn("measured unlimited-tempo fit", segs[0].sync_recovery)

    def test_duration_retry_starts_from_original_every_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "a.wav", root / "b.wav"
            tone(first, 2)
            tone(second, .3)
            segs = [Segment(0, 0, .5, "a", "এক", tts_audio_path=first, sync_issue="unfit"),
                    Segment(1, 1.08, 1.3, "b", "দুই", tts_audio_path=second,
                            stretched_audio_path=second, placed_duration=.3)]
            durations = iter([1.05, .98])
            def fake_render(cmd, **_kwargs):
                self.assertEqual(Path(cmd[cmd.index("-i") + 1]), first)
                tone(Path(cmd[-1]), next(durations))
            with mock.patch("proxmax.sync_fallback.run_ffmpeg", side_effect=fake_render) as runner:
                build_complete_track(engine(), segs, 2.0, root, root / "final.wav")
            self.assertEqual(runner.call_count, 2)
            self.assertGreater(segs[0].tempo_used, 2.0)
            self.assertAlmostEqual(segs[0].placed_duration, .98, places=3)


if __name__ == "__main__":
    unittest.main()
