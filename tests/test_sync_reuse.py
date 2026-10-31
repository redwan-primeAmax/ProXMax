"""A sync retry reuses only unchanged, verified prepared clips."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from proxmax import audio_fx
from proxmax.models import Segment
from proxmax.sync_engine import SyncEngine
from tests._support import FFMPEG


def tone(seconds: float, rate: int = 24000, frequency: float = 320.0):
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (.16 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class TestSyncPreparedClipReuse(unittest.TestCase):
    def _segments(self, root: Path):
        segments = []
        for i in range(5):
            path = root / f"source-{i}.wav"
            audio_fx.write_wav(path, tone(.65 + i * .08), 24000)
            segments.append(Segment(index=i, start=i * 1.4, end=i * 1.4 + .95,
                                    text_en=f"Line {i}", text_bn="বাংলা", tts_audio_path=path))
        return segments

    def test_identical_rebuild_reuses_all_and_matches_fresh_waveform_and_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segments = self._segments(root)
            engine = SyncEngine(max_workers=2, prefer_rubberband=False)
            with mock.patch.object(engine, "_stretch_one", wraps=engine._stretch_one) as stretch:
                engine.build_final_track(segments, 7.2, root / "cached", root / "first.wav")
                first_stats = dict(engine.last_stats)
                self.assertEqual(stretch.call_count, 5)
                engine.build_final_track(segments, 7.2, root / "cached", root / "second.wav")
                second_stats = dict(engine.last_stats)
                self.assertEqual(stretch.call_count, 5)
                self.assertEqual(second_stats["prepared_reused"], 5)
                self.assertEqual(second_stats["prepared_computed"], 0)
            first, _ = audio_fx.read_wav(root / "first.wav")
            second, _ = audio_fx.read_wav(root / "second.wav")
            np.testing.assert_array_equal(first, second)
            for key in ("prepared_reused", "prepared_computed"):
                first_stats.pop(key)
                second_stats.pop(key)
            self.assertEqual(first_stats, second_stats)

            fresh = SyncEngine(max_workers=2, prefer_rubberband=False)
            fresh.build_final_track(segments, 7.2, root / "fresh", root / "fresh.wav")
            rebuilt, _ = audio_fx.read_wav(root / "fresh.wav")
            np.testing.assert_array_equal(second, rebuilt)
            fresh_stats = dict(fresh.last_stats)
            fresh_stats.pop("prepared_reused")
            fresh_stats.pop("prepared_computed")
            self.assertEqual(second_stats, fresh_stats)

    def test_source_window_settings_and_artifact_changes_invalidate_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segments = self._segments(root)
            engine = SyncEngine(max_workers=2, prefer_rubberband=False)
            with mock.patch.object(engine, "_stretch_one", wraps=engine._stretch_one) as stretch:
                engine.build_final_track(segments, 7.2, root / "work", root / "track.wav")
                self.assertEqual(stretch.call_count, 5)
                # Same path with new content and metadata must not reuse that clip.
                audio_fx.write_wav(segments[2].tts_audio_path, tone(1.02), 24000)
                engine.build_final_track(segments, 7.2, root / "work", root / "track.wav")
                self.assertEqual(stretch.call_count, 6)
                self.assertEqual(engine.last_stats["prepared_reused"], 4)
                # Prepared file removal also forces recomputation.
                (root / "work" / "stretched" / "seg_0003.wav").unlink()
                engine.build_final_track(segments, 7.2, root / "work", root / "track.wav")
                self.assertEqual(stretch.call_count, 7)
                # An exact window change forces recalculation for the affected segment.
                segments[4].end += .1
                engine.build_final_track(segments, 7.2, root / "work", root / "track.wav")
                self.assertEqual(stretch.call_count, 8)
                # A different active engine setting invalidates every prior preparation.
                engine.min_tempo = 0.95
                engine.build_final_track(segments, 7.2, root / "work", root / "track.wav")
                self.assertEqual(stretch.call_count, 13)

    def test_post_slowdown_added_silence_is_removed_without_cutting_speech(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            audio_fx.write_wav(source, tone(.90), 24000)
            seg = Segment(0, 0.0, 1.0, "line", "বাংলা", tts_audio_path=source)
            engine = SyncEngine(max_workers=1, prefer_rubberband=False, borrow_silence_ms=0)

            def fake_ffmpeg(command, **_kwargs):
                audio_fx.write_wav(command[-1], np.r_[tone(.96), np.zeros(3600, np.float32)], 24000)

            with mock.patch("proxmax.sync_prepare.run_ffmpeg", side_effect=fake_ffmpeg):
                engine.build_final_track([seg], 1.5, root, root / "track.wav")
            clip, _ = audio_fx.read_wav(seg.stretched_audio_path)
            self.assertEqual(engine.last_stats["unplaced"], 0)
            self.assertLessEqual(seg.stretched_duration, 1.0 + 1 / 24000)
            self.assertAlmostEqual(seg.sync_trimmed_tail_seconds, .125, places=3)
            np.testing.assert_allclose(clip[:int(.96 * 24000)], tone(.96), atol=1e-4)

    def test_voiced_overrun_is_never_cut_to_make_it_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            audio_fx.write_wav(source, tone(1.10), 24000)
            seg = Segment(0, 0.0, .75, "line", "বাংলা", tts_audio_path=source)
            engine = SyncEngine(max_workers=1, prefer_rubberband=False, borrow_silence_ms=0)
            with mock.patch("proxmax.sync_prepare.run_ffmpeg") as render:
                engine.build_final_track([seg], 1.5, root, root / "track.wav")
            render.assert_not_called()
            self.assertEqual(engine.last_stats["unplaced"], 0)
            self.assertAlmostEqual(seg.placed_duration, 1.10, places=3)
            self.assertGreater(seg.video_added_seconds, 0)
            self.assertEqual(seg.tempo_used, 1)


if __name__ == "__main__":
    unittest.main()
