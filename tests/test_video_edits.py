"""Exercise actual FFmpeg picture-tail repeats, speedups, muxing and workers."""

import json
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np

from proxmax import audio_fx
from proxmax.models import Segment
from proxmax.muxer import VideoMuxer
from proxmax.sync_engine import SyncEngine
from proxmax.video_render import _frame_count, _render_piece, render_video_plan
from tests._support import FFMPEG


@unittest.skipUnless(FFMPEG and shutil.which("ffprobe"), "ffmpeg and ffprobe required")
class TestActualVideoEdits(unittest.TestCase):
    width, height, fps = 96, 64, 25

    def make_video(self, path, count=100, audio_seconds=None):
        frames = np.empty((count, self.height, self.width, 3), dtype=np.uint8)
        for i in range(count):
            frames[i].fill(16 + (i % 10) * 18 + (i // 10) * 2)
        cmd = [FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{self.width}x{self.height}", "-r", str(self.fps), "-i", "pipe:0"]
        if audio_seconds is not None:
            cmd += ["-f", "lavfi", "-i", f"sine=frequency=220:duration={audio_seconds}"]
        cmd += ["-frames:v", str(count), "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
                "-pix_fmt", "yuv420p"]
        if audio_seconds is not None:
            cmd += ["-c:a", "aac"]
        cmd += [str(path)]
        result = subprocess.run(cmd, input=frames.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))

    def decode_frames(self, path):
        proc = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-f", "rawvideo",
                               "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode(errors="replace"))
        return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, self.height, self.width).mean(axis=(1, 2))

    def write_tone(self, path, seconds, frequency):
        sr = 8000
        t = np.arange(round(seconds * sr)) / sr
        audio_fx.write_wav(path, (.2 * np.sin(2 * np.pi * frequency * t)).astype(np.float32), sr)

    def test_motion_is_repeated_at_window_end_then_later_video_and_audio_are_shifted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / "source.mp4", root / "dub.mp4"
            self.make_video(source)
            self.write_tone(root / "a.wav", 1.6, 330)
            self.write_tone(root / "b.wav", .4, 600)
            segs = [Segment(0, .4, 1.4, "one", "এক", tts_audio_path=root / "a.wav"),
                    Segment(1, 2, 3, "two", "দুই", tts_audio_path=root / "b.wav")]
            engine = SyncEngine(sample_rate=8000, max_workers=2, slot_gap_ms=0,
                                borrow_silence_ms=0, prefer_rubberband=False)
            audio = engine.build_final_track(segs, 4, root, root / "dub.wav", video_fps=self.fps)
            muxer = VideoMuxer(use_hwaccel=False, video_workers=2)
            muxer.mux(source, audio, output, required_video_duration=engine.video_plan.duration,
                      video_plan=engine.video_plan)
            original, dubbed = self.decode_frames(source), self.decode_frames(output)
            self.assertEqual(len(dubbed), engine.video_plan.output_frames)
            cursor = 0
            for piece in engine.video_plan.pieces:
                actual = dubbed[cursor:cursor + piece.output_frames]
                if piece.kind in {"normal", "repeat"}:
                    np.testing.assert_allclose(actual, original[piece.start:piece.end], atol=3)
                else:
                    self.assertGreater(np.ptp(actual), 10)  # moving video, not a frozen frame
                    self.assertLessEqual(piece.source_frames / piece.output_frames, 1.15)
                cursor += piece.output_frames
            repeat = next(p for p in engine.video_plan.pieces if p.kind == "repeat")
            self.assertEqual((repeat.start, repeat.end), (20, 35))  # last 0.6 s of first window
            self.assertAlmostEqual(segs[1].actual_start, 2.6, places=2)
            self.assertTrue(all(s.tempo_used <= 1 for s in segs))
            self.assertGreater(segs[1].video_speed, 1)
            self.assertEqual(engine.last_stats["overlaps"], 0)
            dub = audio_fx.decode_audio(output, 8000)
            for seg in segs:
                end = seg.actual_start + seg.placed_duration
                tail = dub[int((end - .08) * 8000):int((end - .02) * 8000)]
                self.assertGreater(float(np.max(np.abs(tail))), .03)

    def test_short_window_shifts_later_video_and_report_cues_earlier(self):
        from proxmax.config import AppConfig
        from proxmax.prosody import subtitles_for_unit
        from proxmax.reporting_timeline import _render_unit, _unit_timing_rows

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"
            self.make_video(source)
            self.write_tone(root / "a.wav", .4, 330)
            self.write_tone(root / "b.wav", .3, 600)
            segs = [Segment(0, .2, 1.2, "one", "এক", tts_audio_path=root / "a.wav"),
                    Segment(1, 1.6, 2.2, "two", "দুই", tts_audio_path=root / "b.wav")]
            sync = SyncEngine(sample_rate=8000, slot_gap_ms=0, borrow_silence_ms=0,
                              prefer_rubberband=False)
            audio = sync.build_final_track(segs, 4, root, root / "dub.wav")
            output = root / "out.mp4"
            VideoMuxer(use_hwaccel=False).mux(source, audio, output, video_plan=sync.video_plan,
                                               required_video_duration=sync.video_plan.duration)
            self.assertGreater(segs[0].video_speed, 1)
            self.assertLessEqual(segs[0].video_speed, 1.15)
            self.assertLess(segs[1].actual_start, segs[1].start)
            self.assertAlmostEqual(segs[1].actual_start, segs[1].video_window_start, places=3)
            row = _unit_timing_rows(segs, 4, AppConfig(slot_gap_ms=0, borrow_silence_ms=0))[1]
            self.assertAlmostEqual(row["actual_start"], segs[1].actual_start)
            self.assertAlmostEqual(row["target_start"], segs[1].video_source_start)
            report = "\n".join(_render_unit(row, {}, {}))
            self.assertIn("Edited video window", report)
            cue = subtitles_for_unit(segs[1])[0]
            self.assertAlmostEqual(cue.start, segs[1].actual_start, places=3)
            self.assertAlmostEqual(cue.end, segs[1].actual_start + segs[1].placed_duration, places=3)
            dub = audio_fx.decode_audio(output, 8000)
            end = segs[1].actual_start + segs[1].placed_duration
            self.assertGreater(np.max(np.abs(dub[int((end-.08)*8000):int((end-.02)*8000)])), .03)

    def test_fractional_fps_paths_timestamps_and_audio_stay_aligned(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "space and 'apostrophe"
            root.mkdir()
            self.fps = Fraction(30000, 1001)
            source, output = root / "source's picture.mp4", root / "dub's output.mp4"
            self.make_video(source, count=90)
            self.write_tone(root / "a.wav", 1.05, 330)
            self.write_tone(root / "b.wav", .4, 600)
            segs = [Segment(0, .2, .7, "one", "এক", tts_audio_path=root / "a.wav"),
                    Segment(1, 1.1, 1.9, "two", "দুই", tts_audio_path=root / "b.wav")]
            sync = SyncEngine(sample_rate=8000, slot_gap_ms=0, borrow_silence_ms=0,
                              max_workers=2, prefer_rubberband=False)
            audio = sync.build_final_track(segs, 90 / float(self.fps), root, root / "dub track.wav",
                                           video_fps=self.fps)
            muxer = VideoMuxer(use_hwaccel=False, video_workers=2)
            muxer.mux(source, audio, output, required_video_duration=sync.video_plan.duration,
                      video_plan=sync.video_plan)
            original, dubbed = self.decode_frames(source), self.decode_frames(output)
            self.assertEqual(len(dubbed), sync.video_plan.output_frames)
            cursor = 0
            for piece in sync.video_plan.pieces:
                actual = dubbed[cursor:cursor + piece.output_frames]
                if piece.kind in {"normal", "repeat"}:
                    np.testing.assert_allclose(actual, original[piece.start:piece.end], atol=3)
                cursor += piece.output_frames
            probe = subprocess.run([
                shutil.which("ffprobe"), "-v", "error", "-select_streams", "v:0",
                "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(output),
            ], check=True, capture_output=True, text=True)
            timestamps = [float(f["best_effort_timestamp_time"]) for f in json.loads(probe.stdout)["frames"]]
            self.assertEqual(len(timestamps), sync.video_plan.output_frames)
            for i, pts in enumerate(timestamps):
                self.assertAlmostEqual(pts, i / float(self.fps), delta=.002)
            decoded = audio_fx.decode_audio(output, 8000)
            for seg in segs:
                end = seg.actual_start + seg.placed_duration
                tail = decoded[int((end - .08) * 8000):int((end - .02) * 8000)]
                self.assertGreater(float(np.max(np.abs(tail))), .03)
            self.assertGreaterEqual(segs[1].actual_start, segs[0].actual_start + segs[0].placed_duration)
            self.assertGreater(segs[1].actual_start, segs[1].start)

    def test_zero_length_window_uses_nearest_moving_picture_and_keeps_both_clips(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"
            self.make_video(source, count=50)
            self.write_tone(root / "a.wav", .36, 330)
            self.write_tone(root / "b.wav", .28, 600)
            segs = [Segment(0, .4, .4, "one", "এক", tts_audio_path=root / "a.wav"),
                    Segment(1, .4, 1, "two", "দুই", tts_audio_path=root / "b.wav")]
            sync = SyncEngine(sample_rate=8000, slot_gap_ms=0, borrow_silence_ms=0,
                              prefer_rubberband=False)
            audio = sync.build_final_track(segs, 2, root, root / "dub.wav", video_fps=25)
            output = root / "out.mp4"
            VideoMuxer(use_hwaccel=False).mux(source, audio, output,
                required_video_duration=sync.video_plan.duration, video_plan=sync.video_plan)
            self.assertIn("tiny source window", segs[0].sync_recovery)
            self.assertGreaterEqual(segs[1].actual_start, segs[0].actual_start + segs[0].placed_duration)
            self.assertEqual(_frame_count(output), sync.video_plan.output_frames)
            repeat = next(p for p in sync.video_plan.pieces if p.kind == "repeat")
            self.assertGreater(repeat.source_frames, 1)
            decoded = self.decode_frames(output)
            self.assertGreater(np.ptp(decoded[10:10 + repeat.output_frames]), 10)
            dub = audio_fx.decode_audio(output, 8000)
            for seg in segs:
                end = seg.actual_start + seg.placed_duration
                self.assertGreater(np.max(np.abs(dub[int((end-.08)*8000):int((end-.02)*8000)])), .03)

    def test_many_repeats_reuse_encoded_pieces_and_run_concurrently(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"
            self.make_video(source, 50)
            self.write_tone(root / "a.wav", 2.3, 330)
            seg = Segment(0, .4, .8, "one", "এক", tts_audio_path=root / "a.wav")
            engine = SyncEngine(sample_rate=8000, slot_gap_ms=0, borrow_silence_ms=0,
                                max_workers=1, prefer_rubberband=False)
            audio = engine.build_final_track([seg], 2, root, root / "dub.wav")
            lock, active, highest = threading.Lock(), [0], [0]
            def counted(*args, **kwargs):
                with lock:
                    active[0] += 1
                    highest[0] = max(highest[0], active[0])
                try:
                    time.sleep(.03)
                    return _render_piece(*args, **kwargs)
                finally:
                    with lock:
                        active[0] -= 1
            with mock.patch("proxmax.video_render._render_piece", side_effect=counted) as render:
                render_video_plan(source, engine.video_plan, root / "out.mp4", audio=audio, workers=2)
            self.assertGreaterEqual(highest[0], 2)
            self.assertLess(render.call_count, len(engine.video_plan.pieces))
            self.assertEqual(_frame_count(root / "out.mp4"), engine.video_plan.output_frames)

    def test_failed_nvenc_restarts_all_pieces_with_cpu_encoder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"
            self.make_video(source, count=50)
            self.write_tone(root / "a.wav", .8, 330)
            seg = Segment(0, .2, .6, "one", "এক", tts_audio_path=root / "a.wav")
            sync = SyncEngine(sample_rate=8000, slot_gap_ms=0, borrow_silence_ms=0)
            audio = sync.build_final_track([seg], 2, root, root / "dub.wav")
            encoders = []
            def render(source, piece, fps, output, encoder):
                encoders.append(encoder)
                if encoder == "h264_nvenc":
                    raise RuntimeError("GPU encoder unavailable")
                return _render_piece(source, piece, fps, output, encoder)
            with mock.patch("proxmax.video_render._render_piece", side_effect=render):
                output = render_video_plan(source, sync.video_plan, root / "out.mp4", audio=audio,
                                           workers=2, use_nvenc=True)
            self.assertIn("h264_nvenc", encoders)
            self.assertIn("libx264", encoders)
            self.assertEqual(_frame_count(output), sync.video_plan.output_frames)
            self.assertGreater(audio_fx.decode_audio(output, 8000).size, 0)

    def test_shorter_source_picture_is_extended_with_moving_tail_not_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "short.mp4"
            self.make_video(source, count=25, audio_seconds=2)
            muxer = VideoMuxer(use_hwaccel=False)
            normalized = muxer.ensure_picture_duration(source, 2, root, muxer.video_frame_rate(source))
            original, longer = self.decode_frames(source), self.decode_frames(normalized)
            self.assertEqual(len(original), 25)
            self.assertEqual(len(longer), 50)
            np.testing.assert_allclose(longer[:25], original, atol=3)
            np.testing.assert_allclose(longer[25:], original, atol=3)
            self.assertGreater(np.ptp(longer[25:]), 10)


if __name__ == "__main__":
    unittest.main()
