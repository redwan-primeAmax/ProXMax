"""Verified tiny source tails skip redundant encoding without changing edits."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np

from proxmax import audio_fx
from proxmax.models import Segment
from proxmax.muxer import VideoMuxer
from proxmax.sync_engine import SyncEngine
from proxmax.utils import FFmpegError
from proxmax.video_render import _frame_count, render_video_plan
from proxmax.video_timeline import VideoPiece, VideoPlan
from proxmax.virtual_picture import is_constant_frame_grid, mapped_spans, render_virtual_piece
from tests._support import FFMPEG


class TestVirtualSourceMap(unittest.TestCase):
    def test_tail_mapping_reuses_physical_frames_in_order(self):
        mapping = VideoPlan(Fraction(25), 4,
                            (VideoPiece(0, 4, 4), VideoPiece(2, 4, 2, "repeat")), 6)
        self.assertEqual(mapped_spans(VideoPiece(1, 6, 5), mapping), [(1, 4), (2, 4)])
        self.assertEqual(mapped_spans(VideoPiece(4, 6, 2), mapping), [(2, 4)])
        with self.assertRaisesRegex(FFmpegError, "does not cover"):
            mapped_spans(VideoPiece(1, 7, 6), mapping)

    def test_packet_timestamps_reject_vfr_even_if_stream_metadata_claims_cfr(self):
        stream = '{"streams": [{"time_base":"1/25", "avg_frame_rate":"25/1", "r_frame_rate":"25/1"}]}'
        def result(stdout):
            return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")
        with mock.patch("proxmax.virtual_picture.ensure_tool", return_value="ffprobe"), \
             mock.patch("proxmax.virtual_picture.subprocess.run", side_effect=[
                 result(stream), result("0,1\n2,1\n1,1\n4,1\n")]):
            self.assertFalse(is_constant_frame_grid(Path("source.mp4"), Fraction(25), 4))
        # Packet order may follow B-frame decoding; sorted presentation times
        # still describe a valid constant-frame timeline.
        with mock.patch("proxmax.virtual_picture.ensure_tool", return_value="ffprobe"), \
             mock.patch("proxmax.virtual_picture.subprocess.run", side_effect=[
                 result(stream), result("0,1\n2,1\n1,1\n3,1\n")]):
            self.assertTrue(is_constant_frame_grid(Path("source.mp4"), Fraction(25), 4))

    def test_missing_span_verification_never_publishes_a_padded_frame(self):
        mapping = VideoPlan(Fraction(25), 4,
                            (VideoPiece(0, 4, 4), VideoPiece(2, 4, 2, "repeat")), 6)
        class Result:
            stderr = "no verified tail source frames"
        with mock.patch("proxmax.virtual_picture.ensure_tool", return_value="ffmpeg"), \
             self.assertRaisesRegex(FFmpegError, "verify every decoded"):
            render_virtual_piece(Path("source.mp4"), VideoPiece(1, 6, 5), Fraction(25),
                                 Path("out.mp4"), "libx264", mapping,
                                 run_ffmpeg=lambda *_a, **_k: Result(), frame_count=lambda _p: 5)


@unittest.skipUnless(FFMPEG and shutil.which("ffprobe"), "FFmpeg and ffprobe required")
class TestVirtualSourceEndToEnd(unittest.TestCase):
    width, height = 96, 64

    def source(self, path, fps, variable=False):
        if variable:
            command = [FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i",
                       f"testsrc2=s={self.width}x{self.height}:rate={fps}:duration=6",
                       "-vf", r"select=not(eq(mod(n\,5)\,0))", "-vsync", "vfr"]
            pixels = None
        else:
            # Consecutive frames differ strongly so a missing/duplicated frame
            # cannot hide inside the tolerance for lossy re-encoding.
            count = round(6 * float(fps))
            frames = np.empty((count, self.height, self.width, 3), dtype=np.uint8)
            for i in range(count):
                frames[i].fill(16 + (i % 10) * 18 + (i // 10) * 2)
            pixels = frames.tobytes()
            command = [FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", f"{self.width}x{self.height}", "-r", str(fps), "-i", "pipe:0"]
        command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
                    "-pix_fmt", "yuv420p", str(path)]
        subprocess.run(command, input=pixels, check=True, capture_output=True)

    def frames(self, path):
        data = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-vsync", "0",
                               "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                              check=True, capture_output=True).stdout
        return np.frombuffer(data, dtype=np.uint8).reshape(-1, self.height, self.width)

    def tone(self, path, seconds, frequency):
        sr = 8000
        t = np.arange(round(seconds * sr)) / sr
        audio_fx.write_wav(path, (.2 * np.sin(2 * np.pi * frequency * t)).astype(np.float32), sr)

    def build(self, root, duration, fps):
        root.mkdir()
        self.tone(root / "a.wav", .25, 330)
        self.tone(root / "b.wav", 2.0, 620)
        segments = [Segment(0, .5, 1.3, "first", "এক", tts_audio_path=root / "a.wav"),
                    Segment(1, 4.8, duration - .02, "last", "দুই", tts_audio_path=root / "b.wav")]
        engine = SyncEngine(sample_rate=8000, max_workers=1, slot_gap_ms=0,
                            borrow_silence_ms=0, prefer_rubberband=False)
        audio = engine.build_final_track(segments, duration, root, root / "dub.wav", video_fps=fps)
        return engine, audio, segments

    def compare_materialized_and_virtual(self, root, fps):
        source = root / "source with 'quote.mp4"
        self.source(source, fps)
        count = _frame_count(source)
        duration = count / float(fps) + .072  # same mismatch class as the report
        self.assertTrue(is_constant_frame_grid(source, fps, count))
        old_muxer, new_muxer = VideoMuxer(use_hwaccel=False), VideoMuxer(use_hwaccel=False)
        old_dir, new_dir = root / "old", root / "new"
        old_engine, old_audio, old_segments = self.build(old_dir, duration, fps)
        new_engine, new_audio, new_segments = self.build(new_dir, duration, fps)
        self.assertEqual(old_engine.video_plan, new_engine.video_plan)
        with mock.patch("proxmax.muxer.render_video_plan", wraps=render_video_plan) as renderer:
            materialized = old_muxer.ensure_picture_duration(source, duration, old_dir, fps)
            self.assertNotEqual(materialized, source)
            self.assertGreater(renderer.call_count, 0)
        with mock.patch("proxmax.muxer.render_video_plan", wraps=render_video_plan) as renderer:
            virtual_source = new_muxer.prepare_picture_for_timeline(source, duration, new_dir, fps)
            self.assertEqual(virtual_source, source)
            self.assertEqual(renderer.call_count, 0)  # no entire-source re-encode
        self.assertTrue(new_muxer.last_picture_prepare_stats["virtual_tail"])
        self.assertEqual(_frame_count(materialized), new_muxer._virtual_tail.output_frames)
        old_out, new_out = root / "old.mp4", root / "new.mp4"
        old_muxer.mux(materialized, old_audio, old_out, video_plan=old_engine.video_plan,
                      required_video_duration=old_engine.video_plan.duration)
        new_muxer.mux(virtual_source, new_audio, new_out, video_plan=new_engine.video_plan,
                      required_video_duration=new_engine.video_plan.duration)
        old_frames, new_frames = self.frames(old_out), self.frames(new_out)
        self.assertEqual(len(old_frames), old_engine.video_plan.output_frames)
        self.assertEqual(len(new_frames), old_engine.video_plan.output_frames)
        difference = np.abs(old_frames.astype(np.int16) - new_frames.astype(np.int16)).mean(axis=(1, 2))
        self.assertLess(float(difference.max()), 5.0)
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(new_out)],
                               check=True, capture_output=True, text=True)
        stamps = [float(frame["best_effort_timestamp_time"]) for frame in json.loads(probe.stdout)["frames"]]
        self.assertEqual(len(stamps), old_engine.video_plan.output_frames)
        for index, pts in enumerate(stamps):
            self.assertAlmostEqual(pts, index / float(fps), delta=.003)
        self.assertGreater(new_muxer.last_mux_stats["render"]["virtual_source_tail_frames"], 0)
        self.assertNotIn("virtual_fallback", new_muxer.last_mux_stats)
        for old, new in zip(old_segments, new_segments):
            self.assertAlmostEqual(old.actual_start, new.actual_start, places=6)
            self.assertAlmostEqual(old.placed_duration, new.placed_duration, places=6)
            self.assertEqual(old.video_speed, new.video_speed)
        old_samples, sr = audio_fx.read_wav(old_audio)
        new_samples, sr2 = audio_fx.read_wav(new_audio)
        self.assertEqual(sr, sr2)
        np.testing.assert_array_equal(old_samples, new_samples)
        decoded = audio_fx.decode_audio(new_out, 8000)
        end = new_segments[-1].actual_start + new_segments[-1].placed_duration
        self.assertGreater(np.max(np.abs(decoded[int((end - .08)*sr):int((end - .02)*sr)])), .03)

    def test_cfr_and_fractional_fps_match_materialized_picture_and_full_audio(self):
        for fps in (Fraction(25), Fraction(30000, 1001)):
            with self.subTest(fps=fps), tempfile.TemporaryDirectory() as temp:
                self.compare_materialized_and_virtual(Path(temp), fps)

    def test_speeded_piece_crossing_the_virtual_boundary_matches_materialization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"
            self.source(source, Fraction(25))
            fps, physical = Fraction(25), _frame_count(source)
            duration = physical / float(fps) + .072
            old_muxer, new_muxer = VideoMuxer(use_hwaccel=False), VideoMuxer(use_hwaccel=False)
            materialized = old_muxer.ensure_picture_duration(source, duration, root, fps)
            virtual_source = new_muxer.prepare_picture_for_timeline(source, duration, root, fps)
            mapping = new_muxer._virtual_tail
            virtual_count = round(duration * float(fps))
            start = physical - 20
            source_count = virtual_count - start
            output_count = max(1, (source_count * 20 + 22) // 23)
            pieces = (VideoPiece(0, start, start), VideoPiece(start, virtual_count, output_count, "speed"))
            plan = VideoPlan(fps, virtual_count, pieces, sum(p.output_frames for p in pieces))
            old, new = root / "old.mp4", root / "new.mp4"
            render_video_plan(materialized, plan, old)
            render_video_plan(virtual_source, plan, new, source_map=mapping)
            a, b = self.frames(old), self.frames(new)
            self.assertEqual(len(a), plan.output_frames)
            self.assertEqual(len(b), plan.output_frames)
            delta = np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(axis=(1, 2))
            self.assertLess(float(delta.max()), 4.0)

    def test_virtual_tail_is_rendered_even_if_plan_has_no_other_edits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, audio = root / "source.mp4", root / "dub.wav"
            self.source(source, Fraction(25))
            fps, physical = Fraction(25), _frame_count(source)
            duration = physical / float(fps) + .072
            self.tone(audio, duration, 400)
            muxer = VideoMuxer(use_hwaccel=False)
            prepared = muxer.prepare_picture_for_timeline(source, duration, root, fps)
            count = round(duration * float(fps))
            plan = VideoPlan(fps, count, (VideoPiece(0, count, count),), count)
            self.assertFalse(plan.changed)
            output = root / "out.mp4"
            muxer.mux(prepared, audio, output, video_plan=plan, required_video_duration=plan.duration)
            self.assertEqual(_frame_count(output), count)
            self.assertEqual(muxer.last_mux_stats["mode"], "edited_picture")
            self.assertGreater(muxer.last_mux_stats["render"]["virtual_source_tail_frames"], 0)

    def test_variable_frame_rate_falls_back_to_materialized_picture(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "vfr.mp4"
            self.source(source, Fraction(25), variable=True)
            muxer = VideoMuxer(use_hwaccel=False)
            fps, count = muxer.video_frame_rate(source), _frame_count(source)
            self.assertFalse(is_constant_frame_grid(source, fps, count))
            prepared = muxer.prepare_picture_for_timeline(source, count / float(fps) + .08, root, fps)
            self.assertNotEqual(prepared, source)
            self.assertIsNone(muxer._virtual_tail)
            self.assertGreater(_frame_count(prepared), count)

    def test_virtual_final_verification_failure_uses_materialized_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"
            self.source(source, Fraction(25))
            fps, count = Fraction(25), _frame_count(source)
            duration = count / float(fps) + .072
            engine, audio, _segments = self.build(root / "work", duration, fps)
            muxer = VideoMuxer(use_hwaccel=False)
            prepared = muxer.prepare_picture_for_timeline(source, duration, root, fps)
            original_verify = muxer._verify_picture
            checks = []
            def verify(path, required):
                checks.append(path)
                if len(checks) == 1:
                    raise FFmpegError("synthetic final verification failure")
                return original_verify(path, required)
            output = root / "fallback.mp4"
            with mock.patch.object(muxer, "_verify_picture", side_effect=verify):
                muxer.mux(prepared, audio, output, video_plan=engine.video_plan,
                          required_video_duration=engine.video_plan.duration)
            self.assertEqual(len(checks), 2)
            self.assertEqual(_frame_count(output), engine.video_plan.output_frames)
            self.assertEqual(muxer.last_mux_stats["virtual_fallback"]["error_type"], "FFmpegError")

    def test_virtual_filter_failure_uses_materialized_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"
            self.source(source, Fraction(25))
            fps, count = Fraction(25), _frame_count(source)
            duration = count / float(fps) + .072
            engine, audio, _segments = self.build(root / "work", duration, fps)
            muxer = VideoMuxer(use_hwaccel=False)
            virtual_source = muxer.prepare_picture_for_timeline(source, duration, root, fps)
            def render(*args, **kwargs):
                if kwargs.get("source_map") is not None:
                    raise FFmpegError("synthetic virtual filter failure")
                return render_video_plan(*args, **kwargs)
            output = root / "fallback.mp4"
            with mock.patch("proxmax.muxer.render_video_plan", side_effect=render):
                muxer.mux(virtual_source, audio, output, video_plan=engine.video_plan,
                          required_video_duration=engine.video_plan.duration)
            self.assertEqual(_frame_count(output), engine.video_plan.output_frames)
            self.assertEqual(muxer.last_mux_stats["virtual_fallback"]["error_type"], "FFmpegError")
            self.assertEqual(muxer.last_mux_stats["render"]["encoder"], "libx264")


if __name__ == "__main__":
    unittest.main()
