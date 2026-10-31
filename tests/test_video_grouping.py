"""Bounded FFmpeg grouping must match the original piece-by-piece timeline."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np

from proxmax.muxer import VideoMuxer
from proxmax.video_render import _frame_count, _group_pieces, _render_group, render_video_plan
from proxmax.video_timeline import VideoPiece, VideoPlan
from tests._support import FFMPEG


def plan_for(fps: Fraction) -> VideoPlan:
    pieces = (VideoPiece(0, 16, 16, "normal"), VideoPiece(10, 16, 6, "repeat"),
              VideoPiece(16, 40, 21, "speed"), VideoPiece(40, 64, 24, "normal"))
    return VideoPlan(fps, 80, pieces, sum(p.output_frames for p in pieces))


class TestGroupPlanning(unittest.TestCase):
    def test_repeat_reuse_and_additional_frame_budget_are_bounded(self):
        repeated = VideoPiece(0, 20, 20, "repeat")
        pieces = (VideoPiece(20, 40, 20), repeated, repeated, VideoPiece(40, 140, 100),
                  VideoPiece(140, 180, 40), VideoPiece(180, 220, 40), VideoPiece(220, 230, 10))
        groups = _group_pieces(pieces)
        self.assertEqual([piece for group in groups for piece in group], list(pieces))
        self.assertEqual(sum(group == (repeated,) for group in groups), 2)
        for group in groups:
            self.assertLessEqual(len(group), 4)
            self.assertLessEqual(sum(p.output_frames for p in group[1:]), 60)

    def test_unavailable_gpu_does_not_submit_every_job_before_cpu_fallback(self):
        pieces = tuple(VideoPiece(i, i + 1, 1) for i in range(200))
        plan = VideoPlan(Fraction(25), 200, pieces, 200)
        lock, calls = threading.Lock(), []
        def render(_source, _group, _fps, output, encoder):
            with lock:
                calls.append(encoder)
            if encoder == "h264_nvenc":
                raise RuntimeError("synthetic missing GPU")
            output.touch()
        def join(command, **_kwargs):
            Path(command[-1]).touch()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"
            source.touch()
            stats = {}
            with mock.patch("proxmax.video_render._render_group", side_effect=render), \
                 mock.patch("proxmax.video_render._frame_count", return_value=200), \
                 mock.patch("proxmax.video_render.run_ffmpeg_with_progress", side_effect=join):
                render_video_plan(source, plan, root / "out.mp4", use_nvenc=True, workers=2,
                                  diagnostics=stats)
        self.assertLessEqual(calls.count("h264_nvenc"), 2)
        self.assertGreater(calls.count("libx264"), 2)
        self.assertEqual(stats["encoder"], "libx264")
        self.assertEqual(stats["fallbacks"], 1)
        self.assertEqual(stats["attempts"][0]["error_type"], "RuntimeError")
        self.assertNotIn("synthetic missing GPU", json.dumps(stats))

    def test_output_never_overwrites_source(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.mp4"
            source.write_bytes(b"original")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                render_video_plan(source, plan_for(Fraction(25)), source)
            self.assertEqual(source.read_bytes(), b"original")


@unittest.skipUnless(FFMPEG and shutil.which("ffprobe"), "FFmpeg and ffprobe required")
class TestActualGroupedVideo(unittest.TestCase):
    width, height = 96, 64

    def make_source(self, path: Path, *, variable: bool, fps: Fraction) -> None:
        command = [FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i",
                   f"testsrc2=size={self.width}x{self.height}:rate={fps}:duration=6"]
        if variable:
            command += ["-vf", r"select=not(eq(mod(n\,5)\,0))", "-vsync", "vfr"]
        command += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "15",
                    "-threads", "1", "-pix_fmt", "yuv420p", str(path)]
        subprocess.run(command, check=True, capture_output=True)

    def frames(self, path: Path) -> np.ndarray:
        data = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-vsync", "0",
                               "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                              check=True, capture_output=True).stdout
        return np.frombuffer(data, dtype=np.uint8).reshape(-1, self.height, self.width)

    def compare(self, root: Path, fps: Fraction, variable: bool) -> None:
        source = root / "source's file.mp4"
        self.make_source(source, variable=variable, fps=fps)
        actual_fps = VideoMuxer.video_frame_rate(source) if variable else fps
        plan = plan_for(actual_fps)
        baseline, grouped = root / "individual.mp4", root / "grouped.mp4"
        with mock.patch("proxmax.video_render._GROUP_SIZE", 1):
            render_video_plan(source, plan, baseline, workers=2)
        stats = {}
        render_video_plan(source, plan, grouped, workers=2, diagnostics=stats)
        first, second = self.frames(baseline), self.frames(grouped)
        self.assertEqual(len(first), plan.output_frames)
        self.assertEqual(len(second), plan.output_frames)
        difference = np.abs(first.astype(np.int16) - second.astype(np.int16)).mean(axis=(1, 2))
        self.assertLess(float(difference.max()), 4.0)
        self.assertLess(stats["unique_groups"], len(plan.pieces))
        data = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                               "frame=best_effort_timestamp_time", "-of", "json", str(grouped)],
                              check=True, capture_output=True, text=True)
        stamps = [float(frame["best_effort_timestamp_time"]) for frame in json.loads(data.stdout)["frames"]]
        self.assertEqual(len(stamps), plan.output_frames)
        for index, timestamp in enumerate(stamps):
            self.assertAlmostEqual(timestamp, index / float(actual_fps), delta=.003)

    def test_fractional_fps_and_quoted_path_match_single_piece_rendering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "space and 'quote"
            root.mkdir()
            self.compare(root, Fraction(30000, 1001), variable=False)

    def test_variable_frame_rate_matches_single_piece_seek_semantics(self):
        with tempfile.TemporaryDirectory() as temp:
            self.compare(Path(temp), Fraction(25), variable=True)

    def test_failed_group_filter_falls_back_to_verified_individual_pieces(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"
            self.make_source(source, variable=False, fps=Fraction(25))
            plan = plan_for(Fraction(25))
            def reject_group(source, group, fps, output, encoder):
                if len(group) > 1:
                    raise RuntimeError("synthetic unsupported filter")
                return _render_group(source, group, fps, output, encoder)
            stats = {}
            with mock.patch("proxmax.video_render._render_group", side_effect=reject_group):
                output = render_video_plan(source, plan, root / "out.mp4", workers=2,
                                           diagnostics=stats)
            self.assertEqual(_frame_count(output), plan.output_frames)
            self.assertEqual(stats["strategy"], "individual")
            self.assertEqual(stats["fallbacks"], 1)
            self.assertEqual(stats["attempts"][0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
