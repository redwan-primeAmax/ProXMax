"""Real FFmpeg picture-tail repetition and lossless-copy fast path."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from proxmax import audio_fx
from proxmax.muxer import VideoMuxer
from proxmax.utils import FFmpegError, run_ffmpeg_with_progress
from tests._support import FFMPEG


class TestVideoStreamTiming(unittest.TestCase):
    def test_end_time_accounts_for_nonzero_stream_start(self):
        output = '{"streams":[{"start_time":"-0.200", "duration":"1.000"}]}'
        completed = mock.Mock(stdout=output, returncode=0)
        with mock.patch("proxmax.muxer.ensure_tool", return_value="ffprobe"), \
             mock.patch("proxmax.muxer.subprocess.run", return_value=completed):
            self.assertAlmostEqual(VideoMuxer._video_stream_duration(Path("video.mp4")), .8)

    def test_literal_na_duration_is_not_mistaken_for_a_valid_estimate(self):
        output = '{"streams":[{"start_time":"0.000", "duration":"N/A"}]}'
        completed = mock.Mock(stdout=output, returncode=0)
        with mock.patch("proxmax.muxer.ensure_tool", return_value="ffprobe"), \
             mock.patch("proxmax.muxer.subprocess.run", return_value=completed), \
             self.assertRaisesRegex(FFmpegError, "Cannot verify"):
            VideoMuxer._video_stream_duration(Path("video.mp4"))


@unittest.skipUnless(FFMPEG and shutil.which("ffprobe"), "ffmpeg and ffprobe required")
class TestVideoExtension(unittest.TestCase):
    def make_video(self, path, video_seconds=1.0, audio_seconds=1.0):
        proc = subprocess.run([
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=red:s=128x96:r=25:d={video_seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=220:duration={audio_seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
        ], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def make_audio(self, path, seconds=2.0):
        rate = 8000
        t = np.arange(round(seconds * rate)) / rate
        audio_fx.write_wav(path, (.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), rate)

    def frame(self, path, at):
        proc = subprocess.run([
            FFMPEG, "-v", "error", "-ss", str(at), "-i", str(path),
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ], capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(len(proc.stdout), 128 * 96 * 3)
        return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, 3).mean(axis=0)

    def test_extended_picture_repeats_video_tail_and_contains_full_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, audio, output = root / "in.mp4", root / "dub.wav", root / "out.mp4"
            self.make_video(source)
            self.make_audio(audio)
            muxer = VideoMuxer(use_hwaccel=False)
            with mock.patch("proxmax.video_render.run_ffmpeg_with_progress", wraps=run_ffmpeg_with_progress) as run:
                muxer.mux(source, audio, output, required_video_duration=1.9)
            cmd = run.call_args.args[0]
            self.assertIn("concat", cmd)
            self.assertNotIn("tpad", " ".join(cmd))
            self.assertGreaterEqual(muxer._video_stream_duration(output), 1.9)
            np.testing.assert_allclose(self.frame(output, 1.5), self.frame(source, .8), atol=8)
            decoded = audio_fx.decode_audio(output, 8000)
            self.assertGreater(len(decoded), int(1.9 * 8000))
            self.assertGreater(float(np.max(np.abs(decoded[int(1.8 * 8000):int(1.9 * 8000)]))), .05)

    def test_no_extension_retains_video_copy_fast_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, audio, output = root / "in.mp4", root / "dub.wav", root / "out.mp4"
            self.make_video(source)
            self.make_audio(audio, .8)
            muxer = VideoMuxer(use_hwaccel=False)
            with mock.patch("proxmax.muxer.run_ffmpeg_with_progress", wraps=run_ffmpeg_with_progress) as run:
                muxer.mux(source, audio, output, required_video_duration=.8)
            cmd = run.call_args.args[0]
            self.assertEqual(cmd[cmd.index("-c:v") + 1], "copy")
            self.assertNotIn("-vf", cmd)
            self.assertAlmostEqual(muxer._video_stream_duration(output), muxer._video_stream_duration(source), places=2)

    def test_container_longer_than_video_still_extends_picture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, audio, output = root / "in.mp4", root / "dub.wav", root / "out.mp4"
            self.make_video(source, video_seconds=1, audio_seconds=2)
            self.make_audio(audio, 2)
            muxer = VideoMuxer(use_hwaccel=False)
            muxer.mux(source, audio, output, required_video_duration=1.8)
            self.assertGreaterEqual(muxer._video_stream_duration(output), 1.8)
            np.testing.assert_allclose(self.frame(output, 1.5), self.frame(source, .8), atol=8)

    def test_underestimated_duration_does_not_drop_original_picture_frames(self):
        from proxmax.video_render import render_video_plan
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, audio, output = root / "in.mp4", root / "dub.wav", root / "out.mp4"
            self.make_video(source)
            self.make_audio(audio)
            muxer = VideoMuxer(use_hwaccel=False)
            measure = muxer._video_stream_duration
            def misleading(path):
                return .4 if Path(path) == source else measure(path)
            with mock.patch.object(muxer, "_video_stream_duration", side_effect=misleading), \
                 mock.patch("proxmax.muxer.render_video_plan", wraps=render_video_plan) as render:
                muxer.mux(source, audio, output, required_video_duration=1.8)
            plan = render.call_args.args[1]
            self.assertEqual(plan.source_frames, 25)
            self.assertEqual(sum(p.source_frames for p in plan.pieces if p.kind != "repeat"), 25)
            self.assertGreaterEqual(measure(output), 1.8)

    def test_unknown_or_overestimated_metadata_uses_real_picture_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "short.mp4"
            self.make_video(source, video_seconds=1, audio_seconds=2)
            muxer = VideoMuxer(use_hwaccel=False)
            for error in (FFmpegError("N/A"), 20.0):
                with self.subTest(metadata=error), \
                     mock.patch.object(muxer, "_video_stream_duration", side_effect=error if isinstance(error, Exception) else None,
                                       return_value=None if isinstance(error, Exception) else error):
                    normalized = muxer.ensure_picture_duration(source, 2, root, muxer.video_frame_rate(source))
                self.assertNotEqual(normalized, source)
                self.assertEqual(muxer._video_stream_duration(normalized), 2.0)

    def test_bad_source_duration_estimate_is_caught_and_retried_with_video_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, audio, output = root / "in.mp4", root / "dub.wav", root / "out.mp4"
            self.make_video(source)
            self.make_audio(audio)
            muxer = VideoMuxer(use_hwaccel=False)
            measure = muxer._video_stream_duration
            def misleading(path):
                return 5.0 if Path(path) == source else measure(path)
            with mock.patch.object(muxer, "_video_stream_duration", side_effect=misleading), \
                 mock.patch("proxmax.muxer.run_ffmpeg_with_progress", wraps=run_ffmpeg_with_progress) as run, \
                 mock.patch("proxmax.video_render.run_ffmpeg_with_progress", wraps=run_ffmpeg_with_progress) as join:
                muxer.mux(source, audio, output, required_video_duration=1.8)
            self.assertGreaterEqual(run.call_count, 1)
            self.assertIn("copy", run.call_args_list[0].args[0])
            self.assertIn("concat", join.call_args.args[0])
            self.assertNotIn("tpad", " ".join(join.call_args.args[0]))
            self.assertGreaterEqual(measure(output), 1.8)


if __name__ == "__main__":
    unittest.main()
