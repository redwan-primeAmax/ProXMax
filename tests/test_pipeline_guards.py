"""Focused regression tests for ProXMax."""





from tests._support import *





class TestExistingOutputGuard(unittest.TestCase):
    """আগের আউটপুট "সম্পন্ন" ধরে নেওয়ার আগে সত্যিই বৈধ কি না দেখা হয়।"""

    def _valid_mp4(self, folder: Path) -> Path:
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg/ffprobe নেই")
        out = folder / "video_bn.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=3",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(out),
            ],
            check=True,
        )
        # কৃত্রিমভাবে বড় করা হচ্ছে যাতে min_bytes পাস করে (আসল MP4 ছোট হয়)
        with open(out, "ab") as handle:
            handle.write(b"\0" * (64 * 1024))
        return out

    def test_valid_output_is_skippable(self):
        from proxmax.pipeline import existing_output_is_usable

        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(existing_output_is_usable(self._valid_mp4(Path(tmp))))

    def test_missing_or_tiny_or_garbage_output_is_not_skippable(self):
        from proxmax.pipeline import existing_output_is_usable

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            missing = folder / "nope.mp4"
            tiny = folder / "tiny.mp4"
            tiny.write_bytes(b"\0" * 1024)
            garbage = folder / "garbage.mp4"
            garbage.write_bytes(b"not a video" * 8000)
            self.assertFalse(existing_output_is_usable(missing))
            self.assertFalse(existing_output_is_usable(tiny))
            self.assertFalse(existing_output_is_usable(garbage))

    def test_skip_is_announced_with_force_hint(self):
        from proxmax.pipeline import DubbingPipeline

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            in_dir, out_dir = folder / "in", folder / "out"
            in_dir.mkdir()
            out_dir.mkdir()
            video = in_dir / "video.mp4"
            video.write_bytes(b"\0" * 1024)
            pipeline = DubbingPipeline(
                AppConfig(input_dir=in_dir, output_dir=out_dir, gemini_api_key="test-key")
            )
            # বৈধতা যাচাইটা mock করা হচ্ছে — বাকি সব আসল কোডই চলবে
            (out_dir / "video_bn.mp4").write_bytes(b"\0" * (64 * 1024))
            (out_dir / "video_report.markdown").write_text("# Detailed report\n" + ("diagnostic " * 40), encoding="utf-8")
            with mock.patch(
                "proxmax.pipeline_process.existing_output_is_usable", return_value=True
            ):
                try:
                    with self.assertLogs("proxmax.pipeline_process", level="INFO") as logs:
                        result = pipeline.process_one(video, out_dir)
                finally:
                    pipeline.close()
        self.assertEqual(result.status, "skipped")
        self.assertTrue(any("--force" in line for line in logs.output))


class TestFFmpegSafety(unittest.TestCase):
    def test_nostdin_is_always_added(self):
        from proxmax.utils import ffmpeg_args

        self.assertEqual(ffmpeg_args(["ffmpeg", "-i", "in.mp4"]),
                         ["ffmpeg", "-nostdin", "-i", "in.mp4"])
        # ইতিমধ্যে থাকলে দুবার যোগ হয় না
        self.assertEqual(ffmpeg_args(["ffmpeg", "-nostdin", "-i", "x"]).count("-nostdin"), 1)

    def test_hanging_ffmpeg_is_killed_instead_of_blocking(self):
        from proxmax.utils import FFmpegError, run_ffmpeg_with_progress

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "hang_ffmpeg"
            script.write_text(
                "#!/bin/sh\n"
                "echo out_time_ms=1000000\n"
                "echo progress=continue\n"
                "sleep 30\n",
                encoding="utf-8",
            )
            script.chmod(0o755)
            started = time.monotonic()
            with self.assertRaises(FFmpegError) as ctx:
                run_ffmpeg_with_progress(
                    [str(script), "-i", "x"], "test", duration_seconds=10,
                    stall_timeout=2,
                )
            self.assertLess(time.monotonic() - started, 15)
            self.assertIn("অগ্রগতি", str(ctx.exception))
