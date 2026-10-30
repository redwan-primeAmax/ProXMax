"""Focused regression tests for ProXMax."""





from tests._support import *





@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class TestPipelineEndToEnd(unittest.TestCase):
    def _make_video(self, path: Path, seconds: float = 6.0) -> None:
        proc = subprocess.run(
            [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", f"color=c=navy:s=320x240:r=25:d={seconds}",
             "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}",
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-shortest", str(path)],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_full_pipeline_with_mocks(self):
        from proxmax.audio_extraction import AudioExtractor
        from proxmax.muxer import VideoMuxer
        from proxmax.pipeline import DubbingPipeline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir, output_dir, work_dir = root / "input", root / "output", root / "work"
            input_dir.mkdir()
            video = input_dir / "tiny.mp4"
            self._make_video(video)

            cfg = AppConfig(
                input_dir=str(input_dir), output_dir=str(output_dir),
                work_dir=str(work_dir), save_report=True, unload_stt_before_tts=True,
            ).validate()

            pipeline = DubbingPipeline.__new__(DubbingPipeline)  # ভারী init বাদ
            pipeline.config = cfg
            pipeline.extractor = AudioExtractor(use_hwaccel=False)
            pipeline.stt = _FakeSTT()
            pipeline.translator = _FakeTranslator()
            pipeline.tts = _FakeTTS()
            pipeline._tts = pipeline.tts
            pipeline._profile = None
            pipeline._profile_checked = True
            pipeline.sync_engine = SyncEngine(sample_rate=24000, max_workers=2)
            pipeline.muxer = VideoMuxer(use_hwaccel=False)

            summary = pipeline.run_batch()
            self.assertEqual(len(summary.succeeded), 1, summary.failed)

            out = output_dir / "tiny_bn.mp4"
            self.assertTrue(out.is_file() and out.stat().st_size > 0)
            report = output_dir / "tiny_report.markdown"
            self.assertTrue(report.is_file() and report.stat().st_size > 0)
            report_text = report.read_text(encoding="utf-8")
            self.assertIn("Hello world.", report_text)
            self.assertIn("হ্যালো বিশ্ব।", report_text)
            self.assertFalse((output_dir / "tiny_en.srt").exists())
            self.assertFalse((output_dir / "tiny_bn.srt").exists())

            result = summary.succeeded[0]
            self.assertIsInstance(result.timings, StageTimings)
            self.assertGreater(result.timings.total, 0)
            self.assertGreater(result.timings.tts, 0)
            self.assertGreater(result.timings.translate_tts_wall, 0)
            self.assertGreater(result.timings.mux_processing, 0)
            self.assertGreater(result.timings.output_copy, 0)
            self.assertEqual(result.stats["sync"]["overlaps"], 0)
            self.assertTrue(result.stats["sync_passes"])
            self.assertIn("stt", result.stats)

            # step 4-এ Whisper unload + শেষে সবার release
            self.assertTrue(pipeline.stt.released)
            self.assertTrue(pipeline.translator.closed)
            self.assertTrue(pipeline.tts.closed)

    def test_empty_input_folder_is_graceful(self):
        from proxmax.pipeline import DubbingPipeline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = AppConfig(
                input_dir=str(root / "input"), output_dir=str(root / "output"),
                work_dir=str(root / "work"),
            )
            (root / "input").mkdir()
            pipeline = DubbingPipeline.__new__(DubbingPipeline)
            pipeline.config = cfg
            summary = pipeline.run_batch()
            self.assertEqual(summary.results, [])
