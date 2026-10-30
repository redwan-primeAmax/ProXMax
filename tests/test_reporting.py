"""Focused regression tests for ProXMax."""





from tests._support import *





class TestDetailedReport(unittest.TestCase):
    def test_markdown_report_contains_timeline_and_omits_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output_video = root / "source.mp4", root / "source_bn.mp4"
            output_dir, work_dir = root / "output", root / "work"
            source.write_bytes(b"source")
            output_video.write_bytes(b"dub")
            output_dir.mkdir()
            work_dir.mkdir()

            rate = 16000
            t = np.arange(rate * 4, dtype=np.float32) / rate
            audio_fx.write_wav(work_dir / "audio_16k.wav", .1 * np.sin(2 * np.pi * 220 * t), rate)
            dub = np.zeros(rate * 4, dtype=np.float32)
            dub[:rate] = .1 * np.sin(2 * np.pi * 220 * t[:rate])
            audio_fx.write_wav(work_dir / "final_audio_bn.wav", dub, rate)

            child = Segment(index=0, start=1.0, end=2.0, text_en="Hello world.")
            child.words = [Word(1.0, 1.35, "Hello", .95), Word(1.4, 1.9, "world.", .92)]
            unit = Segment(
                index=0, start=1.0, end=2.0, text_en="Hello world.", text_bn="হ্যালো বিশ্ব।",
                words=list(child.words), tts_audio_path=work_dir / "tts.wav", tts_duration=1.2,
                stretched_audio_path=work_dir / "stretched.wav", stretched_duration=1.1,
                placed_duration=.9, tempo_used=1.2, is_unit=True, children=[child],
            )
            cfg = AppConfig(gemini_api_key="DO_NOT_PUT_THIS_IN_REPORT").validate()
            probe = {
                "format": {"format_name": "mp4", "duration": "4.0", "bit_rate": "1000"},
                "streams": [{"index": 0, "codec_type": "video", "codec_name": "h264",
                             "start_time": "0", "duration": "4", "bit_rate": "900",
                             "width": 320, "height": 240, "avg_frame_rate": "25/1"}],
            }
            timings = StageTimings(extract=.1, stt=.2, translate=.3, tts=.4, sync=.5,
                                   mux=.6, total=1.0, audio_seconds=4.0)
            with mock.patch("proxmax.reporting._probe_media", return_value=probe):
                report = write_detailed_report(
                    video_path=source, output_video_path=output_video, output_dir=output_dir,
                    work_dir=work_dir, segments=[unit], timings=timings,
                    stats={
                        "sync": {"speech_seconds": .9, "silence_seconds": 3.1,
                                 "trimmed": 1, "overlaps": 0},
                        "stt": {"words": 2},
                        "audio_validation": {
                            "status": "review", "method": "synthetic test",
                            "summary": {"checked": 1, "likely_complete": 0, "review": 1, "missing": 0},
                            "units": [{"segment_index": 0, "status": "review", "coverage": .5,
                                       "tail_token_found": False, "recognized_text": "হ্যালো"}],
                        },
                    },
                    config=cfg, total_duration=4.0,
                )

            text = report.read_text(encoding="utf-8")
            self.assertEqual(report.name, "source_report.markdown")
            self.assertIn("Hello world.", text)
            self.assertIn("হ্যালো বিশ্ব।", text)
            self.assertIn("Source word timestamps", text)
            self.assertIn('"gemini_max_paragraphs_per_request": 12', text)
            self.assertIn('"post_mux_bengali_asr_validation": false', text)
            self.assertIn("trimmed=0.200 s", text)
            self.assertIn("Post-mux Bengali speech coverage check", text)
            self.assertIn("**Post-mux Bangla ASR coverage:** review", text)
            self.assertIn("**Post-mux ASR heard:** হ্যালো", text)
            self.assertNotIn("DO_NOT_PUT_THIS_IN_REPORT", text)
            self.assertNotIn(str(root), text)
            self.assertFalse((output_dir / "source_en.srt").exists())
            self.assertFalse((output_dir / "source_bn.srt").exists())


class TestPipelineReportEmission(unittest.TestCase):
    def test_existing_video_without_report_is_reprocessed_and_reported(self):
        from proxmax.pipeline import DubbingPipeline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir, output_dir = root / "input", root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            source = input_dir / "video.mp4"
            output = output_dir / "video_bn.mp4"
            source.write_bytes(b"input")
            output.write_bytes(b"previous valid output")
            cfg = AppConfig(
                input_dir=str(input_dir), output_dir=str(output_dir),
                work_dir=str(root / "work"), skip_existing=True, save_report=True,
            ).validate()
            pipeline = DubbingPipeline.__new__(DubbingPipeline)
            pipeline.config = cfg
            segment = Segment(index=0, start=.2, end=1.2,
                              text_en="A complete sentence.", text_bn="একটি সম্পূর্ণ বাক্য।")

            def fake_steps(_video, work_dir, output_path):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"new dubbed output")
                Path(work_dir).mkdir(parents=True, exist_ok=True)
                return [segment], StageTimings(audio_seconds=2.0), {
                    "audio_seconds": 2.0, "sync": {"trimmed": 0, "overlaps": 0}
                }

            def fake_report(**kwargs):
                path = Path(kwargs["output_dir"]) / "video_report.markdown"
                path.write_text("# synthetic report\n", encoding="utf-8")
                return path

            pipeline._run_steps = mock.Mock(side_effect=fake_steps)
            with mock.patch("proxmax.pipeline_process.existing_output_is_usable", return_value=True), \
                 mock.patch("proxmax.pipeline_process.write_detailed_report", side_effect=fake_report):
                result = pipeline.process_one(source, output_dir)

            self.assertEqual(result.status, "success")
            self.assertEqual(pipeline._run_steps.call_count, 1)
            self.assertEqual(result.stats["report"]["file"], "video_report.markdown")
            self.assertTrue((output_dir / "video_report.markdown").is_file())
            self.assertFalse((output_dir / "video_en.srt").exists())
            self.assertFalse((output_dir / "video_bn.srt").exists())
