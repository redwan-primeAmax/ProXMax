"""Focused regression tests for ProXMax."""





from tests._support import *





class TestUtilsAndSyncMath(unittest.TestCase):
    def test_output_filename(self):
        self.assertEqual(output_filename_for("lecture1.mp4"), "lecture1_bn.mp4")
        self.assertEqual(output_filename_for("/x/y/movie.mkv"), "movie_bn.mp4")

    def test_srt_timestamp(self):
        self.assertEqual(format_srt_timestamp(0), "00:00:00,000")
        self.assertEqual(format_srt_timestamp(61.234), "00:01:01,234")
        self.assertEqual(format_srt_timestamp(3661.5), "01:01:01,500")

    def test_atempo_chain(self):
        from proxmax.sync_math import build_stretch_filter

        self.assertEqual(build_atempo_filter(1.0), "atempo=1.0000")
        self.assertEqual(build_atempo_filter(2.5), "atempo=2.0000,atempo=1.2500")
        self.assertEqual(build_atempo_filter(0.4), "atempo=0.5000,atempo=0.8000")
        self.assertIn("atempo=1.0030", build_stretch_filter(1.003, 24000, False))

    def test_compute_tempo_caps(self):
        # ১× অগ্রাধিকার; overrun হলে দ্রুত করা সর্বোচ্চ ১.২০×।
        self.assertAlmostEqual(compute_tempo(10.0, 5.0), 1.20)
        # ছোট clip ডিফল্টে ধীর হবে, তবে সর্বনিম্ন tempo ০.৯০×।
        self.assertAlmostEqual(compute_tempo(2.5, 5.0), 0.90)
        self.assertAlmostEqual(
            compute_tempo(2.5, 5.0, slow_down_short_clips=False), 1.0
        )
        self.assertAlmostEqual(compute_tempo(10.0, 5.0, max_tempo=2.0), 1.20)
        self.assertAlmostEqual(compute_tempo(10.0, 5.0, max_tempo=1.15), 1.15)
        self.assertAlmostEqual(
            compute_tempo(1.0, 5.0, min_tempo=0.5, slow_down_short_clips=True), 0.90
        )
        self.assertGreaterEqual(
            compute_tempo(1.0, 5.0, slow_down_short_clips=False, fill_extension=0.5), 0.90
        )

    def test_plan_windows_never_overlap(self):
        segments = [
            Segment(index=0, start=0.0, end=3.0, tts_duration=6.0),  # অনেক লম্বা
            Segment(index=1, start=2.5, end=5.0, tts_duration=1.0),
            Segment(index=2, start=5.2, end=8.0, tts_duration=3.0),
        ]
        windows = plan_windows(segments, total_duration=9.0, min_gap_seconds=0.08)
        self.assertEqual(find_overlaps(windows), [])
        for (start, end), seg in zip(windows, segments):
            self.assertGreaterEqual(start, 0.0)
            self.assertGreater(end, start)
            self.assertAlmostEqual(start, seg.start)
        # পরের ক্লিপ শুরুর আগেই আগের উইন্ডো শেষ
        self.assertLessEqual(windows[0][1], windows[1][0])
        self.assertLessEqual(windows[1][1], windows[2][0])

    def test_plan_window_does_not_absorb_a_multi_second_source_pause(self):
        segments = [
            Segment(index=0, start=1.0, end=2.0, tts_duration=3.5),
            Segment(index=1, start=7.0, end=8.0, tts_duration=1.0),
        ]
        windows = plan_windows(
            segments, total_duration=9.0, min_gap_seconds=0.08, borrow_seconds=0.25
        )
        # Five seconds of source silence after line 1 is not a five-second dub slot.
        self.assertAlmostEqual(windows[0][1], 2.25, places=3)
        self.assertLess(windows[0][1], segments[1].start - 3.0)

    def test_find_overlaps_flags_bad_spans(self):
        self.assertEqual(find_overlaps([(0.0, 1.0), (1.0, 2.0)]), [])
        self.assertTrue(find_overlaps([(0.0, 1.5), (1.0, 2.0)]))

    def test_release_cuda_memory_is_safe_without_gpu(self):
        release_cuda_memory()


class TestEdgeSilenceTrim(unittest.TestCase):
    def test_tail_is_trimmed_before_head_and_voiced_frames_remain(self):
        rate = 24000
        t = np.arange(rate, dtype=np.float32) / rate
        speech = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        audio = np.concatenate([
            np.zeros(int(0.2 * rate), dtype=np.float32),
            speech,
            np.zeros(int(0.3 * rate), dtype=np.float32),
        ])
        trimmed, head_seconds, tail_seconds = trim_edge_silence_for_overrun(
            audio, rate, 0.4, threshold_dbfs=-45.0, pad_ms=25
        )
        self.assertAlmostEqual(head_seconds + tail_seconds, 0.4, delta=1 / rate)
        self.assertGreater(tail_seconds, head_seconds)
        self.assertAlmostEqual(trimmed.size / rate, audio.size / rate - 0.4, delta=1 / rate)
        self.assertGreater(float(np.max(np.abs(trimmed))), 0.19)

    def test_voiced_clip_is_not_trimmed_without_detected_edge_silence(self):
        rate = 24000
        t = np.arange(rate, dtype=np.float32) / rate
        audio = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        trimmed, head_seconds, tail_seconds = trim_edge_silence_for_overrun(
            audio, rate, 0.2, threshold_dbfs=-45.0, pad_ms=25
        )
        self.assertEqual(trimmed.size, audio.size)
        self.assertEqual((head_seconds, tail_seconds), (0.0, 0.0))


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class TestSyncEngineBuild(unittest.TestCase):
    def _clip(self, path: Path, seconds: float, freq: float = 220.0) -> Path:
        t = np.arange(int(seconds * 24000)) / 24000.0
        audio_fx.write_wav(path, 0.3 * np.sin(2 * np.pi * freq * t), 24000)
        return path

    def test_final_track_is_full_length_and_never_overlaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            segments = [
                Segment(index=0, start=0.0, end=2.0),
                Segment(index=1, start=2.5, end=4.5),
                Segment(index=2, start=5.0, end=7.0),
            ]
            durations = [3.6, 1.2, 2.0]  # প্রথমটি নিরাপদ window-তে আঁটে না → পুরো clip বাদ
            for seg, dur in zip(segments, durations):
                seg.tts_audio_path = self._clip(work / f"seg_{seg.index}.wav", dur)
                seg.tts_duration = dur

            engine = SyncEngine(max_tempo=1.15, min_tempo=0.90, sample_rate=24000,
                                max_workers=2)
            out = engine.build_final_track(segments, 7.5, work, work / "final.wav")
            self.assertTrue(out.is_file())
            clip, rate = audio_fx.read_wav(out)
            self.assertEqual(rate, 24000)
            self.assertAlmostEqual(clip.size / 24000.0, 7.5, delta=0.3)
            # tempo কখনো 1.15 ছাড়ায়নি
            self.assertLessEqual(engine.last_stats["max_tempo"], 1.15 + 1e-6)
            self.assertEqual(engine.last_stats["overlaps"], 0)
            # কোনো sentence-tail trim হয় না: overrun হলে পুরো clip বাদ ও review flag।
            self.assertTrue(all(s.stretched_duration is not None for s in segments))
            self.assertEqual(engine.last_stats["trimmed"], 0)
            self.assertEqual(engine.last_stats["unplaced"], 1)
            self.assertIsNone(segments[0].placed_duration)
            self.assertIn("not truncated", segments[0].sync_issue)
            self.assertIsNotNone(segments[1].placed_duration)

    def test_no_usable_clips_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SyncEngine(max_workers=1)
            with self.assertRaises(RuntimeError):
                engine.build_final_track([], 5.0, Path(tmp), Path(tmp) / "x.wav")


class TestNoSentenceTruncation(unittest.TestCase):
    def test_overrun_is_left_unplaced_instead_of_cropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "long.wav"
            tone = .2 * np.sin(2 * np.pi * 220 * np.arange(24000) / 24000.0)
            audio_fx.write_wav(source, tone, 24000)
            segment = Segment(index=0, start=0.0, end=.5, text_bn="একটি সম্পূর্ণ বাক্য।")
            segment.tts_audio_path = source
            segment.tts_duration = 1.0
            engine = SyncEngine(max_tempo=1.0, sample_rate=24000, max_workers=1)
            output = engine.build_final_track([segment], 1.0, root, root / "track.wav")
            audio, _rate = audio_fx.read_wav(output)
            self.assertEqual(engine.last_stats["unplaced"], 1)
            self.assertEqual(engine.last_stats["trimmed"], 0)
            self.assertIsNone(segment.placed_duration)
            self.assertIn("not truncated", segment.sync_issue)
            self.assertLess(float(np.max(np.abs(audio))), 1e-6)

    def test_rounding_overrun_within_reserved_gap_is_placed_without_truncation(self):
        segment = Segment(index=0, start=0.0, end=1.0)
        segment.stretched_audio_path = Path("rounded.wav")
        clip = np.full(24000 + 120, 0.1, dtype=np.float32)  # 5 ms beyond the 1 s window
        engine = SyncEngine(sample_rate=24000, slot_gap_ms=80)
        engine.last_stats = {"unplaced": 0, "trimmed": 0, "sync_issues": []}
        with mock.patch("proxmax.sync_assembly.audio_fx.read_wav", return_value=(clip, 24000)):
            track = engine._assemble([(segment, (0.0, 1.0))], 1.1)
        self.assertGreater(float(np.max(np.abs(track))), 0.09)
        self.assertEqual(engine.last_stats["unplaced"], 0)
        self.assertAlmostEqual(segment.placed_duration, 1.005, delta=1 / 24000)

    def test_rounding_tolerance_never_extends_past_the_video_end(self):
        segment = Segment(index=0, start=0.0, end=1.0, sync_tempo_cap_measured=True)
        segment.stretched_audio_path = Path("rounded.wav")
        clip = np.full(24000 + 120, 0.1, dtype=np.float32)
        engine = SyncEngine(sample_rate=24000, slot_gap_ms=80)
        engine.last_stats = {"unplaced": 0, "trimmed": 0, "sync_issues": []}
        with mock.patch("proxmax.sync_assembly.audio_fx.read_wav", return_value=(clip, 24000)):
            track = engine._assemble([(segment, (0.0, 1.0))], 1.0)
        self.assertEqual(engine.last_stats["unplaced"], 1)
        self.assertIsNone(segment.placed_duration)
        self.assertIn("not truncated", segment.sync_issue)
        self.assertLess(float(np.max(np.abs(track))), 1e-6)

    def test_unreadable_stretched_clip_is_reported_as_unplaced(self):
        segment = Segment(index=8, start=0.0, end=1.0, text_bn="শেষ বাক্য।")
        segment.stretched_audio_path = Path("missing-stretched.wav")
        engine = SyncEngine(sample_rate=24000, max_workers=1)
        engine.last_stats = {"unplaced": 0, "trimmed": 0, "sync_issues": []}
        with mock.patch("proxmax.sync_engine.audio_fx.read_wav", side_effect=OSError("broken WAV")):
            track = engine._assemble([(segment, (0.0, 1.0))], 1.0)
        self.assertEqual(track.size, 26400)
        self.assertEqual(engine.last_stats["unplaced"], 1)
        self.assertIn("unreadable", segment.sync_issue)
        self.assertEqual(engine.last_stats["sync_issues"][0]["segment_index"], 8)

    def test_retry_waits_for_sync_fit_failure_and_then_resynthesizes(self):
        from proxmax.pipeline import DubbingPipeline

        class ShrinkingTranslator:
            def __init__(self):
                self.calls = 0

            def shorten_to_durations(self, segments, targets):
                self.calls += 1
                self.targets = list(targets)
                for item in segments:
                    item.text_bn = "এটি লম্বা অনুবাদ।"
                return len(segments)

        class TinyTTS:
            def synthesize(self, segments, output_dir):
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                for item in segments:
                    path = output_dir / f"seg_{item.index:04d}.wav"
                    t = np.arange(24000, dtype=np.float32) / 24000
                    audio_fx.write_wav(path, .1 * np.sin(2 * np.pi * 330 * t), 24000)
                    item.tts_audio_path = path
                    item.tts_duration = 0.6

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_audio = root / "long.wav"
            t = np.arange(72000, dtype=np.float32) / 24000
            audio_fx.write_wav(original_audio, .1 * np.sin(2 * np.pi * 330 * t), 24000)
            segment = Segment(index=0, start=0.0, end=1.0, text_en="Long line", text_bn="এটি অনেক লম্বা অনুবাদ।")
            segment.tts_audio_path = original_audio
            segment.tts_duration = 3.0
            pipeline = DubbingPipeline.__new__(DubbingPipeline)
            pipeline.config = AppConfig(max_tempo=1.15).validate()
            pipeline.translator = ShrinkingTranslator()
            pipeline._tts = TinyTTS()

            # Raw-duration prediction alone cannot trigger a shortening request.
            before_sync = pipeline._repair_overlong_tts([segment], root / "tts", 2.0)
            self.assertEqual(before_sync["overlong_before"], 0)
            self.assertEqual(pipeline.translator.calls, 0)

            # A post-fit issue without a measured cap still cannot trigger rewrite.
            segment.sync_issue = "stretched clip exceeds safe window by 0.7s after silence trim"
            premature = pipeline._repair_overlong_tts([segment], root / "tts", 2.0)
            self.assertEqual(premature["awaiting_tempo_cap"], [0])
            self.assertEqual(pipeline.translator.calls, 0)
            segment.sync_tempo_cap_measured = True
            result = pipeline._repair_overlong_tts([segment], root / "tts", 2.0)
            self.assertEqual(result["overlong_before"], 1)
            self.assertEqual(result["shortened"], 1)
            self.assertEqual(segment.text_bn, "এটি লম্বা অনুবাদ।")
            self.assertEqual(segment.original_text_bn, "এটি অনেক লম্বা অনুবাদ।")
            # The fake provider claimed 0.6 s; the actual WAV is 1.0 s.
            self.assertAlmostEqual(segment.tts_duration, 1.0)
