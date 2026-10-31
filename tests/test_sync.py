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

    def test_compute_tempo_never_speeds_up_and_respects_slowdown_floor(self):
        self.assertEqual(compute_tempo(10.0, 5.0), 1.0)
        self.assertEqual(compute_tempo(2.5, 5.0), .90)
        self.assertEqual(compute_tempo(2.5, 5.0, slow_down_short_clips=False), .90)
        self.assertEqual(compute_tempo(10.0, 5.0, max_tempo=2.0), 1.0)
        self.assertEqual(compute_tempo(1.0, 5.0, min_tempo=.5), .90)
        self.assertAlmostEqual(compute_tempo(4.9, 5.0), .98)

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
            durations = [3.6, 1.2, 2.0]  # প্রথমটির জন্য তার video window বাড়বে
            for seg, dur in zip(segments, durations):
                seg.tts_audio_path = self._clip(work / f"seg_{seg.index}.wav", dur)
                seg.tts_duration = dur

            engine = SyncEngine(max_tempo=1.15, min_tempo=0.90, sample_rate=24000,
                                max_workers=2)
            out = engine.build_final_track(segments, 7.5, work, work / "final.wav")
            self.assertTrue(out.is_file())
            clip, rate = audio_fx.read_wav(out)
            self.assertEqual(rate, 24000)
            self.assertAlmostEqual(clip.size / 24000.0, engine.video_plan.duration, delta=1/24000)
            self.assertTrue(all(.90 <= s.tempo_used <= 1.0 for s in segments))
            self.assertEqual(engine.last_stats["overlaps"], 0)
            self.assertTrue(all(s.stretched_duration is not None for s in segments))
            self.assertEqual(engine.last_stats["trimmed"], 0)
            self.assertEqual(engine.last_stats["unplaced"], 0)
            self.assertTrue(all(s.placed_duration is not None for s in segments))
            self.assertGreater(segments[0].video_added_seconds, 0)
            self.assertLessEqual(max(s.video_speed for s in segments), 1.15)

    def test_no_usable_clips_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SyncEngine(max_workers=1)
            with self.assertRaises(RuntimeError):
                engine.build_final_track([], 5.0, Path(tmp), Path(tmp) / "x.wav")


class TestNoSentenceTruncation(unittest.TestCase):
    def test_overrun_repeats_picture_instead_of_cropping_speech(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "long.wav"
            tone = .2 * np.sin(2 * np.pi * 220 * np.arange(24000) / 24000.0)
            audio_fx.write_wav(source, tone, 24000)
            segment = Segment(index=0, start=0.0, end=.5, text_bn="একটি সম্পূর্ণ বাক্য।", tts_audio_path=source)
            engine = SyncEngine(max_tempo=1.0, sample_rate=24000, max_workers=1,
                                borrow_silence_ms=0)
            output = engine.build_final_track([segment], 1.0, root, root / "track.wav")
            audio, rate = audio_fx.read_wav(output)
            self.assertEqual(engine.last_stats["unplaced"], 0)
            self.assertEqual(engine.last_stats["trimmed"], 0)
            self.assertAlmostEqual(segment.placed_duration, 1.0)
            self.assertEqual(segment.tempo_used, 1.0)
            self.assertGreater(segment.video_added_seconds, .45)
            self.assertGreater(float(np.max(np.abs(audio[rate//2:rate]))), .05)

    def test_unexpected_overrun_does_not_spill_into_reserved_gap(self):
        segment = Segment(index=0, start=0.0, end=1.0)
        segment.stretched_audio_path = Path("rounded.wav")
        clip = np.full(24000 + 120, 0.1, dtype=np.float32)  # 5 ms beyond the planned window
        engine = SyncEngine(sample_rate=24000, slot_gap_ms=80)
        engine.last_stats = {"unplaced": 0, "trimmed": 0, "sync_issues": []}
        with mock.patch("proxmax.sync_assembly.audio_fx.read_wav", return_value=(clip, 24000)):
            track = engine._assemble([(segment, (0.0, 1.0))], 1.1)
        self.assertLess(float(np.max(np.abs(track))), 1e-6)
        self.assertEqual(engine.last_stats["unplaced"], 1)
        self.assertIsNone(segment.placed_duration)
        self.assertIn("planned video window", segment.sync_issue)

    def test_five_millisecond_overrun_adds_one_video_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "voice.wav"
            audio_fx.write_wav(source, np.full(24120, .1, dtype=np.float32), 24000)
            segment = Segment(0, 0, 1, "line", "বাক্য", tts_audio_path=source)
            engine = SyncEngine(sample_rate=24000, borrow_silence_ms=0, slot_gap_ms=0)
            engine.build_final_track([segment], 1.2, root, root / "out.wav", video_fps=25)
            self.assertAlmostEqual(segment.placed_duration, 1.005, places=4)
            self.assertAlmostEqual(segment.video_added_seconds, .04, places=4)
            self.assertEqual(segment.tempo_used, 1.0)
            self.assertEqual(engine.last_stats["unplaced"], 0)

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
        self.assertEqual(track.size, 24000)
        self.assertEqual(engine.last_stats["unplaced"], 1)
        self.assertIn("unreadable", segment.sync_issue)
        self.assertEqual(engine.last_stats["sync_issues"][0]["segment_index"], 8)

    def test_pipeline_has_no_gemini_fit_rewrite_stage(self):
        from proxmax.pipeline import DubbingPipeline
        self.assertFalse(hasattr(DubbingPipeline, "_repair_overlong_tts"))
