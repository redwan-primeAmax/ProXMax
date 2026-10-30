"""TTS throughput, recovery, and audio-integrity regression tests."""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from proxmax import audio_fx
from proxmax.config import AppConfig
from proxmax.models import Segment
from proxmax.text_utils import prepare_for_speech
from proxmax.tts import EdgeTTSEngine, TTSError, build_tts_engine
from tests._support import FFMPEG, REPO_ROOT


def _segments(count: int, offset: int = 0) -> list[Segment]:
    return [
        Segment(index=i + offset, start=float(i), end=i + 0.8,
                text_en=f"Line {i}", text_bn=f"বাংলা লাইন {i}")
        for i in range(count)
    ]


class TestTTSThroughput(unittest.TestCase):
    def test_default_and_configured_concurrency(self):
        self.assertEqual(AppConfig().validate().tts_concurrency, 12)
        self.assertEqual(AppConfig.from_yaml(REPO_ROOT / "config.yaml").validate().tts_concurrency, 12)
        self.assertEqual(EdgeTTSEngine().concurrency, 12)
        with mock.patch.object(EdgeTTSEngine, "prepare"):
            engine = build_tts_engine(AppConfig(tts_concurrency=5).validate())
        self.assertEqual(engine.concurrency, 5)
        with self.assertRaises(ValueError):
            AppConfig(tts_concurrency=0).validate()

    def test_worker_pool_fills_slots_without_one_task_per_segment(self):
        engine = EdgeTTSEngine(concurrency=12)
        segments = _segments(150)
        active = peak = tasks_peak = 0

        async def synthesize_one(seg, output_dir):
            nonlocal active, peak, tasks_peak
            active += 1
            peak = max(peak, active)
            tasks_peak = max(tasks_peak, len(asyncio.all_tasks()))
            await asyncio.sleep(0.001)
            seg.tts_audio_path = output_dir / f"seg_{seg.index}.wav"
            seg.tts_duration = 0.5
            active -= 1

        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(engine, "_resolve_voice", return_value=engine.voice), \
             mock.patch.object(engine, "_synthesize_one", side_effect=synthesize_one), \
             self.assertLogs("proxmax.tts.edge_engine", level="INFO") as logs:
            result = engine.synthesize(segments, Path(tmp))

        self.assertIs(result, segments)
        self.assertEqual(peak, 12)
        self.assertLessEqual(tasks_peak, 13)  # 12 workers and their parent coroutine
        self.assertTrue(all(segment.tts_audio_path and segment.tts_duration == 0.5 for segment in segments))
        self.assertTrue(any("অগ্রগতি" in message for message in logs.output))

    def test_failed_clips_retry_at_lower_concurrency_and_future_batches_slow_down(self):
        engine = EdgeTTSEngine(concurrency=12)
        attempts = {}
        active = 0
        phase_peaks = {1: 0, 2: 0}

        async def synthesize_one(seg, output_dir):
            nonlocal active
            attempts[seg.index] = attempts.get(seg.index, 0) + 1
            phase = attempts[seg.index]
            active += 1
            phase_peaks[phase] = max(phase_peaks[phase], active)
            await asyncio.sleep(0.002)
            if seg.index < 6 and phase == 1:
                seg.tts_audio_path = None
                seg.tts_duration = None
            else:
                seg.tts_audio_path = output_dir / f"seg_{seg.index}.wav"
                seg.tts_duration = 0.5
            active -= 1

        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(engine, "_resolve_voice", return_value=engine.voice), \
             mock.patch.object(engine, "_synthesize_one", side_effect=synthesize_one):
            first = _segments(12)
            engine.synthesize(first, Path(tmp))
            self.assertEqual(phase_peaks, {1: 12, 2: 5})
            self.assertTrue(all(segment.tts_audio_path for segment in first))
            self.assertEqual([attempts[i] for i in range(12)], [2] * 6 + [1] * 6)
            self.assertEqual(engine._active_concurrency, 5)
            self.assertEqual(engine.concurrency, 12)

            phase_peaks[1] = 0
            engine.synthesize(_segments(12, offset=100), Path(tmp))
            self.assertEqual(phase_peaks[1], 5)
            engine.set_budget(None, total_audio_seconds=10.0)
            phase_peaks[1] = 0
            engine.synthesize(_segments(12, offset=200), Path(tmp))
            self.assertEqual(phase_peaks[1], 12)

    def test_isolated_failure_is_retried_without_slowness_for_later_batches(self):
        engine = EdgeTTSEngine(concurrency=12)
        attempts = {}

        async def synthesize_one(seg, output_dir):
            attempts[seg.index] = attempts.get(seg.index, 0) + 1
            await asyncio.sleep(0)
            if seg.index == 0 and attempts[seg.index] == 1:
                seg.tts_audio_path = None
            else:
                seg.tts_audio_path = output_dir / f"seg_{seg.index}.wav"
                seg.tts_duration = 0.5

        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(engine, "_resolve_voice", return_value=engine.voice), \
             mock.patch.object(engine, "_synthesize_one", side_effect=synthesize_one):
            segments = _segments(12)
            engine.synthesize(segments, Path(tmp))
        self.assertTrue(all(segment.tts_audio_path for segment in segments))
        self.assertEqual(attempts[0], 2)
        self.assertTrue(all(attempts[index] == 1 for index in range(1, 12)))
        self.assertEqual(engine._active_concurrency, 12)

    def test_unrecoverable_tts_failure_is_not_reported_as_success(self):
        engine = EdgeTTSEngine(concurrency=12)
        calls = []

        async def fail(seg, _output_dir):
            calls.append(seg.index)
            seg.tts_audio_path = None
            seg.tts_duration = None

        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(engine, "_resolve_voice", return_value=engine.voice), \
             mock.patch.object(engine, "_synthesize_one", side_effect=fail):
            with self.assertRaises(TTSError):
                engine.synthesize(_segments(6), Path(tmp))
        self.assertEqual(len(calls), 12)  # initial pass plus lower-concurrency recovery

    def test_synthesis_works_inside_a_running_notebook_event_loop(self):
        engine = EdgeTTSEngine(concurrency=3)

        async def synthesize_one(seg, output_dir):
            await asyncio.sleep(0)
            seg.tts_audio_path = output_dir / f"seg_{seg.index}.wav"
            seg.tts_duration = 0.5

        async def in_notebook(output_dir):
            segments = _segments(7)
            engine.synthesize(segments, output_dir)
            return segments

        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(engine, "_resolve_voice", return_value=engine.voice), \
             mock.patch.object(engine, "_synthesize_one", side_effect=synthesize_one):
            segments = asyncio.run(in_notebook(Path(tmp)))
        self.assertTrue(all(segment.tts_audio_path and segment.tts_duration for segment in segments))

    def test_existing_segment_retry_behavior_is_preserved(self):
        engine = EdgeTTSEngine(concurrency=1, max_retries=2)
        attempts = []
        sleep_delays = []

        class Communicate:
            def __init__(self, text, voice, **options):
                self.text, self.voice, self.options = text, voice, options

            async def save(self, path):
                attempts.append(self.voice)
                if len(attempts) == 1:
                    raise ConnectionError("temporary failure")
                Path(path).write_bytes(b"audio")

        async def fake_sleep(delay):
            sleep_delays.append(delay)

        def convert(mp3, wav):
            wav.write_bytes(b"converted")
            mp3.unlink()
            return 1.25

        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(engine, "_resolve_voice", return_value=engine.voice), \
             mock.patch("edge_tts.Communicate", Communicate), \
             mock.patch("proxmax.tts.edge_engine.asyncio.sleep", side_effect=fake_sleep), \
             mock.patch.object(engine, "_convert_and_finalize", side_effect=convert):
            segment = _segments(1)[0]
            engine.synthesize([segment], Path(tmp))
            self.assertTrue(segment.tts_audio_path.is_file())
        self.assertEqual(len(attempts), 2)
        self.assertEqual(sleep_delays, [1.0])
        self.assertEqual(segment.tts_duration, 1.25)

    @unittest.skipUnless(FFMPEG, "ffmpeg not installed")
    def test_parallel_audio_conversion_preserves_unique_clips_and_timing(self):
        engine = EdgeTTSEngine(concurrency=4, trim_silence=False)
        event_loop_thread = threading.get_ident()
        finalize_threads = []
        original_finalize = engine._finalize

        def finalize(audio):
            finalize_threads.append(threading.get_ident())
            return original_finalize(audio)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_wav, source_mp3 = root / "source.wav", root / "source.mp3"
            samples = np.arange(9600) / 24000.0
            audio_fx.write_wav(source_wav, (0.2 * np.sin(2 * np.pi * 220 * samples)).astype(np.float32), 24000)
            subprocess.run(
                [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source_wav),
                 "-c:a", "libmp3lame", str(source_mp3)],
                check=True, capture_output=True,
            )
            mp3_bytes = source_mp3.read_bytes()
            seen_texts = []

            class Communicate:
                def __init__(self, text, voice, **_options):
                    self.text, self.voice = text, voice

                async def save(self, path):
                    seen_texts.append((self.text, self.voice, Path(path).name))
                    Path(path).write_bytes(mp3_bytes)

            segments = _segments(8)
            original_times = [(seg.start, seg.end) for seg in segments]
            with mock.patch("edge_tts.Communicate", Communicate), \
                 mock.patch.object(engine, "_resolve_voice", return_value=engine.voice), \
                 mock.patch.object(engine, "_finalize", side_effect=finalize):
                engine.synthesize(segments, root)

            self.assertEqual(len(seen_texts), 8)
            self.assertEqual(
                {name: (text, voice) for text, voice, name in seen_texts},
                {f"seg_{seg.index:04d}.mp3": (prepare_for_speech(seg.text_bn), engine.voice)
                 for seg in segments},
            )
            self.assertEqual(len({seg.tts_audio_path for seg in segments}), 8)
            self.assertEqual([(seg.start, seg.end) for seg in segments], original_times)
            self.assertTrue(finalize_threads)
            self.assertTrue(all(thread_id != event_loop_thread for thread_id in finalize_threads))
            # The original sequential conversion/postprocessing must produce the same samples.
            reference_mp3, reference_wav = root / "reference.mp3", root / "reference.wav"
            reference_mp3.write_bytes(mp3_bytes)
            engine._to_wav(reference_mp3, reference_wav)
            original_audio, original_rate = audio_fx.read_wav(reference_wav)
            audio_fx.write_wav(reference_wav, engine._finalize(original_audio), original_rate)
            expected_audio, _ = audio_fx.read_wav(reference_wav)
            for seg in segments:
                self.assertTrue(seg.tts_audio_path.is_file())
                audio, rate = audio_fx.read_wav(seg.tts_audio_path)
                self.assertEqual(rate, 24000)
                self.assertGreater(np.max(np.abs(audio)), 0.01)
                np.testing.assert_array_equal(audio, expected_audio)
                self.assertAlmostEqual(seg.tts_duration, audio.size / rate, places=6)
            self.assertEqual(list(root.glob("seg_*.mp3")), [])
