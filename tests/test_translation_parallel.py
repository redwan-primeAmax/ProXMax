"""Gemini concurrency, cancellation, pacing, retry and diagnostics regressions."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np

from proxmax import audio_fx
from proxmax.config import AppConfig
from proxmax.gemini_translator import GeminiTranslator
from proxmax.models import Segment
from proxmax.pipeline import DubbingPipeline
from proxmax.sync_engine import SyncEngine
from proxmax.translation_errors import _RateLimiter
from proxmax.translation_quality import TranslationAssessment
from tests._support import FFMPEG, _FakeClient, _RecordingTTS, _lines_from_contents, _make_translator


def pipeline_for(translator, tts, concurrency=2):
    pipeline = DubbingPipeline.__new__(DubbingPipeline)
    pipeline.config = AppConfig(enable_overlap=True, gemini_concurrency=concurrency).validate()
    pipeline.translator, pipeline._tts = translator, tts
    return pipeline


class TestParallelTranslation(unittest.TestCase):
    def test_out_of_order_requests_deliver_tts_in_order_and_reuse_duplicates(self):
        second_done = threading.Event()
        finished = []
        lock = threading.Lock()

        def respond(contents, _call):
            source = _lines_from_contents(contents)[0]
            if source == "First source":
                if not second_done.wait(2):
                    raise AssertionError("second request did not overlap")
                with lock:
                    finished.append("first")
                text = "প্রথম বাংলা বাক্য"
            else:
                with lock:
                    finished.append("second")
                second_done.set()
                text = "দ্বিতীয় বাংলা বাক্য"
            return json.dumps({"translations": [text]}, ensure_ascii=False)

        client = _FakeClient(respond)
        translator = _make_translator(client, batch_size=1)
        tts = _RecordingTTS()
        pipeline = pipeline_for(translator, tts)
        segments = [Segment(i, i, i + .8, text) for i, text in
                    enumerate(("First source", "Second source", "First source"))]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(translator._rate_limiter, "wait"):
            pipeline._translate_and_synthesize(segments, Path(tmp))
        self.assertEqual(finished, ["second", "first"])
        self.assertEqual([index for group in tts.calls for index, _ in group], [0, 1, 2])
        self.assertEqual(segments[0].text_bn, segments[2].text_bn)
        self.assertEqual(len(client.calls), 2)
        metrics = translator.diagnostics.snapshot()
        self.assertEqual((metrics["attempts"], metrics["peak_in_flight"], metrics["cache_hits"]), (2, 2, 1))

    def test_normal_concurrent_requests_obey_shared_pacing(self):
        starts = []
        lock = threading.Lock()
        def respond(_contents, _call):
            with lock:
                starts.append(time.monotonic())
            time.sleep(.02)
            return json.dumps({"translations": ["সাধারণ বাংলা বাক্য"]}, ensure_ascii=False)
        translator = _make_translator(_FakeClient(respond), batch_size=1, requests_per_minute=600)
        pipeline = pipeline_for(translator, _RecordingTTS())
        segments = [Segment(i, i, i + .8, text) for i, text in enumerate(("One", "Two", "Three"))]
        with tempfile.TemporaryDirectory() as tmp:
            pipeline._translate_and_synthesize(segments, Path(tmp))
        self.assertEqual(len(starts), 3)
        self.assertTrue(all(b - a >= .085 for a, b in zip(starts, starts[1:])), starts)
        metrics = translator.diagnostics.snapshot()
        self.assertGreater(metrics["pacing_wait_seconds_sum"], .15)
        self.assertEqual(metrics["peak_in_flight"], 1)

    def test_concurrency_can_be_reduced_to_one(self):
        lock, active, peak = threading.Lock(), [0], [0]
        def respond(_contents, _call):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(.015)
            with lock:
                active[0] -= 1
            return json.dumps({"translations": ["সাধারণ বাংলা বাক্য"]}, ensure_ascii=False)

        translator = _make_translator(_FakeClient(respond), batch_size=1)
        pipeline = pipeline_for(translator, _RecordingTTS(), concurrency=1)
        segments = [Segment(i, i, i + .8, text) for i, text in enumerate(("One", "Two", "Three"))]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(translator._rate_limiter, "wait"):
            pipeline._translate_and_synthesize(segments, Path(tmp))
        self.assertEqual(peak[0], 1)
        self.assertEqual(translator.diagnostics.snapshot()["attempts"], 3)

    def test_simultaneous_schema_failures_do_not_skip_models(self):
        barrier = threading.Barrier(2)
        class Client:
            models = None
            def __init__(self):
                self.models = self
                self.models_seen = []
            def generate_content(self, *, model, contents, config):
                self.models_seen.append(model)
                if config.response_mime_type:
                    barrier.wait(2)
                    error = RuntimeError("400 schema unsupported")
                    error.code = 400
                    raise error
                return type("Response", (), {"text": '{"translations": ["বাংলা বাক্য"]}', "candidates": []})()
        client = Client()
        translator = _make_translator(client, batch_size=1, max_retries=3)
        a, b = Segment(0, 0, 1, "First"), Segment(1, 1, 2, "Second")
        with mock.patch.object(translator._rate_limiter, "wait"), \
             mock.patch("proxmax.gemini_protocol.time.sleep"), ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(translator.translate_segments, [item]) for item in (a, b)]
            for future in futures:
                future.result(timeout=3)
        self.assertFalse(translator._use_schema)
        self.assertEqual(set(client.models_seen), {"gemini-test"})
        self.assertEqual((a.text_bn, b.text_bn), ("বাংলা বাক্য", "বাংলা বাক্য"))
        self.assertEqual(translator.diagnostics.snapshot()["retries"], 2)

    def test_simultaneous_model_failures_switch_only_once(self):
        translator = _make_translator(_FakeClient(lambda *_: "{}"))
        barrier = threading.Barrier(2)
        def switch():
            barrier.wait(2)
            return translator._fallback_after_failure("gemini-test")
        with ThreadPoolExecutor(max_workers=2) as pool:
            values = [future.result(timeout=3) for future in [pool.submit(switch), pool.submit(switch)]]
        self.assertEqual(sum(value is not None for value in values), 1)
        self.assertEqual(len(translator._tried_models), 2)
        self.assertNotEqual(translator.model, "gemini-test")

    def test_recovery_does_not_mutate_shared_batch_size(self):
        entered, release = threading.Event(), threading.Event()
        translator = _make_translator(_FakeClient(lambda *_: "{}"), batch_size=12)
        def blocked(*_args, **_kwargs):
            entered.set()
            self.assertTrue(release.wait(2))
            return [TranslationAssessment("উদ্ধার করা বাংলা বাক্য")]
        segment = Segment(0, 0, 1, "recover")
        with mock.patch.object(translator, "_translate_checked", side_effect=blocked):
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(translator._recover_missing, [segment])
                self.assertTrue(entered.wait(2))
                try:
                    self.assertEqual(translator.batch_size, 12)
                    items = [Segment(i, i, i + .8, f"text {i}") for i in range(8)]
                    self.assertEqual([len(g) for g in translator.batch_segments(items)], [8])
                finally:
                    release.set()
                self.assertEqual(future.result(timeout=2), 1)
        self.assertEqual(segment.text_bn, "উদ্ধার করা বাংলা বাক্য")

    def test_sequential_gemini_worker_cancels_after_tts_failure_without_waiting_for_http(self):
        second_started, release, finished = (threading.Event() for _ in range(3))
        calls = []
        def respond(contents, _call):
            source = _lines_from_contents(contents)[0]
            calls.append(source)
            if source == "First":
                return json.dumps({"translations": ["প্রথম বাংলা"]}, ensure_ascii=False)
            second_started.set()
            self.assertTrue(release.wait(5))
            finished.set()
            raise TimeoutError("synthetic late failure")
        class FailingTTS:
            def synthesize(self, *_args):
                self_started = second_started.wait(3)
                if not self_started:
                    raise AssertionError("second request did not start")
                raise RuntimeError("synthetic TTS failure")
        translator = _make_translator(_FakeClient(respond), batch_size=1, max_retries=100)
        pipeline = pipeline_for(translator, FailingTTS(), concurrency=1)
        segments = [Segment(i, i, i + .8, text) for i, text in enumerate(("First", "Second"))]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(translator._rate_limiter, "wait"):
            started = time.perf_counter()
            try:
                with self.assertRaisesRegex(RuntimeError, "synthetic TTS failure"):
                    pipeline._translate_and_synthesize(segments, Path(tmp))
                self.assertLess(time.perf_counter() - started, 3.0)
                self.assertFalse(finished.is_set())
            finally:
                release.set()
            self.assertTrue(finished.wait(2))
        time.sleep(.05)
        self.assertEqual(calls, ["First", "Second"])

    def test_external_tts_error_cancels_later_retries(self):
        second_started, release_second, second_finished = (threading.Event() for _ in range(3))
        calls = []
        def respond(contents, _call):
            source = _lines_from_contents(contents)[0]
            calls.append(source)
            if source == "First":
                self.assertTrue(second_started.wait(2))
                return json.dumps({"translations": ["প্রথম বাংলা"]}, ensure_ascii=False)
            second_started.set()
            self.assertTrue(release_second.wait(2))
            second_finished.set()
            raise TimeoutError("synthetic failure")
        class FailingTTS:
            def synthesize(self, *_args):
                raise RuntimeError("synthetic TTS failure")
        translator = _make_translator(_FakeClient(respond), batch_size=1, max_retries=100)
        pipeline = pipeline_for(translator, FailingTTS())
        segments = [Segment(i, i, i + .8, text) for i, text in enumerate(("First", "Second"))]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(translator._rate_limiter, "wait"):
            try:
                with self.assertRaisesRegex(RuntimeError, "synthetic TTS failure"):
                    pipeline._translate_and_synthesize(segments, Path(tmp))
            finally:
                release_second.set()
            self.assertTrue(second_finished.wait(2))
        # The canceled request may finish in the background, but must not retry.
        time.sleep(.05)
        self.assertCountEqual(calls, ["First", "Second"])





    def test_101_attempts_are_counted_without_private_text(self):
        def respond(_contents, call):
            if call <= 100:
                raise TimeoutError("private provider message")
            return json.dumps({"translations": ["বাংলা"]}, ensure_ascii=False)
        translator = GeminiTranslator(api_key="secret", client=_FakeClient(respond))
        with mock.patch.object(translator._rate_limiter, "wait"), \
             mock.patch("proxmax.gemini_protocol.time.sleep") as sleep, \
             mock.patch("proxmax.gemini_protocol.logger"):
            self.assertEqual(translator.translate_text("private source"), "বাংলা")
        metrics = translator.diagnostics.snapshot()
        self.assertEqual((metrics["attempts"], metrics["retries"], metrics["failures"]), (101, 100, 100))
        self.assertEqual(metrics["failure_types"], {"TimeoutError": 100})
        self.assertEqual(sleep.call_args_list, [mock.call(1.0)] * 100)
        self.assertNotIn("private", json.dumps(metrics))
        self.assertNotIn("secret", json.dumps(metrics))

    def test_concurrent_batches_each_keep_all_one_second_retries(self):
        lock, calls = threading.Lock(), {"First": 0, "Second": 0}
        first_attempts = threading.Barrier(2)
        def respond(contents, _call):
            source = _lines_from_contents(contents)[0]
            with lock:
                calls[source] += 1
                attempt = calls[source]
            if attempt == 1:
                first_attempts.wait(2)
            if attempt <= 100:
                raise TimeoutError("synthetic error")
            return json.dumps({"translations": ["বাংলা বক্তব্য"]}, ensure_ascii=False)
        translator = GeminiTranslator(api_key="offline", client=_FakeClient(respond), batch_size=1)
        pipeline = pipeline_for(translator, _RecordingTTS())
        segments = [Segment(i, i, i + .8, text) for i, text in enumerate(("First", "Second"))]
        with tempfile.TemporaryDirectory() as temp, \
             mock.patch.object(translator._rate_limiter, "wait"), \
             mock.patch("proxmax.gemini_protocol.time.sleep") as sleep, \
             mock.patch("proxmax.gemini_protocol.logger"):
            pipeline._translate_and_synthesize(segments, Path(temp))
        self.assertEqual(calls, {"First": 101, "Second": 101})
        metrics = translator.diagnostics.snapshot()
        self.assertEqual((metrics["attempts"], metrics["retries"], metrics["failures"]), (202, 200, 200))
        self.assertEqual(sleep.call_args_list, [mock.call(1.0)] * 200)
        self.assertTrue(all(segment.text_bn for segment in segments))

    @unittest.skipUnless(FFMPEG, "FFmpeg required for sync preparation")
    def test_pipeline_resets_metrics_per_video_while_retaining_translation_cache(self):
        def tone(path, seconds):
            rate = 8000
            samples = np.arange(int(rate * seconds)) / rate
            audio_fx.write_wav(path, (.2 * np.sin(2 * np.pi * 330 * samples)).astype(np.float32), rate)
        class Extractor:
            def extract(self, _source, target, **_kwargs):
                tone(target, 2)
                return target
        class STT:
            last_stats = {}
            def transcribe(self, _source):
                return [Segment(0, .1, 1.0, "First source")]
        class TTS:
            def describe(self): return "fake"
            def set_budget(self, *_args): pass
            def synthesize(self, segments, destination):
                Path(destination).mkdir(parents=True, exist_ok=True)
                for segment in segments:
                    path = Path(destination) / "line.wav"
                    tone(path, .3)
                    segment.tts_audio_path, segment.tts_duration = path, .3
        class Muxer:
            def video_frame_rate(self, _source): return Fraction(25)
            def ensure_picture_duration(self, source, *_args): return source
            def mux(self, _source, _audio, output, **_kwargs): Path(output).write_bytes(b"stub")
        translator = _make_translator(_FakeClient(lambda *_: '{"translations": ["বাংলা বাক্য"]}'))
        pipeline = DubbingPipeline.__new__(DubbingPipeline)
        pipeline.config = AppConfig(enable_overlap=False).validate()
        pipeline.extractor, pipeline.stt, pipeline.translator = Extractor(), STT(), translator
        pipeline._tts, pipeline.muxer = TTS(), Muxer()
        pipeline.sync_engine = SyncEngine(sample_rate=8000, max_workers=1, prefer_rubberband=False)
        with tempfile.TemporaryDirectory() as temp, \
             mock.patch("proxmax.pipeline_process.get_media_duration", return_value=2), \
             mock.patch("proxmax.pipeline_process.build_units", return_value=[]):
            root = Path(temp)
            results = []
            for index in range(2):
                work = root / f"work{index}"
                work.mkdir()
                results.append(pipeline._run_steps(root / "source.mp4", work, root / f"output{index}.mp4"))
        first, second = (result[2]["translation"] for result in results)
        self.assertEqual((first["attempts"], first["cache_hits"]), (1, 0))
        self.assertEqual((second["attempts"], second["cache_hits"]), (0, 1))
        self.assertEqual(first["failures"], 0)
        self.assertEqual(second["failures"], 0)

    def test_waiting_new_request_rechecks_after_concurrent_retry(self):
        limiter = _RateLimiter(15)  # 4 seconds between normally paced starts
        clock, sleeps = [0.0], []
        waiting, resume = threading.Event(), threading.Event()
        def sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) == 1:
                waiting.set()
                self.assertTrue(resume.wait(2))
                clock[0] = 4.0
            else:
                clock[0] += seconds
        with mock.patch("proxmax.translation_errors.time.monotonic", side_effect=lambda: clock[0]), \
             mock.patch("proxmax.translation_errors.time.sleep", side_effect=sleep):
            limiter.wait()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(limiter.wait)
                self.assertTrue(waiting.wait(2))
                clock[0] = 1.0
                limiter.record_retry()
                resume.set()
                future.result(timeout=2)
        self.assertEqual(sleeps, [4.0, 1.0])
        self.assertEqual(clock[0], 5.0)


if __name__ == "__main__":
    unittest.main()
