"""Regression coverage for Gemini's fixed-interval retry policy."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from proxmax.config import AppConfig
from proxmax.gemini_translator import GeminiTranslator
from proxmax.models import Segment
from proxmax.pipeline import DubbingPipeline
from proxmax.translator import TranslationError, create_translator, verify_gemini_api
from tests._support import _FakeClient, _FakeResponse, _RecordingTTS, _lines_from_contents


class _APIError(Exception):
    def __init__(self, code: int, message: str, retry_after: str = "120") -> None:
        super().__init__(message)
        self.code = code
        self.response = SimpleNamespace(status_code=code, headers={"retry-after": retry_after})


def _translator(client, **kwargs):
    return GeminiTranslator(api_key="test-key", model="gemini-test", client=client, **kwargs)


class TestGeminiRetry(unittest.TestCase):
    def test_success_on_the_100th_retry(self):
        def respond(_contents, call):
            if call <= 100:
                raise TimeoutError("network timeout")
            return json.dumps({"translations": ["বাংলা"]})

        client = _FakeClient(respond)
        translator = _translator(client)
        with mock.patch("proxmax.gemini_protocol.logger"), mock.patch(
            "proxmax.gemini_protocol.time.sleep"
        ) as sleep:
            self.assertEqual(translator.translate_text("Hello"), "বাংলা")

        self.assertEqual(translator.max_retries, 100)
        self.assertEqual(len(client.calls), 101)
        self.assertEqual(sleep.call_args_list, [mock.call(1.0)] * 100)

    def test_exhaustion_stops_after_100_retries_without_an_extra_sleep(self):
        def respond(_contents, _call):
            raise _APIError(503, "service unavailable")

        client = _FakeClient(respond)
        translator = _translator(client)
        with mock.patch("proxmax.gemini_protocol.logger"), mock.patch(
            "proxmax.gemini_protocol.time.sleep"
        ) as sleep:
            with self.assertRaises(TranslationError) as error:
                translator.translate_text("Hello")

        self.assertEqual(len(client.calls), 101)
        self.assertEqual(sleep.call_args_list, [mock.call(1.0)] * 100)
        self.assertIn("100 বার retry", str(error.exception))
        self.assertIsInstance(error.exception.__cause__, _APIError)

    def test_http_and_unclassified_errors_all_retry_at_one_second(self):
        cases = [
            _APIError(400, "invalid argument"),
            _APIError(401, "unauthenticated"),
            _APIError(403, "permission denied"),
            _APIError(404, "model not found"),
            _APIError(429, "quota exceeded", retry_after="120"),
            _APIError(500, "internal error"),
            ValueError("unexpected API failure"),
        ]
        for failure in cases:
            with self.subTest(failure=str(failure)):
                def respond(_contents, call):
                    if call == 1:
                        raise failure
                    return json.dumps({"translations": ["বাংলা"]})

                client = _FakeClient(respond)
                # Even a very restrictive RPM must not add time between retries.
                translator = _translator(client, max_retries=2, requests_per_minute=1)
                with mock.patch("proxmax.gemini_protocol.time.sleep") as sleep:
                    self.assertEqual(translator.translate_text("Hello"), "বাংলা")
                self.assertEqual(len(client.calls), 2)
                sleep.assert_called_once_with(1.0)

    def test_safety_block_can_recover_after_retry(self):
        def respond(_contents, call):
            if call == 1:
                return _FakeResponse(None, block_reason="SAFETY")
            return json.dumps({"translations": ["বাংলা"]})

        client = _FakeClient(respond)
        translator = _translator(client, max_retries=2, requests_per_minute=1)
        with mock.patch("proxmax.gemini_protocol.time.sleep") as sleep:
            self.assertEqual(translator.translate_text("Hello"), "বাংলা")
        self.assertEqual(len(client.calls), 2)
        sleep.assert_called_once_with(1.0)

    def test_format_failures_can_succeed_on_the_last_retry(self):
        def respond(_contents, call):
            return "invalid output" if call <= 100 else "1. বাংলা"

        client = _FakeClient(respond)
        translator = _translator(client)
        with mock.patch("proxmax.gemini_protocol.logger"), mock.patch(
            "proxmax.gemini_protocol.time.sleep"
        ) as sleep:
            self.assertEqual(translator.translate_text("Hello"), "বাংলা")
        self.assertEqual(len(client.calls), 101)
        self.assertEqual(sleep.call_args_list, [mock.call(1.0)] * 100)
        self.assertIn("numbered translations", client.calls[-1]["contents"])

    def test_partial_response_retries_only_missing_paragraphs_after_one_second(self):
        def respond(_contents, call):
            if call == 1:
                return _FakeResponse(
                    '{"translations": [{"id": 1, "translation": "এক"},',
                    finish_reason="MAX_TOKENS",
                )
            return json.dumps({"translations": ["দুই"]})

        client = _FakeClient(respond)
        translator = _translator(client, requests_per_minute=1)
        segments = [
            Segment(index=0, start=0.0, end=1.0, text_en="One"),
            Segment(index=1, start=1.0, end=2.0, text_en="Two"),
        ]
        with mock.patch("proxmax.gemini_protocol.time.sleep") as sleep:
            translator.translate_segments(segments)
        self.assertEqual([segment.text_bn for segment in segments], ["এক", "দুই"])
        self.assertEqual(_lines_from_contents(client.calls[1]["contents"]), ["Two"])
        sleep.assert_called_once_with(1.0)

    def test_partial_prefix_survives_a_format_exhaustion(self):
        def respond(_contents, call):
            if call == 1:
                return _FakeResponse('{"translations": [{"id": 1, "translation": "এক"},')
            if call == 2:
                return "invalid"
            return json.dumps({"translations": ["দুই"]})

        client = _FakeClient(respond)
        translator = _translator(client, max_retries=1)
        segments = [
            Segment(index=0, start=0.0, end=1.0, text_en="One"),
            Segment(index=1, start=1.0, end=2.0, text_en="Two"),
        ]
        with mock.patch.object(translator._rate_limiter, "wait"), mock.patch(
            "proxmax.gemini_protocol.time.sleep"
        ):
            translator.translate_segments(segments)
        self.assertEqual([segment.text_bn for segment in segments], ["এক", "দুই"])
        self.assertEqual(_lines_from_contents(client.calls[-1]["contents"]), ["Two"])

    def test_later_request_is_not_poisoned_after_exhausted_auth_error(self):
        def respond(_contents, call):
            if call <= 3:
                raise _APIError(401, "invalid API key")
            return json.dumps({"translations": ["বাংলা"]})

        client = _FakeClient(respond)
        translator = _translator(client, max_retries=2)
        with mock.patch.object(translator._rate_limiter, "wait"), mock.patch(
            "proxmax.gemini_protocol.time.sleep"
        ) as sleep:
            with self.assertRaises(TranslationError):
                translator.translate_text("Hello")
            self.assertEqual(translator.translate_text("Hello"), "বাংলা")
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(sleep.call_args_list, [mock.call(1.0)] * 2)

    def test_empty_failed_translation_is_not_cached(self):
        def respond(_contents, call):
            return "invalid" if call <= 2 else json.dumps({"translations": ["বাংলা"]})

        client = _FakeClient(respond)
        translator = _translator(client, max_retries=1)
        with mock.patch.object(translator._rate_limiter, "wait"), mock.patch(
            "proxmax.gemini_protocol.time.sleep"
        ):
            self.assertEqual(translator.translate_text("Hello"), "")
            self.assertNotIn("Hello", translator._cache)
            self.assertEqual(translator.translate_text("Hello"), "বাংলা")
        self.assertEqual(len(client.calls), 3)

    def test_new_batches_still_obey_rpm_after_unpaced_retries(self):
        clock = [0.0]
        starts = []
        sleeps = []

        def respond(_contents, call):
            starts.append(clock[0])
            if call <= 2:
                raise TimeoutError("temporary timeout")
            return json.dumps({"translations": ["বাংলা"]})

        def advance(seconds):
            sleeps.append(seconds)
            clock[0] += seconds

        translator = _translator(_FakeClient(respond), requests_per_minute=15)
        with mock.patch("proxmax.translation_errors.time.monotonic", side_effect=lambda: clock[0]), \
             mock.patch("proxmax.gemini_protocol.time.sleep", side_effect=advance):
            translator.translate_text("One")
            translator.translate_text("Two")
        self.assertEqual(starts, [0.0, 1.0, 2.0, 6.0])
        self.assertEqual(sleeps, [1.0, 1.0, 4.0])

    def test_configured_fixed_delay_respects_legacy_cap(self):
        def respond(_contents, call):
            if call == 1:
                raise TimeoutError("temporary timeout")
            return json.dumps({"translations": ["বাংলা"]})

        translator = _translator(
            _FakeClient(respond), retry_base_delay=0.75,
            retry_max_delay=2.0, retry_cap_delay=0.25,
        )
        with mock.patch("proxmax.gemini_protocol.time.sleep") as sleep:
            self.assertEqual(translator.translate_text("Hello"), "বাংলা")
        sleep.assert_called_once_with(0.25)

    def test_sdk_does_not_add_its_own_retries(self):
        translator = GeminiTranslator(api_key="test-key")
        with mock.patch("google.genai.Client") as client_class:
            translator._ensure_client()
        options = client_class.call_args.kwargs["http_options"]
        self.assertEqual(options.retry_options.attempts, 1)
        translator.close()

    def test_keyboard_interrupt_is_not_retried(self):
        def respond(_contents, _call):
            raise KeyboardInterrupt()

        client = _FakeClient(respond)
        translator = _translator(client)
        with mock.patch("proxmax.gemini_protocol.time.sleep") as sleep:
            with self.assertRaises(KeyboardInterrupt):
                translator.translate_text("Hello")
        self.assertEqual(len(client.calls), 1)
        sleep.assert_not_called()

    def test_factory_pipeline_and_live_api_verification_use_default_policy(self):
        translator = create_translator("gemini", gemini_api_key="test-key")
        self.assertEqual((translator.max_retries, translator.retry_base_delay), (100, 1.0))
        translator.close()

        pipeline = DubbingPipeline(AppConfig(gemini_api_key="test-key"))
        try:
            self.assertEqual((pipeline.translator.max_retries, pipeline.translator.retry_base_delay), (100, 1.0))
        finally:
            pipeline.close()

        with mock.patch.object(
            GeminiTranslator, "verify_api",
            lambda self: {"retries": self.max_retries, "delay": self.retry_base_delay},
        ):
            check = verify_gemini_api(api_key="test-key")
        self.assertEqual(check, {"retries": 100, "delay": 1.0})

    def test_overlap_pipeline_continues_after_more_than_ten_errors(self):
        failures = [0]

        def respond(contents, _call):
            text = _lines_from_contents(contents)[0]
            if text == "Second" and failures[0] < 12:
                failures[0] += 1
                raise _APIError(429, "quota exceeded")
            return json.dumps({"translations": [f"বাংলা {text}"]})

        client = _FakeClient(respond)
        translator = _translator(client, batch_size=1)
        tts = _RecordingTTS()
        pipeline = DubbingPipeline.__new__(DubbingPipeline)
        pipeline.config = AppConfig(enable_overlap=True).validate()
        pipeline.translator = translator
        pipeline._tts = tts
        segments = [
            Segment(index=0, start=0.0, end=1.0, text_en="First"),
            Segment(index=1, start=1.0, end=2.0, text_en="Second"),
        ]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            translator._rate_limiter, "wait"
        ), mock.patch("proxmax.gemini_protocol.logger"), mock.patch(
            "proxmax.gemini_protocol.time.sleep"
        ) as sleep:
            pipeline._translate_and_synthesize(segments, Path(tmp))

        self.assertEqual(failures[0], 12)
        self.assertEqual(sleep.call_args_list, [mock.call(1.0)] * 12)
        self.assertEqual(len(client.calls), 14)
        self.assertEqual([index for group in tts.calls for index, _ in group], [0, 1])
        self.assertTrue(all(segment.text_bn for segment in segments))
