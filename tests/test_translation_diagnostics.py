"""Cancellation isolation and per-request telemetry regressions."""
from __future__ import annotations

import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from proxmax.gemini_translator import GeminiTranslator
from proxmax.models import Segment
from proxmax.translation_errors import TranslationError
from proxmax.translation_quality import TranslationAssessment
from tests._support import _FakeClient, _lines_from_contents, _make_translator


class TestTranslationDiagnostics(unittest.TestCase):
    def test_late_cancelled_request_cannot_block_or_pollute_next_video_metrics(self):
        entered, release = threading.Event(), threading.Event()
        def respond(contents, _call):
            source = _lines_from_contents(contents)[0]
            if source == "Old video":
                entered.set()
                self.assertTrue(release.wait(3))
                raise TimeoutError("private old failure")
            return json.dumps({"translations": ["নতুন বাংলা"]}, ensure_ascii=False)
        translator = _make_translator(_FakeClient(respond), batch_size=1)
        stop = threading.Event()
        old, new = Segment(0, 0, 1, "Old video"), Segment(1, 0, 1, "New video")
        with mock.patch.object(translator._rate_limiter, "wait"), ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(translator.translate_segments_cancellable, [old], stop)
            self.assertTrue(entered.wait(2))
            stop.set()
            translator.diagnostics.reset()
            translator.translate_segments([new])
            before = translator.diagnostics.snapshot()
            self.assertEqual((before["attempts"], before["failures"], before["in_flight"]), (1, 0, 0))
            self.assertEqual(before["prior_in_flight_at_reset"], 1)
            release.set()
            with self.assertRaises(TranslationError):
                future.result(timeout=2)
        after = translator.diagnostics.snapshot()
        self.assertEqual(after, before)
        self.assertFalse(old.text_bn)
        self.assertTrue(new.text_bn)


    def test_cancelled_late_quality_result_cannot_fill_segments_or_cache(self):
        entered, release = threading.Event(), threading.Event()
        translator = _make_translator(_FakeClient(lambda *_: "{}"))
        stop = threading.Event()
        segment = Segment(0, 0, 1, "Private sentence")
        def late(*_args, **_kwargs):
            entered.set()
            self.assertTrue(release.wait(2))
            return [TranslationAssessment("পুরনো বাংলা")]
        with mock.patch.object(translator, "_translate_checked", side_effect=late), \
             ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(translator.translate_segments_cancellable, [segment], stop)
            self.assertTrue(entered.wait(2))
            stop.set()
            release.set()
            with self.assertRaises(TranslationError):
                future.result(timeout=2)
        self.assertFalse(segment.text_bn)
        self.assertNotIn("Private sentence", translator._cache)


    def test_cancelled_quality_recheck_is_not_swallowed_as_a_review_warning(self):
        translator = _make_translator(_FakeClient(lambda *_: "{}"))
        stop = threading.Event()
        segment = Segment(0, 0, 1, "Source sentence")
        def batches(*_args, **_kwargs):
            if not stop.is_set():
                stop.set()
                return ["অঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅ"]
            raise TranslationError("private response")
        with mock.patch.object(translator, "_run_batches", side_effect=batches):
            with self.assertRaises(TranslationError):
                translator.translate_segments_cancellable([segment], stop)
        self.assertFalse(segment.text_bn)
        self.assertFalse(translator._cache)


    def test_retry_events_are_timed_and_scoped_to_the_translation_group(self):
        from proxmax.reporting_runtime import OperationalTrace
        def respond(_contents, call):
            if call <= 2:
                raise TimeoutError("PRIVATE_PROVIDER_RESPONSE")
            return json.dumps({"translations": ["বাংলা বক্তব্য"]}, ensure_ascii=False)
        translator = GeminiTranslator(api_key="offline", client=_FakeClient(respond), batch_size=1)
        segment = Segment(17, 0, 1, "Private source")
        trace = OperationalTrace()
        trace.activate()
        try:
            with mock.patch.object(translator._rate_limiter, "wait"), \
                 mock.patch("proxmax.gemini_protocol.time.sleep"):
                translator.translate_segments([segment])
            snapshot = trace.snapshot()
        finally:
            trace.close()
        attempts = [e for e in snapshot["events"] if e["code"] == "gemini_attempt_finished"]
        retries = [e for e in snapshot["events"] if e["code"] == "gemini_retry_scheduled"]
        self.assertEqual((len(attempts), len(retries)), (3, 2))
        self.assertTrue(all(e["details"]["first_segment"] == 17 for e in attempts + retries))
        self.assertEqual([e["details"]["delay_seconds"] for e in retries], [1.0, 1.0])
        self.assertNotIn("PRIVATE_PROVIDER_RESPONSE", str(snapshot))
        self.assertNotIn("Private source", str(snapshot))


if __name__ == "__main__":
    unittest.main()
