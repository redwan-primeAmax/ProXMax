"""Regressions for reported number/script errors and source-aware fit retries."""

import json
import unittest
from unittest import mock

from proxmax.models import Segment
from proxmax.translation_quality import assess_fit_rewrite, assess_translation
from tests._support import _FakeClient, _make_translator


class TestTranslationQuality(unittest.TestCase):
    def test_reported_126_to_26_error_is_repaired_and_audited(self):
        result = assess_translation("taking us to 126 OBR", "আমাদের ২৬ ওভিআর পর্যন্ত নেবে")
        self.assertIn("১২৬", result.text)
        self.assertFalse(result.issues)
        self.assertIn("২৬ → ১২৬", result.notes[0])

    def test_unrelated_or_multiple_numbers_are_flagged_not_substituted(self):
        result = assess_translation("We have 126 OVR", "আমাদের ৫০ ওভিআর")
        self.assertEqual(result.text, "আমাদের ৫০ ওভিআর")
        self.assertTrue(result.issues)
        result = assess_translation("12 goals and 5 points", "১২ গোল ও ৪ পয়েন্ট")
        self.assertEqual(result.text, "১২ গোল ও ৪ পয়েন্ট")
        self.assertIn("5", result.issues[0])

    def test_spoken_bangla_and_english_loan_numbers_are_not_mistaken_for_loss(self):
        self.assertFalse(assess_translation(
            "25 rated card, level 2 and level 1", "২৫ রেটেড কার্ড লেভেল টু এবং লেভেল ওয়ান"
        ).issues)
        self.assertFalse(assess_translation("1,000 coins", "এক হাজার কয়েন").issues)
        self.assertFalse(assess_translation("-5 points", "মাইনাস পাঁচ পয়েন্ট").issues)
        self.assertFalse(assess_translation("R9 is here", "আর নাইন এখানে").issues)
        self.assertFalse(assess_translation("126 OVR", "ওভিআর-১২৬").issues)
        self.assertTrue(assess_translation("-5 points", "৫ পয়েন্ট").issues)

    def test_only_unexpected_writing_systems_are_flagged(self):
        result = assess_translation("Claim the card", "কার্ডটা क्লেইম করি")
        self.assertIn("DEVANAGARI", result.issues[0])
        self.assertFalse(assess_translation("a Georgian name თბილისი", "თბილისি").issues)
        self.assertFalse(assess_translation("Hello", "হ্যালো বন্ধু").issues)

    def test_clean_batch_results_are_retained_and_only_suspects_retranslated(self):
        translator = _make_translator(_FakeClient(lambda *_: "{}"))
        segments = [
            Segment(0, 0.0, 1.0, "Hello there"),
            Segment(1, 1.0, 2.0, "12 goals and 5 points"),
            Segment(2, 2.0, 3.0, "Claim the card"),
        ]
        with mock.patch.object(translator, "_run_batches", side_effect=[
            ["হ্যালো বন্ধু", "১২ গোল ও ৪ পয়েন্ট", "कार্ড দাবি করো"],
            ["১২ গোল ও ৫ পয়েন্ট", "কার্ড দাবি করো"],
        ]) as batches:
            translator.translate_segments(segments)
        self.assertEqual(batches.call_count, 2)
        self.assertEqual([text for text, _ in batches.call_args_list[1].args[0]],
                         ["12 goals and 5 points", "Claim the card"])
        self.assertEqual(segments[0].text_bn, "হ্যালো বন্ধু")
        self.assertEqual(segments[1].text_bn, "১২ গোল ও ৫ পয়েন্ট")
        self.assertEqual(segments[2].text_bn, "কার্ড দাবি করো")
        self.assertTrue(segments[1].translation_notes)
        self.assertFalse(segments[1].translation_issues)
        self.assertNotIn("Hello there", [text for text, _ in batches.call_args_list[1].args[0]])

    def test_quality_recheck_uses_real_id_protocol_and_existing_retry_policy(self):
        def responder(contents, call):
            if call == 1:
                return json.dumps({"translations": [
                    {"id": 1, "translation": "হ্যালো বন্ধু"},
                    {"id": 2, "translation": "১২ গোল ও ৪ পয়েন্ট"},
                ]}, ensure_ascii=False)
            self.assertIn("12 goals and 5 points", contents)
            self.assertNotIn("Hello there", contents)
            if call == 2:
                raise TimeoutError("synthetic network timeout")
            return json.dumps({"translations": [{"id": 1, "translation": "১২ গোল ও ৫ পয়েন্ট"}]}, ensure_ascii=False)

        client = _FakeClient(responder)
        translator = _make_translator(client, retry_base_delay=1.0, retry_max_delay=1.0)
        segments = [Segment(0, 0.0, 1.0, "Hello there"),
                    Segment(1, 1.0, 2.0, "12 goals and 5 points")]
        with mock.patch("proxmax.gemini_protocol.time.sleep") as sleep:
            translator.translate_segments(segments)
        self.assertEqual(len(client.calls), 3)
        self.assertIn(mock.call(1.0), sleep.call_args_list)
        self.assertEqual([s.text_bn for s in segments], ["হ্যালো বন্ধু", "১২ গোল ও ৫ পয়েন্ট"])
        self.assertFalse(segments[1].translation_issues)

    def test_unresolved_results_are_not_cached_and_corrections_keep_notes_in_cache(self):
        translator = _make_translator(_FakeClient(lambda *_: "{}"))
        line = Segment(0, 0.0, 1.0, "12 goals and 5 points")
        with mock.patch.object(translator, "_run_batches", side_effect=[
            ["১২ গোল ও ৪ পয়েন্ট"], ["১২ গোল ও ৪ পয়েন্ট"],
            ["১২ গোল ও ৫ পয়েন্ট"],
        ]) as batches:
            translator.translate_segments([line])
            self.assertTrue(line.translation_issues)
            self.assertNotIn(line.text_en, translator._cache)
            translator.translate_segments([line])
        self.assertEqual(batches.call_count, 3)
        self.assertFalse(line.translation_issues)

        corrected = Segment(1, 1.0, 2.0, "We reached 126 OVR")
        with mock.patch.object(translator, "_run_batches", return_value=["আমরা ২৬ ওভিআরে পৌঁছেছি"]) as batches:
            translator.translate_segments([corrected])
            copy = Segment(2, 2.0, 3.0, corrected.text_en)
            translator.translate_segments([copy])
        self.assertEqual(batches.call_count, 1)
        self.assertEqual(copy.text_bn, corrected.text_bn)
        self.assertEqual(copy.translation_notes, corrected.translation_notes)
        self.assertTrue(copy.translation_notes)

    def test_fit_rewrite_rejects_missing_or_added_numbers_negation_and_extreme_loss(self):
        source = "We did not earn 126 OVR and 5 stars"
        original = "আমরা ১২৬ ওভিআর এবং ৫ স্টার অর্জন করিনি।"
        cases = [
            "আমরা ২৬ ওভিআর এবং ৫ স্টার অর্জন করিনি।",
            "আমরা ১২৬ ওভিআর এবং ৬ স্টার অর্জন করিনি।",
            "আমরা ১২৬ ওভিআর এবং ৫ স্টার অর্জন করেছি।",
            "১২৬ ওভিআর ৫ স্টার।",
            "আমরা ১২৬ ওভিআর और ৫ স্টার পাইনি।",
        ]
        for candidate in cases:
            with self.subTest(candidate=candidate):
                self.assertTrue(assess_fit_rewrite(source, original, original, candidate).issues)
        good = "আমরা ১২৬ ওভিআর ও ৫ স্টার অর্জন করিনি।"
        self.assertFalse(assess_fit_rewrite(source, original, original, good).issues)

    def test_fit_rewrite_uses_original_as_cumulative_limit(self):
        original = "এটি অনেক লম্বা একটি বাংলা অনুবাদের উদাহরণ।"
        current = "এটি লম্বা বাংলা অনুবাদের একটি উদাহরণ।"
        too_short = "এটি বাংলা উদাহরণ।"
        self.assertTrue(assess_fit_rewrite("A long Bangla example", original, current, too_short).issues)


if __name__ == "__main__":
    unittest.main()
