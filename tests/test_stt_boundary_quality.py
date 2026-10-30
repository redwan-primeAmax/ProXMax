"""Seam decoding must not duplicate or erase source words."""

import unittest

from proxmax.models import Word
from proxmax.stt import SpeechToText
from proxmax.stt_seams import replace_in_window


class TestSTTBoundaryQuality(unittest.TestCase):
    def test_replacement_cannot_insert_words_outside_removed_window(self):
        words = [
            Word(8.7, 8.9, " before"), Word(9.1, 9.4, " old"),
            Word(10.5, 10.8, " tail"), Word(11.1, 11.3, " after"),
        ]
        replacement = [
            Word(8.7, 8.9, " before"), Word(9.1, 9.4, " new"),
            Word(10.5, 10.8, " tail"), Word(11.1, 11.3, " after"),
        ]
        result = replace_in_window(words, replacement, 9.0, 11.0)
        self.assertEqual([w.word.strip() for w in result], ["before", "new", "tail", "after"])

    def test_partial_or_empty_redecode_does_not_delete_originals(self):
        original = [
            Word(9.1, 9.4, " first"), Word(9.5, 9.8, " second"),
            Word(10.0, 10.3, " third"), Word(10.5, 10.8, " last"),
        ]
        partial = [Word(9.5, 9.8, " second")]
        self.assertEqual(replace_in_window(original, partial, 9.0, 11.0), original)
        self.assertEqual(replace_in_window(original, [], 9.0, 11.0), original)
        middle_only = [Word(9.1, 10.8, " hallucination")]
        self.assertEqual(replace_in_window(original, middle_only, 9.0, 11.0), original)

    def test_overlapping_duplicate_is_removed_but_actual_repetition_survives(self):
        words = [
            Word(1.0, 1.3, " seen", .7),
            Word(1.0, 1.3, " seen", .9),  # report's duplicated timestamp
            Word(1.31, 1.52, " very", .8),
            Word(1.50, 1.70, " very", .8),  # quick, real adjacent repetition
            Word(1.72, 1.90, " good", .8),
        ]
        result = SpeechToText._deduplicate_words(words)
        self.assertEqual([word.word.strip() for word in result], ["seen", "very", "very", "good"])
        self.assertEqual(result[0].probability, .9)

    def test_reported_duplicate_with_same_start_and_different_ends(self):
        words = [Word(613.352, 613.432, " seen.", .993),
                 Word(613.352, 613.572, " seen", .998),
                 Word(613.572, 613.672, " in", .989)]
        result = SpeechToText._deduplicate_words(words)
        self.assertEqual([word.word.strip() for word in result], ["seen", "in"])

    def test_nested_repetition_with_different_start_is_not_assumed_duplicate(self):
        words = [Word(1.0, 2.0, " very", .9), Word(1.8, 2.0, " very", .8)]
        self.assertEqual(len(SpeechToText._deduplicate_words(words)), 2)

    def test_overlap_not_just_start_time_controls_deduplication(self):
        words = [
            Word(1.0, 1.5, " be", .7),
            Word(1.02, 1.49, " be", .9),  # overlapping decode of same word
            Word(1.51, 1.9, " be", .8),  # intentional separate repetition
        ]
        result = SpeechToText._deduplicate_words(words)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].probability, .9)
        self.assertEqual(result[1].start, 1.51)


if __name__ == "__main__":
    unittest.main()
