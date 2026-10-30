"""Focused regression tests for ProXMax."""





from tests._support import *





class TestBanglaAudioCoverage(unittest.TestCase):
    def test_stt_qa_decode_uses_requested_bengali_language(self):
        stt = SpeechToText.__new__(SpeechToText)
        stt._load = mock.Mock()
        stt.chunk_min_seconds = 1.0
        stt.chunk_max_seconds = 28.0
        stt.chunk_pad_seconds = 0.1
        stt.vad_min_silence_ms = 300
        stt._decode_words = mock.Mock(return_value=([Word(.1, .4, "শেষ")], "শেষ"))
        chunk = SpeechChunk(start=0.0, end=1.0)
        with mock.patch("proxmax.stt_inference.decode_audio", return_value=np.ones(16000, dtype=np.float32)), \
             mock.patch("proxmax.stt_inference.plan_chunks", return_value=([chunk], "test")):
            words = stt.transcribe_words_in_language("final.wav", language="bn")
        self.assertEqual([word.word for word in words], ["শেষ"])
        self.assertEqual(stt._decode_words.call_args.kwargs["language"], "bn")

    def test_complete_text_and_missing_tail_are_distinguished(self):
        segment = Segment(
            index=3, start=1.0, end=3.0,
            text_en="I scored a goal.", text_bn="আমি একটি গোল করেছি।",
            placed_duration=2.0,
        )
        complete = [
            Word(1.1, 1.3, "আমি"), Word(1.4, 1.6, "একটি"),
            Word(1.7, 2.0, "গোল"), Word(2.1, 2.5, "করেছি।"),
        ]
        result = assess_bengali_coverage([segment], complete, min_coverage=0.55)
        self.assertEqual(result["units"][0]["status"], "likely_complete")
        self.assertTrue(result["units"][0]["tail_token_found"])

        truncated = assess_bengali_coverage([segment], complete[:-1], min_coverage=0.55)
        self.assertEqual(truncated["units"][0]["status"], "review")
        self.assertFalse(truncated["units"][0]["tail_token_found"])

    def test_unplaced_sentence_is_reported_as_missing_not_passed(self):
        segment = Segment(index=0, start=0.0, end=1.0, text_bn="শেষ কথাটি।")
        result = assess_bengali_coverage([segment], [], min_coverage=0.5)
        self.assertEqual(result["units"][0]["status"], "missing")
        self.assertEqual(result["summary"]["missing"], 1)
