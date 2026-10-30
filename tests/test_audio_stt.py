"""Focused regression tests for ProXMax."""





from tests._support import *





class TestAudioFx(unittest.TestCase):
    def test_wav_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.wav"
            tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(2400) / 24000.0)
            audio_fx.write_wav(path, tone, 24000)
            clip, rate = audio_fx.read_wav(path)
            self.assertEqual(rate, 24000)
            self.assertEqual(clip.size, 2400)
            self.assertAlmostEqual(float(np.max(np.abs(clip))), 0.3, places=2)

    def test_levels_and_fade(self):
        silence = np.zeros(2400, dtype=np.float32)
        self.assertLessEqual(audio_fx.rms_dbfs(silence), -99)
        loud = np.ones(2400, dtype=np.float32) * 0.5
        self.assertGreater(audio_fx.rms_dbfs(loud), -12)
        faded = audio_fx.fade(loud, 24000, in_ms=10, out_ms=20)
        self.assertLess(abs(float(faded[0])), 0.05)
        self.assertLess(abs(float(faded[-1])), 0.05)
        normalized = audio_fx.peak_normalize(loud, peak_dbfs=-1.0)
        self.assertAlmostEqual(float(np.max(np.abs(normalized))), audio_fx.db_to_amp(-1.0), places=3)

    def test_silence_length(self):
        self.assertEqual(audio_fx.silence(0.5, 16000).size, 8000)


class TestTextUtils(unittest.TestCase):
    def test_normalise_bangla(self):
        self.assertEqual(normalise_bangla("  হ্যালো    বিশ্ব  "), "হ্যালো বিশ্ব")
        self.assertEqual(normalise_bangla("বাক্য।পরের"), "বাক্য। পরের")
        self.assertEqual(normalise_bangla("**bold** text"), "bold text")

    def test_verbalize_numbers(self):
        spoken = verbalize_numbers("50% ছাত্র")
        self.assertIn("শতাংশ", spoken)
        self.assertNotIn("50", spoken)
        english = verbalize_numbers("2024 সালে 50% মানুষ")
        self.assertNotIn("2024", english)
        self.assertIn("শতাংশ", english)

    def test_prepare_for_speech_pipeline(self):
        prepared = prepare_for_speech("খরচ 100 টাকা।")
        self.assertNotIn("100", prepared)
        self.assertIn("টাকা", prepared)

    def test_bangla_and_ascii_numbers_are_spoken_the_same(self):
        from proxmax.text_utils import verbalize_numbers

        self.assertEqual(verbalize_numbers("১২৫ ওভিআর"), "একশ পঁচিশ ওভিআর")
        self.assertEqual(verbalize_numbers("125 ওভিআর"), "একশ পঁচিশ ওভিআর")
        self.assertEqual(verbalize_numbers("50%"), "পঞ্চাশ শতাংশ")
        self.assertEqual(verbalize_numbers("১২৫%"), "একশ পঁচিশ শতাংশ")

    def test_number_estimate_counts_spoken_length(self):
        from proxmax.text_utils import estimate_speech_seconds

        # "১২৫" লেখায় ৩ অক্ষর, বলায় "একশ পঁচিশ" (৯ অক্ষর) — সময়ও বেশি লাগবে
        self.assertAlmostEqual(
            estimate_speech_seconds("১২৫", 12.0), len("একশ পঁচিশ") / 12.0, places=3
        )
        self.assertGreater(estimate_speech_seconds("১২৫", 12.0), 3 / 12.0)

    def test_estimate_and_split(self):
        self.assertGreater(estimate_speech_seconds("এটি একটি পরীক্ষা।"), 0.4)
        self.assertEqual(estimate_speech_seconds(""), 0.0)
        parts = split_sentences("প্রথম বাক্য। দ্বিতীয় বাক্য! তৃতীয়?")
        self.assertEqual(len(parts), 3)


class TestChunking(unittest.TestCase):
    def test_cuts_only_inside_silence(self):
        # 6 s কথা + 0.5 s নীরবতা — বারবার; মোট 39.5 s
        audio = _speech_with_silence([(6.0, 0.5)] * 6 + [(3.0, 0.0)])
        chunks, method = plan_chunks(
            audio, sample_rate=16000, min_seconds=8.0, max_seconds=20.0,
            min_silence_ms=300,
        )
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn(method, ("vad", "energy"))
        # কোনো chunk-এর শুরু/শেষ কথা বলার মাঝখানে পড়েনি
        regions = energy_regions(audio, 16000)
        for chunk in chunks:
            for start, end in regions:
                if start + 0.05 < chunk.start < end - 0.05:
                    self.fail(f"chunk boundary {chunk.start:.2f}s কথা বলার মাঝখানে!")
                if start + 0.05 < chunk.end < end - 0.05:
                    self.fail(f"chunk end {chunk.end:.2f}s কথা বলার মাঝখানে!")
        self.assertTrue(all(c.duration <= 20.5 for c in chunks))
        self.assertFalse(any(c.forced for c in chunks))

    def test_forced_split_when_no_silence(self):
        # একটানা 45 s কথা — উপায় নেই, কাটতেই হবে; flag থাকবে যাতে seam repair হয়
        t = np.arange(int(45 * 16000)) / 16000.0
        audio = (0.25 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
        chunks, _ = plan_chunks(
            audio, sample_rate=16000, min_seconds=18.0, max_seconds=28.0,
            min_silence_ms=300,
        )
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(any(c.forced for c in chunks))
        self.assertTrue(all(c.duration <= 28.5 for c in chunks))

    def test_quiet_audio_is_split_safely(self):
        # energy থ্রেশহোল্ডের নিচে (খুব নিচু কথা) — তবু chunk ৩০s-এর নিচে থাকতেই হবে
        t = np.arange(int(60 * 16000)) / 16000.0
        audio = (0.002 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
        chunks, _ = plan_chunks(
            audio, sample_rate=16000, min_seconds=18.0, max_seconds=28.0,
        )
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(c.duration <= 28.5 for c in chunks))
        self.assertTrue(all(c.forced for c in chunks))

    def test_regions_to_chunks_small_gaps(self):
        regions = [(0.0, 5.0), (5.1, 9.0), (12.0, 18.0)]
        chunks = regions_to_chunks(
            regions, total_seconds=18.0, min_seconds=6.0, max_seconds=12.0,
        )
        self.assertTrue(chunks)
        self.assertTrue(all(c.duration <= 12.5 for c in chunks))


class TestSegmenter(unittest.TestCase):
    def test_splits_at_punctuation_and_pauses(self):
        tokens = (
            "This is a test. It has two sentences, and one comma. "
            "A pause follows here."
        ).split()
        words = []
        t = 0.0
        from proxmax.models import Word

        for token in tokens:
            words.append(Word(start=t, end=t + 0.4, word=" " + token))
            t += 0.45
        segments = build_sentences(words, max_words=6, max_seconds=4.0, pause_gap=0.7)
        self.assertGreaterEqual(len(segments), 2)
        self.assertTrue(all(len(s.words) <= 10 for s in segments))
        self.assertIn(".", segments[0].text_en)
        for seg in segments:
            self.assertLess(seg.start, seg.end)

    def test_long_run_is_split(self):
        from proxmax.models import Word

        words = [Word(start=i * 0.5, end=i * 0.5 + 0.45, word=f" word{i}") for i in range(40)]
        segments = build_sentences(words, max_words=8, max_seconds=6.0)
        self.assertGreaterEqual(len(segments), 4)
        self.assertTrue(all(len(s.words) <= 12 for s in segments))

    def test_long_run_splits_at_widest_pause(self):
        from proxmax.models import Word
        from proxmax.segmenter import build_sentences

        words, t = [], 0.0
        for i in range(24):
            gap = 0.9 if i == 12 else 0.05   # ঠিক মাঝখানে বড় বিরতি
            t += gap
            words.append(Word(start=t, end=t + 0.3, word=f"w{i}"))
            t += 0.3
        sentences = build_sentences(words, max_words=15, max_seconds=30.0)
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0].end, words[11].end)
        self.assertEqual(sentences[1].start, words[12].start)

    def test_empty_words(self):
        self.assertEqual(build_sentences([]), [])


class TestSttHelpers(unittest.TestCase):
    def test_tail_context(self):
        self.assertEqual(_tail_context("short text", 50), "short text")
        long = "First sentence here. Second sentence follows with words."
        tail = _tail_context(long, 20)
        self.assertLessEqual(len(tail), 20)
        self.assertTrue(long.endswith(tail))

    def test_find_suspicious_boundaries(self):
        from proxmax.models import Word

        chunks = [SpeechChunk(0.0, 10.0), SpeechChunk(10.0, 20.0)]
        cut = [Word(9.5, 10.4, " XY"), Word(10.4, 10.8, "Z")]
        self.assertTrue(find_suspicious_boundaries(cut, chunks))
        clean = [Word(8.0, 8.4, " hello"), Word(10.5, 10.9, " world")]
        self.assertEqual(find_suspicious_boundaries(clean, chunks), [])

    def test_merge_windows(self):
        self.assertEqual(merge_windows([5.0], 2.5, 28.0), [(2.5, 7.5)])
        merged = merge_windows([5.0, 6.0], 2.5, 28.0)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0], (2.5, 8.5))

    def test_replace_in_window(self):
        from proxmax.models import Word

        words = [
            Word(0.0, 0.5, " hello"),
            Word(9.8, 10.3, " split"),
            Word(10.3, 10.8, "word"),
            Word(20.0, 20.5, " end"),
        ]
        replacement = [Word(9.6, 10.6, " splitword")]
        result = replace_in_window(words, replacement, 9.0, 11.0)
        texts = [w.word.strip() for w in result]
        self.assertIn("splitword", texts)
        self.assertIn("hello", texts)
        self.assertIn("end", texts)
        self.assertNotIn("split", texts)

    def test_transcribe_audio_uses_rolling_prompt(self):
        from proxmax.models import Word

        audio = _speech_with_silence([(6.0, 0.6)] * 5)  # ~33 s
        prompts: list = []

        class _Fake(SpeechToText):
            def _load(self):  # মডেল ছাড়াই
                self.device, self.compute_type = "cpu", "int8"

            def _decode_words(self, samples, offset=0.0, initial_prompt=None, language="en"):
                prompts.append(initial_prompt)
                base = offset + 0.2
                words = [
                    Word(base, base + 0.4, " Hello"),
                    Word(base + 0.4, base + 0.9, " world."),
                    Word(base + 1.0, base + 1.4, " Next"),
                    Word(base + 1.4, base + 1.9, " line."),
                ]
                return words, " Hello world. Next line."

        stt = _Fake(
            model_size="tiny", chunk_min_seconds=8.0, chunk_max_seconds=20.0,
            vad_min_silence_ms=300, seam_repair=False, carry_context_chars=30,
        )
        segments = stt.transcribe_audio(audio, 16000)
        self.assertGreaterEqual(stt.last_stats["chunks"], 2)
        self.assertIsNone(prompts[0])                 # প্রথম chunk-এ প্রসঙ্গ নেই
        self.assertIsNotNone(prompts[1])              # পরের chunk-এ আগের লেখা যায়
        self.assertIn("world", prompts[1])
        self.assertTrue(segments)
        self.assertTrue(all(s.text_en for s in segments))
        self.assertTrue(all(s.end > s.start for s in segments))

    def test_hotwords_reach_whisper_and_unknown_kwarg_is_safe(self):
        calls: list = []

        class _Model:
            def __init__(self, accept_hotwords=True):
                self.accept = accept_hotwords

            def transcribe(self, samples, **kwargs):
                calls.append(dict(kwargs))
                if "hotwords" in kwargs and not self.accept:
                    raise TypeError("unexpected keyword argument 'hotwords'")
                return iter([]), None

        samples = np.ones(16000 * 2, dtype=np.float32) * 0.1
        stt = SpeechToText(model_size="tiny", hotwords="FC Mobile, star pass")
        stt._model = _Model(accept_hotwords=True)
        stt._decode_words(samples, 0.0, None, "en")
        self.assertEqual(calls[0].get("hotwords"), "FC Mobile, star pass")

        # পুরোনো faster-whisper: hotwords চেনে না → TypeError খেয়ে আবার চেষ্টা
        calls.clear()
        old = SpeechToText(model_size="tiny", hotwords="FC Mobile")
        old._model = _Model(accept_hotwords=False)
        old._decode_words(samples, 0.0, None, "en")
        self.assertEqual(len(calls), 2)
        self.assertNotIn("hotwords", calls[1])
        self.assertEqual(old.hotwords, "")

        large_inline = ", ".join(f"Custom Game Term {index:04d}" for index in range(300))
        bounded = SpeechToText(model_size="tiny", hotwords=large_inline)
        self.assertLessEqual(len(bounded.hotwords), 1600)

    def test_whisper_decode_reselects_hotwords_from_chunk_context(self):
        calls: list[dict] = []

        class _Model:
            def transcribe(self, samples, **kwargs):
                calls.append(dict(kwargs))
                return iter([]), None

        stt = SpeechToText(
            model_size="tiny", hotwords_file=str(REPO_ROOT / "hotwords.txt")
        )
        stt._model = _Model()
        stt._decode_words(
            np.ones(16000, dtype=np.float32) * 0.1,
            initial_prompt="FC Mobile 27: Aitana Bonmati is in the lineup.",
        )
        prompt = calls[0]["hotwords"]
        self.assertLessEqual(len(prompt), 1600)
        self.assertIn("Aitana Bonmati", prompt)
        self.assertIn("Game Plans", prompt)

    def test_video_name_is_used_as_first_prompt(self):
        from proxmax.models import Word

        audio = _speech_with_silence([(6.0, 0.6)] * 5)
        prompts: list = []

        class _Fake(SpeechToText):
            def _load(self):
                self.device, self.compute_type = "cpu", "int8"

            def _decode_words(self, samples, offset=0.0, initial_prompt=None, language="en"):
                prompts.append(initial_prompt)
                base = offset + 0.2
                return (
                    [Word(base, base + 0.5, " Squad"), Word(base + 0.5, base + 1.0, " builder.")],
                    " Squad builder.",
                )

        stt = _Fake(model_size="tiny", chunk_min_seconds=8.0, chunk_max_seconds=20.0,
                    seam_repair=False, carry_context_chars=40)
        stt.set_context_hint("FC_Mobile_125_OVR_squad.mp4")
        self.assertEqual(stt.context_hint, "FC Mobile 125 OVR squad")
        stt.transcribe_audio(audio, 16000)
        self.assertEqual(prompts[0], "FC Mobile 125 OVR squad")   # প্রথম chunk-এ নাম
        self.assertIn("builder", prompts[1])                      # পরে rolling প্রসঙ্গ

    def test_context_hint_never_breaks_empty_name(self):
        stt = SpeechToText(model_size="tiny")
        stt.set_context_hint("")
        self.assertEqual(stt.context_hint, "")

    def test_transcribe_audio_rejects_silence(self):
        class _Fake(SpeechToText):
            def _load(self):
                self.device, self.compute_type = "cpu", "int8"

            def _decode_words(self, *a, **k):  # pragma: no cover
                return [], ""

        stt = _Fake(model_size="tiny", chunk_min_seconds=5.0, chunk_max_seconds=10.0)
        with self.assertRaises(ValueError):
            stt.transcribe_audio(np.zeros(16000 * 5, dtype=np.float32), 16000)

    def test_release_without_model_is_safe(self):
        stt = SpeechToText(model_size="tiny")
        stt.release()
        self.assertIsNone(stt._model)


class TestWhisperModelPreCache(unittest.TestCase):
    """সেটআপ ধাপে Whisper large-v3 ক্যাশ হলে ডাবিংয়ের সময় আর ডাউনলোড হয় না।"""

    def test_ensure_whisper_model_caches_and_sets_env(self):
        from proxmax.stt import _resolve_cached_model_path, ensure_whisper_model

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "faster-whisper-large-v3"
            model_dir.mkdir()
            (model_dir / "model.bin").write_bytes(b"weights")
            with mock.patch.dict("os.environ", {}, clear=False):
                resolved = ensure_whisper_model(str(model_dir))
                self.assertEqual(Path(resolved), model_dir.resolve())
                self.assertEqual(
                    Path(_resolve_cached_model_path("large-v3")),
                    model_dir.resolve(),
                )
