"""Focused regression tests for ProXMax."""





from tests._support import *





class TestTTSResolution(unittest.TestCase):
    def test_resolve_uses_builtin_voice(self):
        self.assertEqual(resolve_tts_engine(AppConfig()), "edge")
        self.assertEqual(resolve_tts_engine(AppConfig(tts_engine="auto")), "edge")

    def test_edge_is_the_repo_default(self):
        try:
            import yaml  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("PyYAML not installed")
        cfg = AppConfig.from_yaml(REPO_ROOT / "config.yaml")
        self.assertEqual(cfg.tts_engine, "edge")

    def test_build_targets_builtin_engine(self):
        from proxmax.tts import build_tts_engine

        try:
            engine = build_tts_engine(AppConfig(tts_engine="edge"))
        except TTSError as exc:
            self.assertIn("edge-tts", str(exc))
            return
        self.assertIsInstance(engine, EdgeTTSEngine)


class TestMaleVoiceAndGaps(unittest.TestCase):
    """পুরুষ কণ্ঠ জোর করে নিশ্চিত করা + বাক্যের পরে অযথা ফাঁক বন্ধ।"""

    # ---- voice -------------------------------------------------------
    def test_default_voice_is_male(self):
        cfg = AppConfig()
        self.assertEqual(cfg.tts_gender, "male")
        self.assertEqual(cfg.resolved_tts_voice, "bn-BD-PradeepNeural")

    def test_gender_wins_over_stale_female_config(self):
        # পুরোনো config.yaml-এ নারী ভয়েস লেখা থাকলেও পুরুষই ব্যবহার হবে
        cfg = AppConfig(tts_gender="male", tts_voice="bn-BD-NabanitaNeural")
        self.assertEqual(cfg.resolved_tts_voice, "bn-BD-PradeepNeural")

    def test_explicit_custom_voice_is_honoured(self):
        cfg = AppConfig(tts_gender="custom", tts_voice="bn-BD-NabanitaNeural").validate()
        self.assertEqual(cfg.resolved_tts_voice, "bn-BD-NabanitaNeural")
        female = AppConfig(tts_gender="female").validate()
        self.assertEqual(female.resolved_tts_voice, "bn-BD-NabanitaNeural")

    def test_repo_yaml_uses_male_voice(self):
        cfg = AppConfig.from_yaml(Path(__file__).resolve().parents[1] / "config.yaml").validate()
        self.assertEqual(cfg.resolved_tts_voice, "bn-BD-PradeepNeural")

    # ---- TTS padding trim -------------------------------------------
    def test_tts_padding_is_trimmed(self):
        from proxmax.tts.base import BaseTTSEngine

        class _Engine(BaseTTSEngine):
            name = "test"

            def synthesize(self, segments, output_dir):  # pragma: no cover
                return segments

        engine = _Engine(trim_silence=True, trim_pad_ms=20)
        sr = 24000
        speech = 0.3 * np.sin(2 * np.pi * 220 * np.arange(int(1.0 * sr)) / sr)
        padded = np.concatenate([
            np.zeros(int(0.5 * sr), dtype=np.float32),
            speech.astype(np.float32),
            np.zeros(int(0.8 * sr), dtype=np.float32),
        ])
        final = engine._finalize(padded)
        self.assertLess(final.size, padded.size)
        self.assertAlmostEqual(final.size / sr, 1.04, delta=0.12)

    def test_trimming_can_be_disabled(self):
        from proxmax.tts.base import BaseTTSEngine

        class _Engine(BaseTTSEngine):
            name = "test"

            def synthesize(self, segments, output_dir):  # pragma: no cover
                return segments

        engine = _Engine(trim_silence=False)
        sr = 24000
        audio = np.zeros(int(1.5 * sr), dtype=np.float32)
        audio[int(0.4 * sr):int(0.9 * sr)] = 0.4
        self.assertEqual(engine._finalize(audio).size, audio.size)

    # ---- fill short clips by slowing --------------------------------
    def test_short_clip_is_slowed_to_fill_its_slot(self):
        # ২.০s কথা, ২.৫s স্লট → ১২% সীমা পর্যন্ত ধীর (২.২৪s), নীরবতা কমে
        tempo = compute_tempo(2.0, 2.5, fill_extension=0.12)
        self.assertLess(tempo, 1.0)
        self.assertGreaterEqual(tempo, 1.0 / 1.12 - 1e-6)
        self.assertAlmostEqual(2.0 / tempo, 2.24, places=2)

    def test_fill_never_exceeds_extension(self):
        # ৭s কথা, ১২s স্লট → ১২%-এর বেশি ধীর করা হয় না
        tempo = compute_tempo(7.0, 12.0, fill_extension=0.12)
        self.assertAlmostEqual(tempo, 1.0 / 1.12, places=4)

    def test_fill_can_be_disabled(self):
        self.assertEqual(compute_tempo(2.0, 2.5, fill_extension=0.0), 1.0)

    def test_no_fill_when_clip_is_longer_than_slot(self):
        self.assertGreater(compute_tempo(3.0, 2.4, fill_extension=0.12), 1.0)

    def test_gap_filler_does_not_use_multi_second_pause_after_speech(self):
        from proxmax.pipeline import DubbingPipeline

        class _Translator:
            def __init__(self):
                self.calls = 0

            def expand_to_fit(self, segments, targets):
                self.calls += 1
                return 0

        translator = _Translator()
        pipeline = DubbingPipeline.__new__(DubbingPipeline)
        pipeline.config = AppConfig().validate()
        pipeline.translator = translator
        pipeline._tts = None
        first = Segment(index=0, start=1.0, end=2.0, text_en="First line.", text_bn="Short.")
        first.tts_audio_path = Path("unused-first.wav")
        first.tts_duration = 1.0
        second = Segment(index=1, start=6.0, end=7.0, text_en="Second line.", text_bn="Also short.")
        second.tts_audio_path = Path("unused-second.wav")
        second.tts_duration = 1.0

        filled = pipeline._fill_short_slots([first, second], Path("unused"), total_duration=8.0)
        self.assertEqual(filled, 0)
        self.assertEqual(translator.calls, 0)


class TestEdgeVoiceResolution(unittest.TestCase):
    """edge-tts-এ ভয়েস না থাকলে নারী কণ্ঠে নেমে যাওয়া চলবে না — সমান লিঙ্গ।"""

    def _engine(self, voice="bn-BD-PradeepNeural", gender="male"):
        from proxmax.tts.edge_engine import EdgeTTSEngine

        return EdgeTTSEngine(voice=voice, gender=gender)

    def _with_voices(self, voices):
        from proxmax.tts import edge_engine

        patcher = mock.patch.dict(edge_engine._VOICE_LIST, {"voices": voices})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_existing_voice_is_kept(self):
        self._with_voices([
            {"ShortName": "bn-BD-PradeepNeural", "Gender": "Male", "Locale": "bn-BD"},
            {"ShortName": "bn-BD-NabanitaNeural", "Gender": "Female", "Locale": "bn-BD"},
        ])
        self.assertEqual(self._engine()._resolve_voice(), "bn-BD-PradeepNeural")

    def test_missing_voice_falls_back_to_same_gender(self):
        self._with_voices([
            {"ShortName": "bn-IN-Male", "Gender": "Male", "Locale": "bn-IN"},
            {"ShortName": "bn-BD-NabanitaNeural", "Gender": "Female", "Locale": "bn-BD"},
        ])
        self.assertEqual(self._engine("bn-BD-GoneNeural")._resolve_voice(), "bn-IN-Male")

    def test_no_same_gender_voice_raises_instead_of_female(self):
        from proxmax.tts.base import TTSError

        self._with_voices([
            {"ShortName": "bn-BD-NabanitaNeural", "Gender": "Female", "Locale": "bn-BD"},
        ])
        with self.assertRaises(TTSError):
            self._engine("bn-BD-GoneNeural")._resolve_voice()

    def test_female_request_keeps_female(self):
        self._with_voices([
            {"ShortName": "bn-BD-NabanitaNeural", "Gender": "Female", "Locale": "bn-BD"},
        ])
        self.assertEqual(self._engine("bn-BD-NabanitaNeural", "female")._resolve_voice(),
                         "bn-BD-NabanitaNeural")

    def test_unknown_network_keeps_configured_voice(self):
        self._with_voices([])
        self.assertEqual(self._engine()._resolve_voice(), "bn-BD-PradeepNeural")


class TestProsodyUnits(unittest.TestCase):
    """ছোট segment জোড়া লেগে এক clip-এ পুরো বাক্য হয় (মাঝপথে থামা বন্ধ)।"""

    @staticmethod
    def _seg(i, start, end, en, bn=""):
        return Segment(index=i, start=start, end=end, text_en=en, text_bn=bn)

    def test_mid_sentence_fragments_are_merged(self):
        from proxmax.prosody import build_units

        segments = [
            self._seg(0, 0.0, 2.0, "and give away one star pass"),
            self._seg(1, 2.1, 4.0, "to a lucky subscriber."),
        ]
        units = build_units(segments, text_attr="text_en")
        self.assertEqual(len(units), 1)
        self.assertTrue(units[0].is_unit)
        self.assertEqual(len(units[0].children), 2)
        self.assertIn("star pass", units[0].text_en)
        self.assertIn("lucky subscriber", units[0].text_en)

    def test_big_pause_keeps_separate_units(self):
        from proxmax.prosody import build_units

        segments = [
            self._seg(0, 0.0, 2.0, "This is one idea"),
            self._seg(1, 5.0, 7.0, "and this is another."),
        ]
        units = build_units(segments, text_attr="text_en", max_gap=0.6)
        self.assertEqual(len(units), 2)

    def test_limits_are_respected(self):
        from proxmax.prosody import build_units

        segments = [
            self._seg(0, 0.0, 5.0, "first long clause"),
            self._seg(1, 5.1, 10.0, "second long clause"),
            self._seg(2, 10.1, 15.0, "third long clause"),
        ]
        units = build_units(segments, text_attr="text_en", max_seconds=10.5)
        self.assertGreater(len(units), 1)
        for unit in units:
            self.assertLessEqual(unit.end - unit.start, 10.5 + 1e-6)

    def test_bangla_text_is_joined_for_tts(self):
        from proxmax.prosody import build_units

        segments = [
            self._seg(0, 0.0, 2.0, "a", bn="আমি একটি"),
            self._seg(1, 2.05, 4.0, "b", bn="স্টার পাস দেব।"),
        ]
        units = build_units(segments, text_attr="text_bn")
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].text_bn, "আমি একটি স্টার পাস দেব।")

    def test_subtitles_split_a_long_unit_and_keep_timing(self):
        from proxmax.prosody import subtitles_for_unit

        unit = Segment(index=0, start=10.0, end=20.0, text_bn="প্রথম বাক্য এখানে। দ্বিতীয় বাক্য এখানেও আছে।")
        unit.stretched_duration = 10.0
        unit.is_unit = True
        lines = subtitles_for_unit(unit, max_chars=20)
        self.assertGreaterEqual(len(lines), 2)
        self.assertAlmostEqual(lines[0].start, 10.0, places=2)
        self.assertAlmostEqual(lines[-1].end, 20.0, places=2)
        for line in lines:
            self.assertTrue(line.text_bn)
            self.assertLessEqual(line.start + 1e-6, line.end)

    def test_english_text_never_leaks_into_text_bn(self):
        from proxmax.prosody import build_units

        segments = [
            self._seg(0, 0.0, 2.0, "and give away one star pass"),
            self._seg(1, 2.05, 4.0, "to a lucky subscriber."),
        ]
        unit = build_units(segments, text_attr="text_en")[0]
        self.assertEqual(unit.text_bn, "")          # TTS যেন ইংরেজি না পড়ে
        self.assertIn("star pass", unit.text_en)

    def test_units_cover_all_children(self):
        from proxmax.prosody import build_units

        segments = [self._seg(i, i * 1.0, i * 1.0 + 0.9, f"line {i}") for i in range(6)]
        segments[-1].text_en = "last line."
        units = build_units(segments, text_attr="text_en")
        total_children = sum(len(u.children) for u in units)
        self.assertEqual(total_children, len(segments))
        self.assertEqual(units[0].start, 0.0)
        self.assertAlmostEqual(units[-1].end, segments[-1].end, places=3)


class TestSpeechRateCalibration(unittest.TestCase):
    """কণ্ঠের প্রকৃত গতি মেপে অনুবাদের বাজেট ঠিক হয় (১.২৫× তাড়াহুড়ো বন্ধ)।"""

    class _SlowEngine:
        def __init__(self, chars_per_second=12.0):
            self.rate = chars_per_second
            self.calls = 0

        def synthesize(self, segments, output_dir):
            self.calls += 1
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for seg in segments:
                seconds = max(0.4, len(seg.text_bn) / self.rate)
                path = output_dir / f"rate_{seg.index}.wav"
                t = np.arange(int(seconds * 24000)) / 24000.0
                audio_fx.write_wav(path, 0.2 * np.sin(2 * np.pi * 220 * t), 24000)
                seg.tts_audio_path = path
                seg.tts_duration = seconds
            return segments

    def test_measure_rate_recovers_engine_speed(self):
        from proxmax.speech_rate import measure_rate

        rate = measure_rate(self._SlowEngine(11.5))
        self.assertIsNotNone(rate)
        self.assertAlmostEqual(float(rate), 11.5, delta=0.6)

    def test_calibration_falls_back_when_engine_fails(self):
        from proxmax.speech_rate import calibrate

        class Broken:
            def synthesize(self, *_args, **_kwargs):
                raise RuntimeError("no network")

        self.assertEqual(calibrate(Broken(), fallback=12.0), 12.0)

    def test_translator_accepts_measured_rate(self):
        translator = create_translator("gemini", gemini_api_key="test-key")
        try:
            translator.set_char_rate(11.4)
            self.assertAlmostEqual(translator.char_rate, 11.4, places=2)
            payload = translator._payload([("hello", 2.0)])
            data = json.loads(payload)
            self.assertEqual(data["lines"][0]["d"], 2.0)
            self.assertEqual(data["lines"][0]["c"], int(round(2.0 * 11.4)))
            translator.set_char_rate(999)  # অবাস্তব মান উপেক্ষা করা হয়
            self.assertAlmostEqual(translator.char_rate, 11.4, places=2)
        finally:
            translator.close()
