"""Focused regression tests for ProXMax."""





from tests._support import *





class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = AppConfig()
        self.assertEqual(cfg.whisper_model, "large-v3")
        self.assertEqual(cfg.translator_backend, "gemini")
        self.assertEqual(cfg.gemini_model, "gemini-3.5-flash-lite")
        self.assertEqual(cfg.gemini_max_retries, 100)
        self.assertEqual(
            (cfg.gemini_retry_base_delay, cfg.gemini_retry_max_delay, cfg.gemini_retry_cap_delay),
            (1.0, 1.0, 1.0),
        )
        self.assertEqual(cfg.gemini_max_input_tokens, 5000)
        self.assertEqual(cfg.gemini_batch_size, 12)
        self.assertEqual(cfg.tts_unit_max_seconds, 5.0)
        self.assertFalse(cfg.tts_trim_silence)
        self.assertEqual(cfg.tts_engine, "edge")
        self.assertEqual(cfg.stt_hotwords_file, "hotwords.txt")
        self.assertTrue(cfg.save_report)
        self.assertFalse(cfg.validate_dub_audio)
        self.assertAlmostEqual(cfg.audio_validation_min_coverage, 0.58)
        self.assertEqual((cfg.min_tempo, cfg.max_tempo), (0.90, 1.15))
        self.assertLess(cfg.chunk_min_seconds, cfg.chunk_max_seconds)
        self.assertLessEqual(cfg.chunk_max_seconds, 29.5)
        self.assertTrue(cfg.input_dir.endswith("Video/input"))
        self.assertTrue(cfg.output_dir.endswith("Video/output"))
        cfg.validate()

    def test_repo_yaml_matches_code(self):
        try:
            import yaml  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("PyYAML not installed")
        cfg = AppConfig.from_yaml(REPO_ROOT / "config.yaml").validate()
        self.assertEqual(cfg.whisper_model, "large-v3")
        self.assertEqual(cfg.gemini_model, DEFAULT_GEMINI_MODEL)
        self.assertEqual(cfg.gemini_max_retries, 100)
        self.assertEqual(
            (cfg.gemini_retry_base_delay, cfg.gemini_retry_max_delay, cfg.gemini_retry_cap_delay),
            (1.0, 1.0, 1.0),
        )
        self.assertEqual(cfg.gemini_max_input_tokens, 5000)
        self.assertEqual(cfg.gemini_batch_size, 12)
        self.assertEqual(cfg.tts_unit_max_seconds, 5.0)
        self.assertFalse(cfg.tts_trim_silence)
        self.assertEqual(cfg.tts_engine, "edge")
        self.assertEqual(cfg.stt_hotwords_file, "hotwords.txt")
        self.assertTrue(cfg.save_report)
        self.assertFalse(cfg.validate_dub_audio)
        self.assertAlmostEqual(cfg.audio_validation_min_coverage, 0.58)
        self.assertAlmostEqual(cfg.max_tempo, 1.15)
        self.assertAlmostEqual(cfg.min_tempo, 0.90)
        self.assertGreaterEqual(cfg.carry_context_chars, 100)
        self.assertTrue(cfg.seam_repair)
        self.assertGreater(cfg.chunk_min_seconds, 10)

    def test_input_token_budget_accepts_presets_and_custom_values(self):
        for budget in (1000, 2000, 5000, 10000, 20000, 12345):
            with self.subTest(budget=budget):
                self.assertEqual(AppConfig(gemini_max_input_tokens=budget).validate().gemini_max_input_tokens, budget)

    def test_gemini_batch_size_is_capped_for_alignment_safety(self):
        self.assertEqual(AppConfig(gemini_batch_size=2048).validate().gemini_batch_size, 12)

    def test_section_scoped_aliases(self):
        cfg = AppConfig()
        cfg._apply_mapping(
            {
                "stt": {"model": "medium", "chunk_min_seconds": 20.0, "hotwords_file": "custom_hw.txt"},
                "translation": {"backend": "gemini", "max_retries": 6},
                "gemini": {"model": "gemini-test-model", "api_key": "test-key"},
                "tts": {"voice": "bn-BD-PradeepNeural", "max_retries": 2, "engine": "edge"},
                "sync": {"max_tempo": 1.15, "workers": 2},
                "runtime": {"release_gpu_when_done": False, "enable_overlap": False},
                "output": {"save_report": False},
                "quality": {"validate_dub_audio": False, "min_text_coverage": 0.72},
            }
        )
        self.assertEqual(cfg.whisper_model, "medium")
        self.assertEqual(cfg.chunk_min_seconds, 20.0)
        self.assertEqual(cfg.stt_hotwords_file, "custom_hw.txt")
        self.assertEqual(cfg.gemini_model, "gemini-test-model")
        self.assertEqual(cfg.gemini_max_retries, 6)
        self.assertEqual(cfg.tts_max_retries, 2)
        self.assertEqual(cfg.tts_engine, "edge")
        self.assertEqual(cfg.tts_voice, "bn-BD-PradeepNeural")
        self.assertEqual(cfg.sync_workers, 2)
        self.assertAlmostEqual(cfg.max_tempo, 1.15)
        self.assertFalse(cfg.release_gpu_when_done)
        self.assertFalse(cfg.enable_overlap)
        self.assertFalse(cfg.save_report)
        self.assertFalse(cfg.validate_dub_audio)
        self.assertAlmostEqual(cfg.audio_validation_min_coverage, 0.72)

    def test_unknown_keys_are_ignored(self):
        cfg = AppConfig()
        cfg._apply_mapping({"stt": {"batch_size": 8, "nonsense": 1}})
        self.assertFalse(hasattr(cfg, "whisper_batch_size"))
        self.assertEqual(cfg.whisper_model, "large-v3")

    def test_overrides_ignore_none(self):
        cfg = AppConfig().apply_overrides(
            {"whisper_model": "medium", "tts_voice": None, "tts_engine": "edge"}
        )
        self.assertEqual(cfg.whisper_model, "medium")
        self.assertEqual(cfg.tts_engine, "edge")
        self.assertTrue(cfg.resolved_tts_voice.startswith("bn-BD-"))

    def test_invalid_settings_are_rejected(self):
        cases = [
            {"whisper_model": "not-a-model"},
            {"gemini_model": "   "},
            {"gemini_max_retries": 0},
            {"gemini_max_retries": 101},
            {"gemini_max_input_tokens": 999},
            {"max_tempo": 1.16},
            {"min_tempo": 0.89},
            {"translator_backend": "google"},
            {"translator_backend": "deepl"},
            {"tts_engine": "siri"},
            {"tts_engine": "indicf5"},
            {"chunk_max_seconds": 45.0},
            {"chunk_min_seconds": 40.0},
            {"min_tempo": 1.2, "max_tempo": 1.2},
            {"sync_sample_rate": 4000},
            {"audio_validation_min_coverage": 1.1},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    AppConfig(**kwargs).validate()

    def test_work_dir_resolution(self):
        cfg = AppConfig(work_dir="/tmp/proxmax-test-work")
        self.assertEqual(str(cfg.resolved_work_dir), "/tmp/proxmax-test-work")


class TestDiscovery(unittest.TestCase):
    def test_filters_and_sorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in ["b.mp4", "A.MKV", "c.txt", ".hidden.mp4", "d.mov"]:
                (folder / name).write_bytes(b"x")
            (folder / "empty.mp4").write_bytes(b"")
            found = [p.name for p in discover_videos(folder)]
            self.assertEqual(found, ["A.MKV", "b.mp4", "d.mov"])

    def test_missing_folder_raises(self):
        with self.assertRaises(FileNotFoundError):
            discover_videos("/definitely/not/a/real/folder")
