"""Smoke tests for ProXMax — no GPU / no network required.

চালানো:  ``python -m unittest discover -s tests -v``

ভারী মডেল (Whisper, IndicF5), Gemini API ও edge-tts ছোঁয়া হয় না — শুধু
নিজেদের লজিক পরীক্ষা করা হয়: config, chunking, sentence building,
ভয়েস প্রোফাইল, sync-এর non-overlap হিসাব, translation batching/retry,
TTS governor এবং পুরো pipeline-এর সংযোগ। ffmpeg থাকলে শেষে একটি আসল
ছোট্ট ভিডিও বানিয়ে end-to-end ডাবিংও পরীক্ষা করা হয়।
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proxmax import audio_fx                                         # noqa: E402
from proxmax.chunking import (                                       # noqa: E402
    SpeechChunk,
    energy_regions,
    plan_chunks,
    regions_to_chunks,
)
from proxmax.config import (                                          # noqa: E402
    DEFAULT_GEMINI_MODEL,
    AppConfig,
)
from proxmax.discovery import discover_videos                          # noqa: E402
from proxmax.models import Segment, StageTimings                       # noqa: E402
from proxmax.segmenter import build_sentences                          # noqa: E402
from proxmax.stt import (                                              # noqa: E402
    SpeechToText,
    _tail_context,
    find_suspicious_boundaries,
    merge_windows,
    replace_in_window,
)
from proxmax.sync_engine import (                                      # noqa: E402
    SyncEngine,
    build_atempo_filter,
    compute_tempo,
    find_overlaps,
    plan_windows,
)
from proxmax.text_utils import (                                       # noqa: E402
    estimate_speech_seconds,
    normalise_bangla,
    prepare_for_speech,
    split_sentences,
    verbalize_numbers,
)
from proxmax.translator import (                                       # noqa: E402
    GeminiAuthError,
    GeminiTranslator,
    TranslationError,
    create_translator,
    parse_translations,
)
from proxmax.tts import (                                              # noqa: E402
    TTSError,
    EdgeTTSEngine,
    IndicF5Engine,
    SpeedGovernor,
    resolve_tts_engine,
)
from proxmax.tts.governor import SpeedGovernor as _Governor            # noqa: E402
from proxmax.utils import (                                            # noqa: E402
    format_srt_timestamp,
    output_filename_for,
    release_cuda_memory,
)
from proxmax.voice_profile import (                                    # noqa: E402
    VoiceProfileBuilder,
    discover_reference_audio,
    select_reference_window,
)

FFMPEG = shutil.which("ffmpeg")


# ===========================================================================
# Config
# ===========================================================================
class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = AppConfig()
        self.assertEqual(cfg.whisper_model, "large-v3")
        self.assertEqual(cfg.translator_backend, "gemini")
        self.assertEqual(cfg.gemini_model, "gemini-3.5-flash-lite")
        self.assertEqual(cfg.gemini_max_retries, 7)
        self.assertEqual(cfg.tts_engine, "auto")
        self.assertEqual((cfg.min_tempo, cfg.max_tempo), (0.95, 1.25))
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
        self.assertEqual(cfg.gemini_max_retries, 7)
        self.assertEqual(cfg.tts_engine, "auto")
        self.assertEqual(cfg.tts_indicf5_model, "ai4bharat/IndicF5")
        self.assertAlmostEqual(cfg.max_tempo, 1.25)
        self.assertAlmostEqual(cfg.min_tempo, 0.95)
        self.assertTrue(cfg.tts_fallback_to_edge)
        self.assertGreaterEqual(cfg.carry_context_chars, 100)
        self.assertTrue(cfg.seam_repair)
        self.assertGreater(cfg.chunk_min_seconds, 10)

    def test_section_scoped_aliases(self):
        cfg = AppConfig()
        cfg._apply_mapping(
            {
                "stt": {"model": "medium", "chunk_min_seconds": 20.0},
                "translation": {"backend": "gemini", "max_retries": 6},
                "gemini": {"model": "gemini-test-model", "api_key": "test-key"},
                "tts": {"voice": "bn-BD-PradeepNeural", "max_retries": 2, "engine": "edge"},
                "voice": {"target_seconds": 12.0, "highpass_hz": 80},
                "sync": {"max_tempo": 1.3, "workers": 2},
                "runtime": {"release_gpu_when_done": False, "enable_overlap": False},
            }
        )
        self.assertEqual(cfg.whisper_model, "medium")
        self.assertEqual(cfg.chunk_min_seconds, 20.0)
        self.assertEqual(cfg.gemini_model, "gemini-test-model")
        self.assertEqual(cfg.gemini_max_retries, 6)
        self.assertEqual(cfg.tts_max_retries, 2)
        self.assertEqual(cfg.tts_engine, "edge")
        self.assertEqual(cfg.tts_voice, "bn-BD-PradeepNeural")
        self.assertEqual(cfg.voice_target_seconds, 12.0)
        self.assertEqual(cfg.sync_workers, 2)
        self.assertAlmostEqual(cfg.max_tempo, 1.3)
        self.assertFalse(cfg.release_gpu_when_done)
        self.assertFalse(cfg.enable_overlap)

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
        self.assertTrue(cfg.tts_voice.startswith("bn-BD-"))

    def test_invalid_settings_are_rejected(self):
        cases = [
            {"whisper_model": "not-a-model"},
            {"gemini_model": "   "},
            {"gemini_max_retries": 0},
            {"translator_backend": "google"},
            {"translator_backend": "deepl"},
            {"tts_engine": "siri"},
            {"chunk_max_seconds": 45.0},
            {"chunk_min_seconds": 40.0},
            {"min_tempo": 1.2, "max_tempo": 1.2},
            {"tts_nfe_steps": 8, "tts_min_nfe_steps": 16},
            {"voice_target_seconds": 2.0},
            {"sync_sample_rate": 4000},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    AppConfig(**kwargs).validate()

    def test_work_dir_resolution(self):
        cfg = AppConfig(work_dir="/tmp/proxmax-test-work")
        self.assertEqual(str(cfg.resolved_work_dir), "/tmp/proxmax-test-work")
        self.assertTrue(str(cfg.voice_profile_dir).endswith("voice_profile"))


# ===========================================================================
# Discovery / utils / sync math
# ===========================================================================
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


class TestUtilsAndSyncMath(unittest.TestCase):
    def test_output_filename(self):
        self.assertEqual(output_filename_for("lecture1.mp4"), "lecture1_bn.mp4")
        self.assertEqual(output_filename_for("/x/y/movie.mkv"), "movie_bn.mp4")

    def test_srt_timestamp(self):
        self.assertEqual(format_srt_timestamp(0), "00:00:00,000")
        self.assertEqual(format_srt_timestamp(61.234), "00:01:01,234")
        self.assertEqual(format_srt_timestamp(3661.5), "01:01:01,500")

    def test_atempo_chain(self):
        self.assertEqual(build_atempo_filter(1.0), "atempo=1.0000")
        self.assertEqual(build_atempo_filter(2.5), "atempo=2.0000,atempo=1.2500")
        self.assertEqual(build_atempo_filter(0.4), "atempo=0.5000,atempo=0.8000")

    def test_compute_tempo_caps(self):
        # বাংলা দ্বিগুণ লম্বা → 2.0 চাই, কিন্তু 1.25-এ আটকাবে (রোবটিক শোনা বন্ধ)
        self.assertAlmostEqual(compute_tempo(10.0, 5.0, max_tempo=1.25), 1.25)
        # ছোট ক্লিপ ডিফল্টে ধীর করা হয় না
        self.assertAlmostEqual(
            compute_tempo(2.5, 5.0, min_tempo=0.95, slow_down_short_clips=False), 1.0
        )
        self.assertAlmostEqual(compute_tempo(2.5, 5.0, slow_down_short_clips=True), 0.95)
        self.assertAlmostEqual(compute_tempo(5.0, 5.0), 1.0)
        self.assertAlmostEqual(compute_tempo(0.0, 5.0), 1.0)

    def test_plan_windows_never_overlap(self):
        segments = [
            Segment(index=0, start=0.0, end=3.0, tts_duration=6.0),  # অনেক লম্বা
            Segment(index=1, start=2.5, end=5.0, tts_duration=1.0),
            Segment(index=2, start=5.2, end=8.0, tts_duration=3.0),
        ]
        windows = plan_windows(segments, total_duration=9.0, min_gap_seconds=0.08)
        self.assertEqual(find_overlaps(windows), [])
        for (start, end), seg in zip(windows, segments):
            self.assertGreaterEqual(start, 0.0)
            self.assertGreater(end, start)
            self.assertAlmostEqual(start, seg.start)
        # পরের ক্লিপ শুরুর আগেই আগের উইন্ডো শেষ
        self.assertLessEqual(windows[0][1], windows[1][0])
        self.assertLessEqual(windows[1][1], windows[2][0])

    def test_find_overlaps_flags_bad_spans(self):
        self.assertEqual(find_overlaps([(0.0, 1.0), (1.0, 2.0)]), [])
        self.assertTrue(find_overlaps([(0.0, 1.5), (1.0, 2.0)]))

    def test_release_cuda_memory_is_safe_without_gpu(self):
        release_cuda_memory()


# ===========================================================================
# audio_fx / text_utils
# ===========================================================================
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

    def test_estimate_and_split(self):
        self.assertGreater(estimate_speech_seconds("এটি একটি পরীক্ষা।"), 0.4)
        self.assertEqual(estimate_speech_seconds(""), 0.0)
        parts = split_sentences("প্রথম বাক্য। দ্বিতীয় বাক্য! তৃতীয়?")
        self.assertEqual(len(parts), 3)


# ===========================================================================
# Chunking / segmenter — শব্দের মাঝখানে কাটা যাবে না
# ===========================================================================
def _speech_with_silence(
    pattern: list[tuple[float, float]], sample_rate: int = 16000
) -> np.ndarray:
    """Build a waveform from ``[(speech_seconds, silence_seconds), ...]``."""
    pieces = []
    for speech, gap in pattern:
        t = np.arange(int(speech * sample_rate)) / sample_rate
        pieces.append((0.25 * np.sin(2 * np.pi * 180 * t)).astype(np.float32))
        pieces.append(np.zeros(int(gap * sample_rate), dtype=np.float32))
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


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

    def test_empty_words(self):
        self.assertEqual(build_sentences([]), [])


# ===========================================================================
# Voice profile (helpers only — মডেল ছাড়া)
# ===========================================================================
class TestVoiceProfileHelpers(unittest.TestCase):
    def test_discover_reference_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bigger = root / "voice.mp3"
            bigger.write_bytes(b"x" * 500)
            (root / "smaller.wav").write_bytes(b"x" * 10)
            (root / "clip.mp4").write_bytes(b"x" * 9999)
            found = discover_reference_audio(root)
            self.assertEqual(found, bigger)
            explicit = root / "smaller.wav"
            self.assertEqual(discover_reference_audio(root, explicit=explicit), explicit)

    def test_discover_reference_audio_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(discover_reference_audio(tmp))

    def test_select_reference_window_prefers_contiguous_speech(self):
        regions = [(0.0, 2.0), (2.5, 4.0), (20.0, 26.0), (30.0, 32.0)]
        first, last, speech = select_reference_window(
            regions, target_seconds=8.0, min_seconds=3.0, max_seconds=20.0
        )
        self.assertLessEqual(first, last)
        span = regions[last][1] - regions[first][0]
        self.assertGreaterEqual(speech, 3.0)
        self.assertLessEqual(speech, span + 1e-6)
        self.assertAlmostEqual(
            speech, sum(regions[k][1] - regions[k][0] for k in range(first, last + 1)),
            places=3,
        )

    def test_builder_params_include_audio_cleanup(self):
        builder = VoiceProfileBuilder(
            cache_dir="/tmp/x", highpass_hz=70, loudnorm_i=-16.0
        )
        params = builder._params()
        self.assertEqual(params["hp"], 70)
        self.assertAlmostEqual(params["ln"], -16.0)
        self.assertIn("v", params)
        self.assertIn("highpass=f=70", builder._decode_filters())
        self.assertIn("loudnorm", builder._decode_filters())

    def test_builder_without_cleanup_has_no_filters(self):
        builder = VoiceProfileBuilder(cache_dir="/tmp/x")
        self.assertIsNone(builder._decode_filters())


# ===========================================================================
# STT helpers — seam repair ও rolling prompt
# ===========================================================================
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


# ===========================================================================
# TTS — governor ও ইঞ্জিনের নিরাপত্তা
# ===========================================================================
class _ProfileStub:
    """Minimal stand-in for :class:`VoiceProfile` (no audio file needed)."""

    def __init__(self, valid=True):
        self._valid = valid

    def is_valid(self):
        return self._valid


class TestTTSResolution(unittest.TestCase):
    def test_auto_without_reference_uses_builtin_voice(self):
        self.assertEqual(resolve_tts_engine(AppConfig(), None), "edge")

    def test_auto_with_invalid_reference_uses_builtin_voice(self):
        profile = _ProfileStub(valid=False)
        self.assertEqual(resolve_tts_engine(AppConfig(), profile), "edge")

    def test_auto_with_reference_clones(self):
        profile = _ProfileStub(valid=True)
        self.assertEqual(resolve_tts_engine(AppConfig(), profile), "indicf5")

    def test_explicit_engines_win(self):
        self.assertEqual(
            resolve_tts_engine(AppConfig(tts_engine="edge"), _ProfileStub()), "edge"
        )
        self.assertEqual(
            resolve_tts_engine(AppConfig(tts_engine="indicf5"), None), "indicf5"
        )

    def test_hf_token_available_reads_environment(self):
        from proxmax.tts import hf_token_available

        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.get_token = lambda: None  # type: ignore[attr-defined]
        with mock.patch.dict("os.environ", {}, clear=True), \
                mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            self.assertFalse(hf_token_available())
        with mock.patch.dict("os.environ", {"HF_TOKEN": "hf_test"}, clear=True):
            self.assertTrue(hf_token_available())

    def test_explicit_indicf5_without_profile_falls_back(self):
        from proxmax.tts import build_tts_engine

        try:
            engine = build_tts_engine(
                AppConfig(tts_engine="indicf5"), profile=None
            )
        except TTSError as exc:
            # edge-tts ইনস্টল না থাকলে এখানেই থামবে (IndicF5 নয়)
            self.assertIn("edge-tts", str(exc))
            return
        self.assertIsInstance(engine, EdgeTTSEngine)

    def test_auto_is_the_repo_default(self):
        try:
            import yaml  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("PyYAML not installed")
        cfg = AppConfig.from_yaml(REPO_ROOT / "config.yaml")
        self.assertEqual(cfg.tts_engine, "auto")

    def test_build_auto_without_reference_targets_builtin_engine(self):
        from proxmax.tts import build_tts_engine

        try:
            engine = build_tts_engine(AppConfig(tts_engine="auto"), profile=None)
        except TTSError as exc:
            # edge-tts ইনস্টল না থাকলে ঠিক এই path-এই ব্যর্থ হবে (IndicF5 নয়)
            self.assertIn("edge-tts", str(exc))
            return
        self.assertIsInstance(engine, EdgeTTSEngine)

    def test_auto_mode_still_looks_for_a_reference_sample(self):
        # auto = নমুনা থাকলে ক্লোনিং; তাই নমুনা খোঁজা বন্ধ করা যাবে না
        from proxmax.pipeline import DubbingPipeline

        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            out_dir = Path(tmp) / "out"
            in_dir.mkdir()
            out_dir.mkdir()
            pipeline = DubbingPipeline(
                AppConfig(input_dir=in_dir, output_dir=out_dir, gemini_api_key="test-key")
            )
            try:
                with mock.patch(
                    "proxmax.pipeline.hf_token_available", return_value=True
                ), mock.patch(
                    "proxmax.pipeline.discover_reference_audio", return_value=None
                ) as finder:
                    self.assertIsNone(pipeline._ensure_profile())
                self.assertEqual(finder.call_count, 1)
            finally:
                pipeline.close()

    def test_explicit_edge_skips_the_reference_lookup(self):
        from proxmax.pipeline import DubbingPipeline

        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            out_dir = Path(tmp) / "out"
            in_dir.mkdir()
            out_dir.mkdir()
            pipeline = DubbingPipeline(
                AppConfig(
                    input_dir=in_dir,
                    output_dir=out_dir,
                    gemini_api_key="test-key",
                    tts_engine="edge",
                )
            )
            try:
                with mock.patch(
                    "proxmax.pipeline.discover_reference_audio", return_value=None
                ) as finder:
                    self.assertIsNone(pipeline._ensure_profile())
                self.assertEqual(finder.call_count, 0)
            finally:
                pipeline.close()

    def test_missing_hf_token_skips_cloning_and_keeps_going(self):
        # টোকেন না দিলে = ব্যবহারকারী ক্লোনিং চান না → নমুনা খোঁজাও বন্ধ, কাজ চলবে
        from proxmax.pipeline import DubbingPipeline

        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            out_dir = Path(tmp) / "out"
            in_dir.mkdir()
            out_dir.mkdir()
            pipeline = DubbingPipeline(
                AppConfig(input_dir=in_dir, output_dir=out_dir, gemini_api_key="test-key")
            )
            try:
                with mock.patch(
                    "proxmax.pipeline.hf_token_available", return_value=False
                ), mock.patch(
                    "proxmax.pipeline.discover_reference_audio", return_value=None
                ) as finder:
                    self.assertIsNone(pipeline._ensure_profile())
                self.assertEqual(finder.call_count, 0)
            finally:
                pipeline.close()


class TestFFmpegSafety(unittest.TestCase):
    def test_nostdin_is_always_added(self):
        from proxmax.utils import ffmpeg_args

        self.assertEqual(ffmpeg_args(["ffmpeg", "-i", "in.mp4"]),
                         ["ffmpeg", "-nostdin", "-i", "in.mp4"])
        # ইতিমধ্যে থাকলে দুবার যোগ হয় না
        self.assertEqual(ffmpeg_args(["ffmpeg", "-nostdin", "-i", "x"]).count("-nostdin"), 1)

    def test_hanging_ffmpeg_is_killed_instead_of_blocking(self):
        from proxmax.utils import FFmpegError, run_ffmpeg_with_progress

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "hang_ffmpeg"
            script.write_text(
                "#!/bin/sh\n"
                "echo out_time_ms=1000000\n"
                "echo progress=continue\n"
                "sleep 30\n",
                encoding="utf-8",
            )
            script.chmod(0o755)
            started = time.monotonic()
            with self.assertRaises(FFmpegError) as ctx:
                run_ffmpeg_with_progress(
                    [str(script), "-i", "x"], "test", duration_seconds=10,
                    stall_timeout=2,
                )
            self.assertLess(time.monotonic() - started, 15)
            self.assertIn("অগ্রগতি", str(ctx.exception))


class TestTTSGovernor(unittest.TestCase):
    def test_downgrades_when_over_budget(self):
        gov = _Governor(nfe_steps=16, min_nfe_steps=8, time_budget_seconds=5.0,
                        total_audio_seconds=60.0)
        self.assertEqual(gov.current_nfe, 16)
        gov.observe(audio_seconds=2.0, elapsed_seconds=2.0)   # warm-up
        self.assertEqual(gov.current_nfe, 16)
        gov.observe(audio_seconds=2.0, elapsed_seconds=4.0)
        self.assertLess(gov.current_nfe, 16)
        self.assertIn("nfe=", gov.summary())

    def test_without_budget_nfe_stays(self):
        gov = _Governor(nfe_steps=16, min_nfe_steps=8, time_budget_seconds=None)
        for _ in range(5):
            gov.observe(audio_seconds=1.0, elapsed_seconds=10.0)
        self.assertEqual(gov.current_nfe, 16)

    def test_new_budget_resets_counters_but_keeps_learning(self):
        gov = _Governor(nfe_steps=16, min_nfe_steps=8, time_budget_seconds=1.0,
                        total_audio_seconds=50.0)
        gov.observe(2.0, 2.0)
        gov.observe(2.0, 4.0)
        lowered = gov.current_nfe
        gov.set_budget(100.0, 20.0)
        self.assertEqual(gov.synthesized_seconds, 0.0)
        self.assertEqual(gov.current_nfe, lowered)

    def test_speed_governor_exported(self):
        self.assertIs(SpeedGovernor, _Governor)


class TestTTSEngines(unittest.TestCase):
    def test_indicf5_requires_reference(self):
        engine = IndicF5Engine()
        with self.assertRaises(TTSError):
            engine.prepare(None)
        engine.close()

    def test_set_budget_is_safe(self):
        engine = IndicF5Engine()
        engine.set_budget(100.0, 50.0)          # মডেল লোড না করেই
        self.assertEqual(engine.governor.time_budget_seconds, 100.0)
        engine.close()

    def test_postprocess_fades_edges(self):
        engine = IndicF5Engine()
        audio = np.ones(24000, dtype=np.float32) * 0.8
        processed = engine._postprocess(audio)
        self.assertLess(abs(float(processed[0])), 0.1)
        self.assertLess(abs(float(processed[-1])), 0.1)
        self.assertLessEqual(float(np.max(np.abs(processed))), 1.0)


# ===========================================================================
# Gemini translator (fake client — কোনো নেটওয়ার্ক নেই)
# ===========================================================================
class _FakeResponse:
    def __init__(self, text, block_reason=None):
        self._text = text
        self.prompt_feedback = type("FB", (), {"block_reason": block_reason})()

    @property
    def text(self):
        if self._text is None:
            raise RuntimeError("no candidates")
        return self._text


class _FakeClient:
    def __init__(self, responder):
        self.responder = responder
        self.calls = []
        self.closed = 0
        self.models = self

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return _FakeResponse(self.responder(contents, len(self.calls)))

    def close(self):
        self.closed += 1


def _lines_from_contents(contents: str) -> list:
    payload = contents.split("\n", 1)[1]
    return [line["t"] for line in json.loads(payload)["lines"]]


def _make_translator(client, **kwargs):
    defaults = dict(
        api_key="test-key", model="gemini-test", batch_size=25,
        max_retries=3, retry_base_delay=0.01, retry_max_delay=0.02,
        requests_per_minute=6000, client=client,
    )
    defaults.update(kwargs)
    return GeminiTranslator(**defaults)


def _segments(texts):
    return [
        Segment(index=i, start=i * 2.0, end=i * 2.0 + 1.8, text_en=text)
        for i, text in enumerate(texts)
    ]


class TestTranslatorParsing(unittest.TestCase):
    def test_parse_plain_json(self):
        raw = json.dumps({"translations": ["এক", "দুই"]})
        self.assertEqual(parse_translations(raw, 2), ["এক", "দুই"])

    def test_parse_fenced_and_list(self):
        self.assertEqual(parse_translations("```json\n[\"এক\"]\n```", 1), ["এক"])

    def test_parse_object_items(self):
        raw = json.dumps({"translations": [{"bn": "এক"}, {"text": "দুই"}]})
        self.assertEqual(parse_translations(raw, 2), ["এক", "দুই"])

    def test_parse_count_mismatch(self):
        with self.assertRaises(TranslationError):
            parse_translations(json.dumps({"translations": ["এক"]}), 2)

    def test_parse_invalid_json(self):
        with self.assertRaises(TranslationError):
            parse_translations("not json", 1)


class TestTranslatorCalls(unittest.TestCase):
    def test_batches_and_dedupe(self):
        def responder(contents, _n):
            return json.dumps(
                {"translations": [f"বাংলা: {t}" for t in _lines_from_contents(contents)]}
            )

        client = _FakeClient(responder)
        translator = _make_translator(client, batch_size=2)
        segments = _segments(["One.", "Two.", "One.", "Three.", "Four."])
        translator.translate_segments(segments)
        self.assertEqual(segments[0].text_bn, segments[2].text_bn)   # dedupe
        self.assertTrue(all(s.text_bn.startswith("বাংলা:") for s in segments))
        self.assertEqual(len(client.calls), 2)  # ৪টি অনন্য লাইন, ব্যাচ ২

    def test_empty_texts_are_not_sent(self):
        client = _FakeClient(lambda c, n: json.dumps({"translations": []}))
        translator = _make_translator(client)
        segments = _segments(["", "  "])
        translator.translate_segments(segments)
        self.assertEqual(client.calls, [])

    def test_network_error_is_retried(self):
        connect_error = type("ConnectError", (Exception,), {})
        connect_error.__module__ = "httpx"

        def responder(contents, call_no):
            if call_no == 1:
                raise connect_error("connection refused")
            return json.dumps({"translations": ["বাংলা"]})

        client = _FakeClient(responder)
        translator = _make_translator(client, max_retries=3)
        with mock.patch("proxmax.translator.time.sleep"):
            result = translator.translate_text("Hello")
        self.assertEqual(result, "বাংলা")
        self.assertEqual(len(client.calls), 2)

    def test_all_retries_fail_is_graceful(self):
        def responder(contents, _n):
            raise TimeoutError("request timed out")

        client = _FakeClient(responder)
        translator = _make_translator(client, max_retries=2)
        with mock.patch("proxmax.translator.time.sleep"):
            with self.assertRaises(TranslationError):
                translator.translate_text("Hello")
        self.assertEqual(len(client.calls), 2)

    def test_blocked_content_is_not_retried(self):
        class _Blocked(_FakeClient):
            def generate_content(self, *, model, contents, config):
                self.calls.append({"model": model, "contents": contents, "config": config})
                return _FakeResponse(None, block_reason="SAFETY")

        client = _Blocked(lambda c, n: "")
        translator = _make_translator(client, max_retries=5)
        with mock.patch("proxmax.translator.time.sleep"):
            with self.assertRaises(TranslationError):
                translator.translate_text("Hello")
        self.assertEqual(len(client.calls), 1)
        self.assertIsNone(translator._fatal_error)

    def test_shorten_to_fit(self):
        def responder(contents, _n):
            return json.dumps(
                {"translations": ["ছোট" for _ in _lines_from_contents(contents)]}
            )

        client = _FakeClient(responder)
        translator = _make_translator(client)
        segment = Segment(
            index=0, start=0.0, end=0.5,
            text_bn="এটি একটি বেশ লম্বা বাক্য যা সময়ে আঁটবে না",
        )
        changed = translator.shorten_to_fit([segment], max_tempo=1.25)
        self.assertEqual(changed, 1)
        self.assertEqual(segment.text_bn, "ছোট")

    def test_close_releases_client(self):
        client = _FakeClient(lambda c, n: json.dumps({"translations": []}))
        translator = _make_translator(client)
        translator.close()
        translator.close()
        self.assertEqual(client.closed, 1)


class TestTranslatorFactory(unittest.TestCase):
    def test_gemini_backend(self):
        translator = create_translator(
            "gemini", gemini_api_key="test-key", gemini_model="gemini-test"
        )
        self.assertIsInstance(translator, GeminiTranslator)
        self.assertEqual(translator.describe(), "gemini:gemini-test")
        translator.close()

    def test_missing_api_key_raises(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(GeminiAuthError):
                create_translator("gemini", gemini_api_key="")

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            create_translator("google")
        with self.assertRaises(ValueError):
            create_translator("nope")

    def test_env_key_is_used(self):
        with mock.patch.dict("os.environ", {"GEMINI_API_KEY": "env-key"}, clear=True):
            translator = create_translator("gemini")
        self.assertEqual(translator._api_key, "env-key")
        translator.close()


# ===========================================================================
# Sync engine — আসল WAV দিয়ে (ffmpeg থাকলে)
# ===========================================================================
@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class TestSyncEngineBuild(unittest.TestCase):
    def _clip(self, path: Path, seconds: float, freq: float = 220.0) -> Path:
        t = np.arange(int(seconds * 24000)) / 24000.0
        audio_fx.write_wav(path, 0.3 * np.sin(2 * np.pi * freq * t), 24000)
        return path

    def test_final_track_is_full_length_and_never_overlaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            segments = [
                Segment(index=0, start=0.0, end=2.0),
                Segment(index=1, start=2.5, end=4.5),
                Segment(index=2, start=5.0, end=7.0),
            ]
            durations = [3.6, 1.2, 2.0]  # প্রথমটি অনেক লম্বা → tempo cap + trim
            for seg, dur in zip(segments, durations):
                seg.tts_audio_path = self._clip(work / f"seg_{seg.index}.wav", dur)
                seg.tts_duration = dur

            engine = SyncEngine(max_tempo=1.25, min_tempo=0.95, sample_rate=24000,
                                max_workers=2)
            out = engine.build_final_track(segments, 7.5, work, work / "final.wav")
            self.assertTrue(out.is_file())
            clip, rate = audio_fx.read_wav(out)
            self.assertEqual(rate, 24000)
            self.assertAlmostEqual(clip.size / 24000.0, 7.5, delta=0.3)
            # tempo কখনো 1.25 ছাড়ায়নি
            self.assertLessEqual(engine.last_stats["max_tempo"], 1.25 + 1e-6)
            self.assertEqual(engine.last_stats["overlaps"], 0)
            # প্রতিটি ক্লিপ তার উইন্ডোতে শেষ হয়েছে (ওভারল্যাপ অসম্ভব)
            self.assertTrue(all(s.stretched_duration is not None for s in segments))

    def test_no_usable_clips_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SyncEngine(max_workers=1)
            with self.assertRaises(RuntimeError):
                engine.build_final_track([], 5.0, Path(tmp), Path(tmp) / "x.wav")


# ===========================================================================
# অনুবাদ ↔ TTS overlap (কোনো API/GPU ছাড়াই)
# ===========================================================================
class _RecordingTranslator:
    def __init__(self, fail_on_group=None):
        self.translated = []
        self.shorten_calls = 0
        self.fail_on_group = fail_on_group
        self._group_no = 0

    def translate_segments(self, segments):
        self._group_no += 1
        if self.fail_on_group is not None and self._group_no == self.fail_on_group:
            raise RuntimeError("fake translation failure")
        for seg in segments:
            seg.text_bn = f"টি{seg.index}"
        self.translated.append([s.index for s in segments])
        return segments

    def shorten_to_fit(self, segments, **_):
        self.shorten_calls += 1
        for seg in segments:
            if seg.index >= 0 and (seg.text_bn or "").strip():
                seg.text_bn += "s"      # ছোট করা হয়েছে — TTS-এ এটাই যাবে
        return 0

    def describe(self):
        return "recording"

    def close(self):
        pass


class _RecordingTTS:
    def __init__(self):
        self.calls = []
        self.closed = False

    def set_budget(self, *_a, **_k):
        pass

    def synthesize(self, segments, output_dir):
        self.calls.append([(s.index, s.text_bn) for s in segments])
        return segments

    def describe(self):
        return "recording-tts"

    def close(self):
        self.closed = True


class TestPipelineTranslationOverlap(unittest.TestCase):
    def _pipeline(self, translator, tts):
        from proxmax.pipeline import DubbingPipeline

        pipeline = DubbingPipeline.__new__(DubbingPipeline)
        pipeline.config = AppConfig(
            gemini_batch_size=4, enable_overlap=True, translation_shorten_pass=True,
            unload_stt_before_tts=False,
        ).validate()
        pipeline.translator = translator
        pipeline._tts = tts
        pipeline._profile = None
        pipeline._profile_checked = True
        return pipeline

    def test_groups_are_translated_then_synthesized_in_order(self):
        translator = _RecordingTranslator()
        tts = _RecordingTTS()
        pipeline = self._pipeline(translator, tts)
        segments = [
            Segment(index=i, start=i * 1.5, end=i * 1.5 + 1.2, text_en=f"Line {i}.")
            for i in range(10)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            translate_seconds, tts_seconds = pipeline._translate_and_synthesize(
                segments, Path(tmp)
            )

        self.assertGreaterEqual(translate_seconds, 0.0)
        self.assertGreaterEqual(tts_seconds, 0.0)
        self.assertEqual(len(translator.translated), 3)   # ১০ লাইন / ৪
        self.assertEqual(len(tts.calls), 3)
        self.assertEqual(translator.shorten_calls, 3)
        # প্রতিটি গ্রুপ synthesize হওয়ার আগেই অনূদিত ও ছোট করা হয়েছে
        self.assertEqual([idx for group in tts.calls for idx, _ in group], list(range(10)))
        for group in tts.calls:
            for _idx, text in group:
                self.assertTrue(text.endswith("s"), text)
        self.assertTrue(all(s.text_bn.endswith("s") for s in segments))

    def test_translation_failure_stops_gracefully(self):
        translator = _RecordingTranslator(fail_on_group=2)
        tts = _RecordingTTS()
        pipeline = self._pipeline(translator, tts)
        segments = [
            Segment(index=i, start=i * 1.5, end=i * 1.5 + 1.2, text_en=f"Line {i}.")
            for i in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                pipeline._translate_and_synthesize(segments, Path(tmp))
        # প্রথম গ্রুপের TTS শুরু হলেও বাকিগুলো আর হয়নি — কোনো hang নেই
        self.assertLessEqual(len(tts.calls), 1)

    def test_single_group_uses_sequential_path(self):
        translator = _RecordingTranslator()
        tts = _RecordingTTS()
        pipeline = self._pipeline(translator, tts)
        segments = [Segment(index=0, start=0.0, end=2.0, text_en="Hello.")]
        with tempfile.TemporaryDirectory() as tmp:
            pipeline._translate_and_synthesize(segments, Path(tmp))
        self.assertEqual(len(translator.translated), 1)
        self.assertEqual(len(tts.calls), 1)
        self.assertEqual(translator.shorten_calls, 1)


# ===========================================================================
# End-to-end: আসল ffmpeg + mock STT/Translator/TTS
# ===========================================================================
class _FakeSTT:
    def __init__(self):
        self.released = False
        self.last_stats = {"chunks": 1, "words": 5}

    def transcribe(self, audio_path):
        return [
            Segment(index=0, start=0.3, end=2.3, text_en="Hello world."),
            Segment(index=1, start=2.8, end=5.4, text_en="This is a test video."),
        ]

    def transcribe_reference(self, path, language="bn"):  # pragma: no cover
        return "পরীক্ষা"

    def release(self):
        self.released = True


class _FakeTranslator:
    def __init__(self):
        self.closed = False
        self.shorten_calls = 0

    def translate_segments(self, segments):
        for seg in segments:
            seg.text_bn = "হ্যালো বিশ্ব।" if seg.index == 0 else "এটি একটি পরীক্ষামূলক ভিডিও।"
        return segments

    def shorten_to_fit(self, segments, **_):
        self.shorten_calls += 1
        return 0

    def describe(self):
        return "fake:translator"

    def close(self):
        self.closed = True


class _FakeTTS:
    name = "fake-tts"

    def __init__(self):
        self.closed = False

    def set_budget(self, *_args, **_kwargs):
        return None

    def synthesize(self, segments, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for seg in segments:
            path = output_dir / f"seg_{seg.index:04d}.wav"
            # ইচ্ছাকৃতভাবে একটু লম্বা → atempo লাগবে
            dur = max(0.8, seg.duration * 1.4)
            t = np.arange(int(dur * 24000)) / 24000.0
            audio_fx.write_wav(path, 0.3 * np.sin(2 * np.pi * 330 * t), 24000)
            seg.tts_audio_path = path
            seg.tts_duration = dur
        return segments

    def describe(self):
        return "fake:tts"

    def close(self):
        self.closed = True


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class TestPipelineEndToEnd(unittest.TestCase):
    def _make_video(self, path: Path, seconds: float = 6.0) -> None:
        proc = subprocess.run(
            [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", f"color=c=navy:s=320x240:r=25:d={seconds}",
             "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}",
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-shortest", str(path)],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_full_pipeline_with_mocks(self):
        from proxmax.audio_extraction import AudioExtractor
        from proxmax.muxer import VideoMuxer
        from proxmax.pipeline import DubbingPipeline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir, output_dir, work_dir = root / "input", root / "output", root / "work"
            input_dir.mkdir()
            video = input_dir / "tiny.mp4"
            self._make_video(video)

            cfg = AppConfig(
                input_dir=str(input_dir), output_dir=str(output_dir),
                work_dir=str(work_dir), save_srt=True, unload_stt_before_tts=True,
            ).validate()

            pipeline = DubbingPipeline.__new__(DubbingPipeline)  # ভারী init বাদ
            pipeline.config = cfg
            pipeline.extractor = AudioExtractor(use_hwaccel=False)
            pipeline.stt = _FakeSTT()
            pipeline.translator = _FakeTranslator()
            pipeline.tts = _FakeTTS()
            pipeline._tts = pipeline.tts
            pipeline._profile = None
            pipeline._profile_checked = True
            pipeline.sync_engine = SyncEngine(sample_rate=24000, max_workers=2)
            pipeline.muxer = VideoMuxer(use_hwaccel=False)

            summary = pipeline.run_batch()
            self.assertEqual(len(summary.succeeded), 1, summary.failed)

            out = output_dir / "tiny_bn.mp4"
            self.assertTrue(out.is_file() and out.stat().st_size > 0)
            self.assertTrue((output_dir / "tiny_en.srt").is_file())
            self.assertTrue((output_dir / "tiny_bn.srt").is_file())

            result = summary.succeeded[0]
            self.assertIsInstance(result.timings, StageTimings)
            self.assertGreater(result.timings.total, 0)
            self.assertGreater(result.timings.tts, 0)
            self.assertEqual(result.stats["sync"]["overlaps"], 0)
            self.assertIn("stt", result.stats)

            # step 4-এ Whisper unload + শেষে সবার release
            self.assertTrue(pipeline.stt.released)
            self.assertTrue(pipeline.translator.closed)
            self.assertTrue(pipeline.tts.closed)

    def test_empty_input_folder_is_graceful(self):
        from proxmax.pipeline import DubbingPipeline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = AppConfig(
                input_dir=str(root / "input"), output_dir=str(root / "output"),
                work_dir=str(root / "work"),
            )
            (root / "input").mkdir()
            pipeline = DubbingPipeline.__new__(DubbingPipeline)
            pipeline.config = cfg
            summary = pipeline.run_batch()
            self.assertEqual(summary.results, [])


if __name__ == "__main__":
    unittest.main()
