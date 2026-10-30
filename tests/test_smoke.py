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
import re
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
from proxmax.models import Segment, StageTimings, Word                  # noqa: E402
from proxmax.reporting import write_detailed_report                    # noqa: E402
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
    parse_numbered_lines,
    parse_translations,
    salvage_translations,
    verify_gemini_api,
)
from proxmax.tts import (                                              # noqa: E402
    TTSError,
    EdgeTTSEngine,
    resolve_tts_engine,
)
from proxmax.utils import (                                            # noqa: E402
    format_srt_timestamp,
    output_filename_for,
    release_cuda_memory,
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
        self.assertEqual(cfg.tts_engine, "edge")
        self.assertEqual(cfg.stt_hotwords_file, "hotwords.txt")
        self.assertTrue(cfg.save_report)
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
        self.assertEqual(cfg.tts_engine, "edge")
        self.assertEqual(cfg.stt_hotwords_file, "hotwords.txt")
        self.assertTrue(cfg.save_report)
        self.assertAlmostEqual(cfg.max_tempo, 1.25)
        self.assertAlmostEqual(cfg.min_tempo, 0.95)
        self.assertGreaterEqual(cfg.carry_context_chars, 100)
        self.assertTrue(cfg.seam_repair)
        self.assertGreater(cfg.chunk_min_seconds, 10)

    def test_section_scoped_aliases(self):
        cfg = AppConfig()
        cfg._apply_mapping(
            {
                "stt": {"model": "medium", "chunk_min_seconds": 20.0, "hotwords_file": "custom_hw.txt"},
                "translation": {"backend": "gemini", "max_retries": 6},
                "gemini": {"model": "gemini-test-model", "api_key": "test-key"},
                "tts": {"voice": "bn-BD-PradeepNeural", "max_retries": 2, "engine": "edge"},
                "sync": {"max_tempo": 1.3, "workers": 2},
                "runtime": {"release_gpu_when_done": False, "enable_overlap": False},
                "output": {"save_report": False},
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
        self.assertAlmostEqual(cfg.max_tempo, 1.3)
        self.assertFalse(cfg.release_gpu_when_done)
        self.assertFalse(cfg.enable_overlap)
        self.assertFalse(cfg.save_report)

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
            {"translator_backend": "google"},
            {"translator_backend": "deepl"},
            {"tts_engine": "siri"},
            {"tts_engine": "indicf5"},
            {"chunk_max_seconds": 45.0},
            {"chunk_min_seconds": 40.0},
            {"min_tempo": 1.2, "max_tempo": 1.2},
            {"sync_sample_rate": 4000},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    AppConfig(**kwargs).validate()

    def test_work_dir_resolution(self):
        cfg = AppConfig(work_dir="/tmp/proxmax-test-work")
        self.assertEqual(str(cfg.resolved_work_dir), "/tmp/proxmax-test-work")


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

    def test_plan_window_does_not_absorb_a_multi_second_source_pause(self):
        segments = [
            Segment(index=0, start=1.0, end=2.0, tts_duration=3.5),
            Segment(index=1, start=7.0, end=8.0, tts_duration=1.0),
        ]
        windows = plan_windows(
            segments, total_duration=9.0, min_gap_seconds=0.08, borrow_seconds=0.25
        )
        # Five seconds of source silence after line 1 is not a five-second dub slot.
        self.assertAlmostEqual(windows[0][1], 2.25, places=3)
        self.assertLess(windows[0][1], segments[1].start - 3.0)

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


# ===========================================================================
# TTS — ইঞ্জিনের নিরাপত্তা
# ===========================================================================
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


class TestExistingOutputGuard(unittest.TestCase):
    """আগের আউটপুট "সম্পন্ন" ধরে নেওয়ার আগে সত্যিই বৈধ কি না দেখা হয়।"""

    def _valid_mp4(self, folder: Path) -> Path:
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg/ffprobe নেই")
        out = folder / "video_bn.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=3",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(out),
            ],
            check=True,
        )
        # কৃত্রিমভাবে বড় করা হচ্ছে যাতে min_bytes পাস করে (আসল MP4 ছোট হয়)
        with open(out, "ab") as handle:
            handle.write(b"\0" * (64 * 1024))
        return out

    def test_valid_output_is_skippable(self):
        from proxmax.pipeline import existing_output_is_usable

        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(existing_output_is_usable(self._valid_mp4(Path(tmp))))

    def test_missing_or_tiny_or_garbage_output_is_not_skippable(self):
        from proxmax.pipeline import existing_output_is_usable

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            missing = folder / "nope.mp4"
            tiny = folder / "tiny.mp4"
            tiny.write_bytes(b"\0" * 1024)
            garbage = folder / "garbage.mp4"
            garbage.write_bytes(b"not a video" * 8000)
            self.assertFalse(existing_output_is_usable(missing))
            self.assertFalse(existing_output_is_usable(tiny))
            self.assertFalse(existing_output_is_usable(garbage))

    def test_skip_is_announced_with_force_hint(self):
        from proxmax.pipeline import DubbingPipeline

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            in_dir, out_dir = folder / "in", folder / "out"
            in_dir.mkdir()
            out_dir.mkdir()
            video = in_dir / "video.mp4"
            video.write_bytes(b"\0" * 1024)
            pipeline = DubbingPipeline(
                AppConfig(input_dir=in_dir, output_dir=out_dir, gemini_api_key="test-key")
            )
            # বৈধতা যাচাইটা mock করা হচ্ছে — বাকি সব আসল কোডই চলবে
            (out_dir / "video_bn.mp4").write_bytes(b"\0" * (64 * 1024))
            (out_dir / "video_report.markdown").write_text("# Detailed report\n" + ("diagnostic " * 40), encoding="utf-8")
            with mock.patch(
                "proxmax.pipeline.existing_output_is_usable", return_value=True
            ):
                try:
                    with self.assertLogs("proxmax.pipeline", level="INFO") as logs:
                        result = pipeline.process_one(video, out_dir)
                finally:
                    pipeline.close()
        self.assertEqual(result.status, "skipped")
        self.assertTrue(any("--force" in line for line in logs.output))


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


# ===========================================================================
# Gemini translator (fake client — কোনো নেটওয়ার্ক নেই)
# ===========================================================================
class _FakeResponse:
    def __init__(self, text, block_reason=None, finish_reason=None):
        self._text = text
        self.prompt_feedback = type("FB", (), {"block_reason": block_reason})()
        if finish_reason is not None:
            self.candidates = [
                type("C", (), {"finish_reason": type("F", (), {"name": finish_reason})()})()
            ]
        else:
            self.candidates = []

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
        answer = self.responder(contents, len(self.calls))
        return answer if isinstance(answer, _FakeResponse) else _FakeResponse(answer)

    def close(self):
        self.closed += 1


def _lines_from_contents(contents: str) -> list:
    """কোনো request-এর ইনপুট থেকে মূল ইংরেজি লাইনগুলো বের করে (JSON বা নম্বরযুক্ত)।"""
    marker = contents.find('{"lines"')
    if marker >= 0:
        blob = contents[marker:]
        try:
            return [line["t"] for line in json.loads(blob)["lines"]]
        except (ValueError, KeyError, TypeError):
            pass
    payload = contents.split("Input lines:", 1)[-1]
    lines = []
    for line in payload.splitlines():
        match = re.match(r"^\s*(?:\d+[.)]\s*)?(?P<text>.+)$", line)
        if match and match.group("text").strip():
            lines.append(match.group("text").strip())
    return lines


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
            parse_translations("not json", 3)

    def test_parse_survives_wrapping_text_and_fences(self):
        raw = 'Here are the lines:\n```json\n{"translations": ["এক", "দুই"]}\n```\nHope it helps!'
        self.assertEqual(parse_translations(raw, 2), ["এক", "দুই"])

    def test_parse_numbered_lines_when_model_ignores_json(self):
        raw = "1. আমি একটি স্টার পাস দেব।\n2. দ্বিতীয় লাইন।"
        self.assertEqual(parse_translations(raw, 2), ["আমি একটি স্টার পাস দেব।", "দ্বিতীয় লাইন।"])

    def test_truncated_answer_keeps_complete_lines(self):
        raw = '{"translations": ["এক", "দুই", "তিন'
        self.assertEqual(salvage_translations(raw, 3), ["এক", "দুই"])
        self.assertEqual(salvage_translations('{"lines": [{"bn": "এক"}, {"bn": "দুই"}', 3),
                         ["এক", "দুই"])


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

    # ---- ফরম্যাট ভুল: অপেক্ষা নেই, শুধু বাকি লাইন আবার ------------------
    def test_format_error_retries_without_long_sleep(self):
        """ফরম্যাট ভুল হলে ২ সেকেন্ডের বেশি অপেক্ষা করা চলবে না।"""
        def responder(contents, call_no):
            if call_no <= 2:          # প্রথমে ফরম্যাট ভাঙা উত্তর
                return "sorry, I cannot do JSON"
            return "\n".join(       # তৃতীয়বার থেকে নম্বরযুক্ত লাইন (lines মোড)
                f"{index}. বাংলা: {text}"
                for index, text in enumerate(_lines_from_contents(contents), start=1)
            )

        client = _FakeClient(responder)
        translator = _make_translator(client, max_retries=5, requests_per_minute=1_000_000)
        slept = []
        with mock.patch("proxmax.translator.time.sleep", side_effect=slept.append):
            result = translator.translate_text("Hello")
        self.assertTrue(result.startswith("বাংলা:"))
        self.assertTrue(all(delay <= 2.0 for delay in slept), slept)
        self.assertTrue(all(delay <= 0.3 for delay in slept), f"ফরম্যাট ভুলে অপেক্ষা: {slept}")

    def test_format_retry_does_not_wait_for_rate_limit_slot(self):
        """ফরম্যাট ভুলে আবার পাঠানো একই request — নতুন স্লটের জন্য অপেক্ষা নয়।"""
        def responder(contents, call_no):
            if call_no == 1:
                return "not json at all"
            return "\n".join(
                f"{index}. বাংলা: {text}"
                for index, text in enumerate(_lines_from_contents(contents), start=1)
            )

        client = _FakeClient(responder)
        # RPM খুব কম (১) → নতুন request হলে ৬০ সেকেন্ড অপেক্ষা করতে হতো
        translator = _make_translator(client, max_retries=3, requests_per_minute=1)
        slept = []
        with mock.patch("proxmax.translator.time.sleep", side_effect=slept.append):
            result = translator.translate_text("Hello")
        self.assertTrue(result.startswith("বাংলা:"))
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(all(delay <= 0.3 for delay in slept),
                        f"ফরম্যাট retry-তে অপেক্ষা: {slept}")

    def test_retry_delay_never_exceeds_two_seconds(self):
        translator = _make_translator(_FakeClient(lambda c, n: "{}"), max_retries=7)
        for attempt in range(1, 8):
            for status in (None, 429, 500):
                delay = translator._retry_delay(attempt, TimeoutError("timed out"), status)
                self.assertLessEqual(delay, 2.0)
                self.assertGreaterEqual(delay, 0.0)

    def test_model_ignoring_json_falls_back_to_numbered_lines(self):
        """মডেল JSON না দিলে দুবার চেষ্টার পর নম্বরযুক্ত লাইনে চাওয়া হয়।"""
        def responder(contents, call_no):
            if call_no <= 2:
                return "Here is your Bangla text, sorry for the format."
            return "\n".join(
                f"{index}. বাংলা: {text}"
                for index, text in enumerate(_lines_from_contents(contents), start=1)
            )

        client = _FakeClient(responder)
        translator = _make_translator(client, max_retries=5)
        with mock.patch("proxmax.translator.time.sleep"):
            result = translator.translate_text("Hello")
        self.assertTrue(result.startswith("বাংলা:"), result)
        self.assertTrue(any("numbered lines" in c["contents"] for c in client.calls[2:]),
                        [c["contents"][:60] for c in client.calls])

    def test_truncated_answer_only_retries_missing_lines(self):
        """কাটা উত্তরে শুরুর লাইনগুলো রেখে বাকিগুলোই আবার চাওয়া হয়।"""
        def responder(contents, call_no):
            if call_no == 1:
                return _FakeResponse(
                    '{"translations": ["বাংলা: One.", "বাংলা: Two.',
                    finish_reason="MAX_TOKENS",
                )
            return json.dumps({"translations": [f"বাংলা: {t}" for t in _lines_from_contents(contents)]})

        client = _FakeClient(responder)
        translator = _make_translator(client)
        segments = _segments(["One.", "Two.", "Three.", "Four."])
        with mock.patch("proxmax.translator.time.sleep"):
            translator.translate_segments(segments)
        self.assertEqual([s.text_bn for s in segments],
                         ["বাংলা: One.", "বাংলা: Two.", "বাংলা: Three.", "বাংলা: Four."])
        # প্রথম request-এ ৪টি লাইন ছিল, দ্বিতীয়টিতে শুধু বাকি লাইনগুলো
        self.assertEqual(len(_lines_from_contents(client.calls[0]["contents"])), 4)
        self.assertLess(len(_lines_from_contents(client.calls[1]["contents"])), 4)

    def test_one_unparseable_line_is_skipped_not_fatal(self):
        """এক লাইনে বারবার ভুল ফরম্যাট হলেও পুরো ভিডিও ব্যর্থ হবে না।"""
        def responder(contents, _n):
            lines = _lines_from_contents(contents)
            if any("BROKEN" in line for line in lines):
                return "no bengali here, sorry"
            return json.dumps({"translations": [f"বাংলা: {t}" for t in lines]})

        client = _FakeClient(responder)
        translator = _make_translator(client, max_retries=2, batch_size=2)
        segments = _segments(["One.", "BROKEN line.", "Three."])
        with mock.patch("proxmax.translator.time.sleep"):
            translator.translate_segments(segments)
        self.assertEqual(segments[0].text_bn, "বাংলা: One.")
        self.assertTrue(segments[1].text_bn in ("", "বাংলা: BROKEN line."))
        self.assertTrue(segments[2].text_bn.startswith("বাংলা:"))

    def test_thinking_config_is_not_sent_by_default_to_avoid_http_400(self):
        """নতুন Gemini মডেলে thinking_budget=0 পাঠালে 400 INVALID_ARGUMENT আসে — তাই ডিফল্টে বন্ধ।"""
        client = _FakeClient(lambda c, n: json.dumps({"translations": ["বাংলা"]}))
        translator = _make_translator(client)
        translator.translate_text("Hello")
        thinking = getattr(client.calls[0]["config"], "thinking_config", None)
        self.assertIsNone(thinking)

    def test_http_400_on_schema_falls_back_to_plain_json_without_failing(self):
        """HTTP 400 INVALID_ARGUMENT এলে response_schema বন্ধ করে সাধারণ JSON-এ সফল হয়।"""
        class _Http400SchemaClient(_FakeClient):
            def generate_content(self, *, model, contents, config):
                self.calls.append({"model": model, "contents": contents, "config": config})
                if getattr(config, "response_schema", None) is not None:
                    err = RuntimeError("400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Invalid argument'}}")
                    err.code = 400  # type: ignore[attr-defined]
                    raise err
                return _FakeResponse(json.dumps({"translations": ["বাংলা অনুবাদ।"]}))

        client = _Http400SchemaClient(lambda c, n: "")
        translator = _make_translator(client, max_retries=3, requests_per_minute=1_000_000)
        self.assertEqual(translator.translate_text("Hello"), "বাংলা অনুবাদ।")
        self.assertFalse(translator._use_schema)
        self.assertEqual(len(client.calls), 2)

    def test_http_400_or_404_on_model_switches_to_fallback_model(self):
        """প্রাথমিক মডেল 400/404 দিলে স্বয়ংক্রিয়ভাবে ব্যাকআপ মডেলে সুইচ করে।"""
        class _ModelFallbackClient(_FakeClient):
            def generate_content(self, *, model, contents, config):
                self.calls.append({"model": model, "contents": contents, "config": config})
                if model == "gemini-3.5-flash-lite":
                    err = RuntimeError("400 INVALID_ARGUMENT: model gemini-3.5-flash-lite not supported")
                    err.code = 400  # type: ignore[attr-defined]
                    raise err
                return _FakeResponse(json.dumps({"translations": ["স্বাগতম।"]}))

        client = _ModelFallbackClient(lambda c, n: "")
        translator = _make_translator(client, model="gemini-3.5-flash-lite", max_retries=5)
        result = translator.translate_text("Welcome.")
        self.assertEqual(result, "স্বাগতম।")
        self.assertNotEqual(translator.model, "gemini-3.5-flash-lite")

    def test_verify_api_checks_live_translation(self):
        client = _FakeClient(lambda c, n: json.dumps({"translations": ["এফসি মোবাইলে স্বাগতম।"]}))
        translator = _make_translator(client, model="gemini-3.5-flash-lite")
        info = translator.verify_api()
        self.assertTrue(info["ok"])
        self.assertEqual(info["active_model"], "gemini-3.5-flash-lite")
        self.assertIn("স্বাগতম", info["sample_reply"])

    def test_gemini_gets_only_matched_and_bounded_hotword_glossary(self):
        client = _FakeClient(lambda c, n: json.dumps({"translations": ["মেসি ও মাসচেরানো।"]}))
        translator = _make_translator(
            client,
            hotwords_file=str(REPO_ROOT / "hotwords.txt"),
            hotwords="Unrelated Player Name",
        )
        translator.translate_text("FC Mobile 27: Messi and Mascherano.")
        contents = client.calls[0]["contents"]
        self.assertIn("Recognized football/FC Mobile terms in this batch", contents)
        self.assertIn("Mascherano", contents)
        self.assertNotIn("Unrelated Player Name", contents)
        self.assertNotIn("Aitana Bonmati", contents)
        glossary = contents.split("Recognized football/FC Mobile terms in this batch", 1)[1]
        glossary = glossary.split("\n{", 1)[0]
        self.assertLessEqual(len(glossary), 650)

    def test_gemini_glossary_caps_match_count_and_characters(self):
        response = lambda c, n: json.dumps({"translations": ["বাংলা অনুবাদ।"]})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = (
                ([f"ModeTerm{i:02d}" for i in range(30)], 20),
                ([f"VeryLongGameMode{i:02d}_" + "x" * 40 for i in range(30)], None),
            )
            for index, (terms, expected_count) in enumerate(cases):
                hotwords_file = root / f"hotwords-{index}.txt"
                hotwords_file.write_text("\n".join(terms), encoding="utf-8")
                client = _FakeClient(response)
                translator = _make_translator(client, hotwords_file=str(hotwords_file))
                translator.translate_text(" / ".join(terms))

                contents = client.calls[0]["contents"]
                marker = "(keep names/terms accurate in Bangla): "
                self.assertIn(marker, contents)
                glossary = contents.split(marker, 1)[1].split("\n{", 1)[0].rstrip(".")
                matched = glossary.split(", ")
                self.assertLessEqual(len(matched), 20)
                self.assertLessEqual(len(glossary), 600)
                if expected_count is not None:
                    self.assertEqual(len(matched), expected_count)
                else:
                    self.assertLess(len(matched), 20)

    def test_shorten_budget_uses_speech_span_not_following_pause(self):
        translator = _make_translator(_FakeClient(lambda c, n: "{}"))
        long_line = "বাংলা বাক্য অনুবাদ " * 12
        first = Segment(index=0, start=0.0, end=1.0, text_en="A line.", text_bn=long_line)
        second = Segment(index=1, start=5.0, end=6.0, text_en="Next line.", text_bn="ছোট।")
        with mock.patch.object(translator, "_run_batches", return_value=["ছোট।"]) as run:
            changed = translator.shorten_to_fit(
                [first, second], max_tempo=1.25, slot_gap=0.08, borrow=0.25
            )
        self.assertEqual(changed, 1)
        items = run.call_args.args[0]
        self.assertEqual(len(items), 1)
        self.assertAlmostEqual(items[0][1], 1.25, places=3)

    def test_output_budget_grows_with_batch_size(self):
        translator = _make_translator(_FakeClient(lambda c, n: "{}"))
        small = translator._output_token_budget([("short line", 1.0)] * 2)
        big = translator._output_token_budget([("a long line " * 20, 3.0)] * 20)
        self.assertGreaterEqual(small, 1024)
        self.assertGreater(big, small)
        self.assertLessEqual(big, translator.max_output_tokens)

    def test_unsupported_schema_disables_it_and_retries_immediately(self):
        class _NoSchemaClient(_FakeClient):
            def generate_content(self, *, model, contents, config):
                self.calls.append({"model": model, "contents": contents, "config": config})
                if getattr(config, "response_schema", None) is not None:
                    raise ValueError("response_schema is not supported for this model")
                return _FakeResponse(json.dumps({"translations": ["বাংলা"]}))

        client = _NoSchemaClient(lambda c, n: "")
        translator = _make_translator(client, max_retries=3, requests_per_minute=1_000_000)
        slept = []
        with mock.patch("proxmax.translator.time.sleep", side_effect=slept.append):
            self.assertEqual(translator.translate_text("Hello"), "বাংলা")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(slept, [])
        self.assertFalse(translator._use_schema)

    def test_close_releases_client(self):
        client = _FakeClient(lambda c, n: json.dumps({"translations": []}))
        translator = _make_translator(client)
        translator.close()
        translator.close()
        self.assertEqual(client.closed, 1)


class TestTranslatorRecovery(unittest.TestCase):
    """অনূদিত না হওয়া লাইন ছোট ব্যাচে উদ্ধার — নীরব স্লট যেন না থাকে।"""

    def test_missing_lines_are_recovered_in_small_batches(self):
        def responder(contents, _n):
            lines = _lines_from_contents(contents)
            if len(lines) > 4:      # বড় ব্যাচে মডেল ভুল ফরম্যাট দেয়
                return "sorry, I cannot help with that"
            return json.dumps({"translations": [f"বাংলা: {t}" for t in lines]})

        client = _FakeClient(responder)
        translator = _make_translator(client, batch_size=6, max_retries=1)
        segments = _segments([f"Line {i}." for i in range(6)])
        translator.translate_segments(segments)
        self.assertTrue(all((s.text_bn or "").strip() for s in segments),
                        [s.text_bn for s in segments])
        self.assertTrue(all(s.text_bn.startswith("বাংলা:") for s in segments))

    def test_recovery_failure_is_graceful(self):
        def responder(contents, _n):
            return "still not valid"

        client = _FakeClient(responder)
        translator = _make_translator(client, batch_size=3, max_retries=1)
        segments = _segments(["A.", "B."])
        translator.translate_segments(segments)      # exception নয়
        self.assertEqual([s.text_bn for s in segments], ["", ""])


class TestTranslatorExpandPass(unittest.TestCase):
    """বড় ফাঁকের জন্য লাইন ভরাট করা (expand pass)।"""

    def _translator(self, grow_with: str):
        def responder(contents, _n):
            lines = _lines_from_contents(contents)
            return json.dumps({"translations": [line + grow_with for line in lines]})

        return _make_translator(_FakeClient(responder)), _FakeClient(responder)

    def test_short_line_is_expanded_within_slot(self):
        def responder(contents, _n):
            lines = _lines_from_contents(contents)
            return json.dumps({"translations": [line + " আসলে পুরো সময়টা জুড়ে কথা বলা যায়।" for line in lines]})

        client = _FakeClient(responder)
        translator = _make_translator(client)
        seg = Segment(index=0, start=0.0, end=12.6, text_en="x", text_bn="খুব ছোট।")
        seg.tts_duration = 1.0
        changed = translator.expand_to_fit([seg], [12.5])
        self.assertEqual(changed, 1)
        self.assertGreater(len(seg.text_bn), 20)
        self.assertTrue(seg.text_bn.startswith("খুব ছোট।"))

    def test_absurdly_long_answer_is_rejected(self):
        def responder(_contents, _n):
            return json.dumps({"translations": ["অ" * 5000]})

        translator = _make_translator(_FakeClient(responder))
        seg = Segment(index=0, start=0.0, end=4.0, text_en="x", text_bn="ছোট।")
        self.assertEqual(translator.expand_to_fit([seg], [4.0]), 0)
        self.assertEqual(seg.text_bn, "ছোট।")

    def test_failure_keeps_old_text(self):
        client = _FakeClient(lambda c, n: (_ for _ in ()).throw(RuntimeError("boom")))
        translator = _make_translator(client, max_retries=1)
        seg = Segment(index=0, start=0.0, end=9.0, text_en="x", text_bn="ছোট।")
        with mock.patch("proxmax.translator.time.sleep"):
            self.assertEqual(translator.expand_to_fit([seg], [8.0]), 0)
        self.assertEqual(seg.text_bn, "ছোট।")

    def test_empty_texts_are_skipped(self):
        client = _FakeClient(lambda c, n: json.dumps({"translations": ["কিছু"]}))
        translator = _make_translator(client)
        seg = Segment(index=0, start=0.0, end=5.0, text_en="x", text_bn="  ")
        self.assertEqual(translator.expand_to_fit([seg], [5.0]), 0)
        self.assertEqual(client.calls, [])


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
class TestGapFilling(unittest.TestCase):
    """বড় ফাঁক ভরানোর ধাপ: ছোট লাইন ভরাট করে শুধু সেগুলোরই আবার TTS।"""

    class _Translator(_FakeTranslator):
        def __init__(self, expand: bool = True):
            super().__init__()
            self.expand = expand
            self.expand_calls = 0

        def translate_segments(self, segments):
            for seg in segments:
                seg.text_bn = "ছোট।"
            return segments

        def expand_to_fit(self, segments, targets):
            self.expand_calls += 1
            changed = 0
            for seg, target in zip(segments, targets):
                if target - float(seg.tts_duration or 0.0) >= 0.9:
                    seg.text_bn = "এটি এখন অনেক বেশি ভরাট করে লেখা একটি লাইন যাতে সময়টা পুরো ভরে যায়।"
                    changed += 1
            return changed

    class _PaddedTTS(_FakeTTS):
        """edge-tts-এর মতো দুই প্রান্তে নীরবতা জোড়া ক্লিপ (trim হবে)।"""

        def synthesize(self, segments, output_dir):
            from proxmax.tts.base import BaseTTSEngine

            class _Trimmer(BaseTTSEngine):
                name = "trimmer"

                def synthesize(self, *_args, **_kwargs):  # pragma: no cover
                    return []

            engine = _Trimmer(trim_silence=True, trim_pad_ms=25)
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for seg in segments:
                speech = max(0.5, len(seg.text_bn) / 13.0)
                body = (0.3 * np.sin(2 * np.pi * 220 * np.arange(int(speech * 24000)) / 24000)).astype(np.float32)
                padded = np.concatenate([
                    np.zeros(int(0.15 * 24000), dtype=np.float32),
                    body,
                    np.zeros(int(0.6 * 24000), dtype=np.float32),
                ])
                final = engine._finalize(padded)
                path = output_dir / f"seg_{seg.index:04d}.wav"
                audio_fx.write_wav(path, final, 24000)
                seg.tts_audio_path = path
                seg.tts_duration = final.size / 24000
            return segments

    def _pipeline(self, root: Path, translator, tts):
        from proxmax.audio_extraction import AudioExtractor
        from proxmax.muxer import VideoMuxer
        from proxmax.pipeline import DubbingPipeline

        class _Stt:
            last_stats = {"chunks": 1}

            def __init__(self):
                self.released = False

            def transcribe(self, _path):
                return [
                    Segment(0, 0.3, 12.9, "a long english sentence about the squad", ""),
                    Segment(1, 12.9, 18.0, "and another short one here", ""),
                    Segment(2, 18.0, 23.0, "the final line of this clip", ""),
                ]

            def release(self):
                self.released = True

        input_dir, output_dir = root / "in", root / "out"
        input_dir.mkdir()
        video = input_dir / "demo.mp4"
        subprocess.run(
            [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=c=navy:s=320x240:r=25:d=24",
             "-f", "lavfi", "-i", "sine=frequency=220:duration=24",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(video)],
            capture_output=True, text=True,
        )
        cfg = AppConfig(
            input_dir=str(input_dir), output_dir=str(output_dir),
            work_dir=str(root / "work"), save_report=True,
        ).validate()
        pipeline = DubbingPipeline.__new__(DubbingPipeline)
        pipeline.config = cfg
        pipeline.extractor = AudioExtractor(use_hwaccel=False)
        pipeline.stt = _Stt()
        pipeline.translator = translator
        pipeline.tts = tts
        pipeline._tts = tts
        pipeline._profile = None
        pipeline._profile_checked = True
        pipeline.sync_engine = SyncEngine(sample_rate=24000, max_workers=2,
                                          fill_extension=cfg.fill_extension)
        pipeline.muxer = VideoMuxer(use_hwaccel=False)
        return pipeline

    def test_short_lines_are_filled_and_resynthesized(self):
        with tempfile.TemporaryDirectory() as tmp:
            translator = self._Translator()
            pipeline = self._pipeline(Path(tmp), translator, self._PaddedTTS())
            summary = pipeline.run_batch()
            self.assertEqual(len(summary.succeeded), 1, summary.failed)
            self.assertEqual(translator.expand_calls, 1)
            self.assertGreater(summary.succeeded[0].stats.get("filled_lines", 0), 0)
            self.assertEqual(summary.succeeded[0].stats["sync"]["overlaps"], 0)

    def test_expansion_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            translator = self._Translator()
            pipeline = self._pipeline(Path(tmp), translator, self._PaddedTTS())
            pipeline.config.translation_expand_pass = False
            pipeline.config = pipeline.config.validate()
            summary = pipeline.run_batch()
            self.assertEqual(len(summary.succeeded), 1, summary.failed)
            self.assertEqual(translator.expand_calls, 0)
            self.assertNotIn("filled_lines", summary.succeeded[0].stats)


class TestDetailedReport(unittest.TestCase):
    def test_markdown_report_contains_timeline_and_omits_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output_video = root / "source.mp4", root / "source_bn.mp4"
            output_dir, work_dir = root / "output", root / "work"
            source.write_bytes(b"source")
            output_video.write_bytes(b"dub")
            output_dir.mkdir()
            work_dir.mkdir()

            rate = 16000
            t = np.arange(rate * 4, dtype=np.float32) / rate
            audio_fx.write_wav(work_dir / "audio_16k.wav", .1 * np.sin(2 * np.pi * 220 * t), rate)
            dub = np.zeros(rate * 4, dtype=np.float32)
            dub[:rate] = .1 * np.sin(2 * np.pi * 220 * t[:rate])
            audio_fx.write_wav(work_dir / "final_audio_bn.wav", dub, rate)

            child = Segment(index=0, start=1.0, end=2.0, text_en="Hello world.")
            child.words = [Word(1.0, 1.35, "Hello", .95), Word(1.4, 1.9, "world.", .92)]
            unit = Segment(
                index=0, start=1.0, end=2.0, text_en="Hello world.", text_bn="হ্যালো বিশ্ব।",
                words=list(child.words), tts_audio_path=work_dir / "tts.wav", tts_duration=1.2,
                stretched_audio_path=work_dir / "stretched.wav", stretched_duration=1.1,
                placed_duration=.9, tempo_used=1.2, is_unit=True, children=[child],
            )
            cfg = AppConfig(gemini_api_key="DO_NOT_PUT_THIS_IN_REPORT").validate()
            probe = {
                "format": {"format_name": "mp4", "duration": "4.0", "bit_rate": "1000"},
                "streams": [{"index": 0, "codec_type": "video", "codec_name": "h264",
                             "start_time": "0", "duration": "4", "bit_rate": "900",
                             "width": 320, "height": 240, "avg_frame_rate": "25/1"}],
            }
            timings = StageTimings(extract=.1, stt=.2, translate=.3, tts=.4, sync=.5,
                                   mux=.6, total=1.0, audio_seconds=4.0)
            with mock.patch("proxmax.reporting._probe_media", return_value=probe):
                report = write_detailed_report(
                    video_path=source, output_video_path=output_video, output_dir=output_dir,
                    work_dir=work_dir, segments=[unit], timings=timings,
                    stats={"sync": {"speech_seconds": .9, "silence_seconds": 3.1,
                                    "trimmed": 1, "overlaps": 0}, "stt": {"words": 2}},
                    config=cfg, total_duration=4.0,
                )

            text = report.read_text(encoding="utf-8")
            self.assertEqual(report.name, "source_report.markdown")
            self.assertIn("Hello world.", text)
            self.assertIn("হ্যালো বিশ্ব।", text)
            self.assertIn("Source word timestamps", text)
            self.assertIn("trimmed=0.200 s", text)
            self.assertNotIn("DO_NOT_PUT_THIS_IN_REPORT", text)
            self.assertNotIn(str(root), text)
            self.assertFalse((output_dir / "source_en.srt").exists())
            self.assertFalse((output_dir / "source_bn.srt").exists())


class TestPipelineReportEmission(unittest.TestCase):
    def test_existing_video_without_report_is_reprocessed_and_reported(self):
        from proxmax.pipeline import DubbingPipeline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir, output_dir = root / "input", root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            source = input_dir / "video.mp4"
            output = output_dir / "video_bn.mp4"
            source.write_bytes(b"input")
            output.write_bytes(b"previous valid output")
            cfg = AppConfig(
                input_dir=str(input_dir), output_dir=str(output_dir),
                work_dir=str(root / "work"), skip_existing=True, save_report=True,
            ).validate()
            pipeline = DubbingPipeline.__new__(DubbingPipeline)
            pipeline.config = cfg
            segment = Segment(index=0, start=.2, end=1.2,
                              text_en="A complete sentence.", text_bn="একটি সম্পূর্ণ বাক্য।")

            def fake_steps(_video, work_dir, output_path):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"new dubbed output")
                Path(work_dir).mkdir(parents=True, exist_ok=True)
                return [segment], StageTimings(audio_seconds=2.0), {
                    "audio_seconds": 2.0, "sync": {"trimmed": 0, "overlaps": 0}
                }

            def fake_report(**kwargs):
                path = Path(kwargs["output_dir"]) / "video_report.markdown"
                path.write_text("# synthetic report\n", encoding="utf-8")
                return path

            pipeline._run_steps = mock.Mock(side_effect=fake_steps)
            with mock.patch("proxmax.pipeline.existing_output_is_usable", return_value=True), \
                 mock.patch("proxmax.pipeline.write_detailed_report", side_effect=fake_report):
                result = pipeline.process_one(source, output_dir)

            self.assertEqual(result.status, "success")
            self.assertEqual(pipeline._run_steps.call_count, 1)
            self.assertEqual(result.stats["report"]["file"], "video_report.markdown")
            self.assertTrue((output_dir / "video_report.markdown").is_file())
            self.assertFalse((output_dir / "video_en.srt").exists())
            self.assertFalse((output_dir / "video_bn.srt").exists())


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
                work_dir=str(work_dir), save_report=True, unload_stt_before_tts=True,
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
            report = output_dir / "tiny_report.markdown"
            self.assertTrue(report.is_file() and report.stat().st_size > 0)
            report_text = report.read_text(encoding="utf-8")
            self.assertIn("Hello world.", report_text)
            self.assertIn("হ্যালো বিশ্ব।", report_text)
            self.assertFalse((output_dir / "tiny_en.srt").exists())
            self.assertFalse((output_dir / "tiny_bn.srt").exists())

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


class TestHotwords(unittest.TestCase):
    """আলাদা hotwords.txt ফাইল ও FC Mobile / খেলোয়াড় / ইভেন্ট হটওয়ার্ড পরীক্ষা।"""

    def test_hotwords_file_contains_fc_mobile_terms_events_and_players(self):
        from proxmax.hotwords import load_hotwords

        terms = load_hotwords(REPO_ROOT / "hotwords.txt")
        self.assertGreater(len(terms), 1_500)
        joined = " | ".join(terms)
        # FC Mobile UI, controls, and current game terms
        for term in (
            "FC Mobile", "OVR", "Star Pass", "Mascherano", "Skill Boost", "VSA", "H2H",
            "Exchange", "Player Exchange", "Play", "Game Plans", "Gauntlet Mode",
            "Football Centre", "Sprint & Skill", "Swipe Up", "Quick Match",
        ):
            self.assertIn(term, joined)
        # Current mobile PlayStyles, skill moves and events
        for term in (
            "Incisive Through Pass", "Dead Ball", "Finesse Expert", "Guardian",
            "Advanced Rainbow", "Lane Change", "Hocus Pocus", "3rd Anniversary",
            "Ballon d'Or 2026", "Unbreakable", "CONMEBOL Libertadores",
        ):
            self.assertIn(term, joined)
        # Events
        for event in ("TOTY", "UTOTY", "TOTS", "UTOTS", "Ballon d'Or", "Winter Wildcards", "Rulebreakers"):
            self.assertIn(event, joined)
        # Legends & current players
        for player in (
            "Pele", "Maradona", "Lionel Messi", "Cristiano Ronaldo", "Cruyff", "Zidane",
            "Ronaldinho", "Maldini", "Kylian Mbappe", "Erling Haaland", "Jude Bellingham",
            "Vinicius Junior", "Lamine Yamal", "Alexia Putellas", "Aitana Bonmati",
            "Aitana Bonmatí", "Ewa Pajor", "Michael Olise", "Willian Pacho",
        ):
            self.assertIn(player, joined)

    def test_context_prompt_is_bounded_and_selects_late_curated_names(self):
        from proxmax.hotwords import load_hotwords, select_whisper_hotwords

        terms = load_hotwords(REPO_ROOT / "hotwords.txt")
        prompt = select_whisper_hotwords(
            terms, max_chars=500, context="FC Mobile 27: Aitana Bonmati is on the squad."
        )
        self.assertLessEqual(len(prompt), 500)
        self.assertIn("Aitana Bonmati", prompt)
        self.assertIn("FC Mobile 27", prompt)

    def test_custom_hotwords_file_is_self_contained(self):
        from proxmax.hotwords import load_hotwords

        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "custom.txt"
            custom.write_text("Custom Club Term\n", encoding="utf-8")
            terms = load_hotwords(custom)
        self.assertEqual(terms, ["Custom Club Term"])

    def test_repair_transcript_hotwords_fixes_common_mishearings(self):
        from proxmax.hotwords import repair_transcript_hotwords

        raw = "We packed a 105 over card in the toty event with belling ham and mbappe!"
        fixed = repair_transcript_hotwords(raw)
        self.assertIn("105 OVR", fixed)
        self.assertIn("TOTY", fixed)
        self.assertIn("Bellingham", fixed)
        self.assertIn("Mbappe", fixed)

    def test_curated_player_aliases_get_exact_casing(self):
        from proxmax.hotwords import load_hotwords, repair_transcript_hotwords

        fixed = repair_transcript_hotwords(
            "dembélé and pacho", hotwords=load_hotwords(REPO_ROOT / "hotwords.txt")
        )
        self.assertEqual(fixed, "Dembélé and Pacho")

    def test_matched_hotwords_for_texts_extracts_relevant_terms(self):
        from proxmax.hotwords import load_hotwords, matched_hotwords_for_texts

        terms = load_hotwords(REPO_ROOT / "hotwords.txt")
        matched = matched_hotwords_for_texts(
            ["Ronaldinho and Gullit are insane in H2H during TOTY!"], terms
        )
        self.assertIn("Ronaldinho", matched)
        self.assertIn("Gullit", matched)
        self.assertIn("H2H", matched)
        self.assertIn("TOTY", matched)


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


class TestNotebookRequirements(unittest.TestCase):
    """নোটবুকের কাঠামো যাচাই (ব্যাজ, GPU ধাপ, ভয়েস ক্লোনিং, প্রি-লোড, API যাচাই)।"""

    def test_notebook_meets_all_user_constraints(self):
        nb_path = REPO_ROOT / "notebook" / "ProXMax_Colab.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        cells = nb["cells"]
        full_source = "\n".join("".join(c.get("source", [])) for c in cells)

        # ১. নোটবুকের ভেতরে উপরে অতিরিক্ত "Open in Colab" বাটন নেই
        header_source = "".join(cells[0].get("source", []))
        self.assertNotIn("colab-badge.svg", header_source)
        self.assertNotIn("Open In Colab", header_source)

        # ২. GPU যাচাই ধাপ (!nvidia-smi) নেই
        self.assertNotIn("nvidia-smi", full_source)
        self.assertNotIn("GPU যাচাই", full_source)

        # ৩. ভয়েস ক্লোনিং / IndicF5 ধাপ নেই
        self.assertNotIn("IndicF5", full_source)
        self.assertNotIn("কণ্ঠ ক্লোন", full_source)

        # ৪. ডিপেন্ডেন্সি ধাপেই Whisper মডেল প্রি-ডাউনলোড হয়
        self.assertIn("ensure_whisper_model", full_source)

        # ৫. Gemini API key সেলেই লাইভ API যাচাই হয়
        self.assertIn("verify_gemini_api", full_source)


if __name__ == "__main__":
    unittest.main()
