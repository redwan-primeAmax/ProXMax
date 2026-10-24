"""Smoke tests for ProXMax — no GPU / network required.

চালানো:  ``python -m unittest discover -s tests -v``

ভারী মডেল (Whisper), Gemini API ও edge-tts mock করা হয় — শুধু নিজেদের
লজিক (config, discovery, translation batch/retry/parsing, sync, GPU
release ও পুরো pipeline-এর সংযোগ) পরীক্ষা করা হয়। ffmpeg থাকলে শেষ
টেস্টটি আসল ছোট্ট একটি ভিডিও বানিয়ে end-to-end ডাবিং করে দেখে।
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proxmax.config import DEFAULT_GEMINI_MODEL, AppConfig            # noqa: E402
from proxmax.discovery import discover_videos                          # noqa: E402
from proxmax.models import Segment                                     # noqa: E402
from proxmax.stt import SpeechToText                                   # noqa: E402
from proxmax.sync_engine import build_atempo_filter, compute_tempo     # noqa: E402
from proxmax.translator import (                                       # noqa: E402
    GeminiAuthError,
    GeminiTranslator,
    TranslationError,
    create_translator,
    parse_translations,
)
from proxmax.utils import (                                            # noqa: E402
    format_srt_timestamp,
    output_filename_for,
    release_cuda_memory,
)


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = AppConfig()
        self.assertEqual(cfg.whisper_model, "large-v3")
        self.assertEqual(cfg.translator_backend, "gemini")
        self.assertEqual(cfg.gemini_model, "gemini-3.5-flash-light")
        self.assertEqual(cfg.gemini_max_retries, 7)
        self.assertTrue(cfg.input_dir.endswith("Video/input"))
        self.assertTrue(cfg.output_dir.endswith("Video/output"))
        cfg.validate()

    def test_yaml_repo_config_loads(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML not installed")
        cfg = AppConfig.from_yaml(REPO_ROOT / "config.yaml").validate()
        self.assertEqual(cfg.whisper_model, "large-v3")
        self.assertEqual(cfg.translator_backend, "gemini")
        self.assertEqual(cfg.gemini_model, DEFAULT_GEMINI_MODEL)
        self.assertEqual(cfg.gemini_max_retries, 7)
        self.assertGreaterEqual(cfg.gemini_requests_per_minute, 1)
        self.assertTrue(cfg.tts_voice.startswith("bn-BD-"))

    def test_overrides_ignore_none(self):
        cfg = AppConfig().apply_overrides(
            {"whisper_model": "medium", "gemini_model": "custom-model", "tts_voice": None}
        )
        self.assertEqual(cfg.whisper_model, "medium")
        self.assertEqual(cfg.gemini_model, "custom-model")
        self.assertTrue(cfg.tts_voice.startswith("bn-BD-"))

    def test_section_scoped_aliases(self):
        # translation/gemini.max_retries ও tts.max_retries একে অন্যকে overwrite করবে না
        cfg = AppConfig()
        cfg._apply_mapping(
            {
                "stt": {"model": "medium", "batch_size": 4},
                "translation": {"backend": "gemini"},
                "gemini": {
                    "model": "gemini-test-model",
                    "api_key": "test-key",
                    "max_retries": 6,
                    "requests_per_minute": 10,
                },
                "tts": {"voice": "bn-BD-PradeepNeural", "max_retries": 2},
                "runtime": {"release_gpu_when_done": False},
            }
        )
        self.assertEqual(cfg.whisper_model, "medium")
        self.assertEqual(cfg.whisper_batch_size, 4)
        self.assertEqual(cfg.translator_backend, "gemini")
        self.assertEqual(cfg.gemini_model, "gemini-test-model")
        self.assertEqual(cfg.gemini_api_key, "test-key")
        self.assertEqual(cfg.gemini_max_retries, 6)
        self.assertEqual(cfg.gemini_requests_per_minute, 10)
        self.assertEqual(cfg.tts_max_retries, 2)
        self.assertEqual(cfg.tts_voice, "bn-BD-PradeepNeural")
        self.assertFalse(cfg.release_gpu_when_done)

    def test_invalid_model_rejected(self):
        cfg = AppConfig(whisper_model="not-a-model")
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_gemini_settings_rejected(self):
        with self.assertRaises(ValueError):
            AppConfig(gemini_model="   ").validate()
        with self.assertRaises(ValueError):
            AppConfig(gemini_max_retries=0).validate()
        with self.assertRaises(ValueError):
            AppConfig(gemini_requests_per_minute=0).validate()
        with self.assertRaises(ValueError):
            AppConfig(gemini_temperature=5.0).validate()

    def test_google_backend_removed(self):
        with self.assertRaises(ValueError):
            AppConfig(translator_backend="google").validate()


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

    def test_compute_tempo_clamps(self):
        # Bangla দ্বিগুণ লম্বা → tempo 2.0 চাই, কিন্তু max 1.90-তে আটকাবে
        self.assertAlmostEqual(compute_tempo(10.0, 5.0, max_tempo=1.90), 1.90)
        # Bangla অর্ধেক লম্বা → tempo 0.5 চাই, min 0.65-তে আটকাবে
        self.assertAlmostEqual(compute_tempo(2.5, 5.0, min_tempo=0.65), 0.65)
        # slow_down বন্ধ থাকলে ছোট ক্লিপ স্বাভাবিক গতিতেই থাকবে
        self.assertAlmostEqual(
            compute_tempo(2.5, 5.0, slow_down_short_clips=False), 1.0
        )
        self.assertAlmostEqual(compute_tempo(5.0, 5.0), 1.0)

    def test_release_cuda_memory_is_safe_without_gpu(self):
        release_cuda_memory()  # GPU/Torch ছাড়াও কোনো error ওঠা যাবে না


# ---------------------------------------------------------------------------
# Gemini translator: parsing, batching, retry, rate-limit ও graceful failure
# ---------------------------------------------------------------------------
class _FakeAPIError(Exception):
    """google.genai.errors.APIError-এর মতো code সহ exception।"""

    def __init__(self, code: int, message: str = "boom", details=None):
        super().__init__(f"{code} ERROR. {message}")
        self.code = code
        self.details = details


class _FakeFeedback:
    def __init__(self, block_reason=None):
        self.block_reason = block_reason


class _FakeResponse:
    def __init__(self, text, block_reason=None):
        self.text = text
        self.prompt_feedback = _FakeFeedback(block_reason)


class _FakeClient:
    """Minimal stand-in for ``google.genai.Client`` (no network)।"""

    def __init__(self, responder):
        self.responder = responder
        self.calls = []
        self.closed = 0
        self.models = self  # google.genai.Client-এর মতো ``client.models`` API

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return _FakeResponse(self.responder(contents, len(self.calls)))

    def close(self) -> None:
        self.closed += 1


def _lines_from_contents(contents: str):
    payload = json.loads(contents.split("\n", 1)[1])
    return payload["lines"]


def _make_translator(client, **kwargs) -> GeminiTranslator:
    params = dict(
        model=DEFAULT_GEMINI_MODEL,
        batch_size=10,
        max_retries=3,
        retry_base_delay=0.01,
        retry_max_delay=0.02,
        requests_per_minute=60_000,
        timeout=10,
        client=client,
    )
    params.update(kwargs)
    return GeminiTranslator(**params)


class TestGeminiParsing(unittest.TestCase):
    def test_plain_and_dict_variants(self):
        self.assertEqual(parse_translations('{"translations": ["ক", "খ"]}', 2), ["ক", "খ"])
        self.assertEqual(parse_translations('["ক", "খ"]', 2), ["ক", "খ"])
        self.assertEqual(parse_translations('{"lines": ["ক"]}', 1), ["ক"])
        self.assertEqual(
            parse_translations('[{"translation": "ক"}]', 1), ["ক"]
        )

    def test_code_fence_is_tolerated(self):
        raw = '```json\n{"translations": ["হ্যালো"]}\n```'
        self.assertEqual(parse_translations(raw, 1), ["হ্যালো"])

    def test_count_mismatch_raises_format_error(self):
        with self.assertRaises(TranslationError):
            parse_translations('{"translations": ["একটাই"]}', 3)

    def test_invalid_json_raises_format_error(self):
        with self.assertRaises(TranslationError):
            parse_translations("not json at all", 1)


class TestGeminiTranslator(unittest.TestCase):
    def _segments(self, texts):
        return [
            Segment(index=i, start=float(i), end=float(i) + 1, text_en=text)
            for i, text in enumerate(texts)
        ]

    def test_batches_and_preserves_order(self):
        def responder(contents, call_no):
            lines = _lines_from_contents(contents)
            return json.dumps(
                {"translations": [f"অনুবাদ-{line}" for line in lines]},
                ensure_ascii=False,
            )

        client = _FakeClient(responder)
        translator = _make_translator(client, batch_size=2)
        segments = self._segments(["One.", "Two.", "Three."])
        translator.translate_segments(segments)

        self.assertEqual(
            [s.text_bn for s in segments],
            ["অনুবাদ-One.", "অনুবাদ-Two.", "অনুবাদ-Three."],
        )
        self.assertEqual(len(client.calls), 2)  # ৩টি লাইন → ২+১ batch
        # প্রতিটি request-এ ডিফল্ট মডেল ও JSON response schema যায়
        call = client.calls[0]
        self.assertEqual(call["model"], DEFAULT_GEMINI_MODEL)
        self.assertEqual(call["config"].response_mime_type, "application/json")

    def test_duplicate_texts_use_one_api_call(self):
        def responder(contents, call_no):
            lines = _lines_from_contents(contents)
            return json.dumps({"translations": lines}, ensure_ascii=False)

        client = _FakeClient(responder)
        translator = _make_translator(client)
        segments = self._segments(["Hello.", "Hello.", "World."])
        translator.translate_segments(segments)

        self.assertEqual([s.text_bn for s in segments], ["Hello.", "Hello.", "World."])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(_lines_from_contents(client.calls[0]["contents"]), ["Hello.", "World."])

    def test_retry_then_success(self):
        """প্রথম call 429 → অপেক্ষা করে আবার চেষ্টা → সফল।"""

        def responder(contents, call_no):
            if call_no == 1:
                raise _FakeAPIError(429, "rate limit exceeded")
            lines = _lines_from_contents(contents)
            return json.dumps({"translations": [f"বাংলা {x}" for x in lines]})

        client = _FakeClient(responder)
        translator = _make_translator(client, max_retries=3)
        with mock.patch("proxmax.translator.time.sleep") as sleeper:
            result = translator.translate_text("Hello world")

        self.assertEqual(result, "বাংলা Hello world")
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(sleeper.called)

    def test_all_retries_exhausted_raises_clean_error(self):
        def responder(contents, call_no):
            raise _FakeAPIError(503, "service unavailable")

        client = _FakeClient(responder)
        translator = _make_translator(client, max_retries=7)
        with mock.patch("proxmax.translator.time.sleep"):
            with self.assertLogs("proxmax.translator", level="ERROR") as logs:
                with self.assertRaises(TranslationError) as ctx:
                    translator.translate_text("Hello")

        self.assertEqual(len(client.calls), 7)
        self.assertIn("7 বার", str(ctx.exception))
        self.assertTrue(any("503" in line for line in logs.output))

    def test_auth_error_fails_fast_and_is_cached(self):
        def responder(contents, call_no):
            raise _FakeAPIError(403, "API key not valid. Please pass a valid API key.")

        client = _FakeClient(responder)
        translator = _make_translator(client, max_retries=7)
        with mock.patch("proxmax.translator.time.sleep"):
            with self.assertRaises(TranslationError):
                translator.translate_text("Hello")
            with self.assertRaises(TranslationError):
                translator.translate_text("World")

        # non-retryable → ১টি call; দ্বিতীয়বার আর API-তে যায়নি
        self.assertEqual(len(client.calls), 1)

    def test_format_mismatch_splits_batch(self):
        def responder(contents, call_no):
            lines = _lines_from_contents(contents)
            # একাধিক লাইন থাকলে ইচ্ছাকৃতভাবে ভুল সংখ্যক অনুবাদ দেয়
            translations = lines if len(lines) == 1 else lines[:-1]
            return json.dumps({"translations": translations}, ensure_ascii=False)

        client = _FakeClient(responder)
        translator = _make_translator(client, batch_size=3, max_retries=2)
        segments = self._segments(["One.", "Two.", "Three."])
        with mock.patch("proxmax.translator.time.sleep"):
            translator.translate_segments(segments)

        self.assertEqual([s.text_bn for s in segments], ["One.", "Two.", "Three."])
        self.assertGreater(len(client.calls), 3)  # ছোট ব্যাচে ভেঙে হয়েছে

    def test_empty_segment_texts_are_skipped(self):
        def responder(contents, call_no):
            raise AssertionError("API call করা উচিত নয়")

        client = _FakeClient(responder)
        translator = _make_translator(client)
        segments = self._segments(["", "   "])
        translator.translate_segments(segments)

        self.assertEqual([s.text_bn for s in segments], ["", ""])
        self.assertEqual(client.calls, [])

    def test_close_releases_client(self):
        client = _FakeClient(lambda contents, n: '{"translations": []}')
        translator = _make_translator(client)
        translator.close()
        self.assertEqual(client.closed, 1)
        self.assertIsNone(translator._client)
        translator.close()  # বারবার ডাকা নিরাপদ

    def test_network_errors_are_retried(self):
        """httpx-জাতীয় নেটওয়ার্ক error retry যোগ্য।"""
        connect_error = type("ConnectError", (Exception,), {})
        connect_error.__module__ = "httpx"

        def responder(contents, call_no):
            if call_no == 1:
                raise connect_error("connection refused")
            lines = _lines_from_contents(contents)
            return json.dumps({"translations": [f"বাংলা {x}" for x in lines]})

        client = _FakeClient(responder)
        translator = _make_translator(client, max_retries=3)
        with mock.patch("proxmax.translator.time.sleep"):
            result = translator.translate_text("Hello")

        self.assertEqual(result, "বাংলা Hello")
        self.assertEqual(len(client.calls), 2)

    def test_content_block_is_not_retried(self):
        class _BlockedClient(_FakeClient):
            def generate_content(self, *, model, contents, config):
                self.calls.append({"model": model, "contents": contents, "config": config})
                return _FakeResponse(None, block_reason="SAFETY")

        client = _BlockedClient(lambda contents, n: "")
        translator = _make_translator(client, max_retries=7)
        with mock.patch("proxmax.translator.time.sleep"):
            with self.assertRaises(TranslationError) as ctx:
                translator.translate_text("Hello")

        self.assertEqual(len(client.calls), 1)
        self.assertIn("block_reason", str(ctx.exception))
        # ব্লক শুধু ওই অংশের জন্য — translator-কে স্থায়ীভাবে বন্ধ করে না
        self.assertIsNone(translator._fatal_error)

    def test_retry_after_header_is_respected(self):
        exc = _FakeAPIError(
            429,
            "quota",
            details={
                "error": {
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.RetryInfo",
                            "retryDelay": "3s",
                        }
                    ]
                }
            },
        )
        translator = _make_translator(
            _FakeClient(lambda contents, n: '{"translations": []}'),
            retry_max_delay=90.0,
        )
        self.assertAlmostEqual(translator._retry_delay(1, exc, 429), 4.0, places=1)


class TestTranslatorFactory(unittest.TestCase):
    def test_gemini_backend(self):
        translator = create_translator(
            "gemini", gemini_api_key="test-key", gemini_model="gemini-test-model"
        )
        self.assertIsInstance(translator, GeminiTranslator)
        self.assertEqual(translator.model, "gemini-test-model")
        self.assertEqual(translator.describe(), "gemini:gemini-test-model")
        translator.close()

    def test_missing_api_key_raises_with_hint(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(GeminiAuthError) as ctx:
                create_translator("gemini", gemini_api_key="")
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))

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


class TestSttRelease(unittest.TestCase):
    def test_release_without_model_is_safe(self):
        stt = SpeechToText(model_size="tiny")
        stt.release()
        self.assertIsNone(stt._model)
        self.assertIsNone(stt._batched)


# ---------------------------------------------------------------------------
# End-to-end: আসল ffmpeg/pydub দিয়ে, mock STT + mock Translator + mock TTS
# ---------------------------------------------------------------------------
class _FakeSTT:
    def __init__(self):
        self.released = False

    def transcribe(self, audio_path):
        return [
            Segment(index=0, start=0.3, end=2.3, text_en="Hello world."),
            Segment(index=1, start=2.8, end=5.4, text_en="This is a test video."),
        ]

    def release(self):
        self.released = True


class _FakeTranslator:
    def __init__(self):
        self.closed = False

    def translate_segments(self, segments):
        for seg in segments:
            seg.text_bn = "হ্যালো বিশ্ব।" if seg.index == 0 else "এটি একটি পরীক্ষামূলক ভিডিও।"
        return segments

    def describe(self):
        return "fake:translator"

    def close(self):
        self.closed = True


class _FakeTTS:
    """TTS-এর বদলে প্রতিটি segment-এর জন্য sine-tone WAV বানায়।"""

    def synthesize_segments(self, segments, output_dir):
        from proxmax.utils import get_media_duration, run_ffmpeg

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ffmpeg = shutil.which("ffmpeg")
        for seg in segments:
            path = output_dir / f"seg_{seg.index:04d}.wav"
            # মূল segment-এর চেয়ে ইচ্ছাকৃতভাবে একটু লম্বা ক্লিপ → atempo লাগবে
            dur = seg.duration * 1.4
            run_ffmpeg(
                [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", f"sine=frequency=440:duration={dur:.2f}",
                 "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(path)],
                description="fake tts tone",
            )
            seg.tts_audio_path = path
            seg.tts_duration = get_media_duration(path)
        return segments


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
class TestPipelineEndToEnd(unittest.TestCase):
    def test_full_pipeline_with_mocks(self):
        try:
            import pydub  # noqa: F401
        except ImportError:
            self.skipTest("pydub not installed")

        from proxmax.pipeline import DubbingPipeline
        from proxmax.utils import get_media_duration

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir, output_dir, work_dir = root / "input", root / "output", root / "work"
            input_dir.mkdir()

            # ৬ সেকেন্ডের ছোট্ট আসল ভিডিও (color picture + sine audio)
            video = input_dir / "tiny.mp4"
            ffmpeg = shutil.which("ffmpeg")
            proc = subprocess.run(
                [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "color=c=navy:s=320x240:r=25:d=6",
                 "-f", "lavfi", "-i", "sine=frequency=220:duration=6",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-shortest", str(video)],
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)

            cfg = AppConfig(
                input_dir=str(input_dir), output_dir=str(output_dir),
                work_dir=str(work_dir), save_srt=True,
            )
            pipeline = DubbingPipeline.__new__(DubbingPipeline)  # bypass heavy init
            pipeline.config = cfg.validate()
            from proxmax.audio_extraction import AudioExtractor
            from proxmax.muxer import VideoMuxer
            from proxmax.sync_engine import SyncEngine

            pipeline.extractor = AudioExtractor(use_hwaccel=False)
            pipeline.stt = _FakeSTT()
            pipeline.translator = _FakeTranslator()
            pipeline.tts = _FakeTTS()
            pipeline.sync_engine = SyncEngine()
            pipeline.muxer = VideoMuxer(use_hwaccel=False)

            summary = pipeline.run_batch()
            self.assertEqual(len(summary.succeeded), 1, summary.failed)

            out = output_dir / "tiny_bn.mp4"
            self.assertTrue(out.is_file() and out.stat().st_size > 0)
            duration = get_media_duration(out)
            self.assertAlmostEqual(duration, 6.0, delta=1.0)
            self.assertTrue((output_dir / "tiny_en.srt").is_file())
            self.assertTrue((output_dir / "tiny_bn.srt").is_file())

            # কাজ শেষে GPU release বাধ্যতামূলকভাবে ডাকা হয়েছে
            self.assertTrue(pipeline.stt.released)
            self.assertTrue(pipeline.translator.closed)

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
