"""Smoke tests for ProXMax — no GPU / network required.

চালানো:  ``python -m unittest discover -s tests -v``

ভারী মডেল (Whisper) ও অনলাইন সার্ভিস (Google Translate, edge-tts)
mock করে শুধু নিজেদের লজিক — config, discovery, sync, muxing ও পুরো
pipeline-এর সংযোগ — পরীক্ষা করা হয়। ffmpeg থাকলে শেষ টেস্টটি আসল
ছোট্ট একটি ভিডিও বানিয়ে end-to-end ডাবিং করে দেখে।
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proxmax.config import AppConfig                              # noqa: E402
from proxmax.discovery import discover_videos                      # noqa: E402
from proxmax.models import Segment                                 # noqa: E402
from proxmax.sync_engine import build_atempo_filter, compute_tempo  # noqa: E402
from proxmax.translator import split_long_text                     # noqa: E402
from proxmax.utils import format_srt_timestamp, output_filename_for  # noqa: E402


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = AppConfig()
        self.assertEqual(cfg.whisper_model, "large-v3")
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
        self.assertEqual(cfg.translator_backend, "google")
        self.assertTrue(cfg.tts_voice.startswith("bn-BD-"))

    def test_overrides_ignore_none(self):
        cfg = AppConfig().apply_overrides({"whisper_model": "medium", "tts_voice": None})
        self.assertEqual(cfg.whisper_model, "medium")
        self.assertTrue(cfg.tts_voice.startswith("bn-BD-"))

    def test_section_scoped_aliases(self):
        # translation.max_retries ও tts.max_retries একে অন্যকে overwrite করবে না
        cfg = AppConfig()
        cfg._apply_mapping(
            {
                "stt": {"model": "medium", "batch_size": 4},
                "translation": {"backend": "google", "max_retries": 7},
                "tts": {"voice": "bn-BD-PradeepNeural", "max_retries": 2},
            }
        )
        self.assertEqual(cfg.whisper_model, "medium")
        self.assertEqual(cfg.whisper_batch_size, 4)
        self.assertEqual(cfg.translation_max_retries, 7)
        self.assertEqual(cfg.tts_max_retries, 2)
        self.assertEqual(cfg.tts_voice, "bn-BD-PradeepNeural")

    def test_invalid_model_rejected(self):
        cfg = AppConfig(whisper_model="not-a-model")
        with self.assertRaises(ValueError):
            cfg.validate()


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

    def test_split_long_text(self):
        long_text = " ".join(["Hello world."] * 600)  # ~7800 chars
        chunks = split_long_text(long_text, max_chars=4500)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(len(c) <= 4500 for c in chunks))
        self.assertEqual(split_long_text(""), [])


# ---------------------------------------------------------------------------
# End-to-end: আসল ffmpeg/pydub দিয়ে, mock STT + mock Translator + mock TTS
# ---------------------------------------------------------------------------
class _FakeSTT:
    def transcribe(self, audio_path):
        return [
            Segment(index=0, start=0.3, end=2.3, text_en="Hello world."),
            Segment(index=1, start=2.8, end=5.4, text_en="This is a test video."),
        ]


class _FakeTranslator:
    def translate_segments(self, segments):
        for seg in segments:
            seg.text_bn = "হ্যালো বিশ্ব।" if seg.index == 0 else "এটি একটি পরীক্ষামূলক ভিডিও।"
        return segments


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
        from proxmax.utils import get_media_duration, run_ffmpeg

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
