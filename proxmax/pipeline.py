"""Pipeline orchestrator: the whole dubbing flow + batch controller.

একটি ভিডিওর পুরো যাত্রা::

    video ─► [1] Audio Extract ─► [2] STT (faster-whisper)
          ─► [3] Translate En→Bn ─► [4] TTS (edge-tts, async)
          ─► [5] Sync (atempo + assemble) ─► [6] Mux ─► *_bn.mp4

``run_batch()`` input ফোল্ডারের সব ভিডিও একটার পর একটা প্রসেস করে।
Whisper মডেল ও translator একবারই তৈরি হয় এবং ব্যাচের সব ভিডিওতে
পুনরায় ব্যবহার হয় — এটাই বড় speed সাশ্রয়। একটি ভিডিও ব্যর্থ হলে
পুরো ব্যাচ থামে না; পরের ভিডিও চলতে থাকে, শেষে সারসংক্ষেপ দেখানো হয়।

সব সাময়িক ফাইল Colab-এর লোকাল ডিস্কে (``work_dir``) থাকে — Drive-এ
শুধু চূড়ান্ত ``*_bn.mp4`` ও SRT ফাইল কপি হয় (Drive/FUSE ধীর বলে)।
"""

from __future__ import annotations

import re
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional

from .audio_extraction import AudioExtractor
from .config import AppConfig
from .discovery import discover_videos
from .models import BatchSummary, Segment, VideoResult
from .muxer import VideoMuxer
from .stt import SpeechToText
from .sync_engine import SyncEngine
from .translator import create_translator
from .tts import BanglaTTS
from .utils import format_duration, get_logger, output_filename_for, write_srt

logger = get_logger(__name__)

_TOTAL_STEPS = 6


def _safe_dirname(name: str) -> str:
    """Filesystem-safe folder name derived from a video filename."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return safe or "video"


class DubbingPipeline:
    """End-to-end English→Bangla dubbing pipeline (reusable per batch)."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config.validate()
        cfg = self.config
        self.extractor = AudioExtractor(use_hwaccel=cfg.use_hwaccel)
        self.stt = SpeechToText(
            model_size=cfg.whisper_model,
            device=cfg.whisper_device,
            compute_type=cfg.whisper_compute_type,
            batch_size=cfg.whisper_batch_size,
            beam_size=cfg.whisper_beam_size,
            vad_filter=cfg.vad_filter,
            vad_min_silence_duration_ms=cfg.vad_min_silence_duration_ms,
        )
        self.translator = create_translator(
            cfg.translator_backend,
            translation_max_workers=cfg.translation_max_workers,
            translation_max_retries=cfg.translation_max_retries,
            indictrans2_model=cfg.indictrans2_model,
        )
        self.tts = BanglaTTS(
            voice=cfg.tts_voice,
            rate=cfg.tts_rate,
            volume=cfg.tts_volume,
            pitch=cfg.tts_pitch,
            concurrency=cfg.tts_concurrency,
            max_retries=cfg.tts_max_retries,
        )
        self.sync_engine = SyncEngine(
            min_tempo=cfg.min_tempo,
            max_tempo=cfg.max_tempo,
            slow_down_short_clips=cfg.slow_down_short_clips,
            sample_rate=cfg.sync_sample_rate,
            max_workers=cfg.sync_workers,
        )
        self.muxer = VideoMuxer(audio_bitrate=cfg.audio_bitrate, use_hwaccel=cfg.use_hwaccel)

    # ------------------------------------------------------------------
    @contextmanager
    def _step(self, number: int, title: str) -> Iterator[None]:
        logger.info("▶ ধাপ %d/%d: %s", number, _TOTAL_STEPS, title)
        started = time.perf_counter()
        yield
        logger.info("✓ ধাপ %d/%d শেষ (%s) — %.1fs", number, _TOTAL_STEPS, title,
                    time.perf_counter() - started)

    def output_path_for(self, video_path: Path, output_dir: Optional[Path] = None) -> Path:
        folder = output_dir or Path(self.config.output_dir)
        return folder / output_filename_for(video_path)

    # ------------------------------------------------------------------
    def process_one(
        self, video_path: Path | str, output_dir: Optional[Path] = None
    ) -> VideoResult:
        """Dub a single video; never raises — returns a :class:`VideoResult`."""
        video_path = Path(video_path)
        output_path = self.output_path_for(video_path, output_dir)
        started = time.perf_counter()

        if self.config.skip_existing and output_path.is_file() and output_path.stat().st_size > 0:
            logger.info("⏭️  ইতিমধ্যে আউটপুট আছে, বাদ দেওয়া হলো: %s", output_path.name)
            return VideoResult(
                video_path=video_path, status="skipped", output_path=output_path,
                elapsed_seconds=0.0,
            )

        work_dir = self.config.resolved_work_dir / _safe_dirname(video_path.stem)
        try:
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
            work_dir.mkdir(parents=True, exist_ok=True)

            segments = self._run_steps(video_path, work_dir, output_path)
            elapsed = time.perf_counter() - started
            logger.info(
                "✅ সম্পন্ন: %s → %s (মোট %s, %d segment)",
                video_path.name, output_path, format_duration(elapsed), len(segments),
            )
            return VideoResult(
                video_path=video_path, status="success", output_path=output_path,
                segment_count=len(segments), elapsed_seconds=elapsed,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            logger.error(
                "❌ ব্যর্থ: %s — %s: %s", video_path.name, type(exc).__name__, exc
            )
            logger.debug("Full traceback:", exc_info=True)
            return VideoResult(
                video_path=video_path, status="failed", error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=elapsed,
            )
        finally:
            if not self.config.keep_temp_files and work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    def _run_steps(
        self, video_path: Path, work_dir: Path, output_path: Path
    ) -> List[Segment]:
        """Execute steps 1–6 for one video (raises on unrecoverable error)."""
        from .utils import get_media_duration

        # 1) Audio extraction (16 kHz mono WAV) ------------------------------
        with self._step(1, "অডিও extract করা (ffmpeg)"):
            audio_wav = self.extractor.extract(video_path, work_dir / "audio_16k.wav")
            try:
                total_duration = get_media_duration(video_path)
            except Exception:
                total_duration = get_media_duration(audio_wav)

        # 2) Speech-to-Text -----------------------------------------------------
        with self._step(2, "Speech-to-Text (faster-whisper)"):
            segments = self.stt.transcribe(audio_wav)

        # 3) Translation En → Bn --------------------------------------------------
        with self._step(3, "অনুবাদ English → Bangla"):
            segments = self.translator.translate_segments(segments)
            translated = [s for s in segments if (s.text_bn or "").strip()]
            if not translated:
                raise RuntimeError("কোনো segment-এরই বাংলা অনুবাদ পাওয়া যায়নি।")
            if len(translated) < len(segments):
                logger.warning(
                    "%dটি segment-এর অনুবাদ নেই — সেগুলো বাদ দিয়ে এগোনো হচ্ছে।",
                    len(segments) - len(translated),
                )
            segments = translated

        # 4) Bangla TTS (async, concurrent) -----------------------------------------
        with self._step(4, "Bangla Text-to-Speech (edge-tts)"):
            segments = self.tts.synthesize_segments(segments, work_dir / "tts")

        # 5) Sync: time-stretch + assemble final track --------------------------------
        with self._step(5, "Audio-Video Sync (time-stretch)"):
            final_audio = self.sync_engine.build_final_track(
                segments, total_duration, work_dir, work_dir / "final_audio_bn.wav"
            )

        # 6) Mux locally, then copy the finished file to the output (Drive) folder -----
        with self._step(6, "Final Muxing (ভিডিও + Bangla অডিও)"):
            local_output = work_dir / output_filename_for(video_path)
            self.muxer.mux(video_path, final_audio, local_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_output, output_path)
            logger.info("আউটপুট সেভ হয়েছে: %s", output_path)

        # Optional subtitles (যাচাই ও পুনঃব্যবহারের জন্য উপকারী) -------------------
        if self.config.save_srt:
            try:
                stem = video_path.stem
                write_srt(segments, output_path.parent / f"{stem}_en.srt", language="en")
                write_srt(segments, output_path.parent / f"{stem}_bn.srt", language="bn")
                logger.info("সাবটাইটেল সেভ হয়েছে: %s_en.srt, %s_bn.srt", stem, stem)
            except Exception as exc:  # SRT ব্যর্থ হলে মূল কাজ ব্যর্থ ধরা হবে না
                logger.warning("SRT সেভ করা যায়নি: %s", exc)

        return segments

    # ------------------------------------------------------------------
    def run_batch(
        self,
        input_dir: Optional[Path | str] = None,
        output_dir: Optional[Path | str] = None,
    ) -> BatchSummary:
        """Process every video in the input folder, one after another."""
        input_folder = Path(input_dir or self.config.input_dir)
        output_folder = Path(output_dir or self.config.output_dir)
        summary = BatchSummary()
        batch_started = time.perf_counter()

        try:
            videos = discover_videos(input_folder)
        except FileNotFoundError as exc:
            logger.error(str(exc))
            return summary

        if not videos:
            logger.warning(
                "ℹ️  Input ফোল্ডারে কোনো ভিডিও (.mp4/.mkv/.mov/.avi) পাওয়া যায়নি: %s\n"
                "   Google Drive-এ Video/input ফোল্ডারে ভিডিও আপলোড করে আবার রান করুন।",
                input_folder,
            )
            return summary

        output_folder.mkdir(parents=True, exist_ok=True)
        logger.info("=" * 70)
        logger.info(
            "ব্যাচ শুরু: %dটি ভিডিও | মডেল=%s | translator=%s | voice=%s",
            len(videos), self.config.whisper_model,
            self.config.translator_backend, self.config.tts_voice,
        )
        logger.info("=" * 70)

        for position, video in enumerate(videos, start=1):
            logger.info("-" * 70)
            logger.info("🎬 [%d/%d] %s", position, len(videos), video.name)
            logger.info("-" * 70)
            summary.results.append(self.process_one(video, output_folder))

        summary.total_elapsed_seconds = time.perf_counter() - batch_started
        self._log_summary(summary, output_folder)
        return summary

    # Short alias, handy from notebooks: ``pipeline.run()``
    run = run_batch

    # ------------------------------------------------------------------
    @staticmethod
    def _log_summary(summary: BatchSummary, output_folder: Path) -> None:
        logger.info("=" * 70)
        logger.info("🏁 ব্যাচ সম্পন্ন — মোট সময় %s", format_duration(summary.total_elapsed_seconds))
        logger.info(
            "সফল: %d | বাদ দেওয়া (আগে থেকেই ছিল): %d | ব্যর্থ: %d",
            len(summary.succeeded), len(summary.skipped), len(summary.failed),
        )
        for result in summary.results:
            icon = {"success": "✅", "skipped": "⏭️", "failed": "❌"}.get(result.status, "•")
            detail = (
                f"→ {result.output_path.name}" if result.output_path
                else f"— {result.error}"
            )
            logger.info("  %s %s %s", icon, result.video_path.name, detail)
        logger.info("সব আউটপুট এখানে আছে: %s", output_folder)
        logger.info("=" * 70)
