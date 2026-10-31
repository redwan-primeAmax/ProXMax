"""Batch iteration and summary reporting for the dubbing pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from .discovery import discover_videos
from .models import BatchSummary
from .pipeline_helpers import _clock
from .utils import format_duration, get_logger

logger = get_logger(__name__)


class PipelineBatchMixin:
    """Batch processing helpers mixed into :class:`DubbingPipeline`."""

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
        cfg = self.config
        logger.info("=" * 70)
        logger.info(
            "ব্যাচ শুরু: %dটি ভিডিও | faster-whisper=%s | অনুবাদ=%s | কণ্ঠ=%s (%s) | "
            "speech tempo %.2f–1.00; video সর্বোচ্চ %.2f×",
            len(videos), cfg.whisper_model, self.translator.describe(),
            cfg.resolved_tts_voice, cfg.tts_gender, cfg.min_tempo, cfg.video_max_speed,
        )
        logger.info("=" * 70)

        for position, video in enumerate(videos, start=1):
            logger.info("-" * 70)
            logger.info("🎬 [%d/%d] %s", position, len(videos), video.name)
            logger.info("-" * 70)
            summary.results.append(self.process_one(video, output_folder))

        summary.total_elapsed_seconds = time.perf_counter() - batch_started
        self._log_summary(summary, output_folder)

        if cfg.release_gpu_when_done:
            logger.info("GPU মেমরি ছেড়ে দেওয়া হচ্ছে (মডেল আনলোড)…")
            self.release_gpu()
            logger.info("✅ GPU মুক্ত — এখন অন্য কাজে ব্যবহার করা যাবে।")
        return summary

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
            if result.timings is not None and result.status == "success":
                logger.info("     ⏱️  %s", result.timings.as_text())
        logger.info("সব আউটপুট এখানে আছে: %s", output_folder)
        logger.info("=" * 70)
