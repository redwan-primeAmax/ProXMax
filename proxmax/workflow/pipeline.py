"""Reusable end-to-end English-to-Bangla dubbing pipeline."""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from .audio_extraction import AudioExtractor
from .config import AppConfig
from .models import BatchSummary, Segment, StageTimings, VideoResult
from .pipeline_batch import PipelineBatchMixin
from .pipeline_helpers import (
    _MIN_USABLE_OUTPUT_BYTES, _MIN_USABLE_OUTPUT_SECONDS, _TOTAL_STEPS, _clock,
    _report_is_usable, _safe_dirname, _usable, existing_output_is_usable,
)
from .pipeline_process import PipelineProcessingMixin
from .pipeline_translation import PipelineTranslationMixin
from .stt import SpeechToText
from .muxer import VideoMuxer
from .reporting import report_path_for as build_report_path
from .sync_engine import SyncEngine
from .translator import create_translator
from .tts import BaseTTSEngine, build_tts_engine
from .utils import format_duration, get_logger, get_media_duration, output_filename_for

logger = get_logger(__name__)


class DubbingPipeline(
    PipelineBatchMixin, PipelineProcessingMixin, PipelineTranslationMixin
):
    """End-to-end English-to-Bangla dubbing pipeline (reusable per batch)."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config.validate()
        cfg = self.config
        self.extractor = AudioExtractor(
            use_hwaccel=False,
            stall_timeout=cfg.extract_stall_seconds,
            audio_filter=cfg.stt_audio_filter,
        )
        self.stt = SpeechToText(
            model_size=cfg.whisper_model,
            device=cfg.whisper_device,
            compute_type=cfg.whisper_compute_type,
            beam_size=cfg.whisper_beam_size,
            vad_min_silence_ms=cfg.vad_min_silence_ms,
            sentence_max_seconds=cfg.tts_unit_max_seconds,
            chunk_min_seconds=cfg.chunk_min_seconds,
            chunk_max_seconds=cfg.chunk_max_seconds,
            chunk_pad_ms=cfg.chunk_pad_ms,
            carry_context_chars=cfg.carry_context_chars,
            condition_on_previous_text=cfg.condition_on_previous_text,
            seam_repair=cfg.seam_repair,
            seam_window_seconds=cfg.seam_window_seconds,
            max_seam_repairs=cfg.max_seam_repairs,
            no_speech_threshold=cfg.no_speech_threshold,
            compression_ratio_threshold=cfg.compression_ratio_threshold,
            log_prob_threshold=cfg.log_prob_threshold,
            hotwords=cfg.stt_hotwords,
            hotwords_file=cfg.stt_hotwords_file,
        )
        self.translator = create_translator(
            cfg.translator_backend,
            gemini_api_key=cfg.resolved_gemini_api_key(),
            gemini_model=cfg.gemini_model,
            gemini_batch_size=cfg.gemini_batch_size,
            gemini_max_chars_per_request=cfg.gemini_max_chars_per_request,
            gemini_max_input_tokens=cfg.gemini_max_input_tokens,
            gemini_max_retries=cfg.gemini_max_retries,
            gemini_retry_base_delay=cfg.gemini_retry_base_delay,
            gemini_retry_max_delay=cfg.gemini_retry_max_delay,
            gemini_retry_cap_delay=cfg.gemini_retry_cap_delay,
            gemini_requests_per_minute=cfg.gemini_requests_per_minute,
            gemini_temperature=cfg.gemini_temperature,
            gemini_timeout=cfg.gemini_timeout,
            gemini_max_output_tokens=cfg.gemini_max_output_tokens,
            char_rate=cfg.translation_char_rate,
            indictrans2_model=cfg.indictrans2_model,
            hotwords=cfg.stt_hotwords,
            hotwords_file=cfg.stt_hotwords_file,
        )
        self.sync_engine = SyncEngine(
            min_tempo=cfg.min_tempo,
            max_tempo=cfg.max_tempo,
            slow_down_short_clips=cfg.slow_down_short_clips,
            sample_rate=cfg.sync_sample_rate,
            max_workers=cfg.sync_workers,
            slot_gap_ms=cfg.slot_gap_ms,
            borrow_silence_ms=cfg.borrow_silence_ms,
            prefer_rubberband=cfg.prefer_rubberband,
            fill_extension=cfg.fill_extension,
            silence_pad_ms=cfg.tts_trim_pad_ms,
            video_max_speed=cfg.video_max_speed,
        )
        self.muxer = VideoMuxer(audio_bitrate=cfg.audio_bitrate, use_hwaccel=cfg.use_hwaccel,
                                video_workers=cfg.video_workers)

        # ব্যাচ জুড়ে একবারই তৈরি হয়
        self._tts: Optional[BaseTTSEngine] = None

    def release_gpu(self) -> None:
        """সব মডেল আনলোড করে GPU (VRAM) ও API client ছেড়ে দেয়।"""
        for name, closer in (
            ("TTS", getattr(self._tts, "close", None)),
            ("Whisper", getattr(self.stt, "release", None)),
            ("Translator", getattr(self.translator, "close", None)),
        ):
            if closer is None:
                continue
            try:
                closer()
            except Exception as exc:  # pragma: no cover - best effort cleanup
                logger.debug("%s release-এ সমস্যা: %s", name, exc)
        self._tts = None

    def close(self) -> None:
        """Alias of :meth:`release_gpu` — pipeline-এর সব resource ছেড়ে দেয়।"""
        self.release_gpu()

    def _ensure_tts(self, total_speech_seconds: float) -> BaseTTSEngine:
        """TTS ইঞ্জিন তৈরি (বা পুনঃব্যবহার) করে।"""
        cfg = self.config
        total_speech_seconds = max(1.0, float(total_speech_seconds))

        if self._tts is None:
            self._tts = build_tts_engine(
                cfg,
                total_audio_seconds=total_speech_seconds,
            )
        else:
            self._tts.set_budget(None, total_speech_seconds)
        return self._tts

    @contextmanager
    def _step(self, number: int, title: str) -> Iterator[None]:
        logger.info("▶ ধাপ %d/%d: %s", number, _TOTAL_STEPS, title)
        started = time.perf_counter()
        trace = getattr(self, "_trace", None)
        if trace is not None:
            trace.start_stage(number)
        try:
            yield
        except BaseException as exc:
            if trace is not None:
                trace.end_stage(number, exc)
            raise
        else:
            if trace is not None:
                trace.end_stage(number)
            logger.info(
                "✓ ধাপ %d/%d শেষ (%s) — %.1fs", number, _TOTAL_STEPS, title,
                time.perf_counter() - started,
            )

    def output_path_for(self, video_path: Path, output_dir: Optional[Path] = None) -> Path:
        folder = output_dir or Path(self.config.output_dir)
        return folder / output_filename_for(video_path)

    def report_path_for(self, video_path: Path, output_dir: Optional[Path] = None) -> Path:
        folder = output_dir or Path(self.config.output_dir)
        return build_report_path(video_path, folder)


def run_Single_or_batch(
    config: AppConfig,
    single_video: Optional[Path | str] = None,
    progress: Optional[Callable[[VideoResult], None]] = None,
) -> BatchSummary:
    """Convenience runner used by both CLI and Colab notebook."""
    pipeline = DubbingPipeline(config)
    if single_video is not None:
        started = time.perf_counter()
        summary = BatchSummary()
        try:
            res = pipeline.process_one(single_video, Path(config.output_dir))
            summary.results.append(res)
            if progress:
                progress(res)
        finally:
            summary.total_elapsed_seconds = time.perf_counter() - started
            if config.release_gpu_when_done:
                pipeline.release_gpu()
            pipeline._log_summary(summary, Path(config.output_dir))
        return summary
    return pipeline.run_batch()
