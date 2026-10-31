"""Per-video processing and high-level pipeline stage orchestration."""

from __future__ import annotations

import shutil
import time
import wave
from pathlib import Path
from typing import List, Optional

from .models import Segment, StageTimings, VideoResult
from .prosody import build_units, unit_summary
from .pipeline_integrity import recover_invalid_waveforms, recover_missing_speech
from .pipeline_helpers import (
    _TOTAL_STEPS, _report_is_usable, _safe_dirname, existing_output_is_usable, _usable,
)
from .reporting import write_detailed_report
from .utils import format_duration, get_logger, get_media_duration, output_filename_for

logger = get_logger(__name__)


def _decoded_wav_duration(path: Path) -> Optional[float]:
    """Read the actual extracted audio length from its WAV header, without loading it."""
    try:
        with wave.open(str(path), "rb") as source:
            rate = source.getframerate()
            return source.getnframes() / float(rate) if rate > 0 else None
    except (OSError, EOFError, wave.Error, ValueError):
        return None


class PipelineProcessingMixin:
    """Methods for a single video's extraction, translation, sync, and mux."""

    def process_one(
        self, video_path: Path | str, output_dir: Optional[Path] = None
    ) -> VideoResult:
        """Dub a single video; never raises — returns a :class:`VideoResult`."""
        video_path = Path(video_path)
        output_path = self.output_path_for(video_path)
        if output_dir is not None:
            output_path = Path(output_dir) / output_filename_for(video_path)
        started = time.perf_counter()

        report_path = self.report_path_for(video_path, output_path.parent)
        if self.config.skip_existing and output_path.is_file():
            if existing_output_is_usable(output_path):
                report_ready = not self.config.save_report or _report_is_usable(report_path)
                if report_ready:
                    size_mb = output_path.stat().st_size / (1024 * 1024)
                    logger.info(
                        "⏭️  আগের আউটপুট ও রিপোর্ট পাওয়া গেছে (%.1f MB), তাই বাদ দেওয়া হলো: %s\n"
                        "   নতুন করে (নতুন কোড/কণ্ঠ দিয়ে) ডাবিং করতে চাইলে চালান: "
                        "python main.py --force",
                        size_mb, output_path.name,
                    )
                    return VideoResult(
                        video_path=video_path, status="skipped", output_path=output_path,
                        elapsed_seconds=0.0,
                        stats={"report": report_path.name} if self.config.save_report else {},
                    )
                logger.warning(
                    "⚠️ রিপোর্ট নেই, অসম্পূর্ণ অথবা অনুপস্থিত speech দেখায় — ভিডিওটি আবার প্রসেস করা হবে।"
                )
            else:
                logger.warning(
                    "⚠️ আগের আউটপুট অসম্পূর্ণ/পড়া যাচ্ছে না (%s) — আবার তৈরি করা হবে।",
                    output_path.name,
                )
                output_path.unlink(missing_ok=True)

        work_dir = self.config.resolved_work_dir / _safe_dirname(video_path.stem)
        try:
            if self.config.save_report:
                report_path.unlink(missing_ok=True)
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
            work_dir.mkdir(parents=True, exist_ok=True)

            segments, timings, stats = self._run_steps(video_path, work_dir, output_path)
            timings.total = time.perf_counter() - started
            if self.config.save_report:
                report_started = time.perf_counter()
                with self._step(6, "বিস্তারিত Markdown processing report"):
                    report_path = write_detailed_report(
                        video_path=video_path,
                        output_video_path=output_path,
                        output_dir=output_path.parent,
                        work_dir=work_dir,
                        segments=segments,
                        timings=timings,
                        stats=stats,
                        config=self.config,
                        total_duration=timings.audio_seconds,
                    )
                stats["report"] = {
                    "file": report_path.name,
                    "size_bytes": report_path.stat().st_size,
                    "render_seconds": round(time.perf_counter() - report_started, 3),
                }
            elapsed = time.perf_counter() - started
            timings.total = elapsed
            logger.info(
                "✅ সম্পন্ন: %s → %s (মোট %s, %d segment)\n   %s",
                video_path.name, output_path, format_duration(elapsed), len(segments),
                timings.as_text(),
            )
            return VideoResult(
                video_path=video_path, status="success", output_path=output_path,
                segment_count=len(segments), elapsed_seconds=elapsed,
                timings=timings, stats=stats,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            logger.error(
                "❌ ব্যর্থ: %s — %s: %s", video_path.name, type(exc).__name__, exc
            )
            logger.debug("Full traceback:", exc_info=True)
            return VideoResult(
                video_path=video_path, status="failed", output_path=output_path if output_path.is_file() else None,
                error=f"{type(exc).__name__}: {exc}", elapsed_seconds=elapsed,
            )
        finally:
            if not self.config.keep_temp_files and work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)

    def _run_steps(
        self, video_path: Path, work_dir: Path, output_path: Path
    ) -> tuple[List[Segment], StageTimings, dict]:
        """Execute media steps 1–5 for one video (raises on unrecoverable error)."""
        cfg = self.config
        timings = StageTimings()
        stats: dict = {}

        # 1) Audio extraction (16 kHz mono WAV — Whisper-এর ইনপুট) -----------
        marker = time.perf_counter()
        with self._step(1, "অডিও extract করা (ffmpeg)"):
            total_duration = 0.0
            try:
                total_duration = get_media_duration(video_path)
            except Exception as exc:
                logger.warning(
                    "ভিডিওর দৈর্ঘ্য জানা যায়নি (%s) — অডিও বের করার পর মাপা হবে।", exc
                )
            audio_wav = self.extractor.extract(
                video_path,
                work_dir / "audio_16k.wav",
                duration_seconds=total_duration or None,
            )
            decoded_audio_duration = _decoded_wav_duration(audio_wav)
            if total_duration <= 0:
                total_duration = self.extractor.last_duration or 0.0
            if decoded_audio_duration:
                total_duration = max(total_duration, decoded_audio_duration)
            if total_duration <= 0:
                try:
                    total_duration = get_media_duration(audio_wav)
                except Exception:
                    total_duration = 0.0
            if total_duration <= 0:
                raise RuntimeError("ভিডিওর দৈর্ঘ্য শনাক্ত করা যায়নি (duration = 0)।")
        timings.extract = time.perf_counter() - marker
        timings.audio_seconds = total_duration
        logger.info("ভিডিওর দৈর্ঘ্য: %s", format_duration(total_duration))

        # 2) Speech-to-Text -----------------------------------------------------
        marker = time.perf_counter()
        with self._step(2, "Speech-to-Text (faster-whisper, silence-aware + hotwords)"):
            setter = getattr(self.stt, "set_context_hint", None)
            if callable(setter):
                setter(video_path.stem)
            segments = self.stt.transcribe(audio_wav)
            if not segments:
                raise RuntimeError("ভিডিওতে কোনো কথা (speech) পাওয়া যায়নি।")
            stats["stt"] = dict(getattr(self.stt, "last_stats", {}) or {})
            missing_source = [s.index for s in segments if s.words and not (s.text_en or "").strip()]
            if missing_source:
                raise RuntimeError(f"Recognized source words have no transcript in segments {missing_source}")
            english_chars = sum(len((s.text_en or "").strip()) for s in segments)
            if english_chars == 0:
                raise RuntimeError("ভিডিওতে কোনো কথা (speech) পাওয়া যায়নি।")
        timings.stt = time.perf_counter() - marker

        # 2.5) Speech units: পুরো বাক্য একসাথে — TTS-এ মাঝপথে থামা বন্ধ ------
        units = build_units(
            segments,
            max_seconds=cfg.tts_unit_max_seconds,
            max_chars=cfg.tts_unit_max_chars,
            max_gap=cfg.tts_unit_max_gap,
            text_attr="text_en",
        )
        if units:
            logger.info(
                "Speech unit: %s — এক clip-এ পুরো বাক্য (স্বাভাবিক prosody)",
                unit_summary(units),
            )
            segments = units

        if cfg.unload_stt_before_tts:
            logger.info("TTS-এর আগে Whisper আনলোড করা হচ্ছে (VRAM খালি রাখতে)…")
            self.stt.release()

        # 3) Translation + TTS (একসাথে চলে) ---------------------------------
        spoken = sum(float(s.duration) for s in segments)
        tts_dir = work_dir / "tts"
        engine = self._ensure_tts(spoken)
        logger.info(
            "TTS ইঞ্জিন: %s | মোট কথা ≈ %.0fs",
            engine.describe(), spoken,
        )

        marker = time.perf_counter()
        with self._step(3, "অনুবাদ (Gemini) + Bangla TTS (ভয়েস)"):
            translate_seconds, tts_seconds = self._translate_and_synthesize(
                segments, tts_dir
            )
            recovered_translate, recovered_tts = recover_missing_speech(self, segments, tts_dir)
            translate_seconds += recovered_translate
            tts_seconds += recovered_tts
            synthesized = _usable(segments)
            if not synthesized:
                raise RuntimeError("একটি segment-এরও Bangla অডিও তৈরি হয়নি।")

        timings.translate_tts_wall = time.perf_counter() - marker
        timings.translate = translate_seconds
        timings.tts = tts_seconds
        stats["tts"] = {
            "engine": engine.describe(),
            "segments": len(synthesized),
            "total_audio_seconds": round(sum(float(s.tts_duration or 0.0) for s in synthesized), 2),
        }

        # 4) Trim speech, optionally slow it and plan per-window video edits.
        marker = time.perf_counter()
        invalid_tts_seconds = 0.0
        with self._step(4, "Audio-Video Sync (silence trim + video timeline edits)"):
            video_fps = self.muxer.video_frame_rate(video_path)
            picture_measure = getattr(self.muxer, "picture_timeline_duration", None)
            if decoded_audio_duration and callable(picture_measure):
                try:
                    measured_duration = max(decoded_audio_duration, picture_measure(video_path, video_fps))
                    if abs(measured_duration - total_duration) > 1 / float(video_fps):
                        logger.warning("Container duration %.3fs disagrees with measured audio/picture %.3fs; using measured timeline.",
                                       total_duration, measured_duration)
                    total_duration = measured_duration
                    timings.audio_seconds = total_duration
                except Exception as exc:
                    logger.warning("Could not independently measure source picture (%s); using duration hint.", exc)
            picture_source = self.muxer.ensure_picture_duration(video_path, total_duration, work_dir, video_fps)
            sync_passes = []

            def build_track():
                started = time.perf_counter()
                output = self.sync_engine.build_final_track(
                    segments, total_duration, work_dir, work_dir / "final_audio_bn.wav",
                    allow_empty_intermediate=True, video_fps=video_fps,
                )
                current = dict(getattr(self.sync_engine, "last_stats", {}) or {})
                stats["sync"] = current
                sync_passes.append({
                    "seconds": round(time.perf_counter() - started, 3),
                    "unplaced": int(current.get("unplaced", 0) or 0),
                    "prepared_reused": int(current.get("prepared_reused", 0) or 0),
                    "prepared_computed": int(current.get("prepared_computed", 0) or 0),
                })
                return output

            final_audio = build_track()
            if int(stats["sync"].get("unplaced", 0) or 0):
                invalid_tts_seconds, repaired = recover_invalid_waveforms(self, segments, tts_dir)
                if repaired:
                    timings.tts += invalid_tts_seconds
                    stats["tts"]["recovered_invalid_waveforms"] = repaired
                    final_audio = build_track()
            stats["sync_passes"] = sync_passes
            missing = [int(s.index) for s in _usable(segments) if s.placed_duration is None]
            if missing or int(stats["sync"].get("unplaced", 0) or 0) or stats["sync"].get("overlaps"):
                raise RuntimeError(f"Incomplete or overlapping dubbing; unplaced segments: {missing}")
            if self.sync_engine.video_plan is None:
                raise RuntimeError("No verified picture-edit plan was produced")
            timeline_duration = self.sync_engine.video_plan.duration
        timings.sync = max(0.0, time.perf_counter() - marker - invalid_tts_seconds)
        if invalid_tts_seconds:
            stats["tts"]["total_audio_seconds"] = round(
                sum(float(s.tts_duration or 0.0) for s in synthesized), 2
            )

        # 5) Mux locally, then copy the finished file to the output (Drive) ----
        marker = time.perf_counter()
        with self._step(5, "Final Muxing (ভিডিও + Bangla অডিও)"):
            local_output = work_dir / output_filename_for(video_path)
            mux_started = time.perf_counter()
            self.muxer.mux(
                picture_source, final_audio, local_output, required_video_duration=timeline_duration,
                video_plan=self.sync_engine.video_plan,
            )
            timings.mux_processing = time.perf_counter() - mux_started
            output_path.parent.mkdir(parents=True, exist_ok=True)
            copy_started = time.perf_counter()
            shutil.copy2(local_output, output_path)
            timings.output_copy = time.perf_counter() - copy_started
            logger.info("আউটপুট সেভ হয়েছে: %s", output_path)
        timings.mux = time.perf_counter() - marker

        if cfg.validate_dub_audio:
            validation_started = time.perf_counter()
            stats["audio_validation"] = self._validate_dub_audio(
                output_path, work_dir, segments, timeline_duration
            )
            timings.validation = time.perf_counter() - validation_started

        stats["segments"] = sum(
            len(getattr(unit, "children", None) or [unit]) for unit in segments
        )
        stats["units"] = sum(1 for unit in segments if getattr(unit, "is_unit", False)) or len(segments)
        stats["audio_seconds"] = round(total_duration, 2)
        stats["output_timeline_seconds"] = round(timeline_duration, 3)
        return segments, timings, stats
