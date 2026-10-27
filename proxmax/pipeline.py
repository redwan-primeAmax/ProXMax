"""Pipeline orchestrator: the whole dubbing flow + batch controller.

একটি ভিডিওর পুরো যাত্রা::

    video ─► [1] Audio Extract
          ─► [2] STT (faster-whisper, silence-aware chunking + hotwords.txt)
          ─► [3] অনুবাদ (Gemini) + TTS (বিল্ট-ইন বাংলা neural ভয়েস)
          ─► [4] Sync (non-overlap, tempo ≤ 1.25) ─► [5] Mux ─► *_bn.mp4
          ─► [6] SRT

``run_batch()`` input ফোল্ডারের সব ভিডিও একটার পর একটা প্রসেস করে।
Whisper, Gemini client ও TTS ইঞ্জিন একবারই তৈরি হয় এবং ব্যাচের সব ভিডিওতে
পুনরায় ব্যবহার হয় — বড় speed সাশ্রয়।
"""

from __future__ import annotations

import queue
import re
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, List, Optional

from .audio_extraction import AudioExtractor
from .config import AppConfig
from .discovery import discover_videos
from .models import BatchSummary, Segment, StageTimings, VideoResult
from .muxer import VideoMuxer
from .prosody import build_units, subtitles_for_unit, unit_summary
from .speech_rate import calibrate as calibrate_speech_rate
from .stt import SpeechToText
from .sync_engine import SyncEngine
from .translator import create_translator
from .tts import (
    BaseTTSEngine,
    EdgeTTSEngine,
    build_tts_engine,
    resolve_tts_engine,
)
from .utils import (
    format_duration,
    get_logger,
    get_media_duration,
    output_filename_for,
    write_srt,
)

logger = get_logger(__name__)


def _clock(seconds: float) -> str:
    """সেকেন্ডকে ``MM:SS`` আকারে দেখায় (লগে সময় বোঝার জন্য)।"""
    total = max(0, int(round(float(seconds or 0.0))))
    return f"{total // 60:02d}:{total % 60:02d}"


_TOTAL_STEPS = 6


def _safe_dirname(name: str) -> str:
    """Filesystem-safe folder name derived from a video filename."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return safe or "video"


#: এর চেয়ে ছোট ফাইল কখনোই বৈধ ডাবিং করা ভিডিও নয় (অর্ধেক লেখা/ভাঙা ফাইল)
_MIN_USABLE_OUTPUT_BYTES = 20 * 1024
#: এর কম duration-এর MP4 আসলে ডাবিং করা ভিডিও নয়
_MIN_USABLE_OUTPUT_SECONDS = 0.5


def existing_output_is_usable(path: Path, min_bytes: int = _MIN_USABLE_OUTPUT_BYTES) -> bool:
    """আগের আউটপুট ফাইলটি সত্যিই ব্যবহারযোগ্য কি না (skip করার যোগ্য)।"""
    path = Path(path)
    try:
        if not path.is_file() or path.stat().st_size < min_bytes:
            return False
    except OSError:
        return False
    try:
        return get_media_duration(path) >= _MIN_USABLE_OUTPUT_SECONDS
    except Exception:
        return False


def _usable(segments: List[Segment]) -> List[Segment]:
    """Segments that carry speech (STT text or translated text)."""
    return [s for s in segments if (s.text_en or "").strip() or (s.text_bn or "").strip()]


class DubbingPipeline:
    """End-to-end English→Bangla dubbing pipeline (reusable per batch)."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config.validate()
        cfg = self.config
        self.extractor = AudioExtractor(
            use_hwaccel=False,
            stall_timeout=cfg.extract_stall_seconds,
        )
        self.stt = SpeechToText(
            model_size=cfg.whisper_model,
            device=cfg.whisper_device,
            compute_type=cfg.whisper_compute_type,
            beam_size=cfg.whisper_beam_size,
            vad_min_silence_ms=cfg.vad_min_silence_ms,
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
        )
        self.muxer = VideoMuxer(audio_bitrate=cfg.audio_bitrate, use_hwaccel=cfg.use_hwaccel)

        # ব্যাচ জুড়ে একবারই তৈরি হয়
        self._tts: Optional[BaseTTSEngine] = None

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # TTS engine (ব্যাচ জুড়ে একবার)
    # ------------------------------------------------------------------
    def _ensure_tts(self, total_speech_seconds: float) -> BaseTTSEngine:
        """TTS ইঞ্জিন তৈরি (বা পুনঃব্যবহার) করে।"""
        cfg = self.config
        total_speech_seconds = max(1.0, float(total_speech_seconds))

        if self._tts is None:
            self._tts = build_tts_engine(
                cfg,
                total_audio_seconds=total_speech_seconds,
            )
            self._calibrate_speech_rate()
        else:
            self._tts.set_budget(None, total_speech_seconds)
        return self._tts

    # ------------------------------------------------------------------
    def _calibrate_speech_rate(self) -> None:
        """আসল কণ্ঠ দিয়ে বাংলা বলার গতি মেপে অনুবাদের বাজেট ঠিক করে।"""
        cfg = self.config
        if not cfg.translation_calibrate_speech_rate:
            return
        if self._tts is None or not hasattr(self._tts, "synthesize"):
            return
        if not isinstance(self._tts, EdgeTTSEngine):
            return
        try:
            rate = calibrate_speech_rate(
                self._tts,
                fallback=cfg.translation_char_rate,
                sample_rate=cfg.sync_sample_rate,
            )
        except Exception as exc:
            logger.info("কণ্ঠের গতি মাপা যায়নি (%s) — অনুমানকৃত হার ব্যবহার হবে।", exc)
            return
        setter = getattr(self.translator, "set_char_rate", None)
        if callable(setter) and rate > 0:
            source = "measured" if abs(rate - cfg.translation_char_rate) > 1e-6 else "config"
            setter(rate, source=source)

    # ------------------------------------------------------------------
    @contextmanager
    def _step(self, number: int, title: str) -> Iterator[None]:
        logger.info("▶ ধাপ %d/%d: %s", number, _TOTAL_STEPS, title)
        started = time.perf_counter()
        yield
        logger.info(
            "✓ ধাপ %d/%d শেষ (%s) — %.1fs", number, _TOTAL_STEPS, title,
            time.perf_counter() - started,
        )

    def output_path_for(self, video_path: Path, output_dir: Optional[Path] = None) -> Path:
        folder = output_dir or Path(self.config.output_dir)
        return folder / output_filename_for(video_path)

    # ------------------------------------------------------------------
    def process_one(
        self, video_path: Path | str, output_dir: Optional[Path] = None
    ) -> VideoResult:
        """Dub a single video; never raises — returns a :class:`VideoResult`."""
        video_path = Path(video_path)
        output_path = self.output_path_for(video_path)
        if output_dir is not None:
            output_path = Path(output_dir) / output_filename_for(video_path)
        started = time.perf_counter()

        if self.config.skip_existing and output_path.is_file():
            if existing_output_is_usable(output_path):
                size_mb = output_path.stat().st_size / (1024 * 1024)
                logger.info(
                    "⏭️  আগের আউটপুট পাওয়া গেছে (%.1f MB), তাই বাদ দেওয়া হলো: %s\n"
                    "   নতুন করে (নতুন কোড/কণ্ঠ দিয়ে) ডাবিং করতে চাইলে চালান: "
                    "python main.py --force",
                    size_mb, output_path.name,
                )
                return VideoResult(
                    video_path=video_path, status="skipped", output_path=output_path,
                    elapsed_seconds=0.0,
                )
            logger.warning(
                "⚠️ আগের আউটপুট অসম্পূর্ণ/পড়া যাচ্ছে না (%s) — আবার তৈরি করা হবে।",
                output_path.name,
            )
            output_path.unlink(missing_ok=True)

        work_dir = self.config.resolved_work_dir / _safe_dirname(video_path.stem)
        try:
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
            work_dir.mkdir(parents=True, exist_ok=True)

            segments, timings, stats = self._run_steps(video_path, work_dir, output_path)
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
                video_path=video_path, status="failed", error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=elapsed,
            )
        finally:
            if not self.config.keep_temp_files and work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)

    # Alias for callers using process_video
    process_video = process_one

    # ------------------------------------------------------------------
    def _run_steps(
        self, video_path: Path, work_dir: Path, output_path: Path
    ) -> tuple[List[Segment], StageTimings, dict]:
        """Execute steps 1–6 for one video (raises on unrecoverable error)."""
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
            if total_duration <= 0:
                total_duration = self.extractor.last_duration or 0.0
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
            english_chars = sum(len(s.text_en or "") for s in segments)
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
            translated = [s for s in segments if (s.text_bn or "").strip()]
            if not translated:
                raise RuntimeError("কোনো segment-এরই বাংলা অনুবাদ পাওয়া যায়নি।")
            if len(translated) < len(segments):
                lost = [s for s in segments if not (s.text_bn or "").strip()]
                logger.warning(
                    "⚠️ %dটি segment-এর অনুবাদ নেই — ওই সময়টুকুতে কথা থাকবে না "
                    "(স্লট পাশের লাইন ভরাট করবে)। সময়: %s",
                    len(lost),
                    ", ".join(f"{_clock(s.start)}–{_clock(s.end)}" for s in lost[:8])
                    + (" …" if len(lost) > 8 else ""),
                )
                stats["untranslated_seconds"] = round(
                    sum(float(s.duration or 0.0) for s in lost), 2
                )
            synthesized = [s for s in translated if s.tts_audio_path]
            if not synthesized:
                raise RuntimeError(
                    "একটি segment-এরও Bangla অডিও তৈরি হয়নি — TTS ইঞ্জিন যাচাই করুন।"
                )

            # 3.5) বড় ফাঁক ভরানো: স্লটের চেয়ে অনেক ছোট লাইন আরও ভরাট করে লেখানো
            filled = self._fill_short_slots(synthesized, tts_dir, total_duration)
            if filled:
                logger.info("✅ %dটি লাইন ভরাট করে আবার কণ্ঠ দেওয়া হলো (ফাঁক কমল)।", filled)
                stats["filled_lines"] = filled
        timings.translate = translate_seconds
        timings.tts = tts_seconds
        stats["tts"] = {
            "engine": engine.describe(),
            "segments": len(synthesized),
            "total_audio_seconds": round(sum(float(s.tts_duration or 0.0) for s in synthesized), 2),
        }

        # 4) Sync: time-stretch + non-overlapping assembly ---------------------
        marker = time.perf_counter()
        with self._step(4, "Audio-Video Sync (কোনো overlap নেই)"):
            final_audio = self.sync_engine.build_final_track(
                segments, total_duration, work_dir, work_dir / "final_audio_bn.wav"
            )
            stats["sync"] = dict(getattr(self.sync_engine, "last_stats", {}) or {})
            trimmed = int(stats["sync"].get("trimmed") or 0)
            if trimmed:
                logger.warning(
                    "⚠️ %dটি ক্লিপ উইন্ডোর চেয়ে লম্বা ছিল — প্রতিটির শেষটুকু কেটে "
                    "বসানো হয়েছে (দুই কণ্ঠ একসাথে বাজবে না)। অনেক বেশি হলে "
                    "`config.yaml`-এ translation.char_rate একটু বাড়ান বা "
                    "`sync.borrow_silence_ms` বাড়ান।",
                    trimmed,
                )

        timings.sync = time.perf_counter() - marker

        # 5) Mux locally, then copy the finished file to the output (Drive) ----
        marker = time.perf_counter()
        with self._step(5, "Final Muxing (ভিডিও + Bangla অডিও)"):
            local_output = work_dir / output_filename_for(video_path)
            self.muxer.mux(video_path, final_audio, local_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_output, output_path)
            logger.info("আউটপুট সেভ হয়েছে: %s", output_path)
        timings.mux = time.perf_counter() - marker

        # 6) Subtitles (যাচাই ও পুনঃব্যবহারের জন্য) -----------------------------
        if cfg.save_srt:
            with self._step(6, "সাবটাইটেল (SRT) সেভ করা"):
                try:
                    stem = video_path.stem
                    english_lines = [
                        child
                        for unit in segments
                        for child in (getattr(unit, "children", None) or [unit])
                    ]
                    write_srt(
                        english_lines or segments,
                        output_path.parent / f"{stem}_en.srt",
                        language="en",
                    )
                    bangla_lines = [
                        line
                        for unit in segments
                        for line in subtitles_for_unit(unit)
                    ]
                    write_srt(
                        bangla_lines or segments,
                        output_path.parent / f"{stem}_bn.srt",
                        language="bn",
                    )
                    logger.info("সাবটাইটেল সেভ হয়েছে: %s_en.srt, %s_bn.srt", stem, stem)
                except Exception as exc:  # SRT ব্যর্থ হলে মূল কাজ ব্যর্থ ধরা হবে না
                    logger.warning("SRT সেভ করা যায়নি: %s", exc)

        stats["segments"] = sum(
            len(getattr(unit, "children", None) or [unit]) for unit in segments
        )
        stats["units"] = sum(1 for unit in segments if getattr(unit, "is_unit", False)) or len(segments)
        stats["audio_seconds"] = round(total_duration, 2)
        return segments, timings, stats

    # ------------------------------------------------------------------
    # Translation ↔ TTS orchestration
    # ------------------------------------------------------------------
    def _translate_and_synthesize(
        self, segments: List[Segment], tts_dir: Path
    ) -> tuple[float, float]:
        """Translate then synthesize, overlapping the two when possible."""
        cfg = self.config
        todo = _usable(segments)
        if not todo:
            raise RuntimeError("TTS-এর জন্য কোনো speech segment পাওয়া যায়নি।")

        group_size = max(4, int(cfg.gemini_batch_size))
        groups = [todo[i:i + group_size] for i in range(0, len(todo), group_size)]

        overlap = bool(cfg.enable_overlap) and len(groups) > 1

        if not overlap:
            started = time.perf_counter()
            self.translator.translate_segments(todo)
            self._shorten(todo)
            translate_seconds = time.perf_counter() - started
            started = time.perf_counter()
            self._synthesize(todo, tts_dir)
            return translate_seconds, time.perf_counter() - started

        outbox: "queue.SimpleQueue[object]" = queue.SimpleQueue()
        sentinel = object()
        stop = threading.Event()
        errors: List[BaseException] = []
        produced_seconds = [0.0]

        def producer() -> None:
            started = time.perf_counter()
            pending: Optional[List[Segment]] = None
            try:
                for group in groups:
                    if stop.is_set():
                        break
                    self.translator.translate_segments(group)
                    if pending is not None:
                        self._shorten(pending, anchor=group[0])
                        outbox.put(pending)
                    pending = group
                if pending is not None and not stop.is_set():
                    self._shorten(pending)
                    outbox.put(pending)
            except BaseException as exc:
                errors.append(exc)
            finally:
                produced_seconds[0] = time.perf_counter() - started
                outbox.put(sentinel)

        worker = threading.Thread(target=producer, name="proxmax-translate", daemon=True)
        worker.start()

        tts_started = time.perf_counter()
        try:
            while True:
                item = outbox.get()
                if item is sentinel:
                    break
                if errors:
                    break
                self._synthesize(list(item), tts_dir)  # type: ignore[arg-type]
        except BaseException:
            stop.set()
            raise
        finally:
            stop.set()
            worker.join(timeout=60)
        tts_seconds = time.perf_counter() - tts_started

        if errors:
            first = errors[0]
            raise RuntimeError(
                f"অনুবাদ ব্যর্থ হয়েছে ({type(first).__name__}): {first}"
            ) from first
        return produced_seconds[0], tts_seconds

    # ------------------------------------------------------------------
    def _shorten(self, group: List[Segment], anchor: Optional[Segment] = None) -> None:
        """সময়ে না আঁটা লাইনগুলো ছোট করে লেখায় (duration-aware অনুবাদ)।"""
        cfg = self.config
        if not cfg.translation_shorten_pass:
            return
        target = list(group)
        if anchor is not None:
            target.append(
                Segment(index=-1, start=float(anchor.start), end=float(anchor.end), text_bn="।")
            )
        try:
            self.translator.shorten_to_fit(
                target,
                max_tempo=cfg.max_tempo,
                slot_gap=cfg.slot_gap_ms / 1000.0,
                borrow=cfg.borrow_silence_ms / 1000.0,
            )
        except Exception as exc:
            logger.warning("অনুবাদ ছোট করার ধাপ ব্যর্থ (%s) — মূল অনুবাদ রাখা হচ্ছে।", exc)

    # ------------------------------------------------------------------
    def _fill_short_slots(
        self, segments: List[Segment], tts_dir: Path, total_duration: float
    ) -> int:
        """মূল speech interval-এর তুলনায় ছোট ক্লিপ ভরাট করে; পরের নীরবতা ব্যবহার করে না।"""
        cfg = self.config
        if not cfg.translation_expand_pass:
            return 0
        expander = getattr(self.translator, "expand_to_fit", None)
        if not callable(expander):
            return 0
        ordered = [s for s in sorted(segments, key=lambda s: s.start) if s.tts_audio_path]
        if len(ordered) < 2:
            return 0
        slot_gap = max(0.0, cfg.slot_gap_ms / 1000.0)
        borrow = max(0.0, cfg.borrow_silence_ms / 1000.0)
        offenders: List[Segment] = []
        targets: List[float] = []
        for index, seg in enumerate(ordered):
            deadline = (
                float(ordered[index + 1].start) - slot_gap
                if index + 1 < len(ordered) else float(total_duration)
            )
            hard_window = max(0.0, deadline - float(seg.start))
            speech_window = max(0.0, float(seg.end) - float(seg.start))
            # Only the segment's measured speech interval plus a small configured
            # borrow is available. Never treat a 2–5s source pause as extra speech time.
            slot = min(hard_window, max(0.4, speech_window + borrow))
            if slot <= 0.2:
                continue
            gap = slot - float(seg.tts_duration or 0.0)
            if gap >= float(cfg.translation_expand_min_gap):
                offenders.append(seg)
                targets.append(slot)
        if not offenders:
            return 0

        logger.info(
            "🕳️ %dটি লাইনের মূল speech interval-এ বড় ফাঁক (≥%.1fs) — সীমিত সময়ে অনুবাদ ভরাট করা হচ্ছে…",
            len(offenders), cfg.translation_expand_min_gap,
        )
        try:
            changed = expander(offenders, targets)
        except Exception as exc:
            logger.warning("লাইন ভরাট করা যায়নি (%s) — আগের টেক্সটই রাখা হচ্ছে।", exc)
            return 0
        updated = [seg for seg in offenders if (seg.text_bn or "").strip()]
        if not changed or not updated:
            return 0
        try:
            self._synthesize(updated, tts_dir)
        except Exception as exc:
            logger.warning("ভরাট করা লাইনের TTS ব্যর্থ (%s) — পুরোনো অডিওই থাকবে।", exc)
            return 0
        return len([s for s in updated if s.tts_audio_path])

    # ------------------------------------------------------------------
    def _synthesize(self, group: List[Segment], tts_dir: Path) -> None:
        """এক গ্রুপ segment-এর Bangla কণ্ঠ তৈরি করে।"""
        usable = [s for s in group if (s.text_bn or "").strip()]
        if not usable:
            return
        if self._tts is None:
            self._ensure_tts(sum(float(s.duration) for s in usable))
        assert self._tts is not None
        self._tts.synthesize(usable, tts_dir)

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
        cfg = self.config
        logger.info("=" * 70)
        logger.info(
            "ব্যাচ শুরু: %dটি ভিডিও | STT=%s | অনুবাদ=%s | কণ্ঠ=বিল্ট-ইন %s (%s) | tempo %.2f–%.2f",
            len(videos), cfg.whisper_model, self.translator.describe(),
            cfg.resolved_tts_voice, cfg.tts_gender,
            cfg.min_tempo, cfg.max_tempo,
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
            if result.timings is not None and result.status == "success":
                logger.info("     ⏱️  %s", result.timings.as_text())
        logger.info("সব আউটপুট এখানে আছে: %s", output_folder)
        logger.info("=" * 70)


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
