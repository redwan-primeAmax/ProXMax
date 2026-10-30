"""Translation/TTS worker coordination and timing-repair helpers."""

from __future__ import annotations

import copy
import queue
import threading
import time
import wave
from pathlib import Path
from typing import List, Optional, Set

from .audio_quality import assess_bengali_coverage
from .models import Segment
from .pipeline_helpers import _usable
from .pipeline_validation import bengali_validator
from .sync_engine import plan_windows
from .sync_silence import measured_fit_durations
from .translation_quality import assess_fit_rewrite
from .tts.base import TTSError
from .utils import get_logger

logger = get_logger(__name__)


class PipelineTranslationMixin:
    """Translation, TTS generation, and measured overrun repair methods."""

    def _translate_and_synthesize(
        self, segments: List[Segment], tts_dir: Path
    ) -> tuple[float, float]:
        """Translate then synthesize, overlapping the two when possible."""
        cfg = self.config
        todo = _usable(segments)
        if not todo:
            raise RuntimeError("TTS-এর জন্য কোনো speech segment পাওয়া যায়নি।")

        batcher = getattr(self.translator, "batch_segments", None)
        if callable(batcher):
            groups = batcher(todo)
        else:
            group_size = max(1, int(cfg.gemini_batch_size))
            groups = [todo[i:i + group_size] for i in range(0, len(todo), group_size)]

        overlap = bool(cfg.enable_overlap) and len(groups) > 1

        if not overlap:
            started = time.perf_counter()
            self.translator.translate_segments(todo)
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
            try:
                for group in groups:
                    if stop.is_set():
                        break
                    self.translator.translate_segments(group)
                    if stop.is_set():
                        break
                    # No look-ahead is needed: each translated group is ready for TTS.
                    outbox.put(group)
            except BaseException as exc:
                errors.append(exc)
            finally:
                produced_seconds[0] = time.perf_counter() - started
                outbox.put(sentinel)

        worker = threading.Thread(target=producer, name="proxmax-translate", daemon=True)
        worker.start()

        tts_seconds = 0.0
        # Combine ready groups to keep TTS slots busy across Gemini batch boundaries.
        # Keep the work bounded so a later translation error is noticed promptly.
        max_ready_segments = max(1, int(cfg.tts_concurrency)) * 4
        try:
            while True:
                item = outbox.get()
                if item is sentinel:
                    break
                first_group = list(item)  # type: ignore[arg-type]
                ready = list(first_group)
                ready_groups = [first_group]
                producer_done = False
                while len(ready) < max_ready_segments:
                    try:
                        item = outbox.get_nowait()
                    except queue.Empty:
                        break
                    if item is sentinel:
                        producer_done = True
                        break
                    group = list(item)  # type: ignore[arg-type]
                    ready_groups.append(group)
                    ready.extend(group)
                if errors:
                    break
                tts_started = time.perf_counter()
                self._synthesize(ready, tts_dir)
                tts_seconds += time.perf_counter() - tts_started
                # Coalescing must not hide the failure of an entire original group.
                for group in ready_groups:
                    translated = [s for s in group if (s.text_bn or "").strip()]
                    if translated and not any(s.tts_audio_path for s in translated):
                        raise TTSError(
                            f"অনুবাদ batch-এর কোনো TTS ক্লিপ তৈরি হয়নি "
                            f"(segment {translated[0].index}–{translated[-1].index})."
                        )
                if producer_done:
                    break
        except BaseException:
            stop.set()
            raise
        finally:
            stop.set()
            worker.join(timeout=60)

        if errors:
            first = errors[0]
            raise RuntimeError(
                f"অনুবাদ ব্যর্থ হয়েছে ({type(first).__name__}): {first}"
            ) from first
        return produced_seconds[0], tts_seconds

    def _repair_overlong_tts(
        self,
        segments: List[Segment],
        tts_dir: Path,
        total_duration: float,
        only_unfit: bool = True,
        retry_round: int = 1,
        eligible_indexes: Optional[Set[int]] = None,
    ) -> dict:
        """Rewrite only clips still unfit after silence trimming and a measured tempo cap."""
        ordered = [s for s in sorted(segments, key=lambda item: item.start) if s.tts_audio_path]
        if not ordered:
            return {
                "checked": 0, "overlong_before": 0, "overlong_segments": [], "shortened": 0, "shortened_segments": [],
                "retry_failed": [], "not_shorter_audio": [], "still_unfit_after_sync": [],
                "awaiting_tempo_cap": [], "retry_seconds": 0.0, "translation_seconds": 0.0, "tts_seconds": 0.0, "audio_check_seconds": 0.0,
            }
        windows = plan_windows(
            ordered,
            total_duration,
            min_gap_seconds=max(0.0, self.config.slot_gap_ms / 1000.0),
            borrow_seconds=max(0.0, self.config.borrow_silence_ms / 1000.0),
        )
        overlong: List[Segment] = []
        targets: List[float] = []
        awaiting_cap: List[int] = []
        for segment, (start, end) in zip(ordered, windows):
            window_length = max(0.0, end - start)
            safe_window = max(min(0.01, window_length), window_length - 0.04)
            issue = segment.sync_issue or ""
            remains_unfit = "stretched clip exceeds safe window" in issue
            impossible_window = "sync window has no positive duration" in issue
            eligible = eligible_indexes is None or segment.index in eligible_indexes
            if only_unfit and (remains_unfit or impossible_window) and eligible:
                if remains_unfit and not segment.sync_tempo_cap_measured:
                    awaiting_cap.append(int(segment.index))
                    continue
                overlong.append(segment)
                # A zero-length window is impossible regardless of translation;
                # request a modest target and allow the final scheduler to shift.
                targets.append(max(0.15, safe_window) if impossible_window else safe_window)
        summary = {
            "checked": len(ordered),
            "overlong_before": len(overlong),
            "overlong_segments": [int(seg.index) for seg in overlong],
            "shortened": 0,
            "shortened_segments": [],
            "retry_failed": [],
            "not_shorter_audio": [],
            "still_unfit_after_sync": [],
            "awaiting_tempo_cap": awaiting_cap,
            "retry_seconds": 0.0,
            "translation_seconds": 0.0,
            "tts_seconds": 0.0,
            "audio_check_seconds": 0.0,
        }
        if awaiting_cap:
            logger.warning(
                "%dটি unfit clip-এর সর্বোচ্চ tempo মাপা যায়নি; rewrite না করে review-তে রাখা হলো।",
                len(awaiting_cap),
            )
        if not only_unfit or not overlong:
            return summary

        logger.warning(
            "⚠️ %dটি TTS clip নিরাপদ sync window-এর চেয়ে লম্বা — কাটার বদলে সংক্ষিপ্ত অনুবাদ ও TTS retry হবে।",
            len(overlong),
        )
        shorten = getattr(self.translator, "shorten_to_durations", None)
        if not callable(shorten):
            summary["retry_failed"] = [int(seg.index) for seg in overlong]
            for segment in overlong:
                segment.translation_notes.append("Translator has no fit-retry support; prior text/audio retained.")
            return summary

        # Keep measured fit metadata while asking for a rewrite, but synthesize
        # candidates into new files so the original text/audio can be retained.
        replacements = [copy.copy(segment) for segment in overlong]
        for replacement in replacements:
            replacement.translation_issues = list(replacement.translation_issues)
            replacement.translation_notes = list(replacement.translation_notes)
        old_texts = [segment.text_bn for segment in overlong]
        translation_started = time.perf_counter()
        try:
            shorten(replacements, targets)
        except Exception as exc:
            logger.warning("মাপা TTS সময়ে অনুবাদ ছোট করার retry ব্যর্থ (%s)।", exc)
            summary["translation_seconds"] = round(time.perf_counter() - translation_started, 3)
            summary["retry_seconds"] = summary["translation_seconds"]
            summary["retry_failed"] = [int(seg.index) for seg in overlong]
            for segment in overlong:
                segment.translation_notes.append("Fit-retry translation failed; prior text and audio retained.")
            return summary
        summary["translation_seconds"] = round(time.perf_counter() - translation_started, 3)

        changed = []
        for i, replacement in enumerate(replacements):
            original = overlong[i]
            checked = assess_fit_rewrite(
                original.text_en, original.original_text_bn or old_texts[i],
                old_texts[i], replacement.text_bn,
            )
            if checked.issues:
                summary["retry_failed"].append(int(original.index))
                new_notes = replacement.translation_notes[len(original.translation_notes):]
                original.translation_notes.extend(new_notes or [
                    "Fit-retry text rejected; prior text retained: " + "; ".join(checked.issues)
                ])
                continue
            replacement.text_bn = checked.text
            replacement.tts_audio_path = None
            replacement.tts_duration = None
            replacement.stretched_audio_path = None
            replacement.stretched_duration = None
            replacement.placed_duration = None
            replacement.sync_issue = None
            changed.append(i)
        summary["retry_seconds"] = float(summary["translation_seconds"])
        if changed:
            retry_started = time.perf_counter()
            retry_dir = tts_dir / f"fit_retry_{max(1, int(retry_round))}"
            try:
                self._synthesize([replacements[i] for i in changed], retry_dir)
            except Exception as exc:
                summary["tts_seconds"] = round(time.perf_counter() - retry_started, 3)
                logger.warning("সংক্ষিপ্ত text-এর TTS retry ব্যর্থ (%s)। পুরোনো clip থাকবে।", exc)
                summary["retry_failed"].extend(int(overlong[i].index) for i in changed)
                for i in changed:
                    overlong[i].translation_notes.append("Fit-retry synthesis failed; prior text and audio retained.")
            else:
                summary["tts_seconds"] = round(time.perf_counter() - retry_started, 3)
                check_started = time.perf_counter()
                for i in changed:
                    original, replacement = overlong[i], replacements[i]
                    if not replacement.tts_audio_path:
                        summary["retry_failed"].append(int(original.index))
                        original.translation_notes.append("Fit-retry synthesis produced no usable clip; prior text/audio retained.")
                        continue
                    try:
                        pad = int(self.config.tts_trim_pad_ms)
                        threshold = float(getattr(getattr(self, "sync_engine", None), "silence_threshold_dbfs", -45.0))
                        _, old_duration = measured_fit_durations(original.tts_audio_path, pad_ms=pad, threshold_dbfs=threshold)
                        actual_duration, new_duration = measured_fit_durations(replacement.tts_audio_path, pad_ms=pad, threshold_dbfs=threshold)
                    except (OSError, ValueError, RuntimeError, wave.Error, EOFError) as exc:
                        logger.warning("TTS retry-এর audio মাপা যায়নি (%s); আগের clip রাখা হলো।", exc)
                        summary["retry_failed"].append(int(original.index))
                        original.translation_notes.append("Original or fit-retry audio was silent/unreadable; prior text/audio retained.")
                        continue
                    if new_duration >= old_duration - 0.005:
                        summary["not_shorter_audio"].append(int(original.index))
                        summary["retry_failed"].append(int(original.index))
                        original.translation_notes.append("Fit-retry audio was not shorter; prior text/audio retained.")
                        continue
                    original.original_text_bn = original.original_text_bn or old_texts[i]
                    original.text_bn = replacement.text_bn
                    original.tts_audio_path = replacement.tts_audio_path
                    original.tts_duration = actual_duration
                    original.translation_issues = []
                    original.translation_notes.append(
                        "Bangla was rewritten after a measured TTS fit failure; review meaning against the source."
                    )
                    summary["shortened"] += 1
                    summary["shortened_segments"].append(int(original.index))
                summary["audio_check_seconds"] = round(time.perf_counter() - check_started, 3)
            summary["retry_seconds"] = round(
                float(summary["translation_seconds"]) + float(summary["tts_seconds"])
                + float(summary["audio_check_seconds"]), 3
            )

        return summary

    def _validate_dub_audio(
        self, output_path: Path, work_dir: Path, segments: List[Segment], total_duration: float
    ) -> dict:
        """Run a best-effort Bengali ASR coverage check on the muxed output."""
        validation_stt, owned = None, False
        try:
            validation_stt, owned = bengali_validator(self.stt, self.config)
            validator = getattr(validation_stt, "transcribe_words_in_language", None)
            if not callable(validator):
                return {"status": "unavailable", "reason": "STT backend has no multilingual validator"}
            audio_path = self.extractor.extract(
                output_path,
                work_dir / "post_mux_audio_16k.wav",
                duration_seconds=total_duration,
            )
            words = validator(audio_path, language="bn")
            result = assess_bengali_coverage(
                segments,
                words,
                min_coverage=float(self.config.audio_validation_min_coverage),
            )
            logger.info(
                "Post-mux Bangla coverage check: %s | %s/%s unit likely complete",
                result["status"], result["summary"]["likely_complete"], result["summary"]["checked"],
            )
            return result
        except Exception as exc:
            logger.warning(
                "Post-mux Bangla ASR check unavailable (%s); video processing continues.",
                type(exc).__name__,
            )
            return {"status": "unavailable", "reason": type(exc).__name__}
        finally:
            if owned and validation_stt is not None:
                try:
                    validation_stt.release()
                except Exception as exc:
                    logger.warning("Could not release optional Bengali validator: %s", type(exc).__name__)

    def _synthesize(self, group: List[Segment], tts_dir: Path) -> None:
        """এক গ্রুপ segment-এর Bangla কণ্ঠ তৈরি করে।"""
        usable = [s for s in group if (s.text_bn or "").strip()]
        if not usable:
            return
        if self._tts is None:
            self._ensure_tts(sum(float(s.duration) for s in usable))
        assert self._tts is not None
        self._tts.synthesize(usable, tts_dir)
