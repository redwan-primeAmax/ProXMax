"""Translation/TTS worker coordination and timing-repair helpers."""

from __future__ import annotations

import copy
import queue
import threading
import time
from pathlib import Path
from typing import List, Optional

from .models import Segment
from .pipeline_helpers import _usable
from .sync_engine import plan_windows
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

    def _repair_overlong_tts(
        self, segments: List[Segment], tts_dir: Path, total_duration: float
    ) -> dict:
        """Retry measured TTS overruns with shorter text before sync can clip them."""
        ordered = [s for s in sorted(segments, key=lambda item: item.start) if s.tts_audio_path]
        if not ordered:
            return {
                "checked": 0, "overlong_before": 0, "shortened": 0,
                "retry_seconds": 0.0, "translation_seconds": 0.0, "tts_seconds": 0.0,
            }
        windows = plan_windows(
            ordered,
            total_duration,
            min_gap_seconds=max(0.0, self.config.slot_gap_ms / 1000.0),
            borrow_seconds=max(0.0, self.config.borrow_silence_ms / 1000.0),
        )
        overlong: List[Segment] = []
        targets: List[float] = []
        limits: dict[int, float] = {}
        for segment, (start, end) in zip(ordered, windows):
            safe_window = max(0.12, end - start - 0.04)
            max_tts_seconds = safe_window * float(self.config.max_tempo) * 0.96
            limits[id(segment)] = max_tts_seconds
            duration = float(segment.tts_duration or 0.0)
            if duration > max_tts_seconds * 1.02:
                overlong.append(segment)
                targets.append(max(0.4, max_tts_seconds))
        summary = {
            "checked": len(ordered),
            "overlong_before": len(overlong),
            "shortened": 0,
            "still_overlong": [],
            "retry_seconds": 0.0,
            "translation_seconds": 0.0,
            "tts_seconds": 0.0,
        }
        if not overlong:
            return summary

        logger.warning(
            "⚠️ %dটি TTS clip নিরাপদ sync window-এর চেয়ে লম্বা — কাটার বদলে সংক্ষিপ্ত অনুবাদ ও TTS retry হবে।",
            len(overlong),
        )
        shorten = getattr(self.translator, "shorten_to_durations", None)
        if not callable(shorten):
            summary["still_overlong"] = [int(seg.index) for seg in overlong]
            return summary

        replacements = [
            copy.copy(segment) for segment in overlong
        ]
        for replacement in replacements:
            replacement.tts_audio_path = None
            replacement.tts_duration = None
            replacement.stretched_audio_path = None
            replacement.stretched_duration = None
            replacement.placed_duration = None
            replacement.sync_issue = None
        old_texts = [segment.text_bn for segment in overlong]
        translation_started = time.perf_counter()
        try:
            shorten(replacements, targets)
        except Exception as exc:
            logger.warning("মাপা TTS সময়ে অনুবাদ ছোট করার retry ব্যর্থ (%s)।", exc)
            summary["translation_seconds"] = round(time.perf_counter() - translation_started, 3)
            summary["still_overlong"] = [int(seg.index) for seg in overlong]
            return summary
        summary["translation_seconds"] = round(time.perf_counter() - translation_started, 3)

        changed = [
            i for i, replacement in enumerate(replacements)
            if (replacement.text_bn or "").strip()
            and len(replacement.text_bn.strip()) < len(old_texts[i].strip())
        ]
        summary["retry_seconds"] = float(summary["translation_seconds"])
        if changed:
            retry_started = time.perf_counter()
            retry_dir = tts_dir / "fit_retry"
            try:
                self._synthesize([replacements[i] for i in changed], retry_dir)
            except Exception as exc:
                logger.warning("সংক্ষিপ্ত text-এর TTS retry ব্যর্থ (%s)। পুরোনো clip থাকবে।", exc)
            else:
                for i in changed:
                    original, replacement = overlong[i], replacements[i]
                    if replacement.tts_audio_path and replacement.tts_duration:
                        original.text_bn = replacement.text_bn
                        original.tts_audio_path = replacement.tts_audio_path
                        original.tts_duration = replacement.tts_duration
                        summary["shortened"] += 1
            summary["tts_seconds"] = round(time.perf_counter() - retry_started, 3)
            summary["retry_seconds"] = round(
                float(summary["translation_seconds"]) + float(summary["tts_seconds"]), 3
            )

        for segment in overlong:
            limit = limits[id(segment)]
            if float(segment.tts_duration or 0.0) > limit * 1.02:
                summary["still_overlong"].append(int(segment.index))
        if summary["still_overlong"]:
            logger.error(
                "🛑 %dটি line retry-এর পরও sync window-তে আঁটেনি; final track-এ এগুলো কাটা হবে না, report-এ review issue হবে।",
                len(summary["still_overlong"]),
            )
        return summary

    def _validate_dub_audio(
        self, output_path: Path, work_dir: Path, segments: List[Segment], total_duration: float
    ) -> dict:
        """Run a best-effort Bengali ASR coverage check on the muxed output."""
        validator = getattr(self.stt, "transcribe_words_in_language", None)
        if not callable(validator):
            return {"status": "unavailable", "reason": "STT backend has no multilingual validator"}
        try:
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

    def _synthesize(self, group: List[Segment], tts_dir: Path) -> None:
        """এক গ্রুপ segment-এর Bangla কণ্ঠ তৈরি করে।"""
        usable = [s for s in group if (s.text_bn or "").strip()]
        if not usable:
            return
        if self._tts is None:
            self._ensure_tts(sum(float(s.duration) for s in usable))
        assert self._tts is not None
        self._tts.synthesize(usable, tts_dir)
