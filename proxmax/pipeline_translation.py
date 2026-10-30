"""Translation/TTS worker coordination and timing-repair helpers."""

from __future__ import annotations

import copy
import queue
import threading
import time
from pathlib import Path
from typing import List, Optional

from .audio_quality import assess_bengali_coverage
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
            pending: Optional[List[Segment]] = None
            try:
                for group in groups:
                    if stop.is_set():
                        break
                    self.translator.translate_segments(group)
                    if pending is not None:
                        outbox.put(pending)
                    pending = group
                if pending is not None and not stop.is_set():
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

    def _repair_overlong_tts(
        self,
        segments: List[Segment],
        tts_dir: Path,
        total_duration: float,
        only_unfit: bool = True,
    ) -> dict:
        """Shorten only measured clips that remain unfit after edge trim and tempo fitting."""
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
        for segment, (start, end) in zip(ordered, windows):
            safe_window = max(0.12, end - start - 0.04)
            remains_unfit = "stretched clip exceeds safe window" in (segment.sync_issue or "")
            if only_unfit and remains_unfit:
                overlong.append(segment)
                targets.append(max(0.4, safe_window))
        summary = {
            "checked": len(ordered),
            "overlong_before": len(overlong),
            "shortened": 0,
            "retry_failed": [],
            "still_unfit_after_sync": [],
            "retry_seconds": 0.0,
            "translation_seconds": 0.0,
            "tts_seconds": 0.0,
        }
        if not only_unfit or not overlong:
            return summary

        logger.warning(
            "⚠️ %dটি TTS clip নিরাপদ sync window-এর চেয়ে লম্বা — কাটার বদলে সংক্ষিপ্ত অনুবাদ ও TTS retry হবে।",
            len(overlong),
        )
        shorten = getattr(self.translator, "shorten_to_durations", None)
        if not callable(shorten):
            summary["retry_failed"] = [int(seg.index) for seg in overlong]
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
            summary["retry_failed"] = [int(seg.index) for seg in overlong]
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
                summary["retry_failed"] = [int(overlong[i].index) for i in changed]
            else:
                for i in changed:
                    original, replacement = overlong[i], replacements[i]
                    if replacement.tts_audio_path and replacement.tts_duration:
                        original.text_bn = replacement.text_bn
                        original.tts_audio_path = replacement.tts_audio_path
                        original.tts_duration = replacement.tts_duration
                        summary["shortened"] += 1
                    else:
                        summary["retry_failed"].append(int(original.index))
            summary["tts_seconds"] = round(time.perf_counter() - retry_started, 3)
            summary["retry_seconds"] = round(
                float(summary["translation_seconds"]) + float(summary["tts_seconds"]), 3
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
