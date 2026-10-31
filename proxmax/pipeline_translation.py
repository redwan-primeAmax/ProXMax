"""Translation/TTS worker coordination and optional Bengali audio validation."""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import List

from .audio_quality import assess_bengali_coverage
from .models import Segment
from .pipeline_helpers import _usable
from .pipeline_validation import bengali_validator
from .tts.base import TTSError
from .utils import get_logger

logger = get_logger(__name__)


class PipelineTranslationMixin:
    """Translation, TTS generation and optional audio validation."""

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
