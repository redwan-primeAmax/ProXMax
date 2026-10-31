"""Targeted recovery and explicit accounting for untranslated or missing clips."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np

from . import audio_fx
from .models import Segment
from .pipeline_helpers import _usable
from .utils import get_logger

logger = get_logger(__name__)


def _has_audio(segment: Segment) -> bool:
    try:
        path = Path(segment.tts_audio_path) if segment.tts_audio_path else None
        return bool(path and path.is_file() and path.stat().st_size > 44)
    except (OSError, TypeError, ValueError):
        return False


def _indexes(segments: Sequence[Segment]) -> str:
    return ", ".join(str(segment.index) for segment in segments[:20]) + (" …" if len(segments) > 20 else "")


def _reset_tts(segment: Segment) -> None:
    segment.tts_audio_path = segment.tts_duration = segment.tts_raw_duration = None
    segment.tts_silence_trimmed = False
    segment.tts_trimmed_head_seconds = segment.tts_trimmed_internal_seconds = 0.0
    segment.tts_trimmed_tail_seconds = 0.0
    segment.tts_trimmed_internal_cuts = 0


def recover_missing_speech(pipeline, segments: list[Segment], tts_dir: Path) -> tuple[float, float]:
    """Retry just missing translations/audio; fail rather than emit a silent gap.

    This checks existence and a minimal WAV size without rereading every normal
    clip. The sync pass checks decoded waveforms and raises if recovery fails.
    """
    todo = _usable(segments)
    translation_seconds = tts_seconds = 0.0
    missing_text = [segment for segment in todo if not (segment.text_bn or "").strip()]
    if missing_text:
        logger.warning("%dটি অনুবাদ অনুপস্থিত; শুধু ওই segment আবার অনুবাদ করা হচ্ছে।", len(missing_text))
        started = time.perf_counter()
        try:
            pipeline.translator.translate_segments(missing_text)
        except Exception as exc:
            raise RuntimeError(f"Missing translation recovery failed for segments {_indexes(missing_text)}: {exc}") from exc
        translation_seconds += time.perf_counter() - started
        missing_text = [segment for segment in todo if not (segment.text_bn or "").strip()]
        if missing_text:
            raise RuntimeError(f"Missing Bengali translation for segments {_indexes(missing_text)}; no output published")

    missing_audio = [segment for segment in todo if not _has_audio(segment)]
    if missing_audio:
        logger.warning("%dটি TTS অডিও অনুপস্থিত; শুধু ওই segment আবার তৈরি করা হচ্ছে।", len(missing_audio))
        for segment in missing_audio:
            _reset_tts(segment)
        started = time.perf_counter()
        try:
            pipeline._synthesize(missing_audio, tts_dir / "missing_recovery")
        except Exception as exc:
            raise RuntimeError(f"TTS recovery failed for segments {_indexes(missing_audio)}: {exc}") from exc
        tts_seconds += time.perf_counter() - started
        missing_audio = [segment for segment in todo if not _has_audio(segment)]
        if missing_audio:
            raise RuntimeError(f"Missing TTS audio for segments {_indexes(missing_audio)}; no output published")
    return translation_seconds, tts_seconds


def recover_invalid_waveforms(pipeline, segments: Sequence[Segment], tts_dir: Path) -> tuple[float, int]:
    """Retry only unreadable/silent TTS sources discovered by the first sync pass."""
    reasons = ("TTS audio file missing", "TTS audio unreadable", "TTS preparation failed",
               "stretched audio measurement failed", "stretched clip unreadable")
    candidates = [s for s in segments if s.placed_duration is None and
                  any(reason in (s.sync_issue or "") for reason in reasons)]

    def valid(seg: Segment) -> bool:
        try:
            if not seg.tts_audio_path:
                return False
            clip, rate = audio_fx.read_wav(seg.tts_audio_path)
            return bool(rate > 0 and clip.size and np.all(np.isfinite(clip)) and np.any(clip))
        except Exception:
            return False

    invalid = [s for s in candidates if not valid(s)]
    if not invalid:
        return 0.0, 0
    logger.warning("%dটি silent/অপাঠযোগ্য TTS clip আবার তৈরি করা হচ্ছে।", len(invalid))
    for seg in invalid:
        _reset_tts(seg)
    started = time.perf_counter()
    try:
        pipeline._synthesize(invalid, tts_dir / "invalid_recovery")
    except Exception as exc:
        raise RuntimeError(f"Invalid TTS recovery failed for segments {_indexes(invalid)}: {exc}") from exc
    elapsed = time.perf_counter() - started
    still_invalid = [s for s in invalid if not valid(s)]
    if still_invalid:
        raise RuntimeError(f"TTS audio remains silent/unreadable for segments {_indexes(still_invalid)}")
    return elapsed, len(invalid)
