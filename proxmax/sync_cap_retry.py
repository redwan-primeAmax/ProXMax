"""Measure a final allowed tempo when a calculated stretch still overruns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import audio_fx
from .sync_math import build_stretch_filter
from .sync_silence import CONSERVATIVE_SILENCE_DBFS, trim_edge_silence_for_overrun
from .utils import ensure_tool, get_logger

logger = get_logger(__name__)


def suspicious_stretch_shortfall(prepared_seconds: float, tempo: float, rendered_seconds: float) -> bool:
    """Flag unexpectedly large duration loss, allowing FFmpeg frame/silence rounding."""
    expected = prepared_seconds / tempo
    allowance = min(0.06, max(0.01, expected * 0.10))
    return rendered_seconds + allowance < expected


@dataclass(frozen=True)
class CapRetryResult:
    measured: bool = False
    adopted: bool = False
    duration: float = 0.0
    head_seconds: float = 0.0
    tail_seconds: float = 0.0


def retry_at_tempo_cap(
    source: Path,
    out_path: Path,
    segment_index: int,
    max_tempo: float,
    sample_rate: int,
    prefer_rubberband: bool,
    target_seconds: float,
    tolerance_seconds: float,
    current_seconds: float,
    prepared_seconds: float,
    threshold_dbfs: float,
    pad_ms: int,
    runner: Callable,
) -> CapRetryResult:
    """Render from the pre-stretch source, measure, and keep only an improvement.

    A theoretical tempo may not produce the expected duration in FFmpeg. Do not
    rewrite the translation on that basis until a valid waveform at the user's
    allowed cap has actually been measured. A failed render leaves the original
    artifact untouched and does not claim that the cap was reached.
    """
    candidate_path = out_path.with_name(f"{out_path.stem}_max_tempo.wav")
    try:
        candidate_path.unlink(missing_ok=True)
        runner(
            [
                ensure_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(source),
                "-filter:a", build_stretch_filter(max_tempo, sample_rate, prefer_rubberband),
                "-ac", "1", "-c:a", "pcm_s16le", str(candidate_path),
            ],
            description=f"time-stretch cap retry (segment {segment_index}, tempo={max_tempo:.2f})",
            timeout=120.0,
        )
        clip, rate = audio_fx.read_wav(candidate_path)
        if rate <= 0 or clip.size == 0 or not np.any(clip):
            raise ValueError("max-tempo render is empty/silent or has invalid sample rate")
        duration = clip.size / float(rate)
        if suspicious_stretch_shortfall(prepared_seconds, max_tempo, duration):
            raise ValueError("max-tempo render is unexpectedly short; possible speech loss")
        post_head = post_tail = 0.0
        if duration > target_seconds + tolerance_seconds:
            clip, post_head, post_tail = trim_edge_silence_for_overrun(
                clip, rate, duration - target_seconds,
                threshold_dbfs=min(threshold_dbfs, CONSERVATIVE_SILENCE_DBFS), pad_ms=pad_ms,
            )
            if post_head or post_tail:
                audio_fx.write_wav(candidate_path, clip, rate)
                duration = clip.size / float(rate)
        if duration + 0.5 / rate < current_seconds:
            candidate_path.replace(out_path)
            logger.info(
                "Segment %d: মাপা সর্বোচ্চ %.2f×-এ %.3fs থেকে %.3fs।",
                segment_index, max_tempo, current_seconds, duration,
            )
            return CapRetryResult(True, True, duration, post_head, post_tail)
        return CapRetryResult(measured=True)
    except Exception as exc:
        logger.warning(
            "Segment %d-এর সর্বোচ্চ %.2f× পরীক্ষা/মাপা যায়নি (%s); rewrite স্থগিত, review দরকার।",
            segment_index, max_tempo, exc,
        )
        return CapRetryResult()
    finally:
        try:
            candidate_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove temporary cap render: %s", candidate_path.name)
