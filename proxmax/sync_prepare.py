"""Prepare one TTS clip: trim all detected silence and only slow, never speed up."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from . import audio_fx
from .models import Segment
from .sync_math import build_stretch_filter, compute_tempo
from .sync_silence import trim_silence_for_overrun
from .utils import ensure_tool, get_logger, run_ffmpeg

logger = get_logger(__name__)


def _valid_audio(path: Path, sample_rate: int, engine) -> np.ndarray:
    audio, rate = audio_fx.read_wav(path)
    if rate <= 0 or audio.size == 0 or not np.all(np.isfinite(audio)) or not np.any(audio):
        raise ValueError("empty, silent, or invalid waveform")
    if rate != sample_rate:
        audio = engine._resample(audio, rate, sample_rate)
    if not audio.size or not np.any(audio):
        raise ValueError("empty waveform after resampling")
    return audio


def _render_slowdown(engine, seg: Segment, source: Path, prepared_seconds: float,
                     target: float, tempo: float, output: Path) -> tuple[float, float]:
    """Try actual measured renders from the untouched prepared source.

    FFmpeg filters can return different durations, especially for short clips.
    An unsafe/short render is never adopted. Rubberband is tried first when
    available, then atempo. If neither is reliable, keep the original clip and
    record the deviation instead of losing speech or increasing its speed.
    """
    sr = engine.sample_rate
    attempts = [True, False] if engine.prefer_rubberband else [False]
    candidate = output.with_name(f"{output.stem}_slow_candidate.wav")
    for prefer_rubberband in attempts:
        requested = tempo
        for _ in range(3):
            if requested >= 0.999999:
                break
            try:
                candidate.unlink(missing_ok=True)
                run_ffmpeg(
                    [ensure_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
                     "-i", str(source), "-filter:a",
                     build_stretch_filter(requested, sr, prefer_rubberband),
                     "-ac", "1", "-c:a", "pcm_s16le", str(candidate)],
                    description=f"slow speech (segment {seg.index}, {requested:.4f}x)", timeout=120.0,
                )
                audio = _valid_audio(candidate, sr, engine)
                post = trim_silence_for_overrun(
                    audio, sr, audio.size / float(sr),
                    threshold_dbfs=engine.silence_threshold_dbfs, pad_ms=engine.silence_pad_ms,
                )
                audio = post.audio
                measured = audio.size / float(sr)
                expected = prepared_seconds / requested
                tolerance = min(0.06, max(0.01, expected * 0.10))
                if measured + tolerance < expected or measured + 1 / sr < prepared_seconds:
                    raise ValueError("slowdown render unexpectedly short; possible speech loss")
                # Validate the *measured* slowdown as well as the requested
                # filter value: a padded render must not exceed the 0.90x
                # duration bound even if the target window is much longer.
                limit = min(target, prepared_seconds / engine.min_tempo)
                if measured <= limit + 0.5 / sr:
                    audio_fx.write_wav(output, audio, sr)
                    seg.sync_trimmed_head_seconds += post.head_seconds
                    seg.sync_trimmed_internal_seconds += post.internal_seconds
                    seg.sync_trimmed_tail_seconds += post.tail_seconds
                    seg.sync_trimmed_internal_cuts += post.internal_cuts
                    candidate.unlink(missing_ok=True)
                    return requested, measured
                # Retry closer to 1.0, never above 1.0, always from the
                # prepared clip; both target and measured slowdown are bounded.
                requested = min(1.0, requested * measured / limit * 1.002)
            except Exception as exc:
                logger.warning("Segment %d slowdown attempt failed (%s); trying a safe fallback.", seg.index, exc)
                break
    candidate.unlink(missing_ok=True)
    original = _valid_audio(source, sr, engine)
    audio_fx.write_wav(output, original, sr)
    seg.sync_recovery = "slowdown was unreliable; unsped prepared speech retained"
    logger.warning("Segment %d: slowdown could not be verified; original speech retained.", seg.index)
    return 1.0, original.size / float(sr)


def prepare_clip(engine, seg: Segment, window: Tuple[float, float], output: Path) -> Optional[Path]:
    """Write a valid trimmed/optionally slowed WAV or record an explicit issue."""
    try:
        source = Path(seg.tts_audio_path) if seg.tts_audio_path else None
        if source is None or not source.is_file():
            raise ValueError("TTS audio file missing")
        clip = _valid_audio(source, engine.sample_rate, engine)
        sr = engine.sample_rate
        seg.tts_duration = clip.size / float(sr)
        target = max(0.0, float(window[1] - window[0]))
        raw_duration = float(seg.tts_raw_duration or seg.tts_duration)
        seg.sync_overrun_before_seconds = max(0.0, raw_duration - target)
        seg.sync_trimmed_head_seconds = 0.0
        seg.sync_trimmed_internal_seconds = 0.0
        seg.sync_trimmed_tail_seconds = 0.0
        seg.sync_trimmed_internal_cuts = 0
        if not seg.tts_silence_trimmed:
            trimmed = trim_silence_for_overrun(
                clip, sr, clip.size / float(sr),
                threshold_dbfs=engine.silence_threshold_dbfs, pad_ms=engine.silence_pad_ms,
            )
            clip = trimmed.audio
            seg.sync_trimmed_head_seconds = trimmed.head_seconds
            seg.sync_trimmed_internal_seconds = trimmed.internal_seconds
            seg.sync_trimmed_tail_seconds = trimmed.tail_seconds
            seg.sync_trimmed_internal_cuts = trimmed.internal_cuts
        if clip.size == 0 or not np.any(clip):
            raise ValueError("trimmed TTS is empty or silent")
        prepared_duration = clip.size / float(sr)
        tempo = compute_tempo(prepared_duration, target, min_tempo=engine.min_tempo)
        if tempo < 0.999999:
            prepared = output.with_name(f"{output.stem}_trimmed.wav")
            audio_fx.write_wav(prepared, clip, sr)
            try:
                tempo, duration = _render_slowdown(engine, seg, prepared, prepared_duration, target, tempo, output)
            finally:
                prepared.unlink(missing_ok=True)
        else:
            audio_fx.write_wav(output, clip, sr)
            tempo, duration = 1.0, prepared_duration
        seg.tempo_used = tempo
        seg.stretched_duration = duration
        seg.stretched_audio_path = output
        return output
    except Exception as exc:
        seg.sync_issue = f"TTS preparation failed ({type(exc).__name__}: {exc})"
        logger.warning("Segment %d: %s", seg.index, seg.sync_issue)
        return None
