"""Recover every unplaced clip without truncation or overlapping voices.

The ordinary sync pass uses the configured 1.20x-or-less limit. After one
measured translation rewrite, this last resort uses the actual available gap,
measures each new render, and has no application-level tempo cap. An impossible
window or unreliable FFmpeg render is handled by shifting later speech and,
if necessary, extending the video's last frame. Source timestamps never move.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

from . import audio_fx
from .models import Segment
from .sync_cap_retry import suspicious_stretch_shortfall
from .sync_math import build_stretch_filter, find_overlaps
from .utils import ensure_tool, get_logger, run_ffmpeg

logger = get_logger(__name__)
# Below this duration a time-stretch filter cannot reliably represent even a
# short utterance. Such a slot is treated as impossible and timing is shifted.
_MIN_RENDER_WINDOW = 0.15


def _read_clip(path: Path | str | None, sample_rate: int, engine) -> np.ndarray:
    if not path or not Path(path).is_file():
        raise ValueError("TTS or prepared audio file is missing")
    clip, rate = audio_fx.read_wav(path)
    if rate <= 0 or clip.size == 0 or not np.any(clip):
        raise ValueError("audio is empty, silent, or has an invalid sample rate")
    if rate != sample_rate:
        clip = engine._resample(clip, rate, sample_rate)
    if clip.size == 0 or not np.any(clip):
        raise ValueError("audio became empty or silent after resampling")
    return clip


def _render_to_window(engine, segment: Segment, raw: np.ndarray, target: float, directory: Path):
    """Return an actual measured fit, or the uncut original if fitting fails.

    Each attempt starts from the original valid TTS WAV, never a previous
    time-stretched render. Duration checks catch some broken renders, but do not
    prove phonetic or semantic fidelity, particularly at extreme tempos.
    """
    sr = engine.sample_rate
    raw_duration = raw.size / float(sr)
    if target < _MIN_RENDER_WINDOW or raw_duration <= target:
        reason = "no viable window; shifted timing" if target < _MIN_RENDER_WINDOW else "used available gap"
        return Path(segment.tts_audio_path), raw, 1.0, reason

    tempo = raw_duration / target
    candidate = directory / f"seg_{segment.index:04d}_candidate.wav"
    result = directory / f"seg_{segment.index:04d}_fallback.wav"
    for attempt in range(3):
        try:
            if not math.isfinite(tempo):
                raise ValueError("unrepresentable tempo")
            candidate.unlink(missing_ok=True)
            run_ffmpeg(
                [ensure_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
                 "-i", str(segment.tts_audio_path),
                 "-filter:a", build_stretch_filter(tempo, sr, engine.prefer_rubberband),
                 "-ac", "1", "-c:a", "pcm_s16le", str(candidate)],
                description=f"last-resort sync segment {segment.index} ({tempo:.2f}x)",
                timeout=120.0,
            )
            measured = _read_clip(candidate, sr, engine)
            actual = measured.size / float(sr)
            if suspicious_stretch_shortfall(raw_duration, tempo, actual):
                raise ValueError("stretch unexpectedly short; possible speech loss")
            if measured.size <= int(math.floor(target * sr + 1e-7)):
                candidate.replace(result)
                return result, measured, tempo, "measured unlimited-tempo fit"
            # Real FFmpeg output may be longer than raw_duration/tempo. Recalculate
            # from the measured result and retry from the original file, not itself.
            tempo *= max(1.005, actual / target * 1.005)
        except Exception as exc:
            logger.warning("Segment %d fallback render %d failed: %s", segment.index, attempt + 1, exc)
            break
    candidate.unlink(missing_ok=True)
    logger.warning("Segment %d: retaining original audio and shifting timing; no speech was cropped.", segment.index)
    return Path(segment.tts_audio_path), raw, 1.0, "unreliable stretch; original audio retained and timing shifted"


def build_complete_track(engine, segments: Sequence[Segment], total_duration: float,
                         work_dir: Path | str, output_wav: Path | str) -> Path:
    """Place every expected segment, or raise rather than publish missing speech."""
    ordered = sorted(segments, key=lambda seg: (seg.start, seg.index))
    if not ordered or not math.isfinite(total_duration) or total_duration <= 0:
        raise RuntimeError("Cannot recover speech without segments and a valid video duration")
    if len({seg.index for seg in ordered}) != len(ordered):
        raise RuntimeError("Duplicate segment indexes prevent safe audio placement")
    sr = int(engine.sample_rate)
    gap = max(0, int(round(engine.slot_gap_seconds * sr)))
    directory = Path(work_dir) / "stretched_fallback"
    directory.mkdir(parents=True, exist_ok=True)
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    old_stats = dict(engine.last_stats)
    pairs = []
    previous_end = None
    recovered = []
    shifted = []

    for index, seg in enumerate(ordered):
        original_start = max(0, int(round(float(seg.start) * sr)))
        start = max(original_start, (previous_end + gap) if previous_end is not None else 0)
        previously_placed = seg.placed_duration is not None and seg.stretched_audio_path is not None
        clip = None
        if previously_placed:
            try:
                clip = _read_clip(seg.stretched_audio_path, sr, engine)
            except Exception:
                previously_placed = False
        if not previously_placed:
            try:
                raw = _read_clip(seg.tts_audio_path, sr, engine)
            except Exception as exc:
                raise RuntimeError(f"Segment {seg.index}: cannot recover missing/invalid TTS audio: {exc}") from exc
            if index + 1 < len(ordered):
                end = max(0, int(round(float(ordered[index + 1].start) * sr)) - gap)
            else:
                end = max(0, int(round(total_duration * sr)))
            target = max(0, end - start) / float(sr)
            prior_issue = seg.sync_issue or "prepared audio unavailable"
            path, clip, tempo, reason = _render_to_window(engine, seg, raw, target, directory)
            seg.stretched_audio_path = path
            seg.stretched_duration = clip.size / float(sr)
            seg.tempo_used = tempo
            # Fallback always uses the untouched original TTS, not the initial
            # silence-trimmed clip. Do not report initial trim as applied.
            seg.sync_trimmed_head_seconds = seg.sync_trimmed_tail_seconds = 0.0
            seg.sync_trimmed_internal_seconds = 0.0
            seg.sync_trimmed_internal_cuts = 0
            seg.sync_recovery = f"{prior_issue}; {reason}"
            recovered.append({"segment_index": int(seg.index), "reason": reason, "tempo": round(tempo, 3)})
        assert clip is not None
        end_sample = start + clip.size
        seg.placed_start = start / float(sr)
        seg.placed_duration = None  # _assemble records actual placed duration
        seg.sync_issue = None
        pairs.append((seg, (start / float(sr), end_sample / float(sr))))
        if start > original_start:
            shifted.append({"segment_index": int(seg.index), "seconds": round((start - original_start) / sr, 3)})
        previous_end = end_sample

    timeline_duration = max(float(total_duration), (previous_end or 0) / float(sr))
    engine.last_stats = {**old_stats, "unplaced": 0, "sync_issues": [], "overlaps": 0}
    canvas = engine._assemble(pairs, timeline_duration)
    missing = [seg.index for seg in ordered if seg.placed_duration is None]
    spans = [(seg.actual_start, seg.actual_start + float(seg.placed_duration or 0)) for seg in ordered]
    if missing or engine.last_stats.get("unplaced") or find_overlaps(spans):
        raise RuntimeError(f"Could not safely place every audio segment; missing={missing}")
    audio_fx.write_wav(output_wav, canvas, sr)
    speech = sum(float(seg.placed_duration or 0.0) for seg in ordered)
    silence = max(0.0, timeline_duration - speech)
    tempos = [float(seg.tempo_used) for seg in ordered]
    engine.last_stats.update({
        "clips": len(ordered), "unplaced": 0, "overlaps": 0,
        "max_tempo": max(tempos, default=1.0),
        "aggressive_stretch": sum(tempo > 1.10 for tempo in tempos),
        "slowed_to_fill": sum(tempo < 0.98 for tempo in tempos),
        "speech_seconds": round(speech, 2), "silence_seconds": round(silence, 2),
        "silence_percent": round(100.0 * silence / timeline_duration, 1),
        "silence_trimmed_clips": sum(bool(s.sync_trimmed_head_seconds + s.sync_trimmed_internal_seconds + s.sync_trimmed_tail_seconds) for s in ordered),
        "silence_trimmed_head_seconds": round(sum(s.sync_trimmed_head_seconds for s in ordered), 3),
        "silence_trimmed_internal_seconds": round(sum(s.sync_trimmed_internal_seconds for s in ordered), 3),
        "silence_trimmed_internal_cuts": sum(s.sync_trimmed_internal_cuts for s in ordered),
        "silence_trimmed_tail_seconds": round(sum(s.sync_trimmed_tail_seconds for s in ordered), 3),
        "silence_trimmed_seconds": round(sum(s.sync_trimmed_head_seconds + s.sync_trimmed_internal_seconds + s.sync_trimmed_tail_seconds for s in ordered), 3),
        "fallback_segments": recovered, "shifted_segments": shifted,
        "video_extension_seconds": max(0.0, timeline_duration - float(total_duration)),
        "timeline_duration_seconds": timeline_duration,
        "initial_sync_issues": old_stats.get("sync_issues", []),
    })
    return output_wav
