"""Conservative, end-first silence trimming used before duration fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Protect very quiet speech more conservatively than the historical edge-only
# detector. This is still a waveform heuristic, not a speech recognizer.
CONSERVATIVE_SILENCE_DBFS = -55.0
# By default, retain only the speech-protection padding on both sides of a pause.
# Callers may request additional pause preservation if desired.
MIN_INTERNAL_PAUSE_MS = 0


@dataclass(frozen=True)
class SilenceTrimResult:
    audio: np.ndarray
    head_seconds: float = 0.0
    internal_seconds: float = 0.0
    tail_seconds: float = 0.0
    internal_cuts: int = 0


def trim_silence_for_overrun(
    audio: np.ndarray,
    sample_rate: int,
    overrun_seconds: float,
    threshold_dbfs: float = -45.0,
    pad_ms: int = 25,
    min_internal_pause_ms: int = MIN_INTERNAL_PAUSE_MS,
) -> SilenceTrimResult:
    """Remove only the needed detected silence, from the end toward the start.

    Ten-millisecond peak frames at or below a conservative threshold are
    candidates. No above-threshold frame is removed; padding around active
    frames and any requested minimum interior pause are retained. By default,
    only the padding limits interior removal. Quiet speech below the threshold
    can still be misclassified, so
    this cannot replace listening to the result.
    """
    samples = np.asarray(audio, dtype=np.float32)
    rate = int(sample_rate)
    if samples.ndim != 1:
        raise ValueError("Silence trimming expects mono audio")
    if samples.size == 0 or rate <= 0 or overrun_seconds <= 0:
        return SilenceTrimResult(samples)

    frame_size = max(1, int(round(rate * 0.01)))
    frame_count = (samples.size + frame_size - 1) // frame_size
    padded = np.pad(samples, (0, frame_count * frame_size - samples.size))
    peaks = np.max(np.abs(padded.reshape(frame_count, frame_size)), axis=1)
    safe_dbfs = min(float(threshold_dbfs), CONSERVATIVE_SILENCE_DBFS)
    threshold = float(10.0 ** (safe_dbfs / 20.0))
    active = np.flatnonzero(peaks > threshold)
    # Do not remove an entirely silent clip: its contents cannot be assessed.
    if active.size == 0:
        return SilenceTrimResult(samples)

    pad = max(0, int(round(rate * max(0, pad_ms) / 1000.0)))
    first, last = int(active[0]), int(active[-1])
    head_end = max(0, first * frame_size - pad)
    tail_start = min(samples.size, (last + 1) * frame_size + pad)
    # Each candidate is (rightmost removable sample, available samples, kind).
    candidates = [(samples.size, samples.size - tail_start, "tail")]

    silent = peaks <= threshold
    transitions = np.diff(np.concatenate(([False], silent, [False])).astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    minimum_pause = max(2 * pad, int(round(rate * max(0, min_internal_pause_ms) / 1000.0)))
    for start_frame, end_frame in zip(starts, ends):
        # Edge silence is accounted for separately. Interior candidates must
        # have an active frame on each side and never include an active frame.
        if start_frame <= first or end_frame > last:
            continue
        start = int(start_frame) * frame_size
        end = min(samples.size, int(end_frame) * frame_size)
        available = max(0, end - start - minimum_pause)
        if available:
            candidates.append((end - pad, available, "internal"))
    candidates.append((head_end, head_end, "head"))

    remaining = min(samples.size - 1, max(0, int(round(float(overrun_seconds) * rate))))
    cuts: list[tuple[int, int]] = []
    removed = {"head": 0, "internal": 0, "tail": 0}
    internal_cuts = 0
    for end, available, kind in sorted(candidates, key=lambda item: item[0], reverse=True):
        count = min(remaining, available)
        if count <= 0:
            continue
        cuts.append((end - count, end))
        removed[kind] += count
        internal_cuts += int(kind == "internal")
        remaining -= count
        if remaining <= 0:
            break
    if not cuts:
        return SilenceTrimResult(samples)

    pieces = []
    cursor = 0
    for start, end in sorted(cuts):
        pieces.append(samples[cursor:start])
        cursor = end
    pieces.append(samples[cursor:])
    return SilenceTrimResult(
        np.ascontiguousarray(np.concatenate(pieces)),
        head_seconds=removed["head"] / float(rate),
        internal_seconds=removed["internal"] / float(rate),
        tail_seconds=removed["tail"] / float(rate),
        internal_cuts=internal_cuts,
    )


def trim_edge_silence_for_overrun(
    audio: np.ndarray,
    sample_rate: int,
    overrun_seconds: float,
    threshold_dbfs: float = -45.0,
    pad_ms: int = 25,
) -> Tuple[np.ndarray, float, float]:
    """Remove only edge silence, taking from the tail before the head.

    The requested trim is capped by detected silence outside above-threshold
    frames and padding; internal pauses are preserved. Very quiet speech may be
    misclassified as silence. Returns ``(audio, head_seconds, tail_seconds)``.
    """
    samples = np.asarray(audio, dtype=np.float32)
    rate = int(sample_rate)
    if samples.size == 0 or rate <= 0 or overrun_seconds <= 0:
        return samples, 0.0, 0.0

    frame_size = max(1, int(round(rate * 0.01)))
    frame_count = (samples.size + frame_size - 1) // frame_size
    padded_size = frame_count * frame_size
    padded = np.pad(samples, (0, padded_size - samples.size))
    peaks = np.max(np.abs(padded.reshape(frame_count, frame_size)), axis=1)
    threshold = float(10.0 ** (float(threshold_dbfs) / 20.0))
    active = np.flatnonzero(peaks > threshold)
    if active.size == 0:
        return samples, 0.0, 0.0

    pad = max(0, int(round(rate * max(0, pad_ms) / 1000.0)))
    active_start = max(0, int(active[0]) * frame_size - pad)
    active_end = min(samples.size, (int(active[-1]) + 1) * frame_size + pad)
    leading_silence = active_start
    trailing_silence = max(0, samples.size - active_end)

    requested = min(samples.size - 1, int(round(float(overrun_seconds) * rate)))
    trim_tail = min(requested, trailing_silence)
    remaining = requested - trim_tail
    trim_head = min(remaining, leading_silence)
    start = trim_head
    end = samples.size - trim_tail
    if end <= start:
        return samples, 0.0, 0.0

    return (
        np.ascontiguousarray(samples[start:end]),
        trim_head / float(rate),
        trim_tail / float(rate),
    )


def measured_active_seconds(path, pad_ms: int = 25, threshold_dbfs: float = -45.0) -> float:
    """Measure the span remaining after removal of detected edge silence."""
    from .audio_fx import read_wav

    audio, rate = read_wav(path)
    if rate <= 0:
        raise ValueError("Invalid audio sample rate")
    threshold = float(10.0 ** (float(threshold_dbfs) / 20.0))
    if audio.size == 0 or not np.any(np.abs(audio) > threshold):
        raise ValueError("Audio contains no detectable active frames")
    duration = audio.size / float(rate)
    trimmed, _, _ = trim_edge_silence_for_overrun(
        audio, rate, duration, threshold_dbfs=threshold_dbfs, pad_ms=pad_ms
    )
    return trimmed.size / float(rate)


def measured_fit_durations(path, pad_ms: int = 25, threshold_dbfs: float = -45.0) -> tuple[float, float]:
    """Return actual WAV duration and minimum protected duration before stretch.

    Compare regenerated and original clips fairly even when their removable
    internal silence differs; neither input file is modified.
    """
    from .audio_fx import read_wav

    audio, rate = read_wav(path)
    if rate <= 0:
        raise ValueError("Invalid audio sample rate")
    threshold = float(10.0 ** (min(float(threshold_dbfs), CONSERVATIVE_SILENCE_DBFS) / 20.0))
    if audio.size == 0 or not np.any(np.abs(audio) > threshold):
        raise ValueError("Audio contains no detectable active frames")
    actual = audio.size / float(rate)
    result = trim_silence_for_overrun(
        audio, rate, actual, threshold_dbfs=threshold_dbfs, pad_ms=pad_ms
    )
    return actual, result.audio.size / float(rate)
