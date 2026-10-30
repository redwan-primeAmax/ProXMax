"""Fast end-priority silence trimming used before duration fitting."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def trim_edge_silence_for_overrun(
    audio: np.ndarray,
    sample_rate: int,
    overrun_seconds: float,
    threshold_dbfs: float = -45.0,
    pad_ms: int = 25,
) -> Tuple[np.ndarray, float, float]:
    """Remove only edge silence, taking from the tail before the head.

    The requested trim is capped by silence detected outside the active speech
    frames. Internal pauses and the final/initial voiced frames are preserved.
    Returns ``(audio, head_seconds, tail_seconds)``.
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
