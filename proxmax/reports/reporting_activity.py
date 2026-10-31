"""Measured one-second waveform statistics, including decoded final mux audio."""

from __future__ import annotations

import math
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .utils import ensure_tool, run_ffmpeg_with_progress

_THRESHOLD = -45.0


def _measure(samples: np.ndarray, rate: int) -> dict[str, Any]:
    """Measure mono PCM, with time-weighted 100 ms RMS activity."""
    data = np.asarray(samples, dtype=np.float32)
    if not data.size or rate <= 0:
        return {"error": "empty"}
    finite = np.isfinite(data)
    nonfinite = int(data.size - np.count_nonzero(finite))
    if nonfinite:
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    power = float(np.mean(np.square(data, dtype=np.float64)))
    peak = float(np.max(np.abs(data)))
    frame = max(1, rate // 10)
    active_samples = 0
    for offset in range(0, len(data), frame):
        chunk = data[offset:offset + frame]
        rms = math.sqrt(float(np.mean(np.square(chunk, dtype=np.float64))))
        if 20 * math.log10(max(rms, 1e-9)) >= _THRESHOLD:
            active_samples += len(chunk)
    return {
        "rms_dbfs": 20 * math.log10(max(math.sqrt(power), 1e-9)),
        "peak_dbfs": 20 * math.log10(max(peak, 1e-9)),
        "active_percent": 100 * active_samples / len(data),
        "near_clip_samples": int(np.count_nonzero(np.abs(data) >= .999)),
        "nonfinite_samples": nonfinite,
        "duration": len(data) / rate,
    }


def activity_seconds(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Use already-read report PCM; no additional source/dub decode required."""
    samples, rate = summary.get("samples"), summary.get("sample_rate")
    if samples is None or not rate or int(rate) <= 0:
        return []
    rate = int(rate)
    return [_measure(samples[pos:pos + rate], rate) for pos in range(0, len(samples), rate)]


def postmux_seconds(video: Path, work_dir: Path, duration: float | None = None) -> dict[str, Any]:
    """Decode actual output audio to 8 kHz mono, then stream one second at a time.

    The temporary PCM file bounds RAM by one second; FFmpeg preserves the
    output timeline, padding an initially late audio stream to time zero. An
    unavailable decoder is explicit, never represented as measured silence.
    """
    started = time.perf_counter()
    if not video.is_file():
        return {"status": "unavailable", "reason": "output file not found", "seconds": []}
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="report_audio_", dir=str(work_dir)) as temp:
            wav = Path(temp) / "decoded.wav"
            run_ffmpeg_with_progress(
                [ensure_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
                 "-i", str(video), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "8000",
                 "-af", "aresample=async=1:first_pts=0", "-c:a", "pcm_s16le", str(wav)],
                description="Report: decode final mux audio", duration_seconds=duration, stall_timeout=120,
            )
            seconds = []
            with wave.open(str(wav), "rb") as stream:
                rate = stream.getframerate()
                if stream.getnchannels() != 1 or stream.getsampwidth() != 2 or rate <= 0:
                    raise ValueError("unexpected decoded PCM format")
                while True:
                    chunk = stream.readframes(rate)
                    if not chunk:
                        break
                    samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
                    seconds.append(_measure(samples, rate))
            if not seconds:
                raise ValueError("decoded audio is empty")
            return {"status": "ok", "seconds": seconds, "sample_rate": rate,
                    "duration": sum(item["duration"] for item in seconds),
                    "decode_seconds": time.perf_counter() - started}
    except Exception as exc:
        # FFmpeg errors may contain paths, provider details or arbitrary input;
        # expose only their type in a shareable report.
        return {"status": "unavailable", "reason": type(exc).__name__, "seconds": [],
                "decode_seconds": time.perf_counter() - started}
