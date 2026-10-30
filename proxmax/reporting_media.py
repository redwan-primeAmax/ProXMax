"""Media metadata and waveform-activity diagnostics for processing reports."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from . import audio_fx
from .models import Segment
from .reporting_common import _fmt_seconds, _md

_ACTIVITY_THRESHOLD_DBFS = -45.0
_ACTIVITY_FRAME_MS = 100
_MIN_ACTIVITY_RUN_SECONDS = 0.25

def _probe_media(path: Path | str) -> Dict[str, Any]:
    media_path = Path(path)
    if not media_path.is_file():
        return {"error": "media file not found"}
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {"error": "ffprobe is unavailable"}
    command = [
        ffprobe, "-v", "error", "-show_entries",
        "format=format_name,duration,size,bit_rate:"
        "stream=index,codec_type,codec_name,start_time,duration,bit_rate,width,height,"
        "avg_frame_rate,sample_rate,channels,channel_layout",
        "-of", "json", str(media_path),
    ]
    try:
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            timeout=30, check=False,
        )
    except Exception as exc:
        return {"error": f"probe failed: {type(exc).__name__}"}
    if result.returncode:
        detail = (result.stderr or "ffprobe failed")[-500:]
        detail = detail.replace(str(media_path.resolve()), media_path.name)
        detail = detail.replace(str(media_path), media_path.name)
        return {"error": detail}
    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return {"error": "ffprobe returned invalid JSON"}
    return {"format": data.get("format", {}), "streams": data.get("streams", [])}

def _media_section(label: str, path: Path | str, info: Mapping[str, Any]) -> List[str]:
    media_path = Path(path)
    lines = [f"### {label}", f"- **File:** `{_md(media_path.name)}`"]
    if media_path.is_file():
        try:
            lines.append(f"- **Size:** {media_path.stat().st_size:,} bytes")
        except OSError:
            pass
    if info.get("error"):
        lines.append(f"- **Probe status:** {_md(info['error'])}")
        return lines
    fmt = info.get("format", {})
    lines += [
        f"- **Container:** `{_md(fmt.get('format_name', 'n/a'))}`",
        f"- **Container duration:** {_md(fmt.get('duration', 'n/a'))} s",
        f"- **Container bitrate:** {_md(fmt.get('bit_rate', 'n/a'))} bit/s",
        "",
        "| Stream | Type | Codec | Start (s) | Duration (s) | Bitrate (bit/s) | Details |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for stream in info.get("streams", []):
        kind = stream.get("codec_type", "unknown")
        details: List[str] = []
        if kind == "video":
            details += [
                f"{stream.get('width', '?')}×{stream.get('height', '?')}",
                f"{stream.get('avg_frame_rate', 'n/a')} fps",
            ]
        elif kind == "audio":
            details += [
                f"{stream.get('sample_rate', 'n/a')} Hz",
                f"{stream.get('channels', 'n/a')} ch",
                str(stream.get("channel_layout", "")),
            ]
        lines.append(
            f"| {stream.get('index', '?')} | {_md(kind)} | {_md(stream.get('codec_name', 'n/a'))} "
            f"| {_md(stream.get('start_time', 'n/a'))} | {_md(stream.get('duration', 'n/a'))} "
            f"| {_md(stream.get('bit_rate', 'n/a'))} | {_md(', '.join(x for x in details if x))} |"
        )
    if not info.get("streams"):
        lines.append("| — | No streams reported | — | — | — | — | — |")
    return lines

def _safe_config(config: Any) -> Dict[str, Dict[str, Any]]:
    """Expose safe settings; omit API credentials and filesystem paths."""
    if config is None:
        return {}
    values: Dict[str, Dict[str, Any]] = {
        "STT": {
            "model": getattr(config, "whisper_model", None),
            "device": getattr(config, "whisper_device", None),
            "compute_type": getattr(config, "whisper_compute_type", None),
            "beam_size": getattr(config, "whisper_beam_size", None),
            "source_audio_filter": getattr(config, "stt_audio_filter", None),
            "chunk_min_seconds": getattr(config, "chunk_min_seconds", None),
            "chunk_max_seconds": getattr(config, "chunk_max_seconds", None),
            "seam_repair": getattr(config, "seam_repair", None),
        },
        "Translation": {
            "backend": getattr(config, "translator_backend", None),
            "model": getattr(config, "gemini_model", None)
            if getattr(config, "translator_backend", None) == "gemini"
            else getattr(config, "indictrans2_model", None),
            "gemini_input_token_budget": getattr(config, "gemini_max_input_tokens", None),
            "gemini_max_paragraphs_per_request": getattr(config, "gemini_batch_size", None),
            "pre_tts_duration_prediction": False,
            "shorten_retry_policy": "source-aware after measured TTS and initial tempo cap; one rewrite, then uncapped measured recovery with shifts/extension if necessary",
            "text_expansion_to_fill_gaps": False,
        },
        "TTS": {
            "engine": getattr(config, "tts_engine", None),
            "voice": getattr(config, "resolved_tts_voice", None),
            "rate": getattr(config, "tts_rate", None),
            "trim_silence": getattr(config, "tts_trim_silence", None),
            "trim_pad_ms": getattr(config, "tts_trim_pad_ms", None),
            "unit_max_seconds": getattr(config, "tts_unit_max_seconds", None),
            "unit_max_chars": getattr(config, "tts_unit_max_chars", None),
            "unit_max_gap_seconds": getattr(config, "tts_unit_max_gap", None),
        },
        "Sync": {
            "min_tempo": getattr(config, "min_tempo", None),
            "max_tempo": getattr(config, "max_tempo", None),
            "slot_gap_ms": getattr(config, "slot_gap_ms", None),
            "borrow_silence_ms": getattr(config, "borrow_silence_ms", None),
            "fill_extension": getattr(config, "fill_extension", None),
            "sample_rate": getattr(config, "sync_sample_rate", None),
            "reuse_unchanged_prepared_clips": True,
        },
        "Quality": {
            "gemini_translation_checks": "explicit numbers and unexpected scripts; not semantic validation"
            if getattr(config, "translator_backend", None) == "gemini" else "not applied by this backend",
            "source_word_review_probability_below": 0.35,
            "post_mux_bengali_asr_validation": getattr(config, "validate_dub_audio", None),
            "minimum_text_coverage": getattr(config, "audio_validation_min_coverage", None),
        },
        "Output": {
            "audio_bitrate": getattr(config, "audio_bitrate", None),
            "report_format": "Markdown (.markdown)",
        },
    }
    return {
        section: {key: value for key, value in items.items() if value is not None}
        for section, items in values.items()
        if any(value is not None for value in items.values())
    }

def _activity_summary(path: Optional[Path | str], threshold_dbfs: float = _ACTIVITY_THRESHOLD_DBFS) -> Dict[str, Any]:
    if not path or not Path(path).is_file():
        return {"error": "waveform unavailable"}
    try:
        samples, sample_rate = audio_fx.read_wav(path)
    except Exception as exc:
        return {"error": f"waveform read failed: {type(exc).__name__}"}
    if not samples.size or sample_rate <= 0:
        return {"error": "waveform is empty"}
    frame_samples = max(1, int(sample_rate * _ACTIVITY_FRAME_MS / 1000.0))
    levels: List[float] = []
    for start in range(0, samples.size, frame_samples):
        frame = samples[start:start + frame_samples]
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) if frame.size else 0.0
        levels.append(20.0 * math.log10(max(rms, 1e-9)))
    active = [level >= threshold_dbfs for level in levels]
    frame_seconds = frame_samples / float(sample_rate)
    quiet_runs: List[Tuple[float, float, float]] = []
    run_start: Optional[int] = None
    for index, state in enumerate(active + [True]):
        if not state and run_start is None:
            run_start = index
        elif state and run_start is not None:
            start_s = run_start * frame_seconds
            end_s = min(samples.size / float(sample_rate), index * frame_seconds)
            duration = max(0.0, end_s - start_s)
            if duration >= _MIN_ACTIVITY_RUN_SECONDS:
                quiet_runs.append((start_s, end_s, duration))
            run_start = None
    return {
        "duration": samples.size / float(sample_rate),
        "sample_rate": int(sample_rate),
        "threshold_dbfs": float(threshold_dbfs),
        "active_percent": 100.0 * sum(active) / max(1, len(active)),
        "median_dbfs": float(np.median(levels)),
        "p10_dbfs": float(np.percentile(levels, 10)),
        "p90_dbfs": float(np.percentile(levels, 90)),
        "quiet_runs": quiet_runs,
        "samples": samples,
    }

def _window_rms(summary: Mapping[str, Any], start: float, end: float) -> str:
    samples = summary.get("samples")
    rate = summary.get("sample_rate")
    if samples is None or not rate:
        return "n/a"
    left = max(0, int(max(0.0, start) * rate))
    right = min(len(samples), max(left + 1, int(math.ceil(max(0.0, end) * rate))))
    if right <= left:
        return "n/a"
    return f"{audio_fx.rms_dbfs(samples[left:right]):.1f} dBFS"

def _audio_section(label: str, summary: Mapping[str, Any]) -> List[str]:
    lines = [f"### {label}"]
    if summary.get("error"):
        lines.append(f"- {_md(summary['error'])}")
        return lines
    lines += [
        f"- Duration: **{summary['duration']:.3f} s**; analysis sample rate: **{summary['sample_rate']} Hz mono**.",
        f"- Signal frames above {_ACTIVITY_THRESHOLD_DBFS:.0f} dBFS: **{summary['active_percent']:.1f}%**; "
        f"RMS p10/median/p90: **{summary['p10_dbfs']:.1f}/{summary['median_dbfs']:.1f}/{summary['p90_dbfs']:.1f} dBFS**.",
        "- These are waveform-energy measurements, **not speech detection**; background/game audio can count as active signal.",
    ]
    quiet_runs = summary.get("quiet_runs", [])
    if quiet_runs:
        lines += ["", f"Near-silence intervals (RMS below {_ACTIVITY_THRESHOLD_DBFS:.0f} dBFS, ≥{_MIN_ACTIVITY_RUN_SECONDS:.2f} s):"]
        lines += [f"- {_fmt_seconds(start)}–{_fmt_seconds(end)} ({duration:.2f} s)" for start, end, duration in quiet_runs]
    else:
        lines += ["", "No near-silence run exceeded the configured reporting duration."]
    return lines
