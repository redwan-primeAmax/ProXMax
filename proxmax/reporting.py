"""Secret-safe Markdown processing reports for each dubbed video.

রিপোর্টে source/dub media metadata, transcript-derived action timeline,
word timings, TTS/sync placement, এবং waveform-level activity diagnostics থাকে।
কোনো API key বা absolute Drive path লেখা হয় না।
"""

from __future__ import annotations

import datetime as dt
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import audio_fx
from .models import Segment, StageTimings
from .prosody import subtitles_for_unit
from .sync_engine import plan_windows


_ACTIVITY_THRESHOLD_DBFS = -45.0
_ACTIVITY_FRAME_MS = 100
_MIN_ACTIVITY_RUN_SECONDS = 0.25


def report_path_for(video_path: Path | str, output_dir: Path | str) -> Path:
    """Return the single Markdown report path for a processed source video."""
    source = Path(video_path)
    return Path(output_dir) / f"{source.stem}_report.markdown"


def _fmt_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(value):
        return "n/a"
    millis = int(round(max(0.0, value) * 1000.0))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds_int, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d}.{millis:03d}"


def _md(value: Any) -> str:
    """Escape values for safe, readable Markdown table/text cells."""
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


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
    """Expose only processing settings; deliberately omit keys, tokens, and paths."""
    if config is None:
        return {}
    values: Dict[str, Dict[str, Any]] = {
        "STT": {
            "model": getattr(config, "whisper_model", None),
            "device": getattr(config, "whisper_device", None),
            "compute_type": getattr(config, "whisper_compute_type", None),
            "beam_size": getattr(config, "whisper_beam_size", None),
            "chunk_min_seconds": getattr(config, "chunk_min_seconds", None),
            "chunk_max_seconds": getattr(config, "chunk_max_seconds", None),
            "seam_repair": getattr(config, "seam_repair", None),
        },
        "Translation": {
            "backend": getattr(config, "translator_backend", None),
            "model": getattr(config, "gemini_model", None)
            if getattr(config, "translator_backend", None) == "gemini"
            else getattr(config, "indictrans2_model", None),
            "char_rate_chars_per_second": getattr(config, "translation_char_rate", None),
            "speech_rate_calibration": getattr(config, "translation_calibrate_speech_rate", None),
            "shorten_pass": getattr(config, "translation_shorten_pass", None),
            "expand_pass": getattr(config, "translation_expand_pass", None),
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


def _unit_children(unit: Segment) -> List[Segment]:
    children = list(getattr(unit, "children", []) or [])
    return children if children else [unit]


def _sync_windows(segments: Sequence[Segment], total_duration: float, config: Any) -> Dict[int, Tuple[float, float]]:
    usable = [segment for segment in sorted(segments, key=lambda s: s.start) if segment.tts_audio_path]
    if not usable:
        return {}
    try:
        windows = plan_windows(
            usable,
            total_duration,
            min_gap_seconds=float(getattr(config, "slot_gap_ms", 80)) / 1000.0,
            borrow_seconds=float(getattr(config, "borrow_silence_ms", 250)) / 1000.0,
        )
    except Exception:
        return {}
    return {id(segment): window for segment, window in zip(usable, windows)}


def _unit_timing_rows(
    segments: Sequence[Segment],
    total_duration: float,
    config: Any,
) -> List[Dict[str, Any]]:
    windows = _sync_windows(segments, total_duration, config)
    ordered = sorted(segments, key=lambda s: (s.start, s.end, s.index))
    rows: List[Dict[str, Any]] = []
    previous_source_end = 0.0
    previous_dub_end: Optional[float] = None
    for position, unit in enumerate(ordered, start=1):
        window = windows.get(id(unit))
        placed = float(unit.placed_duration) if unit.placed_duration is not None else None
        actual_end = float(unit.start) + placed if placed is not None else None
        next_unit = ordered[position] if position < len(ordered) else None
        next_start = float(next_unit.start) if next_unit is not None else total_duration
        allocated_end = float(window[1]) if window else None
        trimmed = max(0.0, float(unit.stretched_duration or 0.0) - placed) if placed is not None else None
        rows.append({
            "number": position,
            "unit": unit,
            "source_gap_before": max(0.0, float(unit.start) - previous_source_end),
            "dub_gap_before": max(0.0, float(unit.start) - previous_dub_end) if previous_dub_end is not None else max(0.0, float(unit.start)),
            "target_start": float(window[0]) if window else None,
            "target_end": allocated_end,
            "actual_start": float(unit.start) if placed is not None else None,
            "actual_end": actual_end,
            "tts_duration": unit.tts_duration,
            "stretched_duration": unit.stretched_duration,
            "placed_duration": placed,
            "tempo": getattr(unit, "tempo_used", None),
            "trimmed_seconds": trimmed,
            "unused_slot_seconds": max(0.0, allocated_end - actual_end) if allocated_end is not None and actual_end is not None else None,
            "gap_after_audio": max(0.0, next_start - actual_end) if actual_end is not None else None,
        })
        previous_source_end = max(previous_source_end, float(unit.end))
        if actual_end is not None:
            previous_dub_end = actual_end
    return rows


def _render_unit(row: Mapping[str, Any], source_audio: Mapping[str, Any], dub_audio: Mapping[str, Any]) -> List[str]:
    unit: Segment = row["unit"]
    number = int(row["number"])
    start, end = float(unit.start), float(unit.end)
    lines = [f"### Unit {number:03d} — {_fmt_seconds(start)}–{_fmt_seconds(end)}"]
    lines.append(f"- **Spoken-action window (source transcript):** {_fmt_seconds(start)}–{_fmt_seconds(end)} ({max(0.0, end-start):.3f} s).")
    lines.append(f"- **Source gap before:** {float(row['source_gap_before']):.3f} s; **dub gap before previous placed clip:** {float(row['dub_gap_before']):.3f} s.")
    if row.get("target_start") is not None:
        lines.append(
            f"- **Sync target window:** {_fmt_seconds(row['target_start'])}–{_fmt_seconds(row['target_end'])} "
            f"({max(0.0, row['target_end'] - row['target_start']):.3f} s)."
        )
    else:
        lines.append("- **Sync target window:** no generated TTS clip/window.")
    if row.get("actual_start") is not None:
        lines.append(
            f"- **Dub actually placed:** {_fmt_seconds(row['actual_start'])}–{_fmt_seconds(row['actual_end'])} "
            f"({row['placed_duration']:.3f} s); TTS={float(row['tts_duration'] or 0.0):.3f} s; "
            f"stretched={float(row['stretched_duration'] or 0.0):.3f} s; "
            f"tempo={float(row['tempo'] or 1.0):.3f}×; estimated trimmed={float(row['trimmed_seconds'] or 0.0):.3f} s; "
            f"unused target slot={float(row['unused_slot_seconds'] or 0.0):.3f} s; "
            f"gap after={float(row['gap_after_audio'] or 0.0):.3f} s."
        )
    else:
        lines.append("- **Dub actually placed:** no audio was recorded as placed for this unit.")
    lines.append(
        f"- **Window RMS:** source {_window_rms(source_audio, start, end)}; "
        f"dub {_window_rms(dub_audio, float(row.get('actual_start') or start), float(row.get('actual_end') or end))}."
    )
    lines.append(f"- **English transcript:** {_md(unit.text_en or '')}")
    lines.append(f"- **Bengali translation:** {_md(unit.text_bn or '') if (unit.text_bn or '').strip() else '**MISSING**'}")

    children = _unit_children(unit)
    lines.append("- **Original STT sentence/cue pieces:**")
    for child in children:
        lines.append(
            f"  - #{child.index} {_fmt_seconds(child.start)}–{_fmt_seconds(child.end)}: "
            f"{_md(child.text_en or '')}"
        )
    try:
        captions = subtitles_for_unit(unit)
    except Exception:
        captions = []
    if captions:
        lines.append("- **Bengali report cues** (estimated from placed unit duration; these are not separate `.srt` files):")
        for cue in captions:
            lines.append(f"  - {_fmt_seconds(cue.start)}–{_fmt_seconds(cue.end)}: {_md(cue.text_bn)}")

    words = list(getattr(unit, "words", []) or [])
    if not words:
        words = [word for child in children for word in (getattr(child, "words", []) or [])]
    if words:
        lines += ["", "<details>", f"<summary>Source word timestamps and recognizer confidence ({len(words)} words)</summary>", "", "| # | Start | End | Word | Probability |", "|---:|---:|---:|---|---:|"]
        for word_number, word in enumerate(sorted(words, key=lambda w: (w.start, w.end)), start=1):
            probability = f"{float(word.probability):.3f}" if word.probability is not None else "n/a"
            lines.append(
                f"| {word_number} | {_fmt_seconds(word.start)} | {_fmt_seconds(word.end)} "
                f"| {_md(word.word)} | {probability} |"
            )
        lines += ["", "</details>"]
    else:
        lines.append("- **Word timestamps:** not supplied by the STT backend for this unit.")
    return lines


def _quality_warnings(rows: Sequence[Mapping[str, Any]], stats: Mapping[str, Any]) -> List[str]:
    warnings: List[str] = []
    for row in rows:
        unit: Segment = row["unit"]
        label = f"Unit {int(row['number']):03d} ({_fmt_seconds(unit.start)}–{_fmt_seconds(unit.end)})"
        if not (unit.text_bn or "").strip():
            warnings.append(f"{label}: Bengali translation is empty.")
        if (unit.text_bn or "").strip() and not unit.tts_audio_path:
            warnings.append(f"{label}: translated text exists but no TTS audio path was produced.")
        if unit.tts_audio_path and not unit.stretched_audio_path:
            warnings.append(f"{label}: TTS clip was not successfully prepared for sync.")
        trimmed = float(row.get("trimmed_seconds") or 0.0)
        if trimmed > 0.08:
            warnings.append(f"{label}: approximately {trimmed:.3f} s was trimmed at placement; listen for a clipped ending.")
        tempo = float(row.get("tempo") or 1.0)
        if tempo > 1.15:
            warnings.append(f"{label}: audio was sped up to {tempo:.3f}×; verify intelligibility/prosody.")
        if row.get("unused_slot_seconds") is not None and row["unused_slot_seconds"] > 0.8:
            warnings.append(f"{label}: {row['unused_slot_seconds']:.3f} s of the bounded target slot remained unused; this is not automatically an error if it is a genuine pause.")
    if float(stats.get("untranslated_seconds", 0.0) or 0.0) > 0:
        warnings.append(f"Pipeline recorded {float(stats['untranslated_seconds']):.2f} s of untranslated source speech.")
    sync = stats.get("sync", {})
    if isinstance(sync, Mapping) and int(sync.get("trimmed", 0) or 0) > 0:
        warnings.append(f"Sync engine reports {int(sync['trimmed'])} clipped/trimmed clip(s).")
    if not warnings:
        warnings.append("No missing translation/TTS, high-tempo, or recorded-trimming warning was detected. This does not replace listening review.")
    return warnings


def write_detailed_report(
    *,
    video_path: Path | str,
    output_video_path: Path | str,
    output_dir: Path | str,
    work_dir: Path | str,
    segments: Sequence[Segment],
    timings: Optional[StageTimings],
    stats: Optional[Mapping[str, Any]],
    config: Any,
    total_duration: Optional[float] = None,
) -> Path:
    """Write one detailed `.markdown` report; no separate subtitle files are written."""
    source = Path(video_path)
    output_video = Path(output_video_path)
    work = Path(work_dir)
    report = report_path_for(source, output_dir)
    report.parent.mkdir(parents=True, exist_ok=True)

    source_info = _probe_media(source)
    output_info = _probe_media(output_video)
    source_audio = _activity_summary(work / "audio_16k.wav")
    dub_audio = _activity_summary(work / "final_audio_bn.wav")
    source_duration = float(total_duration or 0.0)
    if source_duration <= 0:
        try:
            source_duration = float(source_info.get("format", {}).get("duration", 0.0) or 0.0)
        except (TypeError, ValueError):
            source_duration = 0.0

    data = dict(stats or {})
    rows = _unit_timing_rows(segments, source_duration, config)
    children = [child for unit in segments for child in _unit_children(unit)]
    translated_units = sum(bool((unit.text_bn or "").strip()) for unit in segments)
    tts_units = sum(bool(unit.tts_audio_path) for unit in segments)
    placed_units = sum(unit.placed_duration is not None for unit in segments)
    warnings = _quality_warnings(rows, data)

    source_format = source_info.get("format", {}) if not source_info.get("error") else {}
    output_format = output_info.get("format", {}) if not output_info.get("error") else {}
    try:
        source_container_duration = float(source_format["duration"])
        output_container_duration = float(output_format["duration"])
        duration_delta: Optional[float] = output_container_duration - source_container_duration
    except (KeyError, TypeError, ValueError):
        source_container_duration = output_container_duration = None
        duration_delta = None
    source_container_text = f"{source_container_duration:.3f} s" if source_container_duration is not None else "n/a"
    output_container_text = f"{output_container_duration:.3f} s" if output_container_duration is not None else "n/a"
    duration_delta_text = f"{duration_delta:+.3f} s" if duration_delta is not None else "n/a (probe metadata missing)"

    lines: List[str] = [
        f"# ProXMax Detailed Processing Report — `{_md(source.name)}`",
        "",
        f"- **Generated:** {dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- **Source:** `{_md(source.name)}`",
        f"- **Dubbed output:** `{_md(output_video.name)}`",
        f"- **Report:** `{_md(report.name)}`",
        "- This report replaces separate English/Bengali `.srt` outputs. It contains their timestamped text plus processing diagnostics.",
        "- API keys, tokens, and absolute filesystem/Drive paths are intentionally excluded.",
        "- The action/event timeline below is **derived from spoken transcript text**. No visual action-recognition model is run, so it does not claim to identify on-screen events.",
        "",
        "## Executive summary",
        "",
        f"- **Input audio/video duration:** {source_duration:.3f} s ({_fmt_seconds(source_duration)}).",
        f"- **Container duration delta (dubbed − source):** {duration_delta_text} "
        f"(source {source_container_text}; dubbed {output_container_text}).",
        f"- **Transcript units:** {len(segments)}; **source STT pieces:** {len(children)}; "
        f"**translated units:** {translated_units}; **units with TTS clips:** {tts_units}; **placed units:** {placed_units}.",
        f"- **Approximate generated speech:** {float((data.get('sync') or {}).get('speech_seconds', 0.0) or 0.0):.2f} s; "
        f"**remaining track silence:** {float((data.get('sync') or {}).get('silence_seconds', 0.0) or 0.0):.2f} s "
        "(includes intentional source pauses and any unfilled windows; not all is an error).",
        f"- **Recorded trimmed clips:** {int((data.get('sync') or {}).get('trimmed', 0) or 0)}; "
        f"**overlaps:** {int((data.get('sync') or {}).get('overlaps', 0) or 0)}.",
        "",
        "## Media metadata",
        "",
    ]
    lines += _media_section("Original source", source, source_info)
    lines += [""]
    lines += _media_section("Final dubbed video", output_video, output_info)
    lines += ["", "## Pipeline stage timings", ""]
    if timings is not None:
        lines += [
            "| Stage | Wall time (s) |",
            "|---|---:|",
            f"| Audio extraction | {timings.extract:.3f} |",
            f"| Speech recognition | {timings.stt:.3f} |",
            f"| Translation | {timings.translate:.3f} |",
            f"| TTS synthesis | {timings.tts:.3f} |",
            f"| Sync / waveform assembly | {timings.sync:.3f} |",
            f"| Mux / copy | {timings.mux:.3f} |",
            f"| Processing total before report rendering | {timings.total:.3f} |",
            f"| Audio duration | {timings.audio_seconds:.3f} |",
            f"| Realtime factor | {timings.realtime_factor:.3f} |",
        ]
    else:
        lines.append("Stage timings unavailable.")
    lines += ["", "## Safe processing configuration", "", "Secret fields and absolute paths are not included.", ""]
    safe = _safe_config(config)
    for section, values in safe.items():
        lines.append(f"### {section}")
        lines.append("```json")
        lines.append(json.dumps(values, ensure_ascii=False, indent=2, sort_keys=True, default=str))
        lines.append("```")
        lines.append("")

    lines += [
        "## Audio activity and gap diagnostics",
        "",
        f"The report uses **{_ACTIVITY_THRESHOLD_DBFS:.0f} dBFS** RMS frames of **{_ACTIVITY_FRAME_MS} ms**. This is waveform activity, not VAD/ASR.",
        "",
    ]
    lines += _audio_section("Original extracted audio", source_audio)
    lines += [""]
    lines += _audio_section("Generated Bangla track before mux", dub_audio)
    lines += [
        "",
        "## Transcript-derived action / speech timeline",
        "",
        "Each unit marks what was said and when the generated Bangla clip was scheduled/placed. It is not visual scene/action recognition.",
        "",
    ]
    if rows:
        for row in rows:
            lines += _render_unit(row, source_audio, dub_audio)
            lines.append("")
    else:
        lines.append("No transcript units were available.")

    lines += ["## Processing statistics", "", "```json"]
    lines.append(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    lines += ["```", "", "## Automated review flags", ""]
    lines.extend(f"- {warning}" for warning in warnings)
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- The source transcript and translated text are included for review; transcript text is not proof of the exact spoken words.",
        "- Waveform activity can find near-silence and dropped audio, but it cannot establish pronunciation, semantic correctness, or whether a silence is intentional.",
        "- A timing overlap only compares timestamps; compare the actual source and dubbed audio before changing genuine pauses.",
        "- Audio files are not embedded in this Markdown report.",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
