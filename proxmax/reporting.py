"""Secret-safe Markdown processing reports for each dubbed video."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from .models import Segment, StageTimings
from .reporting_common import _fmt_seconds, _md, report_path_for
from .reporting_media import (
    _ACTIVITY_FRAME_MS, _ACTIVITY_THRESHOLD_DBFS, _activity_summary, _audio_section,
    _media_section, _probe_media, _safe_config, _window_rms,
)
from .reporting_timeline import (
    _low_confidence_words, _quality_warnings, _render_unit, _sync_windows, _unit_children, _unit_timing_rows,
)

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
    translation_issues = sum(bool(unit.translation_issues) for unit in segments)
    translation_notes = sum(bool(unit.translation_notes) for unit in segments)
    rewritten_units = sum(bool(unit.original_text_bn) for unit in segments)
    low_confidence_units = sum(bool(_low_confidence_words(unit)) for unit in segments)
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
        f"- **Source transcription review:** {low_confidence_units} units have at least one recognizer word probability below 0.35. Confidence is an uncalibrated review cue, not proof of an error or accuracy.",
        f"- **Translation review:** {translation_issues} units with unresolved number/script signals; "
        f"{translation_notes} with automatic-correction or fit-retry notes; {rewritten_units} rewritten for fit. "
        "These limited checks do not verify meaning, names, or naturalness.",
        f"- **Approximate generated speech:** {float((data.get('sync') or {}).get('speech_seconds', 0.0) or 0.0):.2f} s; "
        f"**remaining track silence:** {float((data.get('sync') or {}).get('silence_seconds', 0.0) or 0.0):.2f} s "
        "(includes intentional source pauses and any unfilled windows; not all is an error).",
        f"- **Detected edge silence removed:** {float((data.get('sync') or {}).get('silence_trimmed_seconds', 0.0) or 0.0):.3f} s "
        f"(tail {float((data.get('sync') or {}).get('silence_trimmed_tail_seconds', 0.0) or 0.0):.3f} s; "
        f"head {float((data.get('sync') or {}).get('silence_trimmed_head_seconds', 0.0) or 0.0):.3f} s, "
        f"{int((data.get('sync') or {}).get('silence_trimmed_clips', 0) or 0)} clips); "
        f"**unplaced clips:** {int((data.get('sync') or {}).get('unplaced', 0) or 0)}; "
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
            "Translation and TTS can overlap; their worker/call times include fit retries and must not be added to obtain total elapsed time. TTS call time excludes waiting for translation queue items. The initial combined elapsed time excludes later fit retries.",
            "",
            "| Stage | Wall time (s) |",
            "|---|---:|",
            f"| Audio extraction | {timings.extract:.3f} |",
            f"| Speech recognition | {timings.stt:.3f} |",
            f"| Translation worker, including fit retries | {timings.translate:.3f} |",
            f"| TTS calls, including fit retries | {timings.tts:.3f} |",
            f"| Initial translation + TTS combined elapsed | {timings.translate_tts_wall:.3f} |",
            f"| Fit-retry translation | {float((data.get('tts_fit_retry') or {}).get('translation_seconds', 0.0)):.3f} |",
            f"| Fit-retry TTS | {float((data.get('tts_fit_retry') or {}).get('tts_seconds', 0.0)):.3f} |",
            f"| Fit-retry audio comparison | {timings.fit_audio_check:.3f} |",
            f"| Sync / waveform assembly, excluding fit requests and comparison | {timings.sync:.3f} |",
            f"| Mux / copy combined | {timings.mux:.3f} |",
            f"| Mux processing | {timings.mux_processing:.3f} |",
            f"| Output file copy | {timings.output_copy:.3f} |",
            f"| Post-mux speech validation | {timings.validation:.3f} |",
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
    validation = data.get("audio_validation", {})
    validation_units = {
        int(item.get("segment_index", -1)): item
        for item in validation.get("units", [])
        if isinstance(item, Mapping)
    } if isinstance(validation, Mapping) else {}
    lines += ["", "## Post-mux Bengali speech coverage check", ""]
    if isinstance(validation, Mapping):
        summary = validation.get("summary", {})
        lines.append(f"- **Status:** {_md(validation.get('status', 'unavailable'))}.")
        lines.append(f"- **Method:** {_md(validation.get('method', 'not run'))}.")
        if isinstance(summary, Mapping):
            lines.append(
                f"- **Units checked:** {int(summary.get('checked', 0) or 0)}; "
                f"likely complete: {int(summary.get('likely_complete', 0) or 0)}; "
                f"review: {int(summary.get('review', 0) or 0)}; "
                f"missing: {int(summary.get('missing', 0) or 0)}."
            )
    else:
        lines.append("- **Status:** unavailable; validator was not enabled or did not return results.")
    lines += [
        "- This compares a Bengali Whisper transcript against the expected Bangla text. ASR can mishear even clean speech; flags are review cues, not proof.",
        "",
        "## Transcript-derived action / speech timeline",
        "",
        "Each unit marks what was said and when the generated Bangla clip was scheduled/placed. It is not visual scene/action recognition.",
        "",
    ]
    if rows:
        for row in rows:
            unit = row["unit"]
            lines += _render_unit(
                row, source_audio, dub_audio,
                validation_units.get(int(unit.index)),
            )
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
