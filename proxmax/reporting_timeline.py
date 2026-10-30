"""Transcript-unit timing and per-unit Markdown rendering."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .models import Segment
from .prosody import subtitles_for_unit
from .reporting_common import _fmt_seconds, _md
from .reporting_media import _window_rms
from .sync_math import plan_windows

def _unit_children(unit: Segment) -> List[Segment]:
    children = list(getattr(unit, "children", []) or [])
    return children if children else [unit]


def _low_confidence_words(unit: Segment) -> list:
    words = list(unit.words or []) or [word for child in _unit_children(unit) for word in (child.words or [])]
    result = []
    for word in words:
        try:
            if word.probability is not None and float(word.probability) < 0.35:
                result.append(word)
        except (TypeError, ValueError):
            continue
    return result


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
            "tempo_cap": float(getattr(config, "max_tempo", 1.20)),
            "tempo_cap_measured": getattr(unit, "sync_tempo_cap_measured", False),
            "overrun_before_silence_trim": getattr(unit, "sync_overrun_before_seconds", 0.0),
            "trimmed_head_seconds": getattr(unit, "sync_trimmed_head_seconds", 0.0),
            "trimmed_internal_seconds": getattr(unit, "sync_trimmed_internal_seconds", 0.0),
            "trimmed_internal_cuts": getattr(unit, "sync_trimmed_internal_cuts", 0),
            "trimmed_tail_seconds": getattr(unit, "sync_trimmed_tail_seconds", 0.0),
            "trimmed_seconds": trimmed,
            "unused_slot_seconds": max(0.0, allocated_end - actual_end) if allocated_end is not None and actual_end is not None else None,
            "gap_after_audio": max(0.0, next_start - actual_end) if actual_end is not None else None,
        })
        previous_source_end = max(previous_source_end, float(unit.end))
        if actual_end is not None:
            previous_dub_end = actual_end
    return rows

def _render_unit(
    row: Mapping[str, Any],
    source_audio: Mapping[str, Any],
    dub_audio: Mapping[str, Any],
    validation: Optional[Mapping[str, Any]] = None,
) -> List[str]:
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
            f"pre-fit overrun={float(row['overrun_before_silence_trim'] or 0.0):.3f} s; "
            f"unused target slot={float(row['unused_slot_seconds'] or 0.0):.3f} s; "
            f"gap after={float(row['gap_after_audio'] or 0.0):.3f} s."
        )
    else:
        lines.append("- **Dub actually placed:** no audio was recorded as placed for this unit.")
    if unit.tts_audio_path:
        lines.append(
            f"- **Detected silence removed, tail/internal/head:** "
            f"{float(row['trimmed_tail_seconds'] or 0.0):.3f}/"
            f"{float(row['trimmed_internal_seconds'] or 0.0):.3f}/"
            f"{float(row['trimmed_head_seconds'] or 0.0):.3f} s "
            f"({int(row['trimmed_internal_cuts'] or 0)} internal pauses); "
            f"**allowed tempo cap measured:** {bool(row['tempo_cap_measured'])} "
            f"(cap {float(row['tempo_cap']):.2f}×; not needed for every clip)."
        )
    if getattr(unit, "sync_issue", None):
        lines.append(f"- **Sync repair flag:** {_md(unit.sync_issue)}")
    if validation:
        lines.append(
            f"- **Post-mux Bangla ASR coverage:** {validation.get('status', 'n/a')} "
            f"(token coverage {float(validation.get('coverage', 0.0)):.3f}; "
            f"final-word match={bool(validation.get('tail_token_found', False))})."
        )
        if validation.get("recognized_text"):
            lines.append(f"- **Post-mux ASR heard:** {_md(validation['recognized_text'])}")
    lines.append(
        f"- **Window RMS:** source {_window_rms(source_audio, start, end)}; "
        f"dub {_window_rms(dub_audio, float(row.get('actual_start') or start), float(row.get('actual_end') or end))}."
    )
    lines.append(f"- **English transcript:** {_md(unit.text_en or '')}")
    lines.append(f"- **Bengali translation:** {_md(unit.text_bn or '') if (unit.text_bn or '').strip() else '**MISSING**'}")
    if unit.original_text_bn and unit.original_text_bn != unit.text_bn:
        lines.append(f"- **Original Bengali before fit rewrite:** {_md(unit.original_text_bn)}")
    for issue in unit.translation_issues:
        lines.append(f"- **Translation review issue:** {_md(issue)}")
    for note in unit.translation_notes:
        lines.append(f"- **Translation review note:** {_md(note)}")
    uncertain = _low_confidence_words(unit)
    if uncertain:
        examples = ", ".join(f"{_md(word.word.strip())} ({_fmt_seconds(word.start)})" for word in uncertain[:5])
        suffix = " …" if len(uncertain) > 5 else ""
        lines.append(f"- **Low-confidence source words (review cues, not proof):** {examples}{suffix}")

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
        uncertain = _low_confidence_words(unit)
        if uncertain:
            warnings.append(f"{label}: {len(uncertain)} source word(s) have recognizer probability below 0.35; listen to the source.")
        for issue in unit.translation_issues:
            warnings.append(f"{label}: translation review needed: {issue}.")
        if unit.translation_notes:
            warnings.append(f"{label}: translation or fit-retry note; compare with the source.")
        if getattr(unit, "sync_issue", None):
            warnings.append(f"{label}: sync did not place this full clip: {unit.sync_issue}.")
        if (unit.text_bn or "").strip() and not unit.tts_audio_path:
            warnings.append(f"{label}: translated text exists but no TTS audio path was produced.")
        if unit.tts_audio_path and not unit.stretched_audio_path:
            warnings.append(f"{label}: TTS clip was not successfully prepared for sync.")
        trimmed = float(row.get("trimmed_seconds") or 0.0)
        silence_trimmed = sum(float(row.get(key) or 0.0) for key in (
            "trimmed_tail_seconds", "trimmed_internal_seconds", "trimmed_head_seconds"
        ))
        if trimmed > 0.08:
            warnings.append(f"{label}: approximately {trimmed:.3f} s exceeded the placed duration; listen for fit issues.")
        if silence_trimmed > 0.08:
            warnings.append(
                f"{label}: {silence_trimmed:.3f} s of detected silence was removed end-first; above-threshold frames and padding were retained, but quiet speech can be misclassified."
            )
        tempo = float(row.get("tempo") or 1.0)
        if tempo > 1.10:
            warnings.append(f"{label}: audio was sped up ({tempo:.3f}×; allowed cap {float(row.get('tempo_cap', 1.20)):.2f}×); verify intelligibility/prosody.")
        if row.get("unused_slot_seconds") is not None and row["unused_slot_seconds"] > 0.8:
            warnings.append(f"{label}: {row['unused_slot_seconds']:.3f} s of the bounded target slot remained unused; this is not automatically an error if it is a genuine pause.")
    if float(stats.get("untranslated_seconds", 0.0) or 0.0) > 0:
        warnings.append(f"Pipeline recorded {float(stats['untranslated_seconds']):.2f} s of untranslated source speech.")
    sync = stats.get("sync", {})
    if isinstance(sync, Mapping):
        if int(sync.get("silence_trimmed_clips", 0) or 0) > 0:
            warnings.append(
                f"Silence heuristic trimmed {int(sync['silence_trimmed_clips'])} clip(s), from the end toward the start; review unit-level timing details."
            )
        if int(sync.get("unplaced", 0) or 0) > 0:
            warnings.append(f"Sync safely left {int(sync['unplaced'])} full clip(s) unplaced instead of truncating the sentence.")
    audio_validation = stats.get("audio_validation", {})
    if isinstance(audio_validation, Mapping):
        summary = audio_validation.get("summary", {})
        if audio_validation.get("status") == "unavailable":
            warnings.append(f"Post-mux Bengali ASR coverage check unavailable: {audio_validation.get('reason', 'not reported')}.")
        elif isinstance(summary, Mapping) and (int(summary.get("review", 0) or 0) or int(summary.get("missing", 0) or 0)):
            warnings.append(
                f"Post-mux Bengali ASR review: {int(summary.get('review', 0) or 0)} uncertain and "
                f"{int(summary.get('missing', 0) or 0)} missing/unplaced unit(s); heuristic flag, listen to these timestamps."
            )
    if not warnings:
        warnings.append("No missing translation/TTS, high-tempo, or recorded-trimming warning was detected. This does not replace listening review.")
    return warnings
