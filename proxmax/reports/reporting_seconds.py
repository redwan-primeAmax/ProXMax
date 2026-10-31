"""Separate per-second source and edited-output media timelines for reports."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .prosody import subtitles_for_unit
from .reporting_activity import activity_seconds
from .reporting_common import _fmt_seconds, _md
from .reporting_timeline import _unit_children


def _duration(value: Any) -> float:
    try:
        result = float(value or 0.0)
        return max(0.0, result) if math.isfinite(result) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _cover(start: float, end: float, limit: int) -> range:
    """Half-open second bins overlapped by a finite media interval."""
    start, end = _duration(start), _duration(end)
    if end <= start:
        return range(0)
    return range(min(limit, int(start)), min(limit, int(math.ceil(end))))


def _audio_cell(seconds: Sequence[Mapping[str, Any]], index: int) -> str:
    if index >= len(seconds) or seconds[index].get("error"):
        return "unavailable"
    row = seconds[index]
    return (f"{row['rms_dbfs']:.1f}/{row['peak_dbfs']:.1f} dBFS; "
            f"active {row['active_percent']:.0f}%; near-clip {row['near_clip_samples']}; "
            f"samples {row['duration']:.3f}s")


def _packet_cell(packets: Mapping[str, Any], index: int) -> str:
    status = packets.get("status", "unavailable")
    if status == "unavailable":
        return "unavailable"
    if status == "partial" and index >= int(packets.get("last_scanned_second", -1)):
        return "scan truncated"
    row = (packets.get("bins") or {}).get(index)
    if not row:
        return "0 packet starts"
    return (f"{row['count']} packets; PTS {row['first_pts']:.3f}–{row['last_pts']:.3f}s; "
            f"key packets {row['keyframes']}")


def _packet_summary(packets: Mapping[str, Any]) -> str:
    status = packets.get("status", "unavailable")
    if status == "unavailable":
        return f"**unavailable** ({_md(packets.get('reason', 'not scanned'))})"
    return (f"**{_md(status)}**, {packets.get('scanned_packets', '?')} packets, "
            f"{packets.get('missing_pts', '?')} missing PTS, {packets.get('missing_duration', '?')} missing duration, "
            f"origin {_md(packets.get('origin', 'unavailable'))}, {_duration(packets.get('scan_seconds')):.3f} s")


def _packet_end_warning(packets: Mapping[str, Any], end: float, label: str) -> list[str]:
    measured_end = packets.get("end")
    if packets.get("origin") == "measured" and measured_end is not None and measured_end + .03 < end:
        return [f"{label} packet coverage ends before this interval ends: review"]
    return []


def _word_label(word: Any) -> str:
    try:
        probability = f"{float(word.probability):.2f}" if word.probability is not None else "?"
    except (TypeError, ValueError):
        probability = "?"
    return f"{str(word.word).strip()} [{float(word.start):.2f}–{float(word.end):.2f}, p={probability}]"


def _signal(audio: Sequence[Mapping[str, Any]], second: int, label: str) -> list[str]:
    if second >= len(audio):
        return [f"{label}: unavailable/ended"]
    row = audio[second]
    if row.get("error"):
        return [f"{label}: unavailable"]
    result = []
    if row.get("near_clip_samples", 0):
        result.append(f"{label}: near-full-scale samples")
    if row.get("nonfinite_samples", 0):
        result.append(f"{label}: non-finite samples")
    return result


def _source_bins(rows: Sequence[Mapping[str, Any]], count: int):
    units, words, flags = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        unit, number = row["unit"], int(row["number"])
        tag = f"U{number:03d}"
        for second in _cover(unit.start, unit.end, count):
            units[second].append(f"{tag}(segment={unit.index}) [{unit.start:.2f}–{unit.end:.2f}]")
            if unit.translation_issues:
                flags[second].append(f"{tag}: translation review")
            if unit.sync_issue:
                flags[second].append(f"{tag}: sync issue")
            if unit.placed_duration is None:
                flags[second].append(f"{tag}: placement unavailable")
        source_words = list(unit.words or []) or [w for child in _unit_children(unit) for w in (child.words or [])]
        for word in source_words:
            for second in _cover(word.start, word.end, count):
                words[second].append(_word_label(word))
                try:
                    if word.probability is not None and float(word.probability) < .35:
                        flags[second].append(f"{tag}: low-confidence word")
                except (TypeError, ValueError):
                    pass
    return units, words, flags


def _output_bins(rows: Sequence[Mapping[str, Any]], count: int):
    units, cues, flags = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        unit, number = row["unit"], int(row["number"])
        tag = f"U{number:03d}"
        if unit.placed_duration is None:
            continue
        start, end = unit.actual_start, unit.actual_start + float(unit.placed_duration)
        for second in _cover(start, end, count):
            units[second].append(f"{tag}(segment={unit.index}) [{start:.2f}–{end:.2f}], audio {unit.tempo_used:.3f}×")
            if unit.translation_issues:
                flags[second].append(f"{tag}: translation review")
            if unit.sync_issue or unit.sync_recovery:
                flags[second].append(f"{tag}: sync issue/recovery")
        try:
            estimated = subtitles_for_unit(unit)
        except Exception:
            estimated = []
            for second in _cover(start, end, count):
                flags[second].append(f"{tag}: cue generation unavailable")
        for index, cue in enumerate(estimated, 1):
            for second in _cover(cue.start, cue.end, count):
                cues[second].append(f"{tag}:C{index} [{cue.start:.2f}–{cue.end:.2f}] {cue.text_bn}")
    return units, cues, flags


def _picture_bins(plan: Any, count: int):
    bins = defaultdict(list)
    if plan is None:
        return bins
    try:
        fps = float(plan.fps)
        if not math.isfinite(fps) or fps <= 0:
            return bins
        cursor = 0
        for index, piece in enumerate(plan.pieces, 1):
            start, end = cursor / fps, (cursor + piece.output_frames) / fps
            speed = piece.source_frames / piece.output_frames
            label = (f"P{index} {piece.kind} [{start:.2f}–{end:.2f}] "
                     f"source frames {piece.start}–{piece.end - 1}, {speed:.3f}×")
            for second in _cover(start, end, count):
                bins[second].append(label)
            cursor += piece.output_frames
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return defaultdict(list)
    return bins


def media_timeline_lines(rows: Sequence[Mapping[str, Any]], source_audio: Mapping[str, Any],
                         dub_audio: Mapping[str, Any], post: Mapping[str, Any],
                         source_duration: float, plan: Any = None,
                         source_packets: Mapping[str, Any] | None = None,
                         output_packets: Mapping[str, Any] | None = None) -> list[str]:
    """Render every available second without equating source and edited clocks."""
    source_packets = source_packets or {"status": "unavailable"}
    output_packets = output_packets or {"status": "unavailable"}
    source = activity_seconds(source_audio)
    pre = activity_seconds(dub_audio)
    final = post.get("seconds", []) or []
    source_end = max(_duration(source_duration), _duration(source_audio.get("duration")))
    output_end = max(_duration(dub_audio.get("duration")), _duration(getattr(plan, "duration", 0)),
                     _duration(post.get("duration")))
    source_count, output_count = math.ceil(source_end), math.ceil(output_end)
    units, words, source_flags = _source_bins(rows, source_count)
    placed, cues, output_flags = _output_bins(rows, output_count)
    picture = _picture_bins(plan, output_count)
    lines = [
        "## Per-second media diagnostics", "",
        "Source and edited output are **different clocks**; equal second numbers need not show the same content. "
        "All intervals are half-open [start, end), with a shorter final row where needed.",
        "Audio activity is a 100 ms RMS threshold of −45 dBFS; near-clip counts are samples at or above 0.999 full scale. "
        "Levels are measurements, not speech recognition, correctness or proof of clipping.",
        "Source is the extracted/filtered STT waveform; pre-mux is the assembled dub WAV; "
        "post-mux is decoded from the actual output, mono at 8 kHz, aligned to output time zero. "
        "Resampling/AAC can change levels. Unavailable is not silence.",
        f"Post-mux waveform status: **{_md(post.get('status', 'unavailable'))}**; "
        f"reason: {_md(post.get('reason', 'none'))}; decode time: {_duration(post.get('decode_seconds')):.3f} s.",
        "Unit IDs correspond to the numbered detailed units; word probabilities are uncalibrated; "
        "Bengali cue times are estimates, not word-level alignment. Review flags can be false positives.",
        "Picture entries are the planned piece ranges on the normalized source grid, not visual recognition "
        "or exact decoded frame selection; repeated source-tail/blank frames may be synthetic.",
        "Actual picture columns count ffprobe video packets by presentation time relative to container start; "
        "packets are not necessarily decoded frames, and zero packet starts can be normal for VFR. "
        "Scans stop at 500,000 packets and mark later seconds truncated.",
        f"Source packet scan: {_packet_summary(source_packets)}. "
        f"Output packet scan: {_packet_summary(output_packets)}.", "",
        "### Source clock — every second", "",
        "| Source interval | Extracted audio RMS/peak and activity | Actual picture packets | Transcript units | Recognized words (source times) | Review signals |",
        "|---|---|---|---|---|---|",
    ]
    for second in range(source_count):
        flags = list(source_flags[second]) + _signal(source, second, "source audio")
        expected = min(1.0, source_end - second)
        flags += _packet_end_warning(source_packets, min(second + 1, source_end), "source picture")
        if second < len(source) and source[second].get("duration", 0) + .03 < expected:
            flags.append("source waveform ends early within this interval")
        if units[second] and second < len(source) and source[second].get("rms_dbfs", 0) < -55:
            flags.append("near-silent source during transcript: review")
        interval = f"{_fmt_seconds(second)}–{_fmt_seconds(min(second + 1, source_end))}"
        lines.append(f"| {interval} | {_audio_cell(source, second)} | {_packet_cell(source_packets, second)} "
                     f"| {_md('; '.join(units[second]) or 'none')} | {_md('; '.join(words[second]) or 'none supplied')} "
                     f"| {_md('; '.join(dict.fromkeys(flags)) or 'none observed')} |")
    if not source_count:
        lines.append("| unavailable | unavailable | unavailable | unavailable | unavailable | source duration unavailable |")
    lines += ["", "### Edited output clock — every second", "",
              "| Output interval | Pre-mux dub RMS/peak and activity | Decoded post-mux RMS/peak and activity | Actual picture packets | Placed speech | Estimated Bengali cues | Planned picture pieces | Review signals |",
              "|---|---|---|---|---|---|---|---|"]
    for second in range(output_count):
        flags = list(output_flags[second]) + _signal(pre, second, "pre-mux") + _signal(final, second, "post-mux")
        expected = min(1.0, output_end - second)
        flags += _packet_end_warning(output_packets, min(second + 1, output_end), "output picture")
        if second < len(pre) and pre[second].get("duration", 0) + .03 < expected:
            flags.append("pre-mux waveform ends early within this interval")
        if second < len(final) and final[second].get("duration", 0) + .03 < expected:
            flags.append("post-mux waveform ends early within this interval")
        if placed[second] and second < len(pre) and pre[second].get("rms_dbfs", 0) < -55:
            flags.append("near-silent pre-mux during placed unit: review")
        if second < len(pre) and second < len(final):
            if pre[second].get("rms_dbfs", -180) > -45 and final[second].get("rms_dbfs", 0) < -55:
                flags.append("post-mux near-silent while pre-mux active: review")
        interval = f"{_fmt_seconds(second)}–{_fmt_seconds(min(second + 1, output_end))}"
        lines.append(f"| {interval} | {_audio_cell(pre, second)} | {_audio_cell(final, second)} "
                     f"| {_packet_cell(output_packets, second)} | {_md('; '.join(placed[second]) or 'none')} "
                     f"| {_md('; '.join(cues[second]) or 'none')} | {_md('; '.join(picture[second]) or 'unavailable')} "
                     f"| {_md('; '.join(dict.fromkeys(flags)) or 'none observed')} |")
    if not output_count:
        lines.append("| unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | output duration unavailable |")
    lines += ["", "These diagnostics cannot establish whether speech is missing, names are pronounced correctly, "
              "translations preserve meaning, or picture content is visually correct. Listen/watch and review; "
              "an absence of observed flags is not evidence that no silent problem exists.", ""]
    return lines
