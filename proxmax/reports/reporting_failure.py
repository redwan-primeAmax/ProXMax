"""Best-effort failure reports, written before temporary media is cleaned up."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .reporting_common import _md
from .reporting_media import _activity_summary
from .reporting_packets import picture_packet_seconds
from .reporting_runtime import runtime_lines
from .reporting_seconds import media_timeline_lines
from .reporting_timeline import _unit_timing_rows


def write_failure_report(report: Path, source: Path, work: Path, config: Any,
                         trace: Any, error: BaseException) -> Path:
    """Record safe failure type, partial waveforms and observed events if possible."""
    report = Path(report)
    report.parent.mkdir(parents=True, exist_ok=True)
    source_audio = _activity_summary(work / "audio_16k.wav")
    dub_audio = _activity_summary(work / "final_audio_bn.wav")
    duration = max(float(trace.source_duration or 0.0), float(source_audio.get("duration", 0.0)))
    rows = _unit_timing_rows(trace.segments, duration, config)
    lines = [f"# ProXMax FAILED Processing Report — `{_md(Path(source).name)}`", "",
             "**PROCESSING FAILED — this report does not indicate a valid published output.**", "",
             f"- Error type: `{_md(type(error).__name__)}`. Raw exception text and provider responses are withheld.",
             "- Timeline data below is partial; anything not yet reached is unavailable, not a measured zero.",
             "- The final muxed output was not verified, so post-mux audio is unavailable.",
             "- This report can contain source words/transcripts; review it before sharing.", ""]
    lines += media_timeline_lines(rows, source_audio, dub_audio,
                                  {"status": "unavailable", "reason": "processing failed"},
                                  duration, getattr(trace, "video_plan", None), picture_packet_seconds(source))
    lines += runtime_lines(trace.snapshot())
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
