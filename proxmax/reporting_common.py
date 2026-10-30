"""Small shared formatting helpers for Markdown processing reports."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

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
