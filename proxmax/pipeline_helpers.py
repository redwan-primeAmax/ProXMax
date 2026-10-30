"""Small filesystem and duration helpers shared by pipeline stages."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from .models import Segment
from .utils import get_media_duration

_TOTAL_STEPS = 6
_MIN_USABLE_OUTPUT_BYTES = 20 * 1024
_MIN_USABLE_OUTPUT_SECONDS = 0.5

def _clock(seconds: float) -> str:
    """সেকেন্ডকে ``MM:SS`` আকারে দেখায় (লগে সময় বোঝার জন্য)।"""
    total = max(0, int(round(float(seconds or 0.0))))
    return f"{total // 60:02d}:{total % 60:02d}"

def _safe_dirname(name: str) -> str:
    """Filesystem-safe folder name derived from a video filename."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return safe or "video"

def existing_output_is_usable(path: Path, min_bytes: int = _MIN_USABLE_OUTPUT_BYTES) -> bool:
    """আগের আউটপুট ফাইলটি সত্যিই ব্যবহারযোগ্য কি না (skip করার যোগ্য)।"""
    path = Path(path)
    try:
        if not path.is_file() or path.stat().st_size < min_bytes:
            return False
    except OSError:
        return False
    try:
        return get_media_duration(path) >= _MIN_USABLE_OUTPUT_SECONDS
    except Exception:
        return False

def _report_is_usable(path: Path) -> bool:
    """Do not skip an old output whose report records missing speech."""
    try:
        if not path.is_file() or path.stat().st_size < 256:
            return False
        with path.open(encoding="utf-8", errors="replace") as report:
            summary = report.read(16384)
        unplaced = re.search(r"\*\*unplaced clips:\*\*\s*(\d+)", summary, re.IGNORECASE)
        if unplaced and int(unplaced.group(1)):
            return False
        counts = re.search(
            r"\*\*Transcript units:\*\*\s*(\d+).*?\*\*translated units:\*\*\s*(\d+)"
            r".*?\*\*units with TTS clips:\*\*\s*(\d+).*?\*\*placed units:\*\*\s*(\d+)",
            summary, re.IGNORECASE | re.DOTALL,
        )
        return not counts or len(set(map(int, counts.groups()))) == 1
    except OSError:
        return False

def _usable(segments: List[Segment]) -> List[Segment]:
    """Segments that carry speech (STT text or translated text)."""
    return [s for s in segments if (s.text_en or "").strip() or (s.text_bn or "").strip()]
