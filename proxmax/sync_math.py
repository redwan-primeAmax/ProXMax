"""Pure timing and FFmpeg-filter helpers used by the sync engine."""

from __future__ import annotations

import functools
import subprocess
from typing import List, Sequence, Tuple

from .models import Segment
from .utils import ensure_tool, get_logger

logger = get_logger(__name__)

def build_atempo_filter(tempo: float) -> str:
    """Build an ffmpeg ``atempo`` filter chain for any tempo ratio.

    একক ``atempo``-এর কার্যকর পরিসর 0.5–2.0 ধরে নিয়ে বড়/ছোট
    অনুপাতকে একাধিক ধাপে ভাঙা হয় (যেমন 2.5 → ``atempo=2.0,atempo=1.25``),
    যাতে সব ffmpeg সংস্করণেই কাজ করে।
    """
    if tempo <= 0:
        raise ValueError(f"tempo অবশ্যই ধনাত্মক হতে হবে, পাওয়া গেছে: {tempo}")
    parts: List[float] = []
    remaining = float(tempo)
    while remaining > 2.0:
        parts.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        parts.append(0.5)
        remaining /= 0.5
    parts.append(remaining)
    return ",".join(f"atempo={p:.4f}" for p in parts)

@functools.lru_cache(maxsize=1)
def available_filters() -> frozenset:
    """ffmpeg-এ উপলব্ধ filter-এর তালিকা (একবারই যাচাই হয়)।"""
    try:
        ffmpeg = ensure_tool("ffmpeg")
    except Exception:  # pragma: no cover - ffmpeg নেই
        return frozenset()
    proc = subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-filters"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    names = set()
    for line in (proc.stdout or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and not line.lstrip().startswith(("Filters:", "|")):
            names.add(parts[1])
    return frozenset(names)

def build_stretch_filter(tempo: float, sample_rate: int, prefer_rubberband: bool = True) -> str:
    """Time-stretch filter chain — ``rubberband`` থাকলে সেটাই, নইলে ``atempo``।

    ``rubberband`` pitch ও formant দুটোই ধরে রাখে, তাই কণ্ঠ অনেক কম
    "ধাতব" শোনায়। না থাকলে সর্বজনীন ``atempo`` ব্যবহৃত হয়।
    """
    if abs(tempo - 1.0) < 1e-6:
        return f"aresample={sample_rate}"
    # This general filter helper supports arbitrary rates; the dubbing sync
    # caller restricts speech to 0.90–1.00x and validates actual output length.
    if prefer_rubberband and tempo <= 100.0 and "rubberband" in available_filters():
        return f"rubberband=tempo={tempo:.4f},aresample={sample_rate}"
    return f"{build_atempo_filter(tempo)},aresample={sample_rate}"

def compute_tempo(
    tts_duration: float,
    target_duration: float,
    min_tempo: float = 0.90,
    max_tempo: float = 1.0,
    slow_down_short_clips: bool = True,
    fill_extension: float = 0.0,
    fill_gap_seconds: float = 0.05,
) -> float:
    """Slow short speech only as much as needed, never below 0.90x.

    Long speech is never sped up. The legacy max/disable/fill arguments are
    accepted for compatibility but cannot override the no-speedup policy.
    """
    minimum = min(1.0, max(0.90, float(min_tempo)))
    if tts_duration <= 0 or target_duration <= 0:
        return 1.0
    return max(minimum, min(1.0, tts_duration / target_duration))

def plan_windows(
    segments: Sequence[Segment],
    total_duration: float,
    min_gap_seconds: float = 0.08,
    borrow_seconds: float = 0.25,
) -> List[Tuple[float, float]]:
    """প্রতিটি segment-এর জন্য ওভারল্যাপ-মুক্ত সময় উইন্ডো।

    নিয়ম:
      * উইন্ডো কখনো পরের segment শুরুর আগে শেষ হয় (``min_gap`` বাদে),
      * উইন্ডোর মূল দৈর্ঘ্য source-এর মাপা speech interval (``seg.start``–``seg.end``),
      * ``borrow_seconds``-এর বেশি নীরবতা ধার নেওয়া হয় না — পরের বক্তব্যের আগে
        ২–৫ সেকেন্ড ফাঁকা থাকলেও সেটি ভুল করে dub slot হিসেবে ব্যবহৃত হবে না,
      * শুরু কখনো পিছিয়ে যায় না (lip-sync নষ্ট হয়)।

    Returns:
        প্রতিটি segment-এর ``(start, end)`` সেকেন্ড।
    """
    ordered = sorted(segments, key=lambda s: s.start)
    windows: List[Tuple[float, float]] = []
    for index, seg in enumerate(ordered):
        start = max(0.0, float(seg.start))
        if index + 1 < len(ordered):
            hard_end = min(float(ordered[index + 1].start) - min_gap_seconds, total_duration)
        else:
            hard_end = float(total_duration)
        source_end = max(start + 0.15, float(seg.end))
        borrow = max(0.0, float(borrow_seconds))
        # Do not absorb a multi-second source pause. An overlong dub is handled
        # by repeating picture frames in this bounded window, not speeding speech.
        end = min(hard_end, source_end + borrow)
        end = max(end, min(hard_end, start + 0.15))
        windows.append((start, end))
    return windows

def find_overlaps(spans: Sequence[Tuple[float, float]]) -> List[Tuple[int, float]]:
    """Return ``(index, overlap_seconds)`` for any overlapping neighbours."""
    problems: List[Tuple[int, float]] = []
    for i in range(len(spans) - 1):
        overlap = spans[i][1] - spans[i + 1][0]
        if overlap > 1e-4:
            problems.append((i, overlap))
    return problems
