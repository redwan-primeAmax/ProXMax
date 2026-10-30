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
    if abs(tempo - 1.0) < 0.02:
        return f"aresample={sample_rate}"
    if prefer_rubberband and "rubberband" in available_filters():
        return f"rubberband=tempo={tempo:.4f},aresample={sample_rate}"
    return f"{build_atempo_filter(tempo)},aresample={sample_rate}"

def compute_tempo(
    tts_duration: float,
    target_duration: float,
    min_tempo: float = 0.95,
    max_tempo: float = 1.25,
    slow_down_short_clips: bool = False,
    fill_extension: float = 0.0,
    fill_gap_seconds: float = 0.05,
) -> float:
    """Tempo ratio to fit ``tts_duration`` into ``target_duration``.

    tempo > 1 মানে দ্রুত করা (বাংলা অডিও লম্বা), tempo < 1 মানে ধীর করা।

    ``fill_extension`` > 0 হলে ক্লিপ স্লটের চেয়ে ছোট থাকলে সেটি **ধীর করে
    স্লট ভরা হয়** (সর্বোচ্চ ``fill_extension`` গুণ, যেমন ০.১২ = ১২% ধীর),
    যাতে বাক্যের পর অযথা নীরবতা না থাকে। খুব ছোট ক্লিপকে অস্বাভাবিক ধীর
    করা হয় না — ওই সীমার বাইরে গেলে স্বাভাবিক গতিতেই থাকে।
    """
    if tts_duration <= 0 or target_duration <= 0:
        return 1.0
    tempo = tts_duration / target_duration
    if tempo < 1.0:
        if slow_down_short_clips:
            return min(max(tempo, min_tempo), max_tempo)
        if fill_extension <= 0:
            return 1.0
        longest = min(
            max(target_duration - fill_gap_seconds, 0.05),
            tts_duration * (1.0 + float(fill_extension)),
        )
        return min(1.0, max(tts_duration / max(longest, 0.05), 1.0 / (1.0 + float(fill_extension))))
    return min(max(tempo, min_tempo), max_tempo)

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
        # TTS length must not turn a multi-second source pause into extra speech time.
        # If it overruns this bounded source window, shorten/tempo-fit it instead.
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
