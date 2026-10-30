"""Pure seam-boundary helpers used by the Whisper transcription pipeline."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .chunking import SpeechChunk
from .models import Word

def find_suspicious_boundaries(
    words: Sequence[Word], chunks: Sequence[SpeechChunk], tolerance: float = 0.12
) -> List[float]:
    """Chunk সীমানাগুলোর মধ্যে যেগুলোতে শব্দ কাটা পড়ার আশঙ্কা আছে।"""
    boundaries: List[float] = []
    for chunk in chunks[1:]:
        b = chunk.start
        straddle = any(w.start < b - 0.02 and w.end > b + 0.02 for w in words)
        left = [w for w in words if w.end <= b + 0.02]
        right = [w for w in words if w.start >= b - 0.02]
        near_left = bool(left) and (b - left[-1].end) <= tolerance
        near_right = bool(right) and (right[0].start - b) <= tolerance
        if straddle or near_left or near_right:
            boundaries.append(b)
    return boundaries

def merge_windows(
    boundaries: Sequence[float], half_width: float, max_seconds: float
) -> List[Tuple[float, float]]:
    """সীমানাগুলোকে একে অন্যের সাথে মিলিয়ে (ওভারল্যাপ-মুক্ত) উইন্ডো বানায়।"""
    windows: List[Tuple[float, float]] = []
    for b in sorted(boundaries):
        start = max(0.0, b - half_width)
        end = max(0.0, b + half_width)
        if end - start > max_seconds:  # নিরাপত্তা: এক উইন্ডো খুব বড় নয়
            end = start + max_seconds
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append((start, end))
    return windows

def replace_in_window(
    words: Sequence[Word], replacement: Sequence[Word], start: float, end: float
) -> List[Word]:
    """উইন্ডোর ভেতরের শব্দ বদলে দেয় (বাইরের শব্দ অপরিবর্তিত থাকে)।"""
    kept = [
        w for w in words
        if not (start <= (w.start + w.end) / 2.0 <= end)
    ]
    inside = [
        w for w in replacement
        if start - 0.35 <= (w.start + w.end) / 2.0 <= end + 0.35
    ]
    combined = kept + inside
    combined.sort(key=lambda w: (w.start, w.end))
    return combined

def _tail_context(text: str, limit: int) -> str:
    """Prompt carry-এর জন্য শেষ কয়েক অক্ষর (সম্ভব হলে বাক্যের শুরু থেকে)।"""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    tail = text[-limit:]
    for mark in (". ", "! ", "? ", "। "):
        idx = tail.find(mark)
        if 0 <= idx < len(tail) // 2:
            return tail[idx + len(mark):]
    return tail[tail.find(" ") + 1:] if " " in tail else tail
