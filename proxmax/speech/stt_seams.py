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
    """Replace only the same window, rejecting empty or clearly partial redecodes."""
    def in_window(word: Word) -> bool:
        return start <= (word.start + word.end) / 2.0 <= end

    original = [word for word in words if in_window(word)]
    inside = [word for word in replacement if in_window(word)]
    if not inside:
        return list(words)
    if original:
        original_start = min(word.start for word in original)
        original_end = max(word.end for word in original)
        replacement_start = min(word.start for word in inside)
        replacement_end = max(word.end for word in inside)
        # A partial decode must not delete valid material at either edge. Small
        # timestamp differences are expected when Whisper decodes a new window.
        if replacement_start > original_start + 0.35 or replacement_end < original_end - 0.35:
            return list(words)
        if len(original) >= 3 and len(inside) < len(original) * 0.75:
            return list(words)
    kept = [word for word in words if not in_window(word)]
    return sorted(kept + inside, key=lambda word: (word.start, word.end))

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
