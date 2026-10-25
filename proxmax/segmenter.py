"""Sentence segmentation: group word timestamps into whole sentences for TTS.

শব্দ-টাইমস্ট্যাম্প থেকে পুরো বাক্যের ভাগ তৈরি করা হয়। Whisper-এর নিজের segment ভাগ প্রায়ই বাক্যের মাঝখানে শেষ হয় (৩০s
সীমার কারণে)। কিন্তু TTS-এ যদি অর্ধেক বাক্য আলাদা ক্লিপে যায়, তো
কণ্ঠ থেমে গিয়ে আবার শুরু হয় — "রোবট" ভাবের সবচেয়ে বড় কারণ।
তাই এখানে **শব্দ ও পাংচুয়েশন দেখে পূর্ণ বাক্যে** ভাগ করা হয়:

* বাক্য শেষ হয় ``. ! ? । …`` -এ,
* খুব লম্বা বাক্য কমা/সেমিকোলনে ভাগ হয়,
* বড় বিরতি (ডিফল্ট ০.৭s) থাকলে সেখানেও ভাগ হয়,
* খুব ছোট টুকরো প্রতিবেশীর সাথে জোড়া লাগানো হয়।

এর ফলে প্রতিটি segment = একটা সম্পূর্ণ ভাব → স্বাভাবিক শ্বাস-বিরতি,
ভালো অনুবাদ, ভালো lip-sync।
"""

from __future__ import annotations

import re
from typing import List, Sequence

from .models import Segment, Word
from .utils import get_logger

logger = get_logger(__name__)

SENTENCE_END = ".!?।…"
SOFT_BREAK = ",;:—–"

_SPACE_RE = re.compile(r"[ \t]+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?।…)\]])")


def join_words(words: Sequence[Word]) -> str:
    """Word list → পরিষ্কার এক লাইন টেক্সট (স্পেস ও পাংচুয়েশন ঠিক করে)।"""
    text = "".join(w.word for w in words if w.word)
    text = _SPACE_RE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return text.strip()


def _ends_sentence(token: str) -> bool:
    stripped = token.strip().rstrip("\"')”’»")
    return bool(stripped) and stripped[-1] in SENTENCE_END


def _ends_soft(token: str) -> bool:
    stripped = token.strip().rstrip("\"')”’»")
    return bool(stripped) and stripped[-1] in SOFT_BREAK


def build_sentences(
    words: Sequence[Word],
    max_words: int = 15,
    max_seconds: float = 10.0,
    pause_gap: float = 0.7,
    min_words: int = 3,
    min_seconds: float = 0.9,
) -> List[Segment]:
    """Group word-timestamps into full-sentence :class:`Segment` objects.

    Args:
        words: STT থেকে পাওয়া শব্দ (absolute timestamp, যেকোনো ক্রমে)।
        max_words: এক segment-এ সর্বোচ্চ শব্দ।
        max_seconds: এক segment-এর সর্বোচ্চ দৈর্ঘ্য।
        pause_gap: এর চেয়ে বড় নীরবতা পেলে বাক্য ভাগ হয়।
        min_words: এর কম শব্দের টুকরো প্রতিবেশীর সাথে মেশানো হয়।
        min_seconds: এর কম দৈর্ঘ্যের টুকরোও মেশানো হয়।

    Returns:
        index/start/end/text_en/words পূর্ণ :class:`Segment`-এর তালিকা।
    """
    collected = [w for w in words if (w.word or "").strip() and w.end > w.start]
    collected.sort(key=lambda w: (w.start, w.end))
    if not collected:
        return []

    groups: List[List[Word]] = []
    current: List[Word] = []

    for i, word in enumerate(collected):
        current.append(word)
        nxt = collected[i + 1] if i + 1 < len(collected) else None
        gap = (nxt.start - word.end) if nxt else 999.0
        elapsed = word.end - current[0].start

        hard_end = _ends_sentence(word.word)
        too_long = len(current) >= max_words or elapsed >= max_seconds
        long_pause = gap >= pause_gap and len(current) >= min_words

        if hard_end or (nxt is None) or long_pause:
            groups.append(current)
            current = []
            continue

        if too_long:
            # লম্বা হয়ে গেলে শেষ কমা/সেমিকোলনে ভাঙার চেষ্টা — স্বাভাবিক শ্বাসের জায়গা
            split_at = None
            for back in range(len(current) - 1, max(0, len(current) - 6), -1):
                if _ends_soft(current[back].word):
                    split_at = back + 1
                    break
            if split_at and split_at >= min_words:
                groups.append(current[:split_at])
                current = current[split_at:]
            else:
                groups.append(current)
                current = []

    if current:
        groups.append(current)

    # --- খুব ছোট টুকরো প্রতিবেশীর সাথে জোড়া লাগানো ------------------------
    merged: List[List[Word]] = []
    for group in groups:
        duration = group[-1].end - group[0].start
        is_tiny = len(group) < min_words or duration < min_seconds
        if merged and is_tiny and len(merged[-1]) + len(group) <= max_words + 4:
            merged[-1] = merged[-1] + group
        elif merged and (len(merged[-1]) < min_words) and len(merged[-1]) + len(group) <= max_words + 4:
            merged[-1] = merged[-1] + group
        else:
            merged.append(group)

    segments: List[Segment] = []
    for index, group in enumerate(merged):
        start = float(group[0].start)
        end = float(max(w.end for w in group))
        if end <= start:
            continue
        segments.append(
            Segment(
                index=len(segments),
                start=start,
                end=end,
                text_en=join_words(group),
                words=list(group),
            )
        )
    logger.debug(
        "Segmenter: %d শব্দ → %d বাক্য (গড় %.1f শব্দ)",
        len(collected), len(segments),
        (len(collected) / len(segments)) if segments else 0.0,
    )
    return segments
