"""Sentence segmentation: group word timestamps into whole sentences for TTS.

শব্দ-টাইমস্ট্যাম্প থেকে পুরো বাক্যের ভাগ তৈরি করা হয়। Whisper-এর নিজের segment ভাগ প্রায়ই বাক্যের মাঝখানে শেষ হয় (৩০s
সীমার কারণে)। কিন্তু TTS-এ যদি অর্ধেক বাক্য আলাদা ক্লিপে যায়, তো
কণ্ঠ থেমে গিয়ে আবার শুরু হয় — "রোবট" ভাবের সবচেয়ে বড় কারণ।
তাই এখানে **শব্দ ও পাংচুয়েশন দেখে পূর্ণ বাক্যে** ভাগ করা হয়:

* বাক্য শেষ হয় ``. ! ? । …`` -এ,
* পাঁচ সেকেন্ড ছাড়ালে শব্দের সীমানায় ভাগ হয় (কমা/বিরতি আগে বেছে),
* বড় বিরতি (ডিফল্ট ০.৭s) থাকলে সেখানেও ভাগ হয়,
* খুব ছোট টুকরো প্রতিবেশীর সাথে জোড়া লাগানো হয়।

এর ফলে প্রতিটি segment = একটা সম্পূর্ণ ভাব → স্বাভাবিক শ্বাস-বিরতি,
ভালো অনুবাদ, ভালো lip-sync।
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

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


def _best_split_index(words: Sequence[Word], min_words: int = 3,
                      search_words: int = 9, min_pause: float = 0.16) -> Optional[int]:
    """লম্বা বাক্য ভাঙার সবচেয়ে ভালো জায়গা (index) বের করে।

    অগ্রাধিকার:
      1. পাংচুয়েশন (কমা/সেমিকোলন/ড্যাশ) — সবচেয়ে স্বাভাবিক থামা,
      2. সবচেয়ে বড় শব্দ-মধ্যবর্তী নীরবতা (``min_pause``-এর বেশি হলে),
      3. কিছুই না পেলে ``None`` (ডাকার দায়িত্বে শব্দ-সংখ্যায় ভাগ হবে)।

    Args:
        words: বর্তমান বাক্যের শব্দ।
        min_words: ভাগের দুই পাশে ন্যূনতম শব্দ।
        search_words: শেষ কতটি শব্দের মধ্যে জায়গা খোঁজা হবে।
        min_pause: নীরবতা হিসেবে ধরার ন্যূনতম ব্যবধান (সেকেন্ড)।

    Returns:
        ভাগ করার index (``current[:index]`` প্রথম অংশ) বা ``None``।
    """
    if len(words) < min_words * 2:
        return None
    lowest = max(min_words, len(words) - max(min_words, search_words))

    for back in range(len(words) - 1, lowest - 1, -1):
        if _ends_soft(words[back].word):
            return back + 1

    best_index, best_pause = None, min_pause
    for i in range(lowest, len(words) - min_words + 1):
        pause = float(words[i].start) - float(words[i - 1].end)
        if pause > best_pause:
            best_index, best_pause = i, pause
    return best_index


def _split_overlong_group(words: Sequence[Word], max_seconds: float) -> List[List[Word]]:
    """Enforce the duration cap at word boundaries, preferring pauses/punctuation."""
    remaining = list(words)
    pieces: List[List[Word]] = []
    while len(remaining) > 1 and remaining[-1].end - remaining[0].start > max_seconds:
        limit = float(remaining[0].start) + max_seconds
        possible = [
            index for index in range(1, len(remaining))
            if float(remaining[index - 1].end) <= limit
        ]
        if not possible:
            pieces.append(remaining[:1])
            remaining = remaining[1:]
            continue
        soft = [index for index in possible if _ends_soft(remaining[index - 1].word)]
        pauses = [
            (index, float(remaining[index].start) - float(remaining[index - 1].end))
            for index in possible
            if float(remaining[index].start) - float(remaining[index - 1].end) >= 0.16
        ]
        if soft:
            split_at = soft[-1]
        elif pauses:
            split_at = max(pauses, key=lambda item: (item[1], item[0]))[0]
        else:
            split_at = possible[-1]
        pieces.append(remaining[:split_at])
        remaining = remaining[split_at:]
    if remaining:
        pieces.append(remaining)
    return pieces


def build_sentences(
    words: Sequence[Word],
    max_words: int = 18,
    max_seconds: float = 5.0,
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

    max_seconds = max(0.1, float(max_seconds))
    max_words = max(1, int(max_words))
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
            # লম্বা হয়ে গেলে সবচেয়ে প্রাকৃতিক জায়গায় ভাগ করা হয়:
            #   ১) শেষ কমা/সেমিকোলন, ২) সবচেয়ে বড় নীরবতা (শ্বাস নেওয়ার জায়গা),
            #   ৩) কিছুই না পেলে শব্দ-সংখ্যায়।
            split_at = _best_split_index(current, min_words)
            if not split_at and len(current) > max_words:
                # কোনো কমা/বিরতিই নেই — তবু unit অসীম লম্বা হতে দেওয়া যাবে না
                split_at = max(min_words, min(max_words, len(current) - min_words))
            if split_at:
                groups.append(current[:split_at])
                current = current[split_at:]
            else:
                groups.append(current)
                current = []

    if current:
        groups.append(current)

    # The greedy sentence pass can meet its word-count limit on a long word;
    # enforce the hard duration bound at safe word boundaries before merging.
    groups = [
        piece for group in groups for piece in _split_overlong_group(group, max_seconds)
    ]

    # --- খুব ছোট টুকরো প্রতিবেশীর সাথে জোড়া লাগানো ------------------------
    merged: List[List[Word]] = []
    for group in groups:
        duration = group[-1].end - group[0].start
        is_tiny = len(group) < min_words or duration < min_seconds
        combined_words = len(merged[-1]) + len(group) if merged else 0
        combined_duration = (
            float(group[-1].end) - float(merged[-1][0].start) if merged else 0.0
        )
        can_merge = (
            merged
            and combined_words <= max_words + 4
            and combined_duration <= max_seconds
        )
        if can_merge and (is_tiny or len(merged[-1]) < min_words):
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
