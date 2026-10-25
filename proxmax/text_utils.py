"""Bangla text helpers: spoken numbers, punctuation cleanup, duration estimates.

সংখ্যা → কথ্য রূপ, পাংচুয়েশন ও দৈর্ঘ্যের হিসাব। TTS (IndicF5) বাংলা **অক্ষর** পড়ে; সংখ্যা ঠিকভাবে উচ্চারণ করে না
("৭" → সাত, নয়তো একদম চুপ)। তাই সিন্থেসিসের আগে সংখ্যাগুলো বাংলা
কথায় বদলে দেওয়া হয়। একই ফাইলে পাংচুয়েশন পরিষ্কার করার ও সময়
আনুমানিক হিসাব করার ফাংশনও আছে — অনুবাদ ও TTS দুটোই এগুলো শেয়ার করে।
"""

from __future__ import annotations

import re
from typing import List

#: ০–৯৯ পর্যন্ত বাংলা সংখ্যাশব্দ (বাংলাদেশে প্রচলিত রূপ)
_ONES: List[str] = [
    "শূন্য", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়",
    "দশ", "এগারো", "বারো", "তেরো", "চৌদ্দ", "পনেরো", "ষোলো", "সতেরো", "আঠারো", "উনিশ",
    "বিশ", "একুশ", "বাইশ", "তেইশ", "চব্বিশ", "পঁচিশ", "ছাব্বিশ", "সাতাশ", "আঠাশ", "ঊনত্রিশ",
    "ত্রিশ", "একত্রিশ", "বত্রিশ", "তেত্রিশ", "চৌত্রিশ", "পঁয়ত্রিশ", "ছত্রিশ", "সাঁইত্রিশ", "আটত্রিশ", "ঊনচল্লিশ",
    "চল্লিশ", "একচল্লিশ", "বিয়াল্লিশ", "তেতাল্লিশ", "চুয়াল্লিশ", "পঁয়তাল্লিশ", "ছেচল্লিশ", "সাতচল্লিশ", "আটচল্লিশ", "ঊনপঞ্চাশ",
    "পঞ্চাশ", "একান্ন", "বায়ান্ন", "তিপ্পান্ন", "চুয়ান্ন", "পঞ্চান্ন", "ছাপ্পান্ন", "সাতান্ন", "আটান্ন", "ঊনষাট",
    "ষাট", "একষট্টি", "বাষট্টি", "তেষট্টি", "চৌষট্টি", "পঁয়ষট্টি", "ছেষট্টি", "সাতষট্টি", "আটষট্টি", "ঊনসত্তর",
    "সত্তর", "একাত্তর", "বাহাত্তর", "তিয়াত্তর", "চুয়াত্তর", "পঁচাত্তর", "ছিয়াত্তর", "সাতাত্তর", "আটাত্তর", "ঊনআশি",
    "আশি", "একাশি", "বিরাশি", "তিরাশি", "চুরাশি", "পঁচাশি", "ছিয়াশি", "সাতাশি", "আটাশি", "ঊননব্বই",
    "নব্বই", "একানব্বই", "বিরানব্বই", "তিরানব্বই", "চুরানব্বই", "পঁচানব্বই", "ছিয়ানব্বই", "সাতানব্বই", "আটানব্বই", "নিরানব্বই",
]

_BANGLA_DIGIT_MAP = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
#: বাংলা অঙ্ক → ASCII (সংখ্যা কথায় বদলানোর আগে দরকার)
_ASCII_DIGIT_MAP = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

_SCALES = ((10_000_000, "কোটি"), (100_000, "লক্ষ"), (1_000, "হাজার"))

#: সংখ্যা খুঁজে বের করার প্যাটার্ন — হাজার/লক্ষ separator সহ
_NUMBER_RE = re.compile(r"\d{1,3}(?:[,]\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?।…)\]])")
_REPEAT_PUNCT_RE = re.compile(r"([!?।,;:])\1{1,}")
_MARKDOWN_RE = re.compile(r"[*_`#>~]+")

#: Bangla স্বাভাবিক কথার গতি (অক্ষর/সেকেন্ড) — সময় অনুমানের জন্য
DEFAULT_CHARS_PER_SECOND = 14.0


# =============================================================================
# Numbers
# =============================================================================
def number_to_bangla_words(value: int) -> str:
    """Integer → বাংলা কথ্য রূপ (``1234`` → ``এক হাজার দুইশ চৌত্রিশ``)."""
    n = int(value)
    if n < 0:
        return "মাইনাস " + number_to_bangla_words(-n)
    if n < 100:
        return _ONES[n]
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        head = "একশ" if hundreds == 1 else f"{_ONES[hundreds]}শ"
        return f"{head} {number_to_bangla_words(rest)}" if rest else head
    for scale, name in _SCALES:
        if n >= scale:
            quotient, rest = divmod(n, scale)
            head = f"{number_to_bangla_words(quotient)} {name}"
            return f"{head} {number_to_bangla_words(rest)}" if rest else head
    return str(n)  # pragma: no cover - অসম্ভব


def _decimal_to_words(text: str) -> str:
    whole, _, fraction = text.partition(".")
    words = number_to_bangla_words(int(whole))
    if fraction:
        digits = " ".join(_ONES[int(d)] for d in fraction)
        words = f"{words} দশমিক {digits}"
    return words


def _replace_number(match: re.Match) -> str:
    raw = match.group(0)
    # match-এ "%", "টাকা", "tk" থাকতে পারে (যেমন "50%") — শুধু সংখ্যাটুকু রাখা হয়
    cleaned = re.sub(r"[^0-9.,]", "", raw.translate(_ASCII_DIGIT_MAP)).replace(",", "")
    if "." in cleaned:
        return _decimal_to_words(cleaned)
    try:
        return number_to_bangla_words(int(cleaned))
    except ValueError:  # pragma: no cover - regex অনুযায়ী অসম্ভব
        return raw


def verbalize_numbers(text: str) -> str:
    """সব সংখ্যা বাংলা কথায় বদলায় (TTS-এর জন্য)।

    শতকরা চিহ্ন ও সাধারণ একক-সংক্ষেপও প্রসারিত হয়:
    ``50%`` → ``পঞ্চাশ শতাংশ``।
    """
    if not text:
        return ""
    # বাংলা অঙ্ক (১২৫) খুঁজে পাওয়ার জন্য আগে ASCII-তে আনা হয়, নাহলে TTS
    # অঙ্কগুলোকে অক্ষর হিসেবে পড়ে ("এক-দুই-পাঁচ") বা একদম চুপ থাকে।
    result = text.translate(_ASCII_DIGIT_MAP)
    # শতকরা: সংখ্যার সাথে % জোড়া লাগানো থাকলে আগে আলাদা করা হয়
    result = re.sub(
        r"(\d[\d,]*(?:\.\d+)?)\s*%",
        lambda m: _replace_number(m) + " শতাংশ",
        result,
    )
    result = re.sub(
        r"(\d[\d,]*(?:\.\d+)?)\s*(?:টাকা|tk|Tk|TK)\b",
        lambda m: _replace_number(m) + " টাকা",
        result,
    )
    result = _NUMBER_RE.sub(_replace_number, result)
    return result


def to_bangla_digits(text: str) -> str:
    """ASCII অঙ্ক → বাংলা অঙ্ক (SRT/প্রদর্শনের জন্য)।"""
    return (text or "").translate(_BANGLA_DIGIT_MAP)


# =============================================================================
# Text cleanup
# =============================================================================
def normalise_bangla(text: str) -> str:
    """বাংলা টেক্সট পরিষ্কার করে — TTS ও SRT দুই জায়গাতেই নিরাপদ।

    * markdown/emoji বাদ,
    * একাধিক স্পেস ও পুনরাবৃত্ত পাংচুয়েশন ঠিক,
    * পাংচুয়েশনের আগে স্পেস সরানো, পরে একটি স্পেস নিশ্চিত করা।
    """
    if not text:
        return ""
    cleaned = _MARKDOWN_RE.sub("", str(text))
    cleaned = _EMOJI_RE.sub("", cleaned)
    cleaned = cleaned.replace("\r", " ").replace("\n", " ")
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = _REPEAT_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = re.sub(r"\.\.\.+", "…", cleaned)
    cleaned = re.sub(r"([।!?…])(?=\S)", r"\1 ", cleaned)
    return cleaned.strip()


def prepare_for_speech(text: str) -> str:
    """TTS-এ পাঠানোর চূড়ান্ত রূপ: পরিষ্কার + সংখ্যা কথায়।"""
    return normalise_bangla(verbalize_numbers(normalise_bangla(text)))


def estimate_speech_seconds(
    text: str, chars_per_second: float = DEFAULT_CHARS_PER_SECOND
) -> float:
    """বাংলা টেক্সট স্বাভাবিক গতিতে বলতে আনুমানিক কত সেকেন্ড লাগবে।"""
    cleaned = (text or "").strip()
    if not cleaned:
        return 0.0
    # সংখ্যা কথায় বদলে দৈর্ঘ্য মাপা হয়: "১২৫" লেখা হয় ৩ অক্ষরে, কিন্তু
    # বলা হয় "একশ পঁচিশ" (১০ অক্ষর) — নাহলে সময়ের হিসাব কমে যায়, ক্লিপ
    # বড় হয়ে sync-এ দ্রুত করতে হয় (কণ্ঠ তাড়াহুড়ো শোনায়)।
    spoken = verbalize_numbers(cleaned)
    rate = max(6.0, float(chars_per_second))
    return max(0.4, len(spoken) / rate)


def split_sentences(text: str) -> List[str]:
    """বাংলা/ইংরেজি বাক্য-সীমা ধরে টেক্সট ভাগ করে (খালি টুকরো বাদ)।"""
    parts = re.split(r"(?<=[.!?।…])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]
