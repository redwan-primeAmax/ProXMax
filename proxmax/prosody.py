"""Speech units: one clip per whole sentence — natural prosody, no mid-phrase stops.

কেন এই মডিউল
-------------
TTS যদি **প্রতি segment-এ** আলাদা ক্লিপ বানায়, তাহলে প্রতিটি ক্লিপের
শুরু/শেষে কণ্ঠ থেমে আবার শুরু হয়। ছোট ছোট segment (১–৩ সেকেন্ড) থাকলে
শোনা যায় "আমি একটি" … থেমে … "স্টার পাস দিবো" — অডিও শুনতে খারাপ লাগে,
যদিও প্রতিটি শব্দ ঠিকই আছে।

STT-এর segment সবসময় পুরো বাক্য নয় (Whisper ৩০s সীমা, ১৫ শব্দের কাট,
বিরতি ইত্যাদির কারণে)। এখানে পাশাপাশি থাকা segment-গুলোকে **একটা
swings-এ (speech unit)** জোড়া লাগানো হয়:

* দুই segment-এর মাঝে বিরতি ছোট (ডিফল্ট ≤০.৬s) এবং
* আগের segment বাক্যের শেষে পৌঁছায়নি (``.!?।…`` নেই) **অথবা** সেটি খুব
  ছোট (≤৩s, তখন দুই ছোট বাক্য একসাথে পড়লে স্বাভাবিক শোনায়), এবং
* জোড়া লাগালে unit-টি সর্বোচ্চ ১২s / ২৪০ অক্ষরের মধ্যে থাকে।

ফলে অনুবাদ ও TTS **পুরো বাক্য** ধরে হয়: বাংলা অনুবাদ বেশি স্বাভাবিক হয়,
কণ্ঠ একসাথে পুরো বাক্য বলে, আর sync-এ unit-এর নিজস্ব উইন্ডোতে (ভেতরের
ছোট বিরতিসহ) বসে — কোনো জোর করে ফাঁক বসাতে হয় না।
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

from .models import Segment
from .text_utils import split_sentences
from .utils import get_logger

logger = get_logger(__name__)

#: বাক্যের শেষ চিহ্ন (এগুলো থাকলে নতুন unit শুরু করা যায়)
_SENTENCE_END = ".!?।…"
_TERMINAL_RE = re.compile(r"[.!?।…][\"')\]}»”’]*\s*$")
#: unit-এর ডিফল্ট সীমা — এর বেশি হলে বাক্য এসে গেলেও ভাগ করতে হয়
DEFAULT_MAX_SECONDS = 12.0
DEFAULT_MAX_CHARS = 240
DEFAULT_MAX_GAP = 0.6
DEFAULT_GLUE_SHORT_SECONDS = 3.0


def ends_sentence(text: str) -> bool:
    """টেক্সট বাক্যের শেষ চিহ্নে শেষ হয় কি না (``।``/``.``/``?``/``!``/``…``)।"""
    return bool(_TERMINAL_RE.search((text or "").strip()))


def _join_text(left: str, right: str) -> str:
    """দুই টুকরো টেক্সট জোড়া লাগায় (বাড়তি স্পেস ছাড়া)।"""
    left, right = (left or "").strip(), (right or "").strip()
    if not left:
        return right
    if not right:
        return left
    return f"{left} {right}"


def build_units(
    segments: Sequence[Segment],
    max_seconds: float = DEFAULT_MAX_SECONDS,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_gap: float = DEFAULT_MAX_GAP,
    glue_short_seconds: float = DEFAULT_GLUE_SHORT_SECONDS,
    text_attr: str = "text_bn",
) -> List[Segment]:
    """পাশাপাশি segment-গুলোকে বাক্য-ভিত্তিক speech unit-এ জোড়া লাগায়।

    Args:
        segments: STT/অনুবাদ করা segment (start অনুযায়ী সাজানো হবে)।
        max_seconds: এক unit-এর সর্বোচ্চ দৈর্ঘ্য (সেকেন্ড)।
        max_chars: এক unit-এর সর্বোচ্চ অক্ষর (TTS ও sync-এর জন্য নিরাপদ)।
        max_gap: এর বেশি বিরতি থাকলে unit ভাগ হয় (স্বাভাবিক থামা ধরে রাখে)।
        glue_short_seconds: এর চেয়ে ছোট segment (বাক্য শেষ হলেও) পরেরটার
            সাথে জোড়া লাগানো যায় — এক-শব্দের বাক্য আলাদা ক্লিপ হলে
            আবার "মেশিনগান" শোনায়।
        text_attr: কোন টেক্সট দিয়ে unit-এর টেক্সট বানানো হবে
            (``"text_bn"`` বা ``"text_en"``)।

    Returns:
        ``is_unit=True`` সেট করা :class:`Segment`-এর তালিকা; প্রতিটির
        ``children``-এ মূল segment-গুলো থাকে। ইনপুট খালি হলে খালি তালিকা।
    """
    ordered = [s for s in sorted(segments, key=lambda s: s.start) if (s.end - s.start) > 0.02]
    if not ordered:
        return []

    max_seconds = max(2.0, float(max_seconds))
    max_chars = max(40, int(max_chars))
    max_gap = max(0.0, float(max_gap))
    glue_short_seconds = max(0.0, float(glue_short_seconds))

    units: List[Segment] = []
    current: List[Segment] = []

    def flush() -> None:
        if not current:
            return
        first, last = current[0], current[-1]
        text = ""
        for child in current:
            text = _join_text(text, getattr(child, text_attr, "") or "")
        units.append(
            Segment(
                index=len(units),
                start=float(first.start),
                end=float(last.end),
                text_en=_join_text(
                    "", " ".join((c.text_en or "").strip() for c in current).strip()
                ),
                # text_attr="text_en" হলে unit-এর বাংলা এখনো নেই — ইংরেজি টেক্সট
                # ভুল করে text_bn-এ বসালে অনুবাদ ব্যর্থ হলে TTS ইংরেজি পড়ে ফেলবে
                text_bn=text if text_attr == "text_bn" else "",
                words=[w for child in current for w in child.words],
                is_unit=True,
                children=list(current),
            )
        )
        current.clear()

    for segment in ordered:
        if not current:
            current.append(segment)
            continue

        previous = current[-1]
        gap = float(segment.start) - float(previous.end)
        merged_seconds = float(segment.end) - float(current[0].start)
        merged_text = " ".join(
            (getattr(c, text_attr, "") or "") for c in list(current) + [segment]
        )
        continuing = not ends_sentence(getattr(previous, text_attr, "") or previous.text_en)
        short_previous = float(previous.end - previous.start) <= glue_short_seconds
        fits = merged_seconds <= max_seconds and len(merged_text) <= max_chars

        if gap <= max_gap and fits and (continuing or short_previous):
            current.append(segment)
            continue

        flush()
        current.append(segment)

    flush()
    return units


def retime_children(unit: Segment, start: Optional[float] = None, duration: Optional[float] = None) -> None:
    """unit-এর ভেতরের segment-গুলোর সময় ভাগ করে দেয় (সাবটাইটেলের জন্য)।

    unit-এর অডিও একটি একটানা ক্লিপ, তাই ভেতরের প্রতিটি বাক্য/লাইনের
    জন্য নির্দিষ্ট সময় জানা যায় না। এখানে মূল (English) দৈর্ঘ্যের
    অনুপাতে সময় ভাগ করা হয় — SRT প্রায় সবসময় ঠিক জায়গায় থাকে।

    Args:
        unit: ``build_units``-এ বানানো unit।
        start: unit-এর অডিও কোথায় শুরু (ডিফল্ট ``unit.start``)।
        duration: unit-এর অডিও কত সেকেন্ড (ডিফল্ট ``unit.speech_seconds``)।
    """
    children = list(getattr(unit, "children", []) or [])
    if not children:
        return
    start = float(unit.start if start is None else start)
    duration = float(unit.speech_seconds if duration is None else duration)
    duration = max(0.2, duration)

    weights = [max(0.05, float(child.end - child.start)) for child in children]
    total = sum(weights) or 1.0
    cursor = start
    for child, weight in zip(children, weights):
        share = duration * (weight / total)
        child.start = round(cursor, 3)
        child.end = round(cursor + share, 3)
        cursor += share
    # শেষ segment unit-এর শেষ পর্যন্ত ঠিক করা হয় (rounding ত্রুটি এড়াতে)
    children[-1].end = round(start + duration, 3)


def subtitles_for_unit(
    unit: Segment, max_chars: int = 60, min_seconds: float = 0.9
) -> List[Segment]:
    """unit-এর বাংলা টেক্সটকে পড়ার উপযোগী সাবটাইটেল লাইনে ভাগ করে।

    এক unit (পুরো বাক্য) যদি লম্বা হয়, বাংলা বাক্য-সীমা বা কমায় ভাগ করা
    হয়, আর সময় character-সংখ্যার অনুপাতে বণ্টন করা হয়।

    Args:
        unit: speech unit (``text_bn`` পূর্ণ)।
        max_chars: এক সাবটাইটেল লাইনের সর্বোচ্চ অক্ষর।
        min_seconds: লাইন খুব ছোট হয়ে গেলে প্রতিবেশীর সাথে মেশানো হয়।

    Returns:
        নতুন :class:`Segment`-এর তালিকা (``text_bn`` সহ, ``start``/``end``
        unit-এর ভেতরে ভাগ করা)।
    """
    text = (unit.text_bn or "").strip()
    if not text:
        return []
    pieces = [p for p in split_sentences(text) if p]
    if not pieces:
        pieces = [text]

    # বড় টুকরো কমায় ভাগ করা হয় (লম্বা এক লাইন পড়তে কষ্ট হয়)
    refined: List[str] = []
    for piece in pieces:
        if len(piece) <= max_chars:
            refined.append(piece)
            continue
        buffer = ""
        for part in re.split(r"(?<=[,;:])\s+", piece):
            candidate = _join_text(buffer, part)
            if buffer and len(candidate) > max_chars:
                refined.append(buffer)
                buffer = part
            else:
                buffer = candidate
        if buffer:
            refined.append(buffer)

    if len(refined) == 1:
        line = Segment(index=unit.index, start=unit.start, end=unit.end, text_bn=text)
        return [line]

    duration = float(unit.speech_seconds)
    weights = [max(6, len(p)) for p in refined]
    total = float(sum(weights)) or 1.0
    lines: List[Segment] = []
    cursor = float(unit.start)
    for order, (piece, weight) in enumerate(zip(refined, weights)):
        share = max(0.35, duration * (weight / total))
        lines.append(
            Segment(
                index=order,
                start=round(cursor, 3),
                end=round(cursor + share, 3),
                text_bn=piece,
            )
        )
        cursor += share
    lines[-1].end = round(float(unit.start) + duration, 3)

    # খুব ছোট লাইন প্রতিবেশীর সাথে মেশানো (blink করে দেখা যাওয়া ঠেকাতে)
    merged: List[Segment] = []
    for line in lines:
        if merged and (line.end - line.start) < min_seconds:
            merged[-1].end = line.end
            merged[-1].text_bn = _join_text(merged[-1].text_bn, line.text_bn)
        else:
            merged.append(line)
    for order, line in enumerate(merged):
        line.index = order
    return merged


def unit_summary(units: Iterable[Segment]) -> str:
    """লগ/রিপোর্টের জন্য এক লাইনের সারসংক্ষেপ।"""
    units = list(units)
    if not units:
        return "0 unit"
    average = sum(u.end - u.start for u in units) / len(units)
    merged = sum(max(0, len(u.children) - 1) for u in units)
    return f"{len(units)} unit (গড় {average:.1f}s, {merged}টি segment জোড়া লেগেছে)"
