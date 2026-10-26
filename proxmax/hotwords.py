"""Hotwords: নাম-শব্দের আলাদা ফাইল, যা Whisper-এর নির্ভুলতা বাড়ায়।

কেন দরকার
-----------
Whisper মাঝে মাঝে চেনা নাম ভুল লেখে ("Bellon d'Or" → "Bale on the door",
"Ronaldinho" → "Ronaldo in yo")। অথচ এই শব্দগুলোই ভিডিওতে বারবার আসে।
সমাধান: নামগুলোর একটি তালিকা Whisper-এর prompt-এ (``hotwords``) দেওয়া,
যাতে সে জানে কোন শব্দগুলো আশা করা যায়।

ব্যবহারকারীকে কিছু টাইপ করতে হয় না — রিপোর রুটে ``hotwords.txt`` ফাইলটাই
ডিফল্টে পড়া হয় (:mod:`proxmax.config` দেখুন)।

ফাইলের নিয়ম
------------
* এক লাইনে একটি নাম/শব্দ (কমা দিয়ে একাধিকও চলবে)।
* ``#`` দিয়ে শুরু হলে পুরো লাইন মন্তব্য।
* ``[ফুটবল]`` ধাঁচের ``[ ]`` হেডার লাইন বাদ দেওয়া হয় (শুধু গোছানোর জন্য)।
* ডুপ্লিকেট (case-insensitive) সরানো হয়।
* ক্রমই অগ্রাধিকার — Whisper-এর prompt সীমিত, তাই **উপরের লাইনগুলোই
  সবচেয়ে বেশি ব্যবহৃত হবে**; নিজের নাম তালিকার শুরুতে যোগ করলে সেটি বরাবর যাবে।
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

#: কমা/সেমিকোলন/নতুন লাইন — যেকোনোটা দিয়ে একাধিক শব্দ আলাদা করা যায়
_SPLIT_RE = re.compile(r"[,;\n]+|\s{2,}")
#: ``[ফুটবল]`` / ``[players]`` ধাঁচের হেডার
_HEADER_RE = re.compile(r"^\[.*\]$")
#: ফাইলের ভেতরে মন্তব্য
_COMMENT_RE = re.compile(r"(?:^|\s)#.*$")


def clean_entry(text: str) -> str:
    """একটি naam/শব্দ পরিষ্কার করে (বাড়তি স্পেস/উদ্ধৃতি বাদ)।

    Args:
        text: কাঁচা এন্ট্রি।

    Returns:
        পরিষ্কার শব্দ; খালি হলে খালি স্ট্রিং।
    """
    entry = str(text or "").strip().strip("\"'").strip()
    entry = re.sub(r"\s+", " ", entry)
    return entry


def split_hotwords(text: str) -> List[str]:
    """এক লাইন/এক স্ট্রিং থেকে একাধিক hotword আলাদা করে।

    Args:
        text: কমা, সেমিকোলন বা নতুন লাইনে লেখা শব্দের তালিকা।

    Returns:
        পরিষ্কার করার পর ডুপ্লিকেট-মুক্ত তালিকা (ক্রম অপরিবর্তিত)।
    """
    entries: List[str] = []
    seen: set[str] = set()
    for chunk in _SPLIT_RE.split(str(text or "")):
        entry = clean_entry(chunk)
        if not entry or _HEADER_RE.match(entry):
            continue
        key = entry.casefold()
        if key in seen:
            continue
        seen.add(key)
        entries.append(entry)
    return entries


def load_hotword_file(path: Path | str) -> List[str]:
    """``hotwords.txt`` ফাইল পড়ে শব্দের তালিকা ফেরত দেয়।

    ফাইল না থাকলে বা পড়া না গেলে খালি তালিকা — কোনো error নয় (ডাবিং কখনো
    এতে থামবে না)।

    Args:
        path: ফাইলের পাথ।

    Returns:
        ডুপ্লিকেট-মুক্ত hotword তালিকা, ফাইলের ক্রমে।
    """
    file_path = Path(path)
    if not file_path.is_file():
        return []
    entries: List[str] = []
    seen: set[str] = set()
    text = file_path.read_text(encoding="utf-8", errors="replace")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or _HEADER_RE.match(line):
            continue
        line = _COMMENT_RE.sub("", line).strip()
        if not line:
            continue
        line = re.sub(r"^[-*•\d.)\s]+", "", line).strip()
        for entry in split_hotwords(line):
            key = entry.casefold()
            if key in seen or _HEADER_RE.match(entry):
                continue
            seen.add(key)
            entries.append(entry)
    return entries


def join_hotwords(entries: Iterable[str]) -> str:
    """তালিকাকে Whisper-এ দেওয়ার মতো এক স্ট্রিং-এ জোড়া লাগায়।

    Args:
        entries: hotword তালিকা।

    Returns:
        কমা দিয়ে আলাদা করা স্ট্রিং।
    """
    return ", ".join(clean_entry(e) for e in entries if clean_entry(e))


def build_hotword_slices(entries: Iterable[str], max_chars: int = 480) -> List[str]:
    """তালিকাকে ছোট ছোট ভাগে ভাগ করে (Whisper-এর prompt সীমা মেনে)।

    কেন ভাগ করা হয়: Whisper-এর প্রসঙ্গ সীমা ২২৪ টোকেন, তাই ৭০০+ নাম একসাথে
    দেওয়া যায় না। পুরো তালিকাকে পরপর ভাগ করে **প্রতিটি chunk-এ আলাদা ভাগ**
    দেওয়া হয় — এতে পুরো ভিডিও জুড়ে সব নামই একবার প্রাইম হয়, কোনো নাম বাদ
    পড়ে না।

    Args:
        entries: hotword তালিকা।
        max_chars: প্রতি ভাগে সর্বোচ্চ অক্ষর।

    Returns:
        কমা-জোড়া স্ট্রিংয়ের তালিকা (খালি তালিকা = hotwords নেই)।
    """
    limit = max(40, int(max_chars))
    slices: List[str] = []
    current: List[str] = []
    used = 0
    for entry in entries:
        entry = clean_entry(entry)
        if not entry:
            continue
        extra = len(entry) + (2 if current else 0)
        if current and used + extra > limit:
            slices.append(", ".join(current))
            current, used = [], 0
            extra = len(entry)
        if extra > limit:  # খুব লম্বা একক এন্ট্রি — বাদ
            continue
        current.append(entry)
        used += extra
    if current:
        slices.append(", ".join(current))
    return slices
