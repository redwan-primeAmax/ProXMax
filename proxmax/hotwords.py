"""Hotwords loader, Whisper prompt selector, and transcript repair utilities.

``hotwords.txt`` ফাইলে FC Mobile টার্ম, ইভেন্টের নাম এবং ইতিহাসের ও বর্তমানের
সব প্রধান ফুটবল খেলোয়াড়ের নাম রাখা আছে। এই মডিউলটি:
1. ``hotwords.txt`` এবং ইউজারের অতিরিক্ত ``stt_hotwords`` একসাথে লোড করে।
2. Whisper-এর জন্য সবচেয়ে গুরুত্বপূর্ণ hotwords নির্বাচন করে (টোকেন সীমার ভেতরে)।
3. ট্রান্সক্রিপশনের পর সাধারণ ভুল উচ্চারণ/বানান (যেমন "toty" → "TOTY",
   "over" → "OVR" যখন রেটিংয়ের পাশে থাকে, খেলোয়াড়ের নামের কেসিং) ঠিক করে।
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

DEFAULT_HOTWORDS_FILENAME = "hotwords.txt"

#: রেটিং/ইভেন্ট ও সাধারণ ভুল শোনা শব্দগুলো ঠিক করার নিয়ম
_ALIAS_PATTERNS: Sequence[tuple[re.Pattern[str], str]] = (
    (re.compile(r"\b(9\d|1[0-2]\d)\s+over\b", re.IGNORECASE), r"\1 OVR"),
    (re.compile(r"\bu\s*toty\b", re.IGNORECASE), "UTOTY"),
    (re.compile(r"\bu\s*tots\b", re.IGNORECASE), "UTOTS"),
    (re.compile(r"\btoty\b", re.IGNORECASE), "TOTY"),
    (re.compile(r"\btots\b", re.IGNORECASE), "TOTS"),
    (re.compile(r"\bvsa\b", re.IGNORECASE), "VSA"),
    (re.compile(r"\bh2h\b", re.IGNORECASE), "H2H"),
    (re.compile(r"\bsbc\b", re.IGNORECASE), "SBC"),
    (re.compile(r"\bf2p\b", re.IGNORECASE), "F2P"),
    (re.compile(r"\bp2w\b", re.IGNORECASE), "P2W"),
    (re.compile(r"\bea\s+fc\s+mobile\b", re.IGNORECASE), "EA FC Mobile"),
    (re.compile(r"\bfc\s+mobile\b", re.IGNORECASE), "FC Mobile"),
    (re.compile(r"\bballon\s+d'?or\b", re.IGNORECASE), "Ballon d'Or"),
    (re.compile(r"\bmascheranos?\b", re.IGNORECASE), "Mascherano"),
    (re.compile(r"\bskill\s+boosts?\b", re.IGNORECASE), "Skill Boost"),
    (re.compile(r"\bstar\s+pass\b", re.IGNORECASE), "Star Pass"),
    (re.compile(r"\bdivision\s+rivals\b", re.IGNORECASE), "Division Rivals"),
    (re.compile(r"\bwinter\s+wildcards?\b", re.IGNORECASE), "Winter Wildcards"),
    (re.compile(r"\brule\s*breakers?\b", re.IGNORECASE), "Rulebreakers"),
    (re.compile(r"\bthunder\s*struck\b", re.IGNORECASE), "Thunderstruck"),
    (re.compile(r"\bdynasties\b", re.IGNORECASE), "Dynasties"),
    (re.compile(r"\banniversary\s+event\b", re.IGNORECASE), "Anniversary"),
    (re.compile(r"\bhall\s+of\s+legends\b", re.IGNORECASE), "Hall of Legends"),
    (re.compile(r"\btitan\s+trophies\b", re.IGNORECASE), "Trophy Titans"),
    (re.compile(r"\btrophy\s+titans\b", re.IGNORECASE), "Trophy Titans"),
    (re.compile(r"\bkvara(?:tskhelia)?\b", re.IGNORECASE), "Kvaratskhelia"),
    (re.compile(r"\bmbapp[eé]\b", re.IGNORECASE), "Mbappe"),
    (re.compile(r"\bhaaland\b", re.IGNORECASE), "Haaland"),
    (re.compile(r"\bbelling\s*ham\b", re.IGNORECASE), "Bellingham"),
    (re.compile(r"\bvinicius\s+jr\.?\b", re.IGNORECASE), "Vinicius Junior"),
    (re.compile(r"\blamine\s+yamal\b", re.IGNORECASE), "Lamine Yamal"),
    (re.compile(r"\bde\s+bruyne\b", re.IGNORECASE), "De Bruyne"),
    (re.compile(r"\bcristiano\s+ronaldo\b", re.IGNORECASE), "Cristiano Ronaldo"),
    (re.compile(r"\blionel\s+messi\b", re.IGNORECASE), "Lionel Messi"),
    (re.compile(r"\bronaldinho\b", re.IGNORECASE), "Ronaldinho"),
    (re.compile(r"\bmaldini\b", re.IGNORECASE), "Maldini"),
    (re.compile(r"\bcruyff\b", re.IGNORECASE), "Cruyff"),
    (re.compile(r"\bzidane\b", re.IGNORECASE), "Zidane"),
    (re.compile(r"\bgullit\b", re.IGNORECASE), "Gullit"),
    (re.compile(r"\bvan\s+basten\b", re.IGNORECASE), "Van Basten"),
    (re.compile(r"\brijskaard\b|\brijkaard\b", re.IGNORECASE), "Rijkaard"),
    (re.compile(r"\bbeckenbauer\b", re.IGNORECASE), "Beckenbauer"),
    (re.compile(r"\bpuskas\b", re.IGNORECASE), "Puskas"),
    (re.compile(r"\byashin\b", re.IGNORECASE), "Yashin"),
    (re.compile(r"\beusebio\b", re.IGNORECASE), "Eusebio"),
    (re.compile(r"\bgarrincha\b", re.IGNORECASE), "Garrincha"),
    (re.compile(r"\broberto\s+carlos\b", re.IGNORECASE), "Roberto Carlos"),
)


def default_hotwords_path() -> Path:
    """Return the repository-level ``hotwords.txt`` path."""
    return Path(__file__).resolve().parent.parent / DEFAULT_HOTWORDS_FILENAME


def resolve_hotwords_path(path: Optional[str | Path] = None) -> Path:
    """Resolve a user-supplied hotwords file path, falling back to repo ``hotwords.txt``."""
    if path is None or not str(path).strip():
        return default_hotwords_path()
    candidate = Path(str(path).strip()).expanduser()
    if candidate.is_file():
        return candidate
    repo_relative = Path(__file__).resolve().parent.parent / candidate
    if repo_relative.is_file():
        return repo_relative
    return candidate


def parse_hotwords_text(raw: str) -> List[str]:
    """Parse newline- or comma-separated hotwords, ignoring ``#`` comments and duplicates."""
    seen: set[str] = set()
    terms: List[str] = []
    for raw_line in (raw or "").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for part in line.split(","):
            term = " ".join(part.split()).strip()
            if not term:
                continue
            key = term.lower()
            if key not in seen:
                seen.add(key)
                terms.append(term)
    return terms


def load_hotwords(
    hotwords_file: Optional[str | Path] = None,
    extra_hotwords: Optional[str | Iterable[str]] = None,
) -> List[str]:
    """Load deduplicated hotwords from ``hotwords.txt`` plus any inline ``extra_hotwords``."""
    combined: List[str] = []
    seen: set[str] = set()

    def _add_many(items: Iterable[str]) -> None:
        for item in items:
            cleaned = " ".join(str(item).split()).strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key not in seen:
                seen.add(key)
                combined.append(cleaned)

    if extra_hotwords:
        if isinstance(extra_hotwords, str):
            _add_many(parse_hotwords_text(extra_hotwords))
        else:
            _add_many(extra_hotwords)

    file_path = resolve_hotwords_path(hotwords_file)
    if file_path.is_file():
        try:
            _add_many(parse_hotwords_text(file_path.read_text(encoding="utf-8")))
        except OSError:
            pass

    return combined


def select_whisper_hotwords(
    hotwords: Sequence[str],
    max_chars: int = 1600,
) -> str:
    """Build a compact comma-separated hotword string for faster-whisper."""
    if not hotwords or max_chars <= 0:
        return ""
    selected: List[str] = []
    current_len = 0
    for term in hotwords:
        addition = len(term) + (2 if selected else 0)
        if current_len + addition > max_chars:
            break
        selected.append(term)
        current_len += addition
    return ", ".join(selected)


def repair_transcript_hotwords(text: str, hotwords: Optional[Sequence[str]] = None) -> str:
    """Fix common Whisper mishearings and casing for FC Mobile terms, events, and players."""
    if not text:
        return ""
    fixed = text
    for pattern, replacement in _ALIAS_PATTERNS:
        fixed = pattern.sub(replacement, fixed)

    if hotwords:
        for term in hotwords:
            if len(term) < 4:
                continue
            if term.lower() in fixed.lower() and term not in fixed:
                pat = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
                fixed = pat.sub(term, fixed)
    return fixed


def matched_hotwords_for_texts(
    texts: Sequence[str],
    hotwords: Sequence[str],
    max_matches: int = 40,
) -> List[str]:
    """Return hotwords that actually appear in ``texts`` (for Gemini translation context)."""
    if not texts or not hotwords:
        return []
    blob = " ".join(t for t in texts if t).lower()
    if not blob:
        return []
    matches: List[str] = []
    for term in hotwords:
        t_low = term.lower()
        if len(t_low) < 3:
            continue
        if t_low in blob:
            matches.append(term)
            if len(matches) >= max_matches:
                break
    return matches
