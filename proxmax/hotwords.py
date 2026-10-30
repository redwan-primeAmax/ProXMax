"""Hotwords loader, Whisper prompt selector, and transcript repair utilities.

``hotwords.txt`` holds FC Mobile terms, UI labels, events, skill moves,
PlayStyles, and a curated (not exhaustive) set of football names/aliases. The
full local vocabulary is never dumped into a request: Whisper receives a short,
context-aware prompt, while Gemini only receives a bounded glossary of terms
that actually occur in the batch being translated.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .hotword_repair import (
    _ALIAS_PATTERNS, _SAFE_SINGLE_WORDS, _find_matches, _find_phrase,
    _normalise_search_text, matched_hotwords_for_texts, repair_transcript_hotwords,
)

DEFAULT_HOTWORDS_FILENAME = "hotwords.txt"
DEFAULT_WHISPER_PROMPT_CHARS = 1600

#: Terms worth prioritising in the Whisper prompt for FC Mobile commentary.
#: The full list is deliberately not dumped into every prompt.
_WHISPER_PRIORITY_TERMS: Tuple[str, ...] = (
    "FC Mobile", "EA Sports FC Mobile", "FIFA Mobile", "FC Mobile 27",
    "OVR", "Star Pass", "Premium Star Pass", "Division Rivals", "Head to Head",
    "H2H", "VS Attack", "VSA", "Manager Mode", "Game Plans", "Game Plan",
    "Gauntlet Mode", "Football Centre", "Club Challenge", "Exchange",
    "Player Exchange", "League Exchange", "Mascherano", "Dudek", "Skill Boost",
    "Rank Up", "Training Transfer", "Sprint & Skill", "Skill Moves",
    "TOTY", "UTOTY", "TOTS", "UTOTS", "Ballon d'Or", "Ballon d’Or",
    "Incisive Through Pass", "Dead Ball", "Finesse Expert", "Clinical Finisher",
    "Power Shot", "Penalty Expert", "Chip Shot", "Tiki Taka", "Bullet Pass",
    "Whipped Cross", "Rapid", "Trickster", "Anticipate", "Guardian",
    "Accelerator", "Relentless", "Bruiser", "Precision Header", "Deflector",
    "Rush Out", "Lane Change", "Heel to Heel Flick", "Open Up Fake Shot",
    "Advanced Rainbow", "Hocus Pocus", "Roulette", "Elastico", "Pele",
    "Maradona", "Lionel Messi", "Messi", "Cristiano Ronaldo", "CR7", "Ronaldo",
    "Kylian Mbappe", "Mbappe", "Erling Haaland", "Haaland", "Lamine Yamal",
    "Jude Bellingham", "Bellingham", "Vinicius Junior", "Vini Jr", "Neymar",
    "Zidane", "Ronaldinho", "Maldini", "Cruyff", "Gullit", "Beckenbauer",
    "Yashin", "Van Basten", "Kvaratskhelia", "Ousmane Dembele", "Dembélé",
)

#: Common single-word terms/names whose casing is safe to repair in isolation.

#: রেটিং/ইভেন্ট ও সাধারণ ভুল শোনা শব্দগুলো ঠিক করার নিয়ম।

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
            key = term.casefold()
            if key not in seen:
                seen.add(key)
                terms.append(term)
    return terms

def _read_terms(path: Path) -> List[str]:
    """Read one terms file, treating an unreadable optional file as empty."""
    try:
        if path.is_file():
            return parse_hotwords_text(path.read_text(encoding="utf-8"))
    except OSError:
        pass
    return []

def load_hotwords(
    hotwords_file: Optional[str | Path] = None,
    extra_hotwords: Optional[str | Iterable[str]] = None,
) -> List[str]:
    """Load the selected hotword file and user additions, deduplicated.

    The bundled vocabulary is a maintained curated list, not a full player/card
    database. A custom ``hotwords_file`` remains self-contained as well.
    """
    combined: List[str] = []
    seen: set[str] = set()

    def _add_many(items: Iterable[str]) -> None:
        for item in items:
            cleaned = " ".join(str(item).split()).strip()
            if not cleaned:
                continue
            key = cleaned.casefold()
            if key not in seen:
                seen.add(key)
                combined.append(cleaned)

    if extra_hotwords:
        if isinstance(extra_hotwords, str):
            _add_many(parse_hotwords_text(extra_hotwords))
        else:
            _add_many(extra_hotwords)

    file_path = resolve_hotwords_path(hotwords_file)
    _add_many(_read_terms(file_path))

    return combined

def _context_seed(context: str) -> int:
    """A stable seed so different chunk contexts select different vocabulary slices."""
    digest = hashlib.blake2s(context.encode("utf-8", errors="ignore"), digest_size=4).digest()
    return int.from_bytes(digest, "big")

def _append_with_limit(selected: List[str], seen: set[str], term: str, max_chars: int) -> bool:
    """Append one term if it fits; return whether the term was appended."""
    key = term.casefold()
    if not term or key in seen:
        return False
    current_len = sum(len(item) for item in selected) + 2 * max(0, len(selected) - 1)
    addition = len(term) + (2 if selected else 0)
    if current_len + addition > max_chars:
        return False
    selected.append(term)
    seen.add(key)
    return True

def select_whisper_hotwords(
    hotwords: Sequence[str],
    max_chars: int = DEFAULT_WHISPER_PROMPT_CHARS,
    context: str = "",
) -> str:
    """Build a bounded, context-aware Whisper hotword prompt.

    Priority FC terms and exact names found in the video/previous-chunk context
    are included first. The remaining curated vocabulary is sampled across the
    file, rather than always favoring its earliest entries, then capped to the
    character budget.
    """
    if not hotwords or max_chars <= 0:
        return ""

    # Keep input order, but avoid spending the prompt budget on duplicate aliases.
    terms: List[str] = []
    lookup: dict[str, str] = {}
    for item in hotwords:
        term = " ".join(str(item).split()).strip()
        key = term.casefold()
        if term and key not in lookup:
            lookup[key] = term
            terms.append(term)

    selected: List[str] = []
    seen: set[str] = set()
    core_budget = max(1, int(max_chars * 0.72))

    # A short stable core vocabulary, including current mobile UI and popular players.
    for preferred in _WHISPER_PRIORITY_TERMS:
        term = lookup.get(preferred.casefold())
        if term:
            _append_with_limit(selected, seen, term, core_budget)

    # Add exact terms already present in the title/context before the broad vocabulary.
    if context:
        context_matches = matched_hotwords_for_texts(
            [context], terms, max_matches=48, max_chars=min(480, max_chars // 3)
        )
        for term in context_matches:
            _append_with_limit(selected, seen, term, max_chars)

    # Modes, buttons, events and curated names retain file order in the core budget.
    for term in terms:
        current_len = sum(len(item) for item in selected) + 2 * max(0, len(selected) - 1)
        if current_len >= core_budget:
            break
        _append_with_limit(selected, seen, term, core_budget)

    # Sample across the whole vocabulary so players near the end of the file aren't ignored.
    remaining = [term for term in terms if term.casefold() not in seen]
    current_len = sum(len(item) for item in selected) + 2 * max(0, len(selected) - 1)
    diversity_budget = max(0, min(max_chars - current_len, int(max_chars * 0.28)))
    if remaining and diversity_budget > 0:
        average_chars = max(
            10, sum(min(len(term), 32) for term in remaining) // len(remaining) + 2
        )
        sample_count = max(1, diversity_budget // average_chars)
        offset = _context_seed(context) % len(remaining)
        stride = max(1, len(remaining) // sample_count)
        while math.gcd(stride, len(remaining)) != 1:
            stride += 1
        sampled = 0
        cursor = offset
        sample_limit = min(max_chars, current_len + diversity_budget)
        for _ in range(len(remaining)):
            term = remaining[cursor]
            if _append_with_limit(selected, seen, term, sample_limit):
                sampled += 1
            if sampled >= sample_count:
                break
            cursor = (cursor + stride) % len(remaining)

    # Fill any unused budget left by unusually long terms, preserving file order.
    for term in terms:
        _append_with_limit(selected, seen, term, max_chars)
        current_len = sum(len(item) for item in selected) + 2 * max(0, len(selected) - 1)
        if current_len >= max_chars:
            break

    return ", ".join(selected)
