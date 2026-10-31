"""Boundary-aware FC Mobile hotword normalization and transcript repair."""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence, Tuple

_SAFE_SINGLE_WORDS = frozenset(
    term.casefold()
    for term in (
        "OVR", "TOTY", "UTOTY", "TOTS", "UTOTS", "VSA", "H2H", "F2P", "P2W",
        "FCM", "CR7", "Pele", "Pelé", "Messi", "Ronaldo", "Mbappe", "Haaland",
        "Neymar", "Zidane", "Ronaldinho", "Maldini", "Cruyff", "Gullit",
        "Beckenbauer", "Yashin", "Mascherano", "Dudek", "Kvaratskhelia",
        "Bellingham", "Yamal", "Haaland", "Dembele", "Dembélé", "Vitinha", "Pacho",
        "Olise", "Osimhen", "Bonmati", "Bonmatí", "Musiala", "Wirtz", "Saliba",
        "Pedri", "Rodri", "TOTW", "POTM", "UCL", "UEL", "LNY",
        "Hol", "HOL", "PlayStyle", "PlayStyles", "Elastico", "Roulette",
        "Hocus Pocus", "Rapid", "Trickster", "Anticipate", "Guardian", "Accelerator",
        "Relentless", "Bruiser", "Deflector",
    )
)

_ALIAS_PATTERNS: Sequence[tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]]] = (
    (re.compile(r"\b(9\d|1[0-2]\d)\s+over\b", re.IGNORECASE), r"\1 OVR"),
    (re.compile(r"\bu\s*toty\b", re.IGNORECASE), "UTOTY"),
    (re.compile(r"\bu\s*tots\b", re.IGNORECASE), "UTOTS"),
    (re.compile(r"\btoty\b", re.IGNORECASE), "TOTY"),
    (re.compile(r"\btots\b", re.IGNORECASE), "TOTS"),
    (re.compile(r"\btotw\b", re.IGNORECASE), "TOTW"),
    (re.compile(r"\bvsa\b", re.IGNORECASE), "VSA"),
    (re.compile(r"\bh2h\b", re.IGNORECASE), "H2H"),
    (re.compile(r"\bsbc\b", re.IGNORECASE), "SBC"),
    (re.compile(r"\bf2p\b", re.IGNORECASE), "F2P"),
    (re.compile(r"\bp2w\b", re.IGNORECASE), "P2W"),
    (re.compile(r"\bea\s+fc\s+mobile\b", re.IGNORECASE), "EA FC Mobile"),
    (re.compile(r"\bfc\s+mobile\s+(?:twenty\s+)?27\b", re.IGNORECASE), "FC Mobile 27"),
    (re.compile(r"\bfc\s+mobile\b", re.IGNORECASE), "FC Mobile"),
    (re.compile(r"\bfootball\s+cent(?:er|re)\b", re.IGNORECASE), "Football Centre"),
    (re.compile(r"\bgame\s+plans?\b", re.IGNORECASE), lambda m: "Game Plan" + ("s" if m.group(0).lower().endswith("s") else "")),
    (re.compile(r"\bgauntlet\s+mode\b", re.IGNORECASE), "Gauntlet Mode"),
    (re.compile(r"\bsprint\s*(?:&|and)\s*skill\b", re.IGNORECASE), "Sprint & Skill"),
    (re.compile(r"\bplay\s*styles?\b", re.IGNORECASE), lambda m: "PlayStyle" + ("s" if m.group(0).lower().endswith("s") else "")),
    (re.compile(r"\bincisive\s+through[- ]pass\b", re.IGNORECASE), "Incisive Through Pass"),
    (re.compile(r"\bdead\s*ball\b", re.IGNORECASE), "Dead Ball"),
    (re.compile(r"\bballon\s+d[’']?or\b", re.IGNORECASE), "Ballon d'Or"),
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
    (re.compile(r"\btitan\s+trophies\b|\btrophy\s+titans\b", re.IGNORECASE), "Trophy Titans"),
    (re.compile(r"\bkvara(?:tskhelia)?\b", re.IGNORECASE), "Kvaratskhelia"),
    (re.compile(r"\bmbapp[eé]\b", re.IGNORECASE), "Mbappe"),
    (re.compile(r"\bhaaland\b", re.IGNORECASE), "Haaland"),
    (re.compile(r"\bbelling\s*ham\b", re.IGNORECASE), "Bellingham"),
    (re.compile(r"\bvinicius\s+(?:jr\.?|junior)\b", re.IGNORECASE), "Vinicius Junior"),
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

def _normalise_search_text(text: str) -> str:
    """Normalize common apostrophe variants for case-insensitive term matching."""
    return (text or "").casefold().replace("’", "'").replace("‘", "'").replace("ʼ", "'")

def _find_phrase(text: str, phrase: str) -> Optional[int]:
    """Find a phrase at a token boundary; return its start offset if present."""
    if not phrase:
        return None
    start = 0
    while True:
        index = text.find(phrase, start)
        if index < 0:
            return None
        end = index + len(phrase)
        before_ok = index == 0 or not (text[index - 1].isalnum() or text[index - 1] == "_")
        after_ok = end == len(text) or not (text[end].isalnum() or text[end] == "_")
        if before_ok and after_ok:
            return index
        start = index + 1

def _find_matches(
    text: str,
    terms: Sequence[str],
    max_matches: int,
    max_chars: Optional[int],
) -> List[str]:
    """Find longest non-overlapping exact hotword matches in text order."""
    if not text or not terms or max_matches <= 0:
        return []
    blob = _normalise_search_text(text)
    candidates: List[Tuple[int, int, str, int]] = []
    seen: set[str] = set()
    for term in terms:
        cleaned = " ".join(str(term).split()).strip()
        key = cleaned.casefold()
        if len(key) < 3 or key in seen:
            continue
        seen.add(key)
        search_term = _normalise_search_text(key)
        if search_term not in blob:
            continue
        index = _find_phrase(blob, search_term)
        if index is not None:
            candidates.append((index, -len(search_term), cleaned, index + len(search_term)))

    candidates.sort(key=lambda item: (item[0], item[1], item[2].casefold()))
    matches: List[str] = []
    covered: List[Tuple[int, int]] = []
    total_chars = 0
    for start, _neg_len, term, end in candidates:
        if any(start < previous_end and end > previous_start for previous_start, previous_end in covered):
            continue
        addition = len(term) + (2 if matches else 0)
        if max_chars is not None and total_chars + addition > max_chars:
            continue
        matches.append(term)
        covered.append((start, end))
        total_chars += addition
        if len(matches) >= max_matches:
            break
    return matches

def repair_transcript_hotwords(text: str, hotwords: Optional[Sequence[str]] = None) -> str:
    """Fix known mishearings and casing for FC Mobile terms and player names.

    This deliberately performs exact, boundary-aware spelling/casing repairs;
    it does not guess a different player based on a loose fuzzy match.
    """
    if not text:
        return ""
    fixed = text
    for pattern, replacement in _ALIAS_PATTERNS:
        fixed = pattern.sub(replacement, fixed)

    if hotwords:
        candidates: List[Tuple[int, int, str, int]] = []
        blob = _normalise_search_text(fixed)
        seen: set[str] = set()
        for item in hotwords:
            term = " ".join(str(item).split()).strip()
            key = term.casefold()
            if len(term) < 4 or key in seen:
                continue
            seen.add(key)
            # Avoid turning ordinary English words (e.g. "will" or "may") into
            # player-name casing unless the one-word term is a trusted FC/name cue.
            if " " not in term and key not in _SAFE_SINGLE_WORDS:
                continue
            search_term = _normalise_search_text(key)
            if search_term not in blob:
                continue
            index = _find_phrase(blob, search_term)
            if index is not None:
                candidates.append((index, -len(search_term), term, index + len(search_term)))

        candidates.sort(key=lambda item: (item[0], item[1]))
        accepted: List[Tuple[int, int, str]] = []
        covered_end = -1
        for start, _neg_len, term, end in candidates:
            if start < covered_end:
                continue
            accepted.append((start, end, term))
            covered_end = end

        # Apply right-to-left so replacement lengths never invalidate later offsets.
        for start, end, term in reversed(accepted):
            if fixed[start:end] != term:
                fixed = fixed[:start] + term + fixed[end:]
    return fixed

def matched_hotwords_for_texts(
    texts: Sequence[str],
    hotwords: Sequence[str],
    max_matches: int = 40,
    max_chars: Optional[int] = None,
) -> List[str]:
    """Return exact hotwords actually found in text, optionally under a size cap."""
    blob = " ".join(t for t in texts if t)
    if not blob:
        return []
    return _find_matches(blob, hotwords, max_matches=max_matches, max_chars=max_chars)
