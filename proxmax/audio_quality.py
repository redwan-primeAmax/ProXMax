"""Conservative quality checks for dubbed speech and transcript coverage.

The checks distinguish a likely missing/cut-off line from a clean result. They
are heuristic and must not be treated as proof of exact pronunciation.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Mapping, Sequence

from .models import Segment, Word
from .text_utils import prepare_for_speech

_TOKEN_RE = re.compile(r"[\u0980-\u09FF]+|[a-zA-Z0-9]+")


def _tokens(text: str) -> List[str]:
    """Normalize punctuation away while keeping Bangla and Latin names."""
    return [token.casefold() for token in _TOKEN_RE.findall(prepare_for_speech(text or ""))]


def _sequence_coverage(expected: Sequence[str], actual: Sequence[str]) -> float:
    if not expected or not actual:
        return 0.0
    matched = sum(block.size for block in SequenceMatcher(
        a=list(expected), b=list(actual), autojunk=False
    ).get_matching_blocks())
    return min(1.0, matched / len(expected))


def _tail_token_found(expected: Sequence[str], actual: Sequence[str]) -> bool:
    if not expected or not actual:
        return False
    tail = expected[-1]
    return any(
        SequenceMatcher(a=tail, b=word, autojunk=False).ratio() >= 0.62
        for word in actual[-3:]
    )


def assess_bengali_coverage(
    segments: Sequence[Segment],
    recognized_words: Sequence[Word],
    *,
    min_coverage: float = 0.58,
    boundary_tolerance: float = 0.05,
) -> Dict[str, Any]:
    """Compare expected Bangla text with a post-mux Bangla ASR transcript.

    Per-unit scores are signals for review, not a semantic correctness test.
    Source pauses are not filled or judged by this comparison.
    """
    ordered_words = sorted(recognized_words, key=lambda word: (word.start, word.end))
    units: List[Dict[str, Any]] = []
    coverages: List[float] = []

    for segment in sorted(segments, key=lambda item: (item.start, item.end, item.index)):
        expected_tokens = _tokens(segment.text_bn)
        if not expected_tokens:
            continue
        start = max(0.0, float(segment.start) - max(0.0, boundary_tolerance))
        duration = segment.placed_duration
        if duration is None:
            duration = segment.stretched_duration or segment.tts_duration or segment.duration
        end = float(segment.start) + max(0.0, float(duration)) + max(0.0, boundary_tolerance)
        actual_words = [word for word in ordered_words if start <= float(word.start) <= end]
        actual_tokens = [token for word in actual_words for token in _tokens(word.word)]
        coverage = _sequence_coverage(expected_tokens, actual_tokens)
        tail_found = _tail_token_found(expected_tokens, actual_tokens)
        coverages.append(coverage)

        if segment.placed_duration is None or not actual_tokens:
            status = "missing"
        elif coverage < float(min_coverage) or not tail_found:
            status = "review"
        else:
            status = "likely_complete"
        units.append({
            "segment_index": int(segment.index),
            "start": round(float(segment.start), 3),
            "end": round(end, 3),
            "status": status,
            "coverage": round(coverage, 3),
            "tail_token_found": bool(tail_found),
            "expected_text": segment.text_bn,
            "recognized_text": " ".join(word.word.strip() for word in actual_words).strip(),
        })

    counts = {
        "checked": len(units),
        "likely_complete": sum(item["status"] == "likely_complete" for item in units),
        "review": sum(item["status"] == "review" for item in units),
        "missing": sum(item["status"] == "missing" for item in units),
    }
    needs_review = counts["review"] + counts["missing"] > 0
    return {
        "status": "review" if needs_review else "likely_complete",
        "method": "post-mux Bengali Whisper word coverage; heuristic, not forced alignment",
        "minimum_text_coverage": float(min_coverage),
        "units": units,
        "summary": counts,
        "mean_coverage": round(sum(coverages) / len(coverages), 3) if coverages else None,
    }
