"""Deprecated no-op fit hooks retained for callers of the translator API.

Dubbing never asks Gemini to rewrite a translation to fit a time window.
"""

from __future__ import annotations

from typing import List, Sequence

from .models import Segment


class GeminiRepairMixin:
    """Compatibility methods; duration-fitting text rewrites are disabled."""

    def shorten_to_fit(self, segments: List[Segment], max_tempo: float = 1.0,
                       slot_gap: float = 0.08, borrow: float = 0.25) -> int:
        return 0

    def shorten_to_durations(self, segments: List[Segment], targets: Sequence[float]) -> int:
        return 0

    def expand_to_fit(self, segments: List[Segment], targets: Sequence[float],
                      max_overshoot: float = 1.15) -> int:
        return 0
