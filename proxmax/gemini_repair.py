"""Measured-TTS shortening retry; pre-fit prediction and text expansion are disabled."""

from __future__ import annotations

from typing import List, Sequence

from .models import Segment
from .text_utils import normalise_bangla
from .translation_errors import TranslationError
from .translator_config import SHORTEN_INSTRUCTION
from .utils import get_logger

logger = get_logger(__name__)


class GeminiRepairMixin:
    """Translation length-repair methods mixed into :class:`GeminiTranslator`."""

    def shorten_to_fit(
        self,
        segments: List[Segment],
        max_tempo: float = 1.15,
        slot_gap: float = 0.08,
        borrow: float = 0.25,
    ) -> int:
        """Legacy compatibility hook; pre-TTS duration prediction is disabled."""
        return 0

    def shorten_to_durations(
        self, segments: List[Segment], targets: Sequence[float]
    ) -> int:
        """Measured TTS overshoot-এর জন্য line-level সংক্ষিপ্ত পুনঃঅনুবাদ।"""
        paired_segments = [
            segment for segment, _target in zip(segments, targets)
            if (segment.text_bn or "").strip()
        ]
        if not paired_segments:
            return 0
        try:
            shortened = self._run_batches(
                [(normalise_bangla(segment.text_bn), 0.0) for segment in paired_segments],
                instruction=SHORTEN_INSTRUCTION,
                task="Make each Bangla translation more concise while preserving its meaning, facts, and names.",
                description="  TTS fit retry (Gemini)",
            )
        except TranslationError as exc:
            logger.warning("মাপা TTS সময়ে অনুবাদ ছোট করা যায়নি (%s)।", exc)
            return 0

        changed = 0
        for segment, candidate in zip(paired_segments, shortened):
            cleaned = normalise_bangla(candidate)
            if cleaned and len(cleaned) < len(segment.text_bn.strip()):
                segment.text_bn = cleaned
                changed += 1
        return changed

    def expand_to_fit(
        self,
        segments: List[Segment],
        targets: Sequence[float],
        max_overshoot: float = 1.15,
    ) -> int:
        """Legacy compatibility hook; translations are never expanded to fill gaps."""
        return 0
