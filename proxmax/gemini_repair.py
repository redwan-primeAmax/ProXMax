"""Measured-TTS shortening retry; pre-fit prediction and text expansion are disabled."""

from __future__ import annotations

import json
from typing import List, Sequence

from .models import Segment
from .text_utils import normalise_bangla
from .translation_errors import TranslationError
from .translation_quality import assess_fit_rewrite
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
        """Rephrase measured unfit clips using the source, not Bangla alone."""
        pairs = [
            (segment, float(target)) for segment, target in zip(segments, targets)
            if (segment.text_bn or "").strip()
        ]
        if not pairs:
            return 0

        items = []
        for segment, target in pairs:
            reference = normalise_bangla(segment.original_text_bn or segment.text_bn)
            payload = {
                "english_source": (segment.text_en or "").strip(),
                "current_bangla": normalise_bangla(segment.text_bn),
                "reference_bangla": reference,
            }
            measured = float(segment.stretched_duration or 0.0)
            if measured > target > 0:
                # A heuristic goal based on a real failed fit, never an absolute timestamp.
                payload["target_percent"] = max(65, min(96, int(100 * target / measured)))
            items.append((json.dumps(payload, ensure_ascii=False), 0.0))
        try:
            shortened = self._run_batches(
                items,
                instruction=SHORTEN_INSTRUCTION,
                task="Make each current Bangla translation more concise without losing source meaning.",
                description="  TTS fit retry (Gemini)",
            )
        except TranslationError as exc:
            logger.warning("মাপা TTS সময়ে অনুবাদ ছোট করা যায়নি (%s)।", exc)
            for segment, _target in pairs:
                segment.translation_notes.append("Fit-retry translation failed; prior text retained.")
            return 0

        changed = 0
        for (segment, _target), candidate in zip(pairs, shortened):
            current = normalise_bangla(segment.text_bn)
            reference = normalise_bangla(segment.original_text_bn or current)
            result = assess_fit_rewrite(segment.text_en, reference, current, candidate)
            if result.issues:
                logger.warning("Segment %d-এর সংক্ষিপ্ত অনুবাদ গ্রহণ করা হয়নি: %s", segment.index,
                               "; ".join(result.issues))
                segment.translation_notes.append(
                    "Fit-retry text rejected; prior text retained: " + "; ".join(result.issues)
                )
                continue
            segment.text_bn = result.text
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
