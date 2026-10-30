"""Measured-duration shortening and gap-filling translation operations."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .models import Segment
from .text_utils import estimate_speech_seconds, normalise_bangla
from .translation_errors import TranslationError
from .translation_parsing import normalise_text
from .translator_config import DURATION_TOLERANCE, EXPAND_INSTRUCTION, SHORTEN_INSTRUCTION
from .utils import get_logger

logger = get_logger(__name__)


class GeminiRepairMixin:
    """Translation length-repair methods mixed into :class:`GeminiTranslator`."""

    def shorten_to_fit(
        self,
        segments: List[Segment],
        max_tempo: float = 1.25,
        slot_gap: float = 0.08,
        borrow: float = 0.25,
    ) -> int:
        """সময়ে না আঁটা বাংলা লাইনগুলো ছোট করে আবার লেখায়।"""
        ordered = sorted([s for s in segments if (s.text_bn or "").strip()], key=lambda s: s.start)
        offenders: List[Segment] = []
        targets: List[float] = []
        for i, seg in enumerate(ordered):
            next_start = ordered[i + 1].start if i + 1 < len(ordered) else None
            speech_window = max(0.0, float(seg.end) - float(seg.start))
            if next_start is not None:
                hard_window = max(0.0, float(next_start) - float(seg.start) - slot_gap)
                speech_window = min(speech_window, hard_window)
                borrowable = min(max(0.0, float(borrow)), max(0.0, hard_window - speech_window))
            else:
                # Without a later segment timestamp, don't borrow the video's trailing silence.
                borrowable = 0.0
            available = min(
                max(0.0, speech_window + borrowable),
                max(0.0, (float(next_start) - float(seg.start) - slot_gap))
                if next_start is not None else max(0.0, speech_window),
            )
            if available <= 0.2:
                continue
            budget = max(0.4, available * max_tempo)
            estimated = estimate_speech_seconds(seg.text_bn, self.char_rate)
            if estimated > budget * (1 + DURATION_TOLERANCE):
                offenders.append(seg)
                targets.append(max(0.4, available))
        if not offenders:
            return 0

        logger.info(
            "অনুবাদ সময়ে আঁটছে না এমন %d/%d লাইন ছোট করে লেখানো হচ্ছে…",
            len(offenders), len(ordered),
        )
        items = [(normalise_bangla(seg.text_bn), target) for seg, target in zip(offenders, targets)]
        try:
            shortened = self._run_batches(
                items,
                instruction=SHORTEN_INSTRUCTION,
                task="Shorten each Bangla line to fit its speaking time.",
                description="  Shorten (Gemini)",
            )
        except TranslationError as exc:
            logger.warning(
                "লাইন ছোট করার চেষ্টা ব্যর্থ (%s) — মূল অনুবাদই রাখা হচ্ছে; "
                "sync সামান্য দ্রুত গতিতে মিলিয়ে নেবে।", exc,
            )
            return 0

        changed = 0
        for seg, new_text in zip(offenders, shortened):
            cleaned = normalise_bangla(new_text)
            if cleaned and len(cleaned) < len(seg.text_bn):
                seg.text_bn = cleaned
                changed += 1
        logger.info("ছোট করা হয়েছে %dটি লাইন।", changed)
        return changed

    def shorten_to_durations(
        self, segments: List[Segment], targets: Sequence[float]
    ) -> int:
        """Measured TTS overshoot-এর জন্য line-level সংক্ষিপ্ত পুনঃঅনুবাদ।"""
        pairs = [
            (segment, max(0.4, float(target)))
            for segment, target in zip(segments, targets)
            if (segment.text_bn or "").strip()
        ]
        if not pairs:
            return 0
        try:
            shortened = self._run_batches(
                [(normalise_bangla(seg.text_bn), target) for seg, target in pairs],
                instruction=SHORTEN_INSTRUCTION,
                task="Shorten each Bangla line to fit its measured speech duration while preserving every key fact.",
                description="  TTS fit retry (Gemini)",
            )
        except TranslationError as exc:
            logger.warning("মাপা TTS সময়ে অনুবাদ ছোট করা যায়নি (%s)।", exc)
            return 0

        changed = 0
        for (segment, _target), candidate in zip(pairs, shortened):
            cleaned = normalise_bangla(candidate)
            if cleaned and len(cleaned) < len(segment.text_bn.strip()):
                segment.text_bn = cleaned
                changed += 1
        return changed

    def expand_to_fit(
        self,
        segments: List[Segment],
        targets: Sequence[float],
        max_overshoot: float = 1.25,
    ) -> int:
        """স্লটের চেয়ে অনেক ছোট বাংলা লাইন **ভরাট করে** আবার লেখায়।"""
        pairs = [
            (seg, float(target))
            for seg, target in zip(segments, targets)
            if (seg.text_bn or "").strip()
        ]
        if not pairs:
            return 0
        items = [(normalise_bangla(seg.text_bn), target) for seg, target in pairs]
        try:
            expanded = self._run_batches(
                items,
                instruction=EXPAND_INSTRUCTION,
                task="Make each Bangla line long enough for its speaking time.",
                description="  Fill gaps (Gemini)",
            )
        except TranslationError as exc:
            logger.warning(
                "লাইন ভরাট করা যায়নি (%s) — আগের অনুবাদই রাখা হচ্ছে; বড় ফাঁক "
                "থাকলে sync ধীর করে ভরাট করবে।", exc,
            )
            return 0

        changed = 0
        for (seg, target), new_text in zip(pairs, expanded):
            cleaned = normalise_bangla(new_text)
            if not cleaned:
                continue
            old_len = max(1, len(seg.text_bn))
            wanted = max(float(target) * float(self.char_rate), old_len * 1.2)
            if old_len < len(cleaned) <= wanted * float(max_overshoot):
                seg.text_bn = cleaned
                changed += 1
            elif len(cleaned) > old_len:
                logger.debug(
                    "ভরাট করা লাইন স্লটের জন্য বেশি লম্বা (%d অক্ষর; সীমা %d) — বাদ।",
                    len(cleaned), int(wanted * float(max_overshoot)),
                )
        logger.info("অনুবাদ ভরাট করা হয়েছে %d/%d লাইনে।", changed, len(pairs))
        return changed

    def set_char_rate(self, chars_per_second: float, source: str = "measured") -> None:
        """কণ্ঠের প্রকৃত গতি বসায় (ক্যালিব্রেশনের পর ডাকা হয়)।"""
        rate = float(chars_per_second)
        if not 6.0 <= rate <= 24.0:
            return
        self.char_rate = rate
        self.rate_source = source
        logger.info("অনুবাদের সময়-বাজেট এখন %.1f অক্ষর/সেকেন্ড (%s)", rate, source)
