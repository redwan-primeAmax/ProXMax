"""Gemini batching, caching, translation, and missing-line recovery."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import Segment
from .translation_errors import TranslationError, _PartialBatchError, _ResponseFormatError, _error_message
from .translation_parsing import normalise_text
from .translation_quality import TranslationAssessment, assess_translation
from .translator_config import SYSTEM_INSTRUCTION
from .utils import get_logger

logger = get_logger(__name__)


class GeminiOperationsMixin:
    """Batch operations mixed into :class:`GeminiTranslator`."""

    def _run_batches(
        self,
        items: Sequence[Tuple[str, float]],
        instruction: str = SYSTEM_INSTRUCTION,
        task: str = "Translate each input paragraph into Bangla.",
        description: str = "  Translate (Gemini)",
        depth: int = 0,
    ) -> List[str]:
        """সব ব্যাচ চালিয়ে ফলাফল একই ক্রমে ফেরত দেয়।"""
        results: List[str] = []
        for batch in self._make_batches(items, instruction=instruction, task=task):
            if self._estimate_input_tokens(batch, instruction, task) > self.max_input_tokens:
                self._skipped_lines += len(batch)
                logger.warning(
                    "⚠️ একটি সম্পূর্ণ chunk-ও input-token cap (%d) ছাড়ায় — অংশ কেটে পাঠানো হবে না; chunk বাদ।",
                    self.max_input_tokens,
                )
                results.extend([""] * len(batch))
                continue
            try:
                results.extend(
                    self._call_with_retry(batch, instruction=instruction, task=task)
                )
            except _PartialBatchError as exc:
                if depth >= 6:
                    raise
                results.extend(exc.translations)
                rest = list(batch)[len(exc.translations):]
                logger.warning(
                    "♻️ %d/%d লাইন পাওয়া গেছে — বাকি %d লাইন আবার চাওয়া হচ্ছে…",
                    len(exc.translations), len(batch), len(rest),
                )
                results.extend(
                    self._run_batches(rest, instruction, task, description, depth + 1)
                )
            except _ResponseFormatError as exc:
                if len(batch) <= 1 or depth >= 6:
                    logger.error(
                        "🩹 একটি লাইনের উত্তর বারবার ভুল ফরম্যাটে এসেছে — লাইনটি বাদ "
                        "দেওয়া হলো (ভিডিওর বাকি অংশ চলবে): %s", _error_message(exc, 200),
                    )
                    self._skipped_lines += 1
                    results.append("")
                    continue
                middle = len(batch) // 2
                logger.warning("🔀 %d লাইনের ব্যাচ দুই ভাগে ভেঙে আবার চেষ্টা…", len(batch))
                results.extend(
                    self._run_batches(batch[:middle], instruction, task, description, depth + 1)
                )
                results.extend(
                    self._run_batches(batch[middle:], instruction, task, description, depth + 1)
                )
        return results

    def _translate_checked(
        self, items: Sequence[Tuple[str, float]], description: str = "  Translate (Gemini)"
    ) -> List[TranslationAssessment]:
        """Repair obvious digit errors and recheck only lines with unresolved issues."""
        translations = self._run_batches(items, description=description)
        checked = [assess_translation(source, text) for (source, _), text in zip(items, translations)]
        for result in checked:
            for note in result.notes:
                logger.warning("অনুবাদের স্বয়ংক্রিয় সংশোধন review করুন: %s", note)
        suspect = [index for index, result in enumerate(checked) if result.text and result.issues]
        if not suspect:
            return checked

        logger.warning("⚠️ %dটি অনুবাদে সংখ্যা/লিপির সমস্যা — শুধু সেগুলো আবার যাচাই হচ্ছে।", len(suspect))
        try:
            retried = self._run_batches(
                [items[index] for index in suspect],
                task=("Translate each input paragraph faithfully into Bangla. Preserve all "
                      "explicit digits exactly and avoid unrelated writing systems."),
                description="  Translation quality recheck (Gemini)",
            )
        except TranslationError as exc:
            # _run_batches already used the configured retry policy for API errors.
            logger.warning("অনুবাদের quality recheck ব্যর্থ (%s) — আগের ফল review-এ থাকবে।", exc)
            retried = []
        for index, candidate in zip(suspect, retried):
            second = assess_translation(items[index][0], candidate)
            if second.text and len(second.issues) < len(checked[index].issues):
                checked[index] = TranslationAssessment(
                    second.text, second.issues,
                    ("Retranslated after a number/script review signal; compare with the source.",) + second.notes,
                )
        for index in suspect:
            if checked[index].issues:
                logger.warning("অনুবাদ review দরকার: %s", "; ".join(checked[index].issues))
        return checked

    def verify_api(self) -> Dict[str, Any]:
        """API key এবং মডেল সত্যিই কাজ করছে কি না তা একটি ছোট টেস্ট কলের মাধ্যমে যাচাই করে।

        নোটবুকে ``GEMINI_API_KEY`` দেওয়ার সাথে সাথেই এটি কল করা হয়, যাতে ডাবিং শুরু
        করার আগেই ব্যবহারকারী নিশ্চিত হতে পারেন যে API key ও মডেল সচল আছে।
        """
        requested_model = self.model
        sample = self.translate_text("Welcome to FC Mobile.", duration=2.0)
        if not sample or not sample.strip():
            raise TranslationError(
                "Gemini API থেকে খালি উত্তর এসেছে — আপনার API key বা মডেল যাচাই করুন।"
            )
        return {
            "ok": True,
            "requested_model": requested_model,
            "active_model": self.model,
            "sample_reply": sample.strip(),
        }

    def translate_text(self, text: str, duration: float = 0.0) -> str:
        """Translate text only; the legacy duration argument is never sent to Gemini."""
        original = normalise_text(text)
        if not original:
            return ""
        with self._cache_lock:
            cached = self._cache.get(original)
        if cached is not None:
            return cached
        checked = self._translate_checked([(original, 0.0)])[0]
        if checked.text and not checked.issues:
            with self._cache_lock:
                self._cache[original] = checked.text
                self._cache_notes[original] = checked.notes
        return checked.text

    def _make_batches(
        self,
        items: Sequence[Tuple[str, float]],
        instruction: str = SYSTEM_INSTRUCTION,
        task: str = "Translate each input paragraph into Bangla.",
    ) -> List[List[Tuple[str, float]]]:
        """Pack whole paragraphs under token, item-count, and optional legacy char limits."""
        batches: List[List[Tuple[str, float]]] = []
        current: List[Tuple[str, float]] = []
        chars = 0
        for item in items:
            text = item[0]
            candidate = current + [item]
            over_limit = (
                len(candidate) > self.batch_size
                or (self.max_chars_per_request > 0
                    and chars + len(text) > self.max_chars_per_request)
                or self._estimate_input_tokens(candidate, instruction, task) > self.max_input_tokens
            )
            if current and over_limit:
                batches.append(current)
                current, chars = [], 0
                candidate = [item]
            current.append(item)
            chars += len(text)
            # Keep an over-budget paragraph intact as a singleton; _run_batches
            # will skip the whole paragraph instead of truncating its text.
            if len(current) == 1 and self._estimate_input_tokens(current, instruction, task) > self.max_input_tokens:
                batches.append(current)
                current, chars = [], 0
        if current:
            batches.append(current)
        return batches

    def batch_segments(self, segments: Sequence[Segment]) -> List[List[Segment]]:
        """Return pipeline groups that match the token-budget request boundaries."""
        items = [(normalise_text(segment.text_en), 0.0) for segment in segments]
        item_batches = self._make_batches(items)
        groups: List[List[Segment]] = []
        cursor = 0
        for batch in item_batches:
            groups.append(list(segments[cursor:cursor + len(batch)]))
            cursor += len(batch)
        return groups

    def translate_segments(self, segments: List[Segment]) -> List[Segment]:
        """সব segment অনুবাদ করে; timestamp অপরিবর্তিত থাকে।"""
        if not segments:
            return segments

        started = time.perf_counter()
        calls_before = self._api_calls
        groups: Dict[str, List[Segment]] = {}
        items: List[Tuple[str, float]] = []

        for seg in segments:
            seg.translation_issues = []
            seg.translation_notes = []
            seg.original_text_bn = None
            text = normalise_text(seg.text_en)
            if not text:
                seg.text_bn = ""
                continue
            with self._cache_lock:
                cached = self._cache.get(text)
                cached_notes = self._cache_notes.get(text, ())
            if cached is not None:
                seg.text_bn = cached
                seg.translation_notes = list(cached_notes)
                continue
            if text not in groups:
                groups[text] = []
                items.append((text, 0.0))
            groups[text].append(seg)

        if items:
            checked = self._translate_checked(
                items, description=f"  Translate (Gemini, {len(items)} lines)"
            )
            with self._cache_lock:
                for (text, _duration), result in zip(items, checked):
                    if result.text and not result.issues:
                        self._cache[text] = result.text
                        self._cache_notes[text] = result.notes
                    for seg in groups[text]:
                        seg.text_bn = result.text
                        seg.translation_issues = list(result.issues)
                        seg.translation_notes = list(result.notes)

        missing = [s for s in segments
                   if not (s.text_bn or "").strip() and normalise_text(s.text_en)]
        if missing:
            self._recover_missing(missing)

        logger.info(
            "Gemini অনুবাদ শেষ: %d segment | নতুন লাইন: %d | API call: %d | cache: %d | "
            "বাদ পড়া লাইন: %d | সময় %.1fs",
            len(segments), len(items), self._api_calls - calls_before,
            len(self._cache), self._skipped_lines, time.perf_counter() - started,
        )
        if self._truncated_hits:
            logger.info(
                "ℹ️ %dটি উত্তরে output-token সীমা ছুঁয়েছিল — বাজেট/ব্যাচ স্বয়ংক্রিয়ভাবে "
                "সমন্বয় করা হয়েছে।", self._truncated_hits,
            )
        return segments

    def _recover_missing(self, segments: Sequence[Segment]) -> int:
        """অনূদিত না হওয়া লাইনগুলো ছোট ব্যাচে আবার অনুবাদ করে বসায়।"""
        by_text: Dict[str, List[Segment]] = {}
        for seg in segments:
            text = normalise_text(seg.text_en)
            if text:
                by_text.setdefault(text, []).append(seg)
        if not by_text:
            return 0

        logger.info(
            "♻️ %dটি লাইন অনূদিত ছিল — ছোট ব্যাচে আবার চেষ্টা করা হচ্ছে (নীরবতা এড়াতে)…",
            len(by_text),
        )
        items = [(text, 0.0) for text in by_text]
        saved_batch = self.batch_size
        recovered = 0
        try:
            self.batch_size = min(4, saved_batch)
            checked = self._translate_checked(items, description="  Retry missing")
        except Exception as exc:
            logger.warning("উদ্ধার-পাস ব্যর্থ (%s) — ওই লাইনগুলো বাদ দিয়ে এগোনো হচ্ছে।", exc)
            return 0
        finally:
            self.batch_size = saved_batch

        with self._cache_lock:
            for (text, _duration), result in zip(items, checked):
                if not result.text:
                    continue
                if not result.issues:
                    self._cache[text] = result.text
                    self._cache_notes[text] = result.notes
                for seg in by_text[text]:
                    seg.text_bn = result.text
                    seg.translation_issues = list(result.issues)
                    seg.translation_notes = list(result.notes)
                recovered += 1
        if recovered:
            logger.info("✅ %dটি লাইন উদ্ধার করা হলো।", recovered)
        return recovered
