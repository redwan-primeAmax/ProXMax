"""Gemini batching, caching, translation, and missing-line recovery."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import Segment
from .text_utils import estimate_speech_seconds, normalise_bangla
from .translation_errors import TranslationError, _PartialBatchError, _ResponseFormatError, _error_message
from .translation_parsing import normalise_text
from .translator_config import SYSTEM_INSTRUCTION
from .utils import get_logger

logger = get_logger(__name__)


class GeminiOperationsMixin:
    """Batch operations mixed into :class:`GeminiTranslator`."""

    def _run_batches(
        self,
        items: Sequence[Tuple[str, float]],
        instruction: str = SYSTEM_INSTRUCTION,
        task: str = "Translate each line into Bangla.",
        description: str = "  Translate (Gemini)",
        depth: int = 0,
    ) -> List[str]:
        """সব ব্যাচ চালিয়ে ফলাফল একই ক্রমে ফেরত দেয়।"""
        results: List[str] = []
        for batch in self._make_batches(items):
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
                    "♻️ %d/%d লাইন পাওয়া গেছে — বাকি %d লাইন সঙ্গে সঙ্গে আবার চাওয়া হচ্ছে…",
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
                logger.warning("🔀 %d লাইনের ব্যাচ দুই ভাগে ভেঙে সঙ্গে সঙ্গে আবার চেষ্টা…", len(batch))
                results.extend(
                    self._run_batches(batch[:middle], instruction, task, description, depth + 1)
                )
                results.extend(
                    self._run_batches(batch[middle:], instruction, task, description, depth + 1)
                )
        return results

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
        """Translate a single English string into Bangla."""
        original = normalise_text(text)
        if not original:
            return ""
        with self._cache_lock:
            cached = self._cache.get(original)
        if cached is not None:
            return cached
        hint = duration if duration > 0 else estimate_speech_seconds(original, self.char_rate)
        result = normalise_bangla(self._run_batches([(original, hint)])[0])
        with self._cache_lock:
            self._cache[original] = result
        return result

    def _make_batches(self, items: Sequence[Tuple[str, float]]) -> List[List[Tuple[str, float]]]:
        """লাইনগুলোকে সংখ্যা ও অক্ষর-সীমা অনুযায়ী batch-এ ভাগ করে।"""
        batches: List[List[Tuple[str, float]]] = []
        current: List[Tuple[str, float]] = []
        chars = 0
        for text, duration in items:
            if current and (
                len(current) >= self.batch_size
                or chars + len(text) > self.max_chars_per_request
            ):
                batches.append(current)
                current, chars = [], 0
            current.append((text, duration))
            chars += len(text)
        if current:
            batches.append(current)
        return batches

    def translate_segments(self, segments: List[Segment]) -> List[Segment]:
        """সব segment অনুবাদ করে; timestamp অপরিবর্তিত থাকে।"""
        if not segments:
            return segments

        started = time.perf_counter()
        calls_before = self._api_calls
        groups: Dict[str, List[Segment]] = {}
        items: List[Tuple[str, float]] = []

        for seg in segments:
            text = normalise_text(seg.text_en)
            if not text:
                seg.text_bn = ""
                continue
            with self._cache_lock:
                cached = self._cache.get(text)
            if cached is not None:
                seg.text_bn = cached
                continue
            if text not in groups:
                groups[text] = []
                items.append((text, seg.duration))
            groups[text].append(seg)

        if items:
            translations = self._run_batches(
                items, description=f"  Translate (Gemini, {len(items)} lines)"
            )
            with self._cache_lock:
                for (text, _duration), translated in zip(items, translations):
                    cleaned = normalise_bangla(translated)
                    self._cache[text] = cleaned
                    for seg in groups[text]:
                        seg.text_bn = cleaned

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
        items = [(text, group[0].duration) for text, group in by_text.items()]
        saved_batch = self.batch_size
        recovered = 0
        try:
            self.batch_size = min(4, saved_batch)
            translations = self._run_batches(items, description="  Retry missing")
        except Exception as exc:
            logger.warning("উদ্ধার-পাস ব্যর্থ (%s) — ওই লাইনগুলো বাদ দিয়ে এগোনো হচ্ছে।", exc)
            return 0
        finally:
            self.batch_size = saved_batch

        with self._cache_lock:
            for (text, _duration), translated in zip(items, translations):
                cleaned = normalise_bangla(translated)
                if not cleaned:
                    continue
                self._cache[text] = cleaned
                for seg in by_text[text]:
                    seg.text_bn = cleaned
                recovered += 1
        if recovered:
            logger.info("✅ %dটি লাইন উদ্ধার করা হলো।", recovered)
        return recovered
