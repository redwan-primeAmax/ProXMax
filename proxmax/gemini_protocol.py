"""Gemini SDK request construction, capability fallback, and retry logic."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .hotwords import matched_hotwords_for_texts
from .translation_errors import (
    TranslationError, _CapabilityError, _ContentBlockedError, _PartialBatchError,
    _ResponseFormatError, _error_message, _status_code,
)
from .translation_parsing import parse_numbered_lines, parse_translations, salvage_translations
from .translator_config import (
    SYSTEM_INSTRUCTION,
    _AUTH_HINTS, _RESPONSE_SCHEMA,
)
from .gemini_token_budget import estimate_gemini_input_tokens
from .utils import get_logger

logger = get_logger(__name__)


class GeminiProtocolMixin:
    """Protocol and retry methods mixed into :class:`GeminiTranslator`."""

    def _ensure_client(self) -> Any:
        """Create (once) the ``google-genai`` client; fail with a clear message."""
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:  # pragma: no cover - dependency is pinned
                raise TranslationError(
                    "google-genai প্যাকেজ নেই। ইনস্টল করুন:\n"
                    "    pip install -r requirements.txt"
                ) from exc

            http_options = types.HttpOptions(
                timeout=self.timeout * 1000,
                retry_options=types.HttpRetryOptions(attempts=1),
            )
            self._client = genai.Client(api_key=self._api_key, http_options=http_options)
            logger.info(
                "Gemini client প্রস্তুত | model=%s | timeout=%ds | RPM=%d | "
                "batch=%d | সর্বোচ্চ retry=%d",
                self.model, self.timeout, self.requests_per_minute,
                self.batch_size, self.max_retries,
            )
        return self._client

    def _payload(
        self, items: Sequence[Tuple[str, float]], include_timing: bool = False
    ) -> str:
        """Render every request as separate text-only paragraphs, never with timing hints."""
        return "\n\n".join(
            f"Paragraph {index}:\n{text}" for index, (text, _duration) in enumerate(items, 1)
        )

    def _payload_numbered(
        self, items: Sequence[Tuple[str, float]], include_timing: bool = False
    ) -> str:
        """Fallback input remains text-only, blank-line-separated paragraphs."""
        return self._payload(items)

    @staticmethod
    def _uses_timing_payload(instruction: str) -> bool:
        """Timing metadata is intentionally excluded from all Gemini requests."""
        return False

    def _glossary_hint(self, items: Sequence[Tuple[str, float]]) -> str:
        """Add only matching, bounded FC Mobile terms to this request."""
        if not self.hotwords_list:
            return ""
        matched = matched_hotwords_for_texts(
            [text for text, _duration in items], self.hotwords_list,
            max_matches=20, max_chars=600,
        )
        if not matched:
            return ""
        return (
            "\nRecognized football/FC Mobile terms in this batch "
            f"(keep names/terms accurate in Bangla): {', '.join(matched)}."
        )

    def _estimate_input_tokens(
        self,
        items: Sequence[Tuple[str, float]],
        instruction: str = SYSTEM_INSTRUCTION,
        task: str = "Translate each input paragraph into Bangla.",
    ) -> int:
        """Fast local estimate of system instruction, task, glossary, and full payload."""
        count_hint = (
            f"IMPORTANT: the input has paragraph IDs 1 through {len(items)}. Return one "
            "translation object for every ID, preserve the exact IDs, and do not omit "
            f"or duplicate one.{self._glossary_hint(items)}"
        )
        text = f"{instruction}\n{task}\n{count_hint}\nInput paragraphs:\n{self._payload(items)}"
        return estimate_gemini_input_tokens(text)

    def _output_token_budget(self, items: Sequence[Tuple[str, float]]) -> int:
        """এই ব্যাচের উত্তরের জন্য কত output token দরকার তার হিসাব।"""
        chars = sum(len(text) for text, _d in items)
        needed = int(chars * 2.4) + 640
        return max(1024, min(int(self.max_output_tokens), max(needed, 2048)))

    def _thinking_config(self, types: Any) -> Any:
        """thinking কনফিগ (শুধু স্পষ্টভাবে চালু থাকলে)।"""
        if not self._use_thinking:
            return None
        try:
            return types.ThinkingConfig(thinking_budget=0)
        except (AttributeError, TypeError):  # পুরোনো SDK
            self._use_thinking = False
            return None

    def _generate(
        self,
        items: Sequence[Tuple[str, float]],
        instruction: str = SYSTEM_INSTRUCTION,
        task: str = "Translate each input paragraph into Bangla.",
        mode: str = "json",
    ) -> List[str]:
        """এক request-এ একাধিক paragraph প্রসেস করে (retry ছাড়া)।"""
        from google.genai import types

        client = self._ensure_client()
        kwargs: Dict[str, Any] = {
            "system_instruction": instruction,
            "temperature": self.temperature if mode == "json" else 0.0,
            "max_output_tokens": self._output_token_budget(items),
        }
        label = "Input paragraphs:"
        glossary_hint = self._glossary_hint(items)
        count_hint = (
            f"IMPORTANT: the input has paragraph IDs 1 through {len(items)}. Return one "
            "translation object for every ID, preserve the exact IDs, and do not omit "
            f"or duplicate one.{glossary_hint}"
        )
        payload = self._payload(items)
        if mode == "json":
            contents = f"{task}\n{count_hint}\n{label}\n{payload}"
            if self._use_schema:
                kwargs["response_mime_type"] = "application/json"
                kwargs["response_schema"] = _RESPONSE_SCHEMA
        elif mode == "json_plain":
            contents = (
                f"{task}\n"
                'Return ONLY JSON shaped like {"translations": '
                '[{"id": 1, "translation": "<Bangla>"}]}. Use every input ID once. '
                "No other text.\n"
                f"{count_hint}\n{label}\n{payload}"
            )
        else:
            contents = (
                f"{task}\n"
                f"Answer with exactly {len(items)} numbered translations; each line must "
                f"start with that paragraph's ID (1 through {len(items)}). "
                "No JSON, timestamps, or explanation.\n"
                f"{count_hint}\n{label}\n{self._payload_numbered(items)}"
            )

        thinking = self._thinking_config(types)
        if thinking is not None:
            kwargs["thinking_config"] = thinking
        try:
            config = types.GenerateContentConfig(**kwargs)
        except TypeError:
            if "thinking_config" in kwargs:
                kwargs.pop("thinking_config", None)
                self._use_thinking = False
                config = types.GenerateContentConfig(**kwargs)
            else:
                raise

        try:
            response = client.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except Exception as exc:  # noqa: BLE001 - কোনো প্যারামিটার অসমর্থিত হলে
            message = _error_message(exc).lower()
            status = _status_code(exc)
            is_auth = any(h in message for h in _AUTH_HINTS)

            if ("thinking" in message or "budget" in message) and self._use_thinking:
                self._use_thinking = False
                raise _CapabilityError(
                    "এই মডেলে thinking বন্ধ করার অপশন নেই — বাদ দিয়ে আবার চেষ্টা।"
                ) from exc
            if ("schema" in message or "response_mime" in message) and self._use_schema:
                self._use_schema = False
                raise _CapabilityError(
                    "এই মডেলে structured output নেই — সাধারণ JSON-এ আবার চেষ্টা।"
                ) from exc
            # HTTP 400 INVALID_ARGUMENT যদি thinking বা schema-র কারণে আসে (যেখানে এরর বার্তায়
            # 'thinking'/'schema' শব্দটি সরাসরি নাও থাকতে পারে), সেগুলো বন্ধ করে আবার চেষ্টা।
            if not is_auth and (
                status == 400
                or "invalid_argument" in message
                or "invalid argument" in message
                or "bad request" in message
            ):
                if self._use_thinking:
                    self._use_thinking = False
                    raise _CapabilityError(
                        "HTTP 400 পাওয়া গেছে — thinking_config বাদ দিয়ে আবার চেষ্টা।"
                    ) from exc
                if self._use_schema and mode == "json":
                    self._use_schema = False
                    raise _CapabilityError(
                        "HTTP 400 পাওয়া গেছে — response_schema বাদ দিয়ে সাধারণ JSON-এ আবার চেষ্টা।"
                    ) from exc
            raise

        truncated = self._finish_reason(response) == "MAX_TOKENS"
        try:
            raw = response.text
        except Exception as exc:  # কোনো candidate নেই (blocked বা empty)
            feedback = getattr(response, "prompt_feedback", None)
            block_reason = getattr(feedback, "block_reason", None)
            if block_reason:
                raise _ContentBlockedError(
                    f"Gemini নিরাপত্তার কারণে উত্তর ব্লক করেছে (block_reason: {block_reason})।"
                ) from exc
            raise _ResponseFormatError(f"Gemini-র উত্তর পড়া যায়নি: {exc}") from exc
        if not isinstance(raw, str) or not raw.strip():
            feedback = getattr(response, "prompt_feedback", None)
            block_reason = getattr(feedback, "block_reason", None)
            if block_reason:
                raise _ContentBlockedError(
                    f"Gemini নিরাপত্তার কারণে উত্তর ব্লক করেছে (block_reason: {block_reason})।"
                )
            detail = f" (prompt_feedback: {feedback})" if feedback else ""
            raise _ResponseFormatError(f"Gemini খালি উত্তর দিয়েছে{detail}")

        try:
            if mode == "lines":
                return parse_numbered_lines(raw, len(items), require_numbered=True)
            return parse_translations(
                raw, len(items), require_ids=len(items) > 1
            )
        except _ResponseFormatError as exc:
            recovered = salvage_translations(
                raw, len(items), require_ids=len(items) > 1
            )
            if 0 < len(recovered) < len(items):
                reason = "উত্তর মাঝপথে কেটে গেছে" if truncated else "উত্তর অসম্পূর্ণ"
                raise _PartialBatchError(recovered, len(items), reason) from exc
            if truncated:
                self._truncated_hits += 1
                logger.warning(
                    "✂️ উত্তর output-token সীমায় কেটে গেছে (%d লাইন, বাজেট %d টোকেন) — "
                    "ব্যাচ ছোট করে আবার চেষ্টা।",
                    len(items), kwargs["max_output_tokens"],
                )
            raise

    @staticmethod
    def _finish_reason(response: Any) -> str:
        """উত্তর কেন থেমেছে (``STOP`` / ``MAX_TOKENS`` …) — নিরাপদে পড়া হয়।"""
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return ""
        reason = getattr(candidates[0], "finish_reason", "")
        name = getattr(reason, "name", None)
        return str(name or reason or "").upper()

    def _retry_delay(self, attempt: int, exc: BaseException, status: Optional[int]) -> float:
        """Use the same configured delay for every error, subject to the legacy cap."""
        return min(self.retry_base_delay, self.retry_cap_delay)

    def _call_with_retry(
        self,
        items: Sequence[Tuple[str, float]],
        instruction: str = SYSTEM_INSTRUCTION,
        task: str = "Translate each line into Bangla.",
    ) -> List[str]:
        """Try once, then retry every error up to ``max_retries`` times."""
        last_error: Optional[BaseException] = None
        format_failures = 0
        translated: List[str] = []
        remaining = list(items)

        for retry in range(self.max_retries + 1):
            mode = self.FORMAT_MODES[min(format_failures, len(self.FORMAT_MODES) - 1)]
            try:
                if retry == 0:
                    self._rate_limiter.wait()
                else:
                    # The retry delay governs retries; RPM still paces new batches.
                    self._rate_limiter.record_retry()
                self._api_calls += 1
                return translated + self._generate(
                    remaining, instruction=instruction, task=task, mode=mode
                )
            except _PartialBatchError as exc:
                if 0 < len(exc.translations) < len(remaining):
                    translated.extend(exc.translations)
                    remaining = remaining[len(exc.translations):]
                    format_failures = 0
                    last_error = exc
                    logger.warning(
                        "♻️ %dটি লাইন পাওয়া গেছে — বাকি %dটি আবার চাওয়া হবে।",
                        len(exc.translations), len(remaining),
                    )
                else:
                    last_error = _ResponseFormatError(str(exc))
                    format_failures += 1
            except _CapabilityError as exc:
                last_error = exc
                logger.info("ℹ️ %s", exc)
            except _ResponseFormatError as exc:
                last_error = exc
                format_failures += 1
                logger.warning(
                    "⚠️ Gemini উত্তর সঠিক ফরম্যাটে আসেনি (চেষ্টা %d/%d, বিন্যাস=%s) — %s",
                    retry + 1, self.max_retries + 1, mode, _error_message(exc, 300),
                )
            except Exception as exc:  # noqa: BLE001 - API and network failures share the policy
                last_error = exc
                status = _status_code(exc)
                msg_lower = _error_message(exc).lower()
                is_auth = status in (401, 403) or any(h in msg_lower for h in _AUTH_HINTS)
                if not is_auth and (
                    status in (400, 404)
                    or "not found" in msg_lower
                    or "not supported" in msg_lower
                    or "invalid_argument" in msg_lower
                    or "invalid argument" in msg_lower
                ):
                    fallback_model = self._next_fallback_model()
                    if fallback_model:
                        logger.warning(
                            "⚠️ মডেল '%s'-এ ত্রুটি (status=%s) — '%s' দিয়ে আবার চেষ্টা হবে।",
                            self.model, status, fallback_model,
                        )
                        self.model = fallback_model
                        self._use_schema = True
                        format_failures = 0
                logger.error(
                    "❌ Gemini API error (চেষ্টা %d/%d) | %s | status=%s | %s",
                    retry + 1, self.max_retries + 1, type(exc).__name__,
                    status if status is not None else "-", _error_message(exc),
                )

            if retry < self.max_retries:
                delay = self._retry_delay(retry + 1, last_error, _status_code(last_error))
                logger.warning(
                    "⏳ %g সেকেন্ড পর retry %d/%d হবে…",
                    delay, retry + 1, self.max_retries,
                )
                if delay > 0:
                    time.sleep(delay)

        final = (
            f"Gemini প্রথম অনুরোধের পর {self.max_retries} বার retry করেও ব্যর্থ হয়েছে "
            f"(model={self.model}): {_error_message(last_error) if last_error else 'অজানা ত্রুটি'}। "
            "API key, মডেল, নেটওয়ার্ক ও quota যাচাই করুন।"
        )
        logger.error("🛑 %s — এই অংশের কাজ ব্যর্থ হলো।", final)
        if translated and isinstance(last_error, _ResponseFormatError):
            # Keep already validated translations when the remainder needs splitting.
            raise _PartialBatchError(translated, len(items), final) from last_error
        if isinstance(last_error, _ResponseFormatError):
            raise _ResponseFormatError(final) from last_error
        raise TranslationError(final) from last_error
