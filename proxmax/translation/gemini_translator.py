"""Concrete Gemini translator assembled from focused mixins."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import DEFAULT_GEMINI_MODEL
from .gemini_operations import GeminiOperationsMixin
from .gemini_protocol import GeminiProtocolMixin
from .gemini_repair import GeminiRepairMixin
from .hotwords import load_hotwords
from .text_utils import DEFAULT_CHARS_PER_SECOND
from .translation_errors import GeminiAuthError, TranslationError, _RateLimiter
from .translation_metrics import TranslationMetrics
from .reporting_runtime import record_event
from .translation_parsing import normalise_text
from .translator_base import BaseTranslator
from .translator_config import (
    DEFAULT_GEMINI_BATCH_SIZE, DEFAULT_MAX_CHARS_PER_REQUEST, DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY, FALLBACK_GEMINI_MODELS, MAX_GEMINI_BATCH_SIZE,
    MAX_GEMINI_RETRIES,
)
from .utils import get_logger

logger = get_logger(__name__)


class GeminiTranslator(
    GeminiOperationsMixin, GeminiRepairMixin, GeminiProtocolMixin, BaseTranslator
):
    """Gemini API translator with text-only paragraph batching and retries."""

    FORMAT_MODES = ("json", "json_plain", "lines")
    # Independent pipeline batches may run concurrently with shared pacing.
    supports_parallel_batches = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_GEMINI_MODEL,
        model_name: Optional[str] = None,
        batch_size: int = DEFAULT_GEMINI_BATCH_SIZE,
        max_chars_per_request: int = DEFAULT_MAX_CHARS_PER_REQUEST,
        max_input_tokens: int = 5000,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay: float = DEFAULT_RETRY_DELAY,
        retry_max_delay: float = DEFAULT_RETRY_DELAY,
        retry_cap_delay: float = DEFAULT_RETRY_DELAY,
        requests_per_minute: int = 15,
        temperature: float = 0.2,
        timeout: int = 120,
        max_output_tokens: int = 8192,
        char_rate: float = DEFAULT_CHARS_PER_SECOND,
        client: Any = None,
        hotwords: str = "",
        hotwords_file: Optional[str | Path] = None,
    ) -> None:
        resolved_key = normalise_text(
            api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        if not resolved_key and client is None:
            raise GeminiAuthError(
                "Gemini API key পাওয়া যায়নি। নোটবুকের 'Gemini কনফিগারেশন' সেলে "
                "GEMINI_API_KEY বসান, অথবা environment variable সেট করুন:\n"
                "    export GEMINI_API_KEY='your-key'\n"
                "API key নিন: https://aistudio.google.com/app/apikey"
            )
        chosen_model = normalise_text(model_name or model)
        if not chosen_model:
            raise ValueError("Gemini মডেলের নাম খালি হতে পারে না।")

        self.model = chosen_model
        requested_batch_size = max(1, int(batch_size))
        self.batch_size = min(requested_batch_size, MAX_GEMINI_BATCH_SIZE)
        if requested_batch_size > MAX_GEMINI_BATCH_SIZE:
            logger.warning(
                "Gemini batch_size=%d বড় — paragraph alignment-এর জন্য %d-এ সীমিত করা হয়েছে।",
                requested_batch_size, MAX_GEMINI_BATCH_SIZE,
            )
        self.max_chars_per_request = max(0, int(max_chars_per_request))
        self.max_input_tokens = max(1000, int(max_input_tokens))
        self.max_retries = min(MAX_GEMINI_RETRIES, max(1, int(max_retries)))
        self.retry_base_delay = max(0.001, float(retry_base_delay))
        self.retry_max_delay = max(self.retry_base_delay, float(retry_max_delay))
        # Legacy cap remains configurable, but retries use a fixed interval.
        self.retry_cap_delay = min(max(0.0, float(retry_cap_delay)), self.retry_max_delay)
        self.requests_per_minute = max(1, int(requests_per_minute))
        self.temperature = min(2.0, max(0.0, float(temperature)))
        self.timeout = max(10, int(timeout))
        self.max_output_tokens = max(256, int(max_output_tokens))
        self.char_rate = max(6.0, float(char_rate))
        #: এই গতি কোথা থেকে এলো — ``"config"`` (অনুমান) বা ``"measured"`` (মাপা)
        self.rate_source = "config"
        #: মডেল/SDK কোন কোন সুবিধা দেয় —
        #: ⚠️ thinking_budget=0 ডিফল্টে বন্ধ রাখা হয়েছে কারণ নতুন Gemini মডেলগুলোতে
        #: thinking_budget=0 পাঠালে HTTP 400 INVALID_ARGUMENT এরর আসে।
        self._use_schema = True
        self._use_thinking = False
        #: কতবার উত্তর output-token সীমায় কেটেছে (লগ/ডায়াগনস্টিকসের জন্য)
        self._truncated_hits = 0
        #: ফরম্যাট ঠিক না হওয়ায় কতটি লাইন বাদ পড়ল
        self._skipped_lines = 0
        self._tried_models: set[str] = {self.model}

        if hotwords_file is not None or hotwords:
            self.hotwords_list: List[str] = load_hotwords(
                hotwords_file, extra_hotwords=hotwords
            )
            logger.info(
                "Gemini glossary-তে %dটি local hotword আছে; request-এ শুধু text-এ মেলা term পাঠানো হবে (সর্বোচ্চ ২০টি/৬০০ অক্ষর)।",
                len(self.hotwords_list),
            )
        else:
            self.hotwords_list = []

        self._api_key = resolved_key
        self._client = client
        self._client_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._request_context = threading.local()
        self._rate_limiter = _RateLimiter(self.requests_per_minute)
        self._cache: Dict[str, str] = {}
        self._cache_notes: Dict[str, Tuple[str, ...]] = {}
        self._cache_lock = threading.Lock()
        self._api_calls = 0
        self.diagnostics = TranslationMetrics()

    @property
    def model_name(self) -> str:
        return self.model

    @model_name.setter
    def model_name(self, value: str) -> None:
        self.model = value

    def _next_fallback_model(self) -> Optional[str]:
        """বর্তমান মডেল কাজ না করলে পরবর্তী স্ট্যাবল ব্যাকআপ মডেল নির্বাচন করে।"""
        for candidate in FALLBACK_GEMINI_MODELS:
            if candidate not in self._tried_models:
                self._tried_models.add(candidate)
                return candidate
        return None

    def _paced_wait(self) -> None:
        started = time.perf_counter()
        try:
            stop = getattr(self._request_context, "stop", None)
            if stop is None:
                self._rate_limiter.wait()
            else:
                self._rate_limiter.wait(stop)
        finally:
            self.diagnostics.add("pacing_wait_seconds_sum", time.perf_counter() - started)

    def _check_cancelled(self) -> None:
        stop = getattr(self._request_context, "stop", None)
        if stop is not None and stop.is_set():
            raise TranslationError("Translation cancelled after another pipeline error")

    def translate_segments_cancellable(self, segments, stop: threading.Event):
        """Stop future retries if TTS or another translation batch has failed."""
        self._request_context.stop = stop
        try:
            return self.translate_segments(segments)
        finally:
            self._request_context.stop = None

    def _generate_attempt(self, items, *, instruction, task, mode, retry: bool):
        with self._state_lock:
            self._api_calls += 1
            self._request_context.model = self.model
        token = self.diagnostics.begin(retry)
        started = time.perf_counter()
        record_event("gemini_attempt_started", retry=retry, count=len(items))
        error = None
        try:
            return self._generate(items, instruction=instruction, task=task, mode=mode)
        except BaseException as exc:
            error = exc
            raise
        finally:
            elapsed = time.perf_counter() - started
            self.diagnostics.end(token, elapsed, error)
            record_event("gemini_attempt_finished", retry=retry, count=len(items), elapsed_seconds=elapsed,
                         status="failed" if error else "success",
                         error_type=type(error).__name__ if error else None)

    def _disable_capability(self, request_model: str, name: str) -> None:
        with self._state_lock:
            if self.model == request_model:
                setattr(self, name, False)

    def _fallback_after_failure(self, failed_model: str) -> Optional[str]:
        """Switch once per failed model even when two requests fail together."""
        with self._state_lock:
            if self.model != failed_model:
                return None  # another request already changed the model
            candidate = self._next_fallback_model()
            if candidate:
                self.model = candidate
                self._use_schema = True
            return candidate

    def describe(self) -> str:
        return f"gemini:{self.model}"

    def close(self) -> None:
        """Close the HTTP client and drop the reference."""
        client, self._client = self._client, None
        if client is not None and hasattr(client, "close"):
            try:
                client.close()
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("Gemini client বন্ধ করতে সমস্যা: %s", exc)

    unload = close
