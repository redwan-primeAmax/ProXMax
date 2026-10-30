"""Translation exceptions, retry classification, and request pacing."""

from __future__ import annotations

import re
import threading
import time
from typing import List, Optional

from .translator_config import (
    _AUTH_HINTS, _NETWORK_HINTS, _NETWORK_MODULES, _RETRYABLE_STATUS, _RETRY_DELAY_RE,
)

class TranslationError(RuntimeError):
    """Raised when translation fails beyond recovery."""

class GeminiAuthError(TranslationError):
    """API key সেট না থাকলে অনুরোধ পাঠানোর আগেই জানানো হয়।"""

class _ResponseFormatError(TranslationError):
    """Gemini-র উত্তর প্রত্যাশিত ফরম্যাটে না এলে (retry যোগ্য)।"""

class _PartialBatchError(_ResponseFormatError):
    """উত্তর কাঁচা/কাটা, কিন্তু শুরুর কয়েকটি লাইন ঠিকই পাওয়া গেছে।"""

    def __init__(self, translations: List[str], expected: int, detail: str = "") -> None:
        self.translations = list(translations)
        self.expected = int(expected)
        super().__init__(
            f"{len(self.translations)}/{expected} লাইন পাওয়া গেছে"
            + (f" — {detail}" if detail else "")
        )

class _CapabilityError(TranslationError):
    """মডেল কোনো প্যারামিটার না চিনলে সেটি বাদ দিয়ে retry করা হয়।"""

class _ContentBlockedError(TranslationError):
    """Gemini নিরাপত্তার কারণে কোনো অংশ অনুবাদ করতে না চাইলে।"""

def _status_code(exc: BaseException) -> Optional[int]:
    """কোনো exception থেকে HTTP status code বের করার সর্বাত্মক চেষ্টা।"""
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value
    match = re.search(r"\b(400|401|403|404|408|409|425|429|500|502|503|504)\b", str(exc))
    if match:
        return int(match.group(1))
    return None

def _error_message(exc: BaseException, limit: int = 800) -> str:
    """Exception-এর পূর্ণ (কিন্তু সীমিত) বার্তা — terminal log-এর জন্য।"""
    text = str(exc).strip() or repr(exc)
    return text if len(text) <= limit else text[:limit] + "…"

def _retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Server-এর বলা অপেক্ষার সময় (``RetryInfo`` / ``Retry-After``) বের করে।"""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            header = headers.get("retry-after")
            if header:
                return max(0.0, float(header))
        except (TypeError, ValueError):
            pass

    details = getattr(exc, "details", None)
    if details is not None:
        try:
            haystack = details if isinstance(details, str) else json.dumps(details, default=str)
        except (TypeError, ValueError):
            haystack = str(details)
        match = _RETRY_DELAY_RE.search(haystack)
        if match:
            try:
                return max(0.0, float(match.group("seconds")))
            except ValueError:
                return None
    return None

def _is_retryable(exc: BaseException, status: Optional[int]) -> bool:
    """এই error-এ আবার চেষ্টা করা অর্থবহ কি না।"""
    if status is not None:
        if status in _RETRYABLE_STATUS:
            return True
        if 400 <= status < 500:
            return False
    lowered = _error_message(exc).lower()
    if any(hint in lowered for hint in _AUTH_HINTS):
        return False
    if any(hint in lowered for hint in _NETWORK_HINTS):
        return True
    module = (type(exc).__module__ or "").split(".")[0]
    if module in _NETWORK_MODULES:
        return True
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))

class _RateLimiter:
    """Simple, thread-safe requests-per-minute limiter."""

    def __init__(self, requests_per_minute: int) -> None:
        rpm = max(1, int(requests_per_minute))
        self._interval = 60.0 / rpm
        self._next_allowed = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Reserve a normally paced request without blocking retry bookkeeping."""
        with self._lock:
            now = time.monotonic()
            scheduled = max(now, self._next_allowed)
            self._next_allowed = scheduled + self._interval
        if scheduled > now:
            time.sleep(scheduled - now)

    def record_retry(self) -> None:
        """Record an unpaced retry so later new requests still respect the RPM interval."""
        with self._lock:
            self._next_allowed = max(self._next_allowed, time.monotonic() + self._interval)

    def refund(self) -> None:
        """Legacy helper; actual retries are recorded rather than refunded."""
        with self._lock:
            self._next_allowed = max(0.0, self._next_allowed - self._interval)
