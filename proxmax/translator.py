"""Translation: English → Bangla with the Google Gemini API (duration-aware).

ডিফল্ট (এবং একমাত্র প্রধান) backend হলো **Gemini API** — ``google-genai``
SDK দিয়ে। পুরনো Google Translate (``deep-translator``) সম্পূর্ণ সরিয়ে
দেওয়া হয়েছে।

Backend দুটি:

* ``gemini`` (ডিফল্ট) — Google Gemini API।
* ``indictrans2`` (ঐচ্ছিক) — AI4Bharat-এর IndicTrans2 লোকাল মডেল।

এই সংস্করণে অনুবাদ **সময়-সচেতন (duration-aware)**:

* প্রতিটি লাইনের সাথে তার সময়সীমা (``d`` সেকেন্ড) পাঠানো হয়, আর
  অনুরোধ করা হয় এমন বাংলা যেন স্বাভাবিক গতিতে পড়লে ওই সময়ে শেষ হয়
  (±১০%)। ফলে পরে অডিওকে ১.৫–১.৯ গুণ দ্রুত করার দরকার পড়ে না —
  কণ্ঠ আর "রোবট/চিপমাঙ্ক" শোনায় না।
* বাংলা পাংচুয়েশন (। ? ,) ঠিকভাবে চাওয়া হয়, তাই SRT ও TTS দুটোই
  স্বাভাবিক বাক্যে চলে।
* একই টেক্সট একবারই API-তে যায় (dedupe) এবং response schema থাকায়
  উত্তর সবসময় বৈধ JSON।
* কোনো লাইন সময়ে না আঁটলে :meth:`GeminiTranslator.shorten_to_fit`
  দিয়ে শুধু সেই লাইনগুলোই আরেকবার ছোট করে লেখানো হয়।

Error handling: প্রতিটি ব্যর্থ call-এ terminal-এ পূর্ণ error log, server
যে অপেক্ষার সময় বলে (``RetryInfo``/``Retry-After``) সেটি সহ exponential
backoff, সর্বোচ্চ ``gemini_max_retries`` (ডিফল্ট ৭) বার চেষ্টা। সব
চেষ্টা ব্যর্থ হলে পরিষ্কার বাংলা error দিয়ে নিয়ন্ত্রিতভাবে থামে।
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import DEFAULT_GEMINI_MODEL
from .models import Segment
from .text_utils import (
    DEFAULT_CHARS_PER_SECOND,
    estimate_speech_seconds,
    normalise_bangla,
)
from .utils import get_logger, iter_progress

logger = get_logger(__name__)

#: এক request-এ সর্বোচ্চ কতটি লাইন (config দিয়ে বদলানো যায়)
DEFAULT_GEMINI_BATCH_SIZE = 25
#: এক request-এ মোট English অক্ষরের নিরাপদ সীমা
DEFAULT_MAX_CHARS_PER_REQUEST = 12_000
#: ডিফল্ট retry সংখ্যা (ব্যবহারকারীর নির্দেশ অনুযায়ী ৬–৭)
DEFAULT_MAX_RETRIES = 7
#: সময়-বাজেটের সহনশীলতা (±১০%)
DURATION_TOLERANCE = 0.10

#: Gemini-কে দেওয়া নির্দেশনা — শুধু অনুবাদের কাজ, বাড়তি কিছু নয়
SYSTEM_INSTRUCTION = (
    "You are a professional English to Bengali (Bangla) dubbing translator. "
    "Every input line has a speaking time in seconds ('d'). Translate each line into "
    "natural, spoken Bangla that a narrator can read within about d seconds at a normal "
    "pace (roughly 14 Bangla characters per second), keeping within plus or minus ten "
    "percent of d. Prefer shorter wording over losing meaning. Never add, remove, merge, "
    "split, summarise or explain content, and keep the same number of lines in the same "
    "order. Keep names, numbers, units, code and technical terms accurate. Use proper "
    "Bangla punctuation (। for a full stop, ? and ! where needed, commas for short "
    "pauses). If a line is already Bangla or empty, return it as is."
)

SHORTEN_INSTRUCTION = (
    "You shorten Bengali (Bangla) lines for dubbing. Each input line has a target "
    "speaking time in seconds ('d'). Rewrite each line so it can be spoken within d "
    "seconds at a normal pace, keeping the original meaning, tone and facts. Remove "
    "filler words; do not add new information; do not merge or split lines; keep the "
    "same number of lines in the same order. Return only the shortened Bangla lines."
)

#: Gemini থেকে সবসময় বৈধ JSON পাওয়ার জন্য response schema
_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "translations": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        }
    },
    "required": ["translations"],
}

#: HTTP status যেগুলো আবার চেষ্টা করলে সফল হতে পারে
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
#: API key / permission সংক্রান্ত সংকেত (এগুলো retry করলে লাভ নেই)
_AUTH_HINTS = ("api key", "api_key", "permission", "unauthenticated", "credential", "forbidden")
#: সাময়িক নেটওয়ার্ক সমস্যার সংকেত — এগুলো retry করলে কাজ হতে পারে
_NETWORK_HINTS = (
    "timeout", "timed out", "connection", "connect", "temporarily",
    "unavailable", "deadline", "network", "ssl", "reset", "eof", "broken pipe",
)
#: এই module-গুলোর exception মানেই সাধারণত নেটওয়ার্ক সমস্যা
_NETWORK_MODULES = frozenset({"httpx", "httpcore", "requests", "urllib3", "aiohttp", "socket"})

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_RETRY_DELAY_RE = re.compile(r"retryDelay\"?\s*[:=]\s*\"?(?P<seconds>\d+(?:\.\d+)?)s", re.IGNORECASE)


class TranslationError(RuntimeError):
    """Raised when translation fails beyond recovery."""


class GeminiAuthError(TranslationError):
    """API key না থাকলে বা অনুমতি না থাকলে — retry করে লাভ নেই।"""


class _ResponseFormatError(TranslationError):
    """Gemini-র উত্তর প্রত্যাশিত JSON ফরম্যাটে না এলে (retry যোগ্য)।"""


class _ContentBlockedError(TranslationError):
    """Gemini নিরাপত্তার কারণে কোনো অংশ অনুবাদ করতে না চাইলে (retry করে লাভ নেই)।"""


# =============================================================================
# Helpers
# =============================================================================
def normalise_text(text: Optional[str]) -> str:
    """Whitespace ঠিক করে এক লাইনের টেক্সট বানায় (``None`` → ``""``)."""
    return " ".join((text or "").split())


def parse_translations(raw: str, expected: int) -> List[str]:
    """Gemini-র JSON উত্তর থেকে ঠিক ``expected`` সংখ্যক অনুবাদ বের করে।

    মডেল মাঝে মাঝে কোড-ফেন্স বা অতিরিক্ত কী দেয়; সেগুলো সহনশীলভাবে
    সামলে নেওয়া হয়। সংখ্যা না মিললে :class:`_ResponseFormatError`
    ওঠে, যাতে caller ব্যাচ ভেঙে আবার চেষ্টা করতে পারে।

    Args:
        raw: মডেলের ফেরত দেওয়া টেক্সট (JSON)।
        expected: ইনপুটে কতটি লাইন ছিল।

    Raises:
        _ResponseFormatError: অবৈধ JSON বা লাইন-সংখ্যা না মিললে।
    """
    cleaned = _FENCE_RE.sub("", raw or "").strip()
    try:
        data = json.loads(cleaned)
    except (TypeError, ValueError) as exc:
        raise _ResponseFormatError(
            f"Gemini বৈধ JSON দেয়নি (প্রথম ২০০ অক্ষর: {cleaned[:200]!r})"
        ) from exc

    values: Any = data
    if isinstance(data, dict):
        for key in ("translations", "lines", "output", "results"):
            candidate = data.get(key)
            if isinstance(candidate, list):
                values = candidate
                break
        else:
            lists = [v for v in data.values() if isinstance(v, list)]
            values = lists[0] if len(lists) == 1 else data

    if not isinstance(values, list):
        raise _ResponseFormatError(
            f"Gemini-র উত্তরে অনুবাদের list পাওয়া যায়নি (পাওয়া গেছে: {type(values).__name__})"
        )

    translations: List[str] = []
    for item in values:
        if isinstance(item, str):
            translations.append(item.strip())
        elif isinstance(item, dict):
            text = next(
                (item[k] for k in ("bn", "translation", "text", "bangla") if isinstance(item.get(k), str)),
                None,
            )
            if text is None:
                raise _ResponseFormatError("Gemini-র উত্তরের একটি item-এ টেক্সট নেই")
            translations.append(text.strip())
        else:
            raise _ResponseFormatError(
                f"Gemini-র উত্তরের item অবৈধ টাইপের: {type(item).__name__}"
            )

    if len(translations) != expected:
        raise _ResponseFormatError(
            f"অনুবাদের সংখ্যা মেলেনি: দরকার {expected}, পাওয়া গেছে {len(translations)}"
        )
    return translations


def _status_code(exc: BaseException) -> Optional[int]:
    """কোনো exception থেকে HTTP status code বের করার সর্বাত্মক চেষ্টা।"""
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


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
        """পরবর্তী request-এর জন্য প্রয়োজনীয় সময় অপেক্ষা করে।"""
        with self._lock:
            now = time.monotonic()
            delay = self._next_allowed - now
            if delay > 0:
                time.sleep(delay)
                now = time.monotonic()
            self._next_allowed = max(now, self._next_allowed) + self._interval


# =============================================================================
# Translator interface
# =============================================================================
class BaseTranslator(ABC):
    """Common interface for all translation backends."""

    @abstractmethod
    def translate_text(self, text: str, duration: float = 0.0) -> str:
        """Translate a single English string into Bangla.

        Args:
            text: English লাইন।
            duration: ওই লাইনটি বলতে কত সেকেন্ড পাওয়া যাবে (০ = অজানা)।
        """

    def translate_segments(self, segments: List[Segment]) -> List[Segment]:
        """Fill ``text_bn`` of every segment in-place (sequential default)."""
        for seg in iter_progress(segments, desc="  Translate", total=len(segments)):
            seg.text_bn = self.translate_text(seg.text_en, seg.duration)
        return segments

    def shorten_to_fit(self, segments: List[Segment], **_: Any) -> int:
        """সময়ে না আঁটা লাইন ছোট করে লেখায় (ডিফল্ট: কিছুই করে না)।"""
        return 0

    def describe(self) -> str:
        """Human-readable backend description (log-এর জন্য)।"""
        return type(self).__name__

    def close(self) -> None:
        """Release backend resources (client, model ইত্যাদি)। ডিফল্ট: কিছুই নয়।"""


# =============================================================================
# Backend 1 (default): Google Gemini API
# =============================================================================
class GeminiTranslator(BaseTranslator):
    """Gemini API translator with batching, duration hints and retries.

    Args:
        api_key: Gemini API key। না দিলে ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``।
        model: মডেলের নাম (ডিফল্ট ``gemini-3.5-flash-lite``)।
        batch_size: এক request-এ সর্বোচ্চ কতটি লাইন।
        max_chars_per_request: এক request-এ সর্বোচ্চ অক্ষর।
        max_retries: সর্বোচ্চ কতবার চেষ্টা (ডিফল্ট ৭)।
        retry_base_delay: প্রথম retry-এর আগে অপেক্ষা (সেকেন্ড)।
        retry_max_delay: এক retry-তে সর্বোচ্চ অপেক্ষা (সেকেন্ড)।
        requests_per_minute: rate limit (ডিফল্ট ১৫ RPM)।
        temperature: সৃজনশীলতা (কম = বেশি নির্ভুল)।
        timeout: এক request-এর timeout (সেকেন্ড)।
        max_output_tokens: উত্তরে সর্বোচ্চ টোকেন।
        char_rate: বাংলা কথার গতি (অক্ষর/সেকেন্ড) — সময় অনুমানের জন্য।
        client: টেস্টের জন্য প্রস্তুত client inject করার সুযোগ।

    Raises:
        GeminiAuthError: API key কোথাও পাওয়া না গেলে।
        ValueError: বাকি প্যারামিটার অবৈধ হলে।
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_GEMINI_MODEL,
        batch_size: int = DEFAULT_GEMINI_BATCH_SIZE,
        max_chars_per_request: int = DEFAULT_MAX_CHARS_PER_REQUEST,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay: float = 4.0,
        retry_max_delay: float = 90.0,
        requests_per_minute: int = 15,
        temperature: float = 0.2,
        timeout: int = 120,
        max_output_tokens: int = 8192,
        char_rate: float = DEFAULT_CHARS_PER_SECOND,
        client: Any = None,
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
        model = normalise_text(model)
        if not model:
            raise ValueError("Gemini মডেলের নাম খালি হতে পারে না।")

        self.model = model
        self.batch_size = max(1, int(batch_size))
        self.max_chars_per_request = max(500, int(max_chars_per_request))
        self.max_retries = min(10, max(1, int(max_retries)))
        self.retry_base_delay = max(0.5, float(retry_base_delay))
        self.retry_max_delay = max(self.retry_base_delay, float(retry_max_delay))
        self.requests_per_minute = max(1, int(requests_per_minute))
        self.temperature = min(2.0, max(0.0, float(temperature)))
        self.timeout = max(10, int(timeout))
        self.max_output_tokens = max(256, int(max_output_tokens))
        self.char_rate = max(6.0, float(char_rate))

        self._api_key = resolved_key
        self._client = client
        self._client_lock = threading.Lock()
        self._rate_limiter = _RateLimiter(self.requests_per_minute)
        self._cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()
        self._fatal_error: Optional[str] = None
        self._api_calls = 0

    # ------------------------------------------------------------------
    # SDK / client
    # ------------------------------------------------------------------
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

            # SDK-এর নিজস্ব retry বন্ধ — আমাদের স্পষ্ট, log-সহ retry নীতিই দেখা যাবে
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

    @staticmethod
    def _payload(items: Sequence[Tuple[str, float]]) -> str:
        """(text, duration) জোড়াকে JSON-এ রূপান্তর (মডেলের ইনপুট)।"""
        lines = [{"d": round(float(duration), 2), "t": text} for text, duration in items]
        return json.dumps({"lines": lines}, ensure_ascii=False)

    def _generate(
        self,
        items: Sequence[Tuple[str, float]],
        instruction: str = SYSTEM_INSTRUCTION,
        task: str = "Translate each line into Bangla.",
    ) -> List[str]:
        """এক request-এ একাধিক লাইন প্রসেস করে (retry ছাড়া)।"""
        from google.genai import types

        client = self._ensure_client()
        contents = f"{task}\n{self._payload(items)}"
        config = types.GenerateContentConfig(
            system_instruction=instruction,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
        )

        response = client.models.generate_content(
            model=self.model, contents=contents, config=config
        )
        try:
            raw = response.text
        except Exception as exc:  # কোনো candidate নেই (blocked বা empty)
            feedback = getattr(response, "prompt_feedback", None)
            block_reason = getattr(feedback, "block_reason", None)
            if block_reason:
                # নিরাপত্তা-ব্লক — বারবার retry করলে শুধু সময় নষ্ট হবে
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
        return parse_translations(raw, len(items))

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------
    def _retry_delay(self, attempt: int, exc: BaseException, status: Optional[int]) -> float:
        """Exponential backoff + jitter; server-এর বলা সময়কে প্রাধান্য দেয়।"""
        hinted = _retry_after_seconds(exc)
        if hinted is not None:
            delay = hinted + 1.0
        else:
            delay = self.retry_base_delay * (2 ** (attempt - 1))
            if status == 429:
                delay *= 1.5
            delay += random.uniform(0.0, min(3.0, delay * 0.1))
        return min(max(delay, 0.5), self.retry_max_delay)

    def _call_with_retry(
        self,
        items: Sequence[Tuple[str, float]],
        instruction: str = SYSTEM_INSTRUCTION,
        task: str = "Translate each line into Bangla.",
    ) -> List[str]:
        """ব্যর্থ হলে অপেক্ষা করে আবার চেষ্টা; শেষে পরিষ্কার error দেয়।"""
        if self._fatal_error:
            raise TranslationError(self._fatal_error)

        last_error: Optional[BaseException] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self._rate_limiter.wait()
                self._api_calls += 1
                return self._generate(items, instruction=instruction, task=task)
            except _ContentBlockedError as exc:
                logger.error("🛑 %s — এই অংশের কাজ বাদ দেওয়া হলো।", exc)
                raise TranslationError(str(exc)) from exc
            except _ResponseFormatError as exc:
                last_error = exc
                logger.error(
                    "⚠️ Gemini উত্তর সঠিক ফরম্যাটে আসেনি (চেষ্টা %d/%d) — %s",
                    attempt, self.max_retries, _error_message(exc),
                )
                if attempt < self.max_retries:
                    delay = self._retry_delay(attempt, exc, None)
                    logger.warning(
                        "⏳ %.1fs অপেক্ষা করে আবার চেষ্টা করা হবে (%d/%d)…",
                        delay, attempt + 1, self.max_retries,
                    )
                    time.sleep(delay)
            except Exception as exc:  # noqa: BLE001 - সব API/নেটওয়ার্ক ত্রুটি এক জায়গায়
                last_error = exc
                status = _status_code(exc)
                retryable = _is_retryable(exc, status)
                logger.error(
                    "❌ Gemini API error (চেষ্টা %d/%d) | %s | status=%s | %s",
                    attempt, self.max_retries, type(exc).__name__,
                    status if status is not None else "-",
                    _error_message(exc),
                )
                if not retryable:
                    message = (
                        f"Gemini API-তে অপুনরুদ্ধারযোগ্য ত্রুটি ({type(exc).__name__}, "
                        f"status={status if status is not None else '-'}): {_error_message(exc)}\n"
                        f"মডেল: {self.model} | API key ঠিক আছে কিনা এবং মডেলের নাম "
                        f"বৈধ কিনা যাচাই করুন।"
                    )
                    self._fatal_error = message
                    logger.error("🛑 %s", message)
                    raise TranslationError(message) from exc
                if attempt < self.max_retries:
                    delay = self._retry_delay(attempt, exc, status)
                    logger.warning(
                        "⏳ %.1fs অপেক্ষা করে আবার চেষ্টা করা হবে (%d/%d)…",
                        delay, attempt + 1, self.max_retries,
                    )
                    time.sleep(delay)

        final = (
            f"Gemini {self.max_retries} বার চেষ্টা করেও ব্যর্থ হয়েছে "
            f"(model={self.model}): {_error_message(last_error) if last_error else 'অজানা ত্রুটি'}"
        )
        logger.error("🛑 %s — এই অংশের কাজ ব্যর্থ হলো।", final)
        if isinstance(last_error, _ResponseFormatError):
            raise _ResponseFormatError(final) from last_error
        raise TranslationError(final) from last_error

    def _run_batches(
        self,
        items: Sequence[Tuple[str, float]],
        instruction: str = SYSTEM_INSTRUCTION,
        task: str = "Translate each line into Bangla.",
        description: str = "  Translate (Gemini)",
        depth: int = 0,
    ) -> List[str]:
        """সব ব্যাচ চালিয়ে ফলাফল একই ক্রমে ফেরত দেয় (format ভুল হলে ভাগ করে)।"""
        results: List[str] = []
        for batch in self._make_batches(items):
            try:
                results.extend(
                    self._call_with_retry(batch, instruction=instruction, task=task)
                )
            except _ResponseFormatError:
                if len(batch) <= 1 or depth >= 4:
                    raise
                middle = len(batch) // 2
                logger.warning("🔀 %d লাইনের ব্যাচ দুই ভাগে ভেঙে আবার চেষ্টা…", len(batch))
                results.extend(
                    self._run_batches(batch[:middle], instruction, task, description, depth + 1)
                )
                results.extend(
                    self._run_batches(batch[middle:], instruction, task, description, depth + 1)
                )
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
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
        """সব segment অনুবাদ করে; timestamp অপরিবর্তিত থাকে।

        একই টেক্সট একাধিকবার থাকলে একবারই API-তে যায় (cache ও dedupe)।

        Raises:
            TranslationError: কোনো ব্যাচ সর্বোচ্চ retry-এর পরেও ব্যর্থ হলে।
        """
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

        logger.info(
            "Gemini অনুবাদ শেষ: %d segment | নতুন লাইন: %d | API call: %d | cache: %d | সময় %.1fs",
            len(segments), len(items), self._api_calls - calls_before,
            len(self._cache), time.perf_counter() - started,
        )
        return segments

    def shorten_to_fit(
        self,
        segments: List[Segment],
        max_tempo: float = 1.25,
        slot_gap: float = 0.08,
        borrow: float = 0.25,
    ) -> int:
        """সময়ে না আঁটা বাংলা লাইনগুলো ছোট করে আবার লেখায়।

        প্রতি segment-এর হাতে থাকা সময় = পরের segment শুরুর আগ পর্যন্ত
        (``slot_gap`` বাদে) + ``borrow``। যদি ``max_tempo``-তেও লাইনটি
        না আঁটে, তবেই সেটি পুনরায় লেখানো হয় — সাধারণত ১০–২০% লাইনে।

        Returns:
            কতটি লাইন ছোট করে লেখানো হলো।
        """
        ordered = sorted([s for s in segments if (s.text_bn or "").strip()], key=lambda s: s.start)
        offenders: List[Segment] = []
        targets: List[float] = []
        for i, seg in enumerate(ordered):
            next_start = ordered[i + 1].start if i + 1 < len(ordered) else None
            available = (
                (next_start - seg.start - slot_gap) if next_start is not None else seg.duration
            )
            budget = max(0.4, (available + borrow) * max_tempo)
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


# =============================================================================
# Backend 2 (optional): IndicTrans2 (AI4Bharat) — offline, En→Bn specialist
# =============================================================================
class IndicTrans2Translator(BaseTranslator):
    """IndicTrans2 local model — trained specifically for Indic languages.

    প্রয়োজন: ``pip install -r requirements-indictrans2.txt``
    """

    _SRC_TAG = "eng_Latn"
    _TGT_TAG = "ben_Beng"

    def __init__(
        self,
        model_name: str = "ai4bharat/indictrans2-en-indic-1B",
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise TranslationError(
                "IndicTrans2 backend-এর জন্য প্রয়োজনীয় প্যাকেজ নেই। আগে চালান:\n"
                "    pip install -r requirements-indictrans2.txt"
            ) from exc

        IndicProcessor = None
        for import_path in (
            "IndicTransToolkit.processor",
            "IndicTransToolkit.IndicTransToolkit.processor",
        ):
            try:
                module = __import__(import_path, fromlist=["IndicProcessor"])
                IndicProcessor = module.IndicProcessor
                break
            except Exception:
                continue
        if IndicProcessor is None:
            raise TranslationError(
                "IndicTransToolkit পাওয়া যায়নি। ইনস্টল করুন:\n"
                "    pip install -r requirements-indictrans2.txt"
            )

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "IndicTrans2 লোড হচ্ছে: %s | device=%s (প্রথমবার ডাউনলোডে সময় লাগবে)…",
            self.model_name, self._device,
        )
        self._processor = IndicProcessor(inference=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
        ).to(self._device)
        self._model.eval()
        logger.info("IndicTrans2 প্রস্তুত।")

    def translate_text(self, text: str, duration: float = 0.0) -> str:
        original = normalise_text(text)
        return normalise_bangla(self._translate_batch([original])[0]) if original else ""

    def translate_segments(self, segments: List[Segment]) -> List[Segment]:
        if not segments:
            return segments
        started = time.perf_counter()
        texts = [seg.text_en for seg in segments]
        for i in iter_progress(
            range(0, len(texts), self.batch_size),
            desc="  Translate(IndicTrans2)",
            total=(len(texts) + self.batch_size - 1) // self.batch_size,
            unit="batch",
        ):
            batch = texts[i : i + self.batch_size]
            for seg, translated in zip(segments[i : i + self.batch_size], self._translate_batch(batch)):
                seg.text_bn = normalise_bangla(translated)
        logger.info(
            "IndicTrans2 অনুবাদ শেষ: %d segment (%.1fs)।",
            len(segments), time.perf_counter() - started,
        )
        return segments

    def _translate_batch(self, texts: List[str]) -> List[str]:
        self._load()
        import torch

        processed = self._processor.preprocess_batch(
            texts, src_lang=self._SRC_TAG, tgt_lang=self._TGT_TAG
        )
        encoded = self._tokenizer(
            processed, truncation=True, padding="longest",
            return_tensors="pt", return_attention_mask=True,
        ).to(self._device)
        with torch.no_grad():
            generated = self._model.generate(
                **encoded,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )
        with self._tokenizer.as_target_tokenizer():
            decoded = self._tokenizer.batch_decode(
                generated.detach().cpu().tolist(),
                skip_special_tokens=True, clean_up_tokenization_spaces=True,
            )
        return self._processor.postprocess_batch(decoded, lang=self._TGT_TAG)

    def describe(self) -> str:
        return f"indictrans2:{self.model_name}"

    def close(self) -> None:
        """Unload the local model and free GPU/CPU memory."""
        from .utils import release_cuda_memory

        self._model = None
        self._tokenizer = None
        self._processor = None
        release_cuda_memory()


# =============================================================================
# Factory
# =============================================================================
def create_translator(
    backend: str = "gemini",
    *,
    gemini_api_key: str = "",
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    gemini_batch_size: int = DEFAULT_GEMINI_BATCH_SIZE,
    gemini_max_chars_per_request: int = DEFAULT_MAX_CHARS_PER_REQUEST,
    gemini_max_retries: int = DEFAULT_MAX_RETRIES,
    gemini_retry_base_delay: float = 4.0,
    gemini_retry_max_delay: float = 90.0,
    gemini_requests_per_minute: int = 15,
    gemini_temperature: float = 0.2,
    gemini_timeout: int = 120,
    gemini_max_output_tokens: int = 8192,
    char_rate: float = DEFAULT_CHARS_PER_SECOND,
    indictrans2_model: str = "ai4bharat/indictrans2-en-indic-1B",
) -> BaseTranslator:
    """Factory: config-এর backend নাম থেকে translator তৈরি করে।

    Raises:
        ValueError: অজানা backend নাম দিলে।
        GeminiAuthError: Gemini backend-এ API key না থাকলে।
    """
    name = normalise_text(backend).lower() or "gemini"
    if name == "gemini":
        return GeminiTranslator(
            api_key=gemini_api_key,
            model=gemini_model,
            batch_size=gemini_batch_size,
            max_chars_per_request=gemini_max_chars_per_request,
            max_retries=gemini_max_retries,
            retry_base_delay=gemini_retry_base_delay,
            retry_max_delay=gemini_retry_max_delay,
            requests_per_minute=gemini_requests_per_minute,
            temperature=gemini_temperature,
            timeout=gemini_timeout,
            max_output_tokens=gemini_max_output_tokens,
            char_rate=char_rate,
        )
    if name == "indictrans2":
        return IndicTrans2Translator(model_name=indictrans2_model)
    raise ValueError(
        f"অজানা translator backend: '{backend}' (সমর্থিত: gemini, indictrans2)"
    )
