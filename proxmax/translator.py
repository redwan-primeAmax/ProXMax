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
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import DEFAULT_GEMINI_MODEL
from .hotwords import load_hotwords, matched_hotwords_for_texts
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

#: প্রাথমিক মডেল কোনো কারণে 400/404 দিলে যেসব স্ট্যাবল মডেলে স্বয়ংক্রিয়ভাবে চেষ্টা করা হবে
FALLBACK_GEMINI_MODELS: Tuple[str, ...] = (
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
)

#: Gemini-কে দেওয়া নির্দেশনা — শুধু অনুবাদের কাজ, বাড়তি কিছু নয়
SYSTEM_INSTRUCTION = (
    "You are a professional English-to-Bengali (Bangla) dubbing writer for YouTube "
    "videos. Translate each input line into Bangla that sounds like a real person "
    "speaking, not like a textbook. "
    "Each input line has a speaking time in seconds ('d'); aim for a line that a "
    "narrator reads comfortably in about d seconds at a normal conversational pace "
    "(roughly the number of Bangla characters given as 'c'). "
    "Rules: "
    "1) ALWAYS finish every sentence with proper Bangla punctuation (। ? !) — never end "
    "a line mid-word or mid-phrase, even if that makes it slightly longer than 'c'. "
    "A complete, natural sentence is more important than the exact character count. "
    "Do not come in far BELOW 'c' either: if your line would be much shorter than c, "
    "say the same thing with fuller everyday phrasing (a natural connective, a spoken "
    "filler like তো/আসলে/মানে, or the implied detail spelled out) so the speech fills "
    "the time. Filling the slot is better than leaving a long silence. "
    "2) If a line must be shortened, compress by choosing shorter everyday words and "
    "dropping filler; never drop facts, numbers, names or the main idea. "
    "3) Keep football player names, club names, FC Mobile event names, and game terms "
    "(OVR, TOTY, UTOTY, TOTS, UTOTS, Icon, Heroes, Star Pass, Mascherano, Skill Boost, "
    "VSA, H2H, Ballon d'Or) accurate — write player names and terms in Bangla script "
    "the way Bangla speakers say them (e.g. স্টার পাস, ওভিআর, ব্যালন ডি'অর, মেসি, রোনালদো, "
    "এমবাপ্পে), but do not turn ordinary English verbs into Bangla-letter "
    "transliterations (translate 'giveaway' as ভাগ করে দেওয়া/উপহার, not গিভঅ্যাওয়ে). "
    "4) Numbers take as long to say as their Bengali words (125 is spoken "
    "'একশ পঁচিশ'), so count a number as its spoken length, not its digits. "
    "5) Use conversational Bangla as spoken by young Bangladeshi speakers; short "
    "sentences, natural rhythm, no literary or overly formal words. "
    "6) Never add, remove, merge, split, summarise or explain content; keep the same "
    "number of lines in the same order. "
    "7) If a line is already Bangla or empty, return it as is. "
    "Return only the translations."
)

EXPAND_INSTRUCTION = (
    "You lengthen Bengali (Bangla) dubbing lines that are too SHORT for their "
    "speaking slot. Each input line's 't' is the existing Bangla line, 'd' is its "
    "speaking time in seconds and 'c' roughly how many Bangla characters fit in it. "
    "Rewrite each line so that spoken at a "
    "natural pace it fills close to d seconds (about c characters), WITHOUT adding any "
    "new fact, number, name, opinion or promise. Do it the way a live commentator "
    "fills time: natural spoken connectives (তো, আসলে, মানে, দেখুন, বন্ধুরা), a "
    "descriptive detail that is already implied, or a fuller but equivalent phrasing. "
    "Keep every fact of the original line, keep it one complete sentence ending with "
    "। ? or !, do not merge or split lines, keep the same number of lines in the same "
    "order. Never pad with meaningless repeated words. Return only the Bangla lines."
)

SHORTEN_INSTRUCTION = (
    "You shorten Bengali (Bangla) dubbing lines. Each input line has a target speaking "
    "time in seconds ('d') and roughly how many Bangla characters fit ('c'). Rewrite "
    "each line so it is spoken comfortably within d seconds, keeping the original "
    "meaning, tone, numbers and names. Remove filler and use shorter everyday words. "
    "Never cut a word or leave a sentence unfinished — every output line must be a "
    "complete sentence ending with ।, ? or !. Do not add new information; do not merge "
    "or split lines; keep the same number of lines in the same order. Return only the "
    "shortened Bangla lines."
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
_AUTH_HINTS = (
    "api key", "api_key", "apikey", "permission_denied", "permission denied",
    "unauthenticated", "credential", "forbidden", "api_key_invalid",
)
#: সাময়িক নেটওয়ার্ক সমস্যার সংকেত — এগুলো retry করলে কাজ হতে পারে
_NETWORK_HINTS = (
    "timeout", "timed out", "connection", "connect", "temporarily",
    "unavailable", "deadline", "network", "ssl", "reset", "eof", "broken pipe",
)
#: এই module-গুলোর exception মানেই সাধারণত নেটওয়ার্ক সমস্যা
_NETWORK_MODULES = frozenset({"httpx", "httpcore", "requests", "urllib3", "aiohttp", "socket"})

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_JSON_START_RE = re.compile(r"[\[{]")
#: বাংলা হরফ (অনুবাদের ফল অবশ্যই এগুলোতে থাকবে)
_BANGLA_RE = re.compile(r"[\u0980-\u09FF]")
_RETRY_DELAY_RE = re.compile(r"retryDelay\"?\s*[:=]\s*\"?(?P<seconds>\d+(?:\.\d+)?)s", re.IGNORECASE)


class TranslationError(RuntimeError):
    """Raised when translation fails beyond recovery."""


class GeminiAuthError(TranslationError):
    """API key না থাকলে বা অনুমতি না থাকলে — retry করে লাভ নেই।"""


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
    """এই মডেল/সংস্করণ কোনো প্যারামিটার চেনে না (যেমন thinking/schema/400)।

    ভুলটা অনুরোধের *বিন্যাসে*, Gemini-র উত্তরে নয় — প্যারামিটারটি বাদ দিয়ে
    সঙ্গে সঙ্গে আবার চেষ্টা করা হয়, কোনো অপেক্ষা ছাড়াই।
    """


class _ContentBlockedError(TranslationError):
    """Gemini নিরাপত্তার কারণে কোনো অংশ অনুবাদ করতে না চাইলে (retry করে লাভ নেই)।"""


# =============================================================================
# Helpers
# =============================================================================
def normalise_text(text: Optional[str]) -> str:
    """Whitespace ঠিক করে এক লাইনের টেক্সট বানায় (``None`` → ``""``)."""
    return " ".join((text or "").split())


def _strip_fences(text: str) -> str:
    """Markdown code-fence (```json … ```) সরিয়ে ফেলে — যেখানেই থাকুক।"""
    cleaned = _FENCE_RE.sub("", text or "")
    cleaned = re.sub(r"```[A-Za-z]*", "", cleaned)
    return cleaned.replace("```", "").strip()


def _json_blobs(text: str) -> List[str]:
    """টেক্সট থেকে সম্ভাব্য JSON টুকরোগুলো (প্রথম ``{``/``[`` থেকে শুরু)।"""
    cleaned = _strip_fences(text)
    blobs = [cleaned]
    match = _JSON_START_RE.search(cleaned)
    if match and match.start() > 0:
        blobs.append(cleaned[match.start():])
    return blobs


def _decode_json(text: str) -> Any:
    """JSON ডিকোড করার সর্বাত্মক চেষ্টা (পেছনে বাড়তি লেখা থাকলেও চলে)।"""
    decoder = json.JSONDecoder()
    last_error: Optional[Exception] = None
    for blob in _json_blobs(text):
        if not blob:
            continue
        try:
            return json.loads(blob)
        except (TypeError, ValueError) as exc:
            last_error = exc
        start = _JSON_START_RE.search(blob)
        if start:
            try:
                value, _end = decoder.raw_decode(blob[start.start():])
                return value
            except (TypeError, ValueError) as exc:
                last_error = exc
    raise ValueError(str(last_error) if last_error else "JSON পাওয়া যায়নি")


_STRING_ITEM_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
_OBJECT_TEXT_KEYS = ("bn", "translation", "text", "bangla")


def _salvage_string_array(text: str) -> List[str]:
    """কাটা/অসম্পূর্ণ JSON থেকে **শুরুর সম্পূর্ণ** স্ট্রিংগুলো উদ্ধার করে।"""
    cleaned = _strip_fences(text)
    match = _JSON_START_RE.search(cleaned)
    if not match:
        return []
    body = cleaned[match.start():]

    if body.startswith("{"):
        key = re.search(r'"(?:translations|lines|output|outputs|results)"\s*:\s*\[', body)
        if not key:
            return []
        body = body[key.end() - 1:]

    dict_mode = bool(re.match(r'\[\s*\{', body))
    values = _STRING_ITEM_RE.findall(body)
    if dict_mode:
        values = [v for v in values if v.strip() not in _OBJECT_TEXT_KEYS]
    items: List[str] = []
    for value in values:
        try:
            items.append(json.loads(f'"{value}"').strip())
        except (TypeError, ValueError):
            items.append(value.strip())
    return [item for item in items if item]


_NUMBERED_LINE_RE = re.compile(r"^\s*(?:[-*•]\s*)?(?P<num>\d{1,3})\s*[.):\-–]\s*(?P<text>.+?)\s*$")


def _looks_like_bangla(lines: Sequence[str]) -> bool:
    """ফলাফলে অন্তত একটি বাংলা হরফ আছে কি না (মডেলের বাড়তি লেখা আলাদা করতে)।"""
    return any(_BANGLA_RE.search(line or "") for line in lines)


def parse_numbered_lines(raw: str, expected: int) -> List[str]:
    """সাধারণ নম্বরযুক্ত লাইন (``1. বাংলা``) থেকে অনুবাদ পড়ে।"""
    cleaned = _strip_fences(raw or "")
    cleaned = re.sub(r"^\s*```.*$", "", cleaned, flags=re.M)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    numbered: Dict[int, str] = {}
    for line in lines:
        match = _NUMBERED_LINE_RE.match(line)
        if match:
            numbered.setdefault(int(match.group("num")), match.group("text").strip())
    ordered = [numbered[index] for index in range(1, expected + 1)
               if index in numbered]
    if len(ordered) == expected and _looks_like_bangla(ordered):
        return ordered
    plain = [re.sub(r"^\s*(?:[-*•]\s*)", "", line).strip() for line in lines]
    plain = [line for line in plain if line]
    if len(plain) == expected and _looks_like_bangla(plain):
        return plain
    snippet = " ".join((raw or "").split())[:160]
    raise _ResponseFormatError(
        f"ফরম্যাট বোঝা যায়নি: {expected} লাইন দরকার, নম্বরযুক্ত পাওয়া গেছে "
        f"{len(numbered)}টি (মোট লাইন {len(lines)}টি) — উত্তর: {snippet!r}"
    )


def _values_from_json(data: Any) -> Any:
    """JSON থেকে অনুবাদের list-টি বের করে (dict/একাধিক কী সহনশীলভাবে)।"""
    values: Any = data
    if isinstance(data, dict):
        for key in ("translations", "lines", "output", "results", "outputs"):
            candidate = data.get(key)
            if isinstance(candidate, list):
                values = candidate
                break
        else:
            lists = [v for v in data.values() if isinstance(v, list)]
            values = lists[0] if len(lists) == 1 else data
    return values


def _normalise_items(values: Any) -> List[str]:
    """list-এর প্রতিটি item-কে স্ট্রিং-এ আনে (dict হলে text কী থেকে)।"""
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
    return translations


def parse_translations(raw: str, expected: int) -> List[str]:
    """Gemini-র উত্তর থেকে ঠিক ``expected`` সংখ্যক অনুবাদ বের করে।"""
    cleaned = _strip_fences(raw or "")
    try:
        data = _decode_json(cleaned)
    except ValueError:
        return parse_numbered_lines(cleaned, expected)

    translations = _normalise_items(_values_from_json(data))
    if translations and not _looks_like_bangla(translations):
        raise _ResponseFormatError(
            "উত্তরে বাংলা টেক্সট নেই (মডেল হয়তো অনুবাদ না করে অন্য কিছু লিখেছে)"
        )
    if len(translations) != expected:
        trimmed = list(translations)
        while len(trimmed) > expected and not trimmed[-1].strip():
            trimmed.pop()
        if len(trimmed) != expected:
            raise _ResponseFormatError(
                f"অনুবাদের সংখ্যা মেলেনি: দরকার {expected}, পাওয়া গেছে {len(trimmed)}"
            )
        translations = trimmed
    return translations


def salvage_translations(raw: str, expected: int) -> List[str]:
    """কাটা JSON থেকে শুরুর সম্পূর্ণ অনুবাদগুলো উদ্ধার করে (সর্বোচ্চ ``expected``)।"""
    recovered = [t for t in _salvage_string_array(raw) if t]
    return recovered[:expected]


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
        """পরবর্তী request-এর জন্য প্রয়োজনীয় সময় অপেক্ষা করে।"""
        with self._lock:
            now = time.monotonic()
            delay = self._next_allowed - now
            if delay > 0:
                time.sleep(delay)
                now = time.monotonic()
            self._next_allowed = max(now, self._next_allowed) + self._interval

    def refund(self) -> None:
        """ব্যর্থ (ফরম্যাট ভুল) request-এর স্লট ফিরিয়ে দেয়।"""
        with self._lock:
            self._next_allowed = max(0.0, self._next_allowed - self._interval)


# =============================================================================
# Translator interface
# =============================================================================
class BaseTranslator(ABC):
    """Common interface for all translation backends."""

    @abstractmethod
    def translate_text(self, text: str, duration: float = 0.0) -> str:
        """Translate a single English string into Bangla."""

    def translate_segments(self, segments: List[Segment]) -> List[Segment]:
        """Fill ``text_bn`` of every segment in-place (sequential default)."""
        for seg in iter_progress(segments, desc="  Translate", total=len(segments)):
            seg.text_bn = self.translate_text(seg.text_en, seg.duration)
        return segments

    def shorten_to_fit(self, segments: List[Segment], **_: Any) -> int:
        """সময়ে না আঁটা লাইন ছোট করে লেখায় (ডিফল্ট: কিছুই করে না)।"""
        return 0

    def expand_to_fit(self, segments: List[Segment], targets: Sequence[float]) -> int:
        """স্লটের চেয়ে ছোট লাইন ভরাট করে লেখায় (ডিফল্ট: কিছুই করে না)।"""
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
    """Gemini API translator with batching, duration hints and retries."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_GEMINI_MODEL,
        model_name: Optional[str] = None,
        batch_size: int = DEFAULT_GEMINI_BATCH_SIZE,
        max_chars_per_request: int = DEFAULT_MAX_CHARS_PER_REQUEST,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 2.0,
        retry_cap_delay: float = 2.0,
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
        self.batch_size = max(1, int(batch_size))
        self.max_chars_per_request = max(500, int(max_chars_per_request))
        self.max_retries = min(10, max(1, int(max_retries)))
        self.retry_base_delay = max(0.001, float(retry_base_delay))
        self.retry_max_delay = max(self.retry_base_delay, float(retry_max_delay))
        #: এক retry-তে সর্বোচ্চ অপেক্ষা (ব্যবহারকারীর নির্দেশ: ২ সেকেন্ডের বেশি নয়)
        self.retry_cap_delay = min(max(0.0, float(retry_cap_delay)), max(self.retry_max_delay, 0.0))
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
        else:
            self.hotwords_list = []

        self._api_key = resolved_key
        self._client = client
        self._client_lock = threading.Lock()
        self._rate_limiter = _RateLimiter(self.requests_per_minute)
        self._cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()
        self._fatal_error: Optional[str] = None
        self._api_calls = 0

    @property
    def model_name(self) -> str:
        return self.model

    @model_name.setter
    def model_name(self, value: str) -> None:
        self.model = value

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

    def _payload(self, items: Sequence[Tuple[str, float]]) -> str:
        """(text, duration) জোড়াকে JSON-এ রূপান্তর (মডেলের ইনপুট)।"""
        lines = [
            {
                "d": round(float(duration), 2),
                "c": max(8, int(round(float(duration) * self.char_rate))),
                "t": text,
            }
            for text, duration in items
        ]
        return json.dumps({"lines": lines}, ensure_ascii=False)

    #: ব্যর্থ হলে যে ক্রমে ফরম্যাট সহজ করা হয় (সবচেয়ে কড়া → সবচেয়ে সহজ)
    FORMAT_MODES = ("json", "json_plain", "lines")

    def _payload_numbered(self, items: Sequence[Tuple[str, float]]) -> str:
        """নম্বরযুক্ত সাধারণ টেক্সট — JSON ব্যর্থ হলে ব্যবহৃত নিরাপদ বিন্যাস।"""
        lines = [f"{i}. {text}" for i, (text, _d) in enumerate(items, start=1)]
        return "\n".join(lines)

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

    def _next_fallback_model(self) -> Optional[str]:
        """বর্তমান মডেল কাজ না করলে পরবর্তী স্ট্যাবল ব্যাকআপ মডেল নির্বাচন করে।"""
        for candidate in FALLBACK_GEMINI_MODELS:
            if candidate not in self._tried_models:
                self._tried_models.add(candidate)
                return candidate
        return None

    def _generate(
        self,
        items: Sequence[Tuple[str, float]],
        instruction: str = SYSTEM_INSTRUCTION,
        task: str = "Translate each line into Bangla.",
        mode: str = "json",
    ) -> List[str]:
        """এক request-এ একাধিক লাইন প্রসেস করে (retry ছাড়া)।"""
        from google.genai import types

        client = self._ensure_client()
        kwargs: Dict[str, Any] = {
            "system_instruction": instruction,
            "temperature": self.temperature if mode == "json" else 0.0,
            "max_output_tokens": self._output_token_budget(items),
        }
        glossary_hint = ""
        if self.hotwords_list:
            matched = matched_hotwords_for_texts(
                [text for text, _d in items], self.hotwords_list, max_matches=35
            )
            if matched:
                glossary_hint = (
                    f"\nRecognized football/FC Mobile terms in this batch "
                    f"(keep names/terms accurate in Bangla): {', '.join(matched)}."
                )

        count_hint = (
            f"IMPORTANT: the input has exactly {len(items)} lines — return exactly "
            f"{len(items)} items, one per line, in the same order.{glossary_hint}"
        )
        if mode == "json":
            contents = f"{task}\n{count_hint}\n{self._payload(items)}"
            if self._use_schema:
                kwargs["response_mime_type"] = "application/json"
                kwargs["response_schema"] = _RESPONSE_SCHEMA
        elif mode == "json_plain":
            contents = (
                f"{task}\n"
                f'Return ONLY a JSON object like {{"translations": ["<bangla 1>", "<bangla 2>"]}} '
                f"with exactly {len(items)} strings in the same order. No other text.\n"
                f"{count_hint}\nInput lines:\n{self._payload(items)}"
            )
        else:
            contents = (
                f"{task}\n"
                f"Answer with exactly {len(items)} numbered lines, each like "
                f"`1. <bangla translation>`. No JSON, no explanation.\n"
                f"{count_hint}\nInput lines:\n{self._payload_numbered(items)}"
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
            # 'thinking'/'schema' শব্দটি সরাসরি নাও থাকতে পারে), তাৎক্ষণিকভাবে সেগুলো বন্ধ করে আবার চেষ্টা।
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
                return parse_numbered_lines(raw, len(items))
            return parse_translations(raw, len(items))
        except _ResponseFormatError as exc:
            recovered = salvage_translations(raw, len(items))
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

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------
    def _retry_delay(self, attempt: int, exc: BaseException, status: Optional[int]) -> float:
        """পরের চেষ্টার আগে অপেক্ষা — **সর্বোচ্চ ২ সেকেন্ড**, দ্বিগুণ নয়।"""
        if isinstance(exc, _ResponseFormatError):
            return random.uniform(0.0, 0.2)
        hinted = _retry_after_seconds(exc)
        if hinted is not None:
            delay = hinted
        else:
            delay = self.retry_base_delay * (1.5 ** (attempt - 1))
            if status == 429:
                delay = max(delay, 1.0)
        return min(max(delay, 0.0), self.retry_cap_delay)

    def _call_with_retry(
        self,
        items: Sequence[Tuple[str, float]],
        instruction: str = SYSTEM_INSTRUCTION,
        task: str = "Translate each line into Bangla.",
    ) -> List[str]:
        """ব্যর্থ হলে (প্রায়) সঙ্গে সঙ্গে আবার চেষ্টা; শেষে পরিষ্কার error দেয়।"""
        if self._fatal_error:
            raise TranslationError(self._fatal_error)

        last_error: Optional[BaseException] = None
        format_failures = 0
        same_request = False  # True = আগের request-এরই পুনরায় চেষ্টা (নতুন নয়)
        for attempt in range(1, self.max_retries + 1):
            mode = self.FORMAT_MODES[min(format_failures, len(self.FORMAT_MODES) - 1)]
            try:
                if not same_request:
                    self._rate_limiter.wait()
                same_request = False
                self._api_calls += 1
                return self._generate(items, instruction=instruction, task=task, mode=mode)
            except _CapabilityError as exc:
                last_error = exc
                logger.info("ℹ️ %s", exc)
                same_request = True
                self._rate_limiter.refund()
                continue
            except _ContentBlockedError as exc:
                logger.error("🛑 %s — এই অংশের কাজ বাদ দেওয়া হলো।", exc)
                raise TranslationError(str(exc)) from exc
            except _PartialBatchError:
                raise
            except _ResponseFormatError as exc:
                last_error = exc
                format_failures += 1
                logger.warning(
                    "⚠️ Gemini উত্তর সঠিক ফরম্যাটে আসেনি (চেষ্টা %d/%d, বিন্যাস=%s) — %s",
                    attempt, self.max_retries, mode, _error_message(exc, 300),
                )
                if attempt < self.max_retries:
                    same_request = True
                    self._rate_limiter.refund()
                    delay = self._retry_delay(attempt, exc, None)
                    if delay >= 0.05:
                        time.sleep(delay)
            except Exception as exc:  # noqa: BLE001 - সব API/নেটওয়ার্ক ত্রুটি এক জায়গায়
                last_error = exc
                status = _status_code(exc)
                msg_lower = _error_message(exc).lower()
                is_auth = any(h in msg_lower for h in _AUTH_HINTS)

                # মডেল নাম 400/404 দিলে এবং API key বৈধ থাকলে স্ট্যাবল ব্যাকআপ মডেলে সুইচ করা
                if (
                    not is_auth
                    and (
                        status in (400, 404)
                        or "not found" in msg_lower
                        or "not supported" in msg_lower
                        or "invalid_argument" in msg_lower
                        or "invalid argument" in msg_lower
                    )
                ):
                    fallback_model = self._next_fallback_model()
                    if fallback_model:
                        logger.warning(
                            "⚠️ মডেল '%s'-এ ত্রুটি (status=%s) — স্বয়ংক্রিয়ভাবে '%s' মডেলে পরিবর্তন করে আবার চেষ্টা করা হচ্ছে...",
                            self.model, status, fallback_model,
                        )
                        self.model = fallback_model
                        self._use_schema = True
                        same_request = True
                        self._rate_limiter.refund()
                        continue

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
                    if delay > 0:
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

    # Alias for callers using translate()
    translate = translate_segments

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


# =============================================================================
# Backend 2 (optional): IndicTrans2 (AI4Bharat) — offline, En→Bn specialist
# =============================================================================
class IndicTrans2Translator(BaseTranslator):
    """IndicTrans2 local model — trained specifically for Indic languages."""

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

    unload = close


# =============================================================================
# Factory & verification helper
# =============================================================================
def create_translator(
    backend: str = "gemini",
    *,
    gemini_api_key: str = "",
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    gemini_batch_size: int = DEFAULT_GEMINI_BATCH_SIZE,
    gemini_max_chars_per_request: int = DEFAULT_MAX_CHARS_PER_REQUEST,
    gemini_max_retries: int = DEFAULT_MAX_RETRIES,
    gemini_retry_base_delay: float = 0.5,
    gemini_retry_max_delay: float = 2.0,
    gemini_retry_cap_delay: float = 2.0,
    gemini_requests_per_minute: int = 15,
    gemini_temperature: float = 0.2,
    gemini_timeout: int = 120,
    gemini_max_output_tokens: int = 8192,
    char_rate: float = DEFAULT_CHARS_PER_SECOND,
    indictrans2_model: str = "ai4bharat/indictrans2-en-indic-1B",
    hotwords: str = "",
    hotwords_file: Optional[str | Path] = None,
) -> BaseTranslator:
    """Factory: config-এর backend নাম থেকে translator তৈরি করে।"""
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
            retry_cap_delay=gemini_retry_cap_delay,
            requests_per_minute=gemini_requests_per_minute,
            temperature=gemini_temperature,
            timeout=gemini_timeout,
            max_output_tokens=gemini_max_output_tokens,
            char_rate=char_rate,
            hotwords=hotwords,
            hotwords_file=hotwords_file,
        )
    if name == "indictrans2":
        return IndicTrans2Translator(model_name=indictrans2_model)
    raise ValueError(
        f"অজানা translator backend: '{backend}' (সমর্থিত: gemini, indictrans2)"
    )


def verify_gemini_api(
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_GEMINI_MODEL,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Convenience helper to immediately test a Gemini API key in Colab."""
    translator = GeminiTranslator(
        api_key=api_key,
        model=model_name,
        max_retries=3,
        timeout=timeout,
    )
    try:
        return translator.verify_api()
    finally:
        translator.close()
