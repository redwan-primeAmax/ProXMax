"""Gemini prompt, batching, and response-format constants."""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple

DEFAULT_GEMINI_BATCH_SIZE = 2048

DEFAULT_MAX_CHARS_PER_REQUEST = 0  # token budget supersedes the legacy character cap

DEFAULT_MAX_RETRIES = 7

DURATION_TOLERANCE = 0.10

FALLBACK_GEMINI_MODELS: Tuple[str, ...] = (
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
)

SYSTEM_INSTRUCTION = (
    "You are a professional English-to-Bengali (Bangla) dubbing writer for YouTube "
    "videos. Each input paragraph is one complete translation chunk. Translate it "
    "into natural spoken Bangla, not textbook language. Do not include timestamps, "
    "duration labels, or explanations. "
    "Rules: "
    "1) Keep every paragraph separate and in the same order; never merge, split, "
    "summarise, or omit a paragraph. Return exactly one translation per paragraph. "
    "2) Preserve every fact, number, name, opinion, and the main idea. Use concise, "
    "complete sentences; never end mid-word or mid-phrase. Finish sentences with "
    "proper Bangla punctuation (। ? !). "
    "3) Keep football player names, club names, FC Mobile event names, and game terms "
    "(OVR, TOTY, UTOTY, TOTS, UTOTS, Icon, Heroes, Star Pass, Mascherano, Skill Boost, "
    "VSA, H2H, Ballon d'Or) accurate — write player names and terms in Bangla script "
    "the way Bangla speakers say them (e.g. স্টার পাস, ওভিআর, ব্যালন ডি'অর, মেসি, রোনালদো, "
    "এমবাপ্পে), but do not transliterate ordinary English verbs (translate 'giveaway' "
    "as ভাগ করে দেওয়া/উপহার, not গিভঅ্যাওয়ে). "
    "4) Numbers should be written so a Bangladeshi Bangla voice can pronounce them "
    "naturally; preserve their exact value. "
    "5) Use conversational Bangla as spoken by young Bangladeshi speakers, with "
    "natural rhythm and no literary or overly formal wording. "
    "6) If a paragraph is already Bangla or empty, return it as is. "
    "Return only the translations in the requested structured format."
)

EXPAND_INSTRUCTION = (
    "Do not lengthen or pad Bengali (Bangla) dubbing lines. Keep each translation "
    "concise, complete, and faithful to its meaning. Return only the Bangla paragraphs."
)

SHORTEN_INSTRUCTION = (
    "Make each Bengali (Bangla) dubbing paragraph more concise while preserving its "
    "meaning, tone, numbers, names, and all key facts. Remove filler and prefer shorter "
    "everyday wording. Never cut a word or leave a sentence unfinished; every output "
    "must be a complete sentence ending with ।, ? or !. Do not add information or merge "
    "or split paragraphs. Return exactly one shortened Bangla paragraph per input, in "
    "the same order, with no timestamps or explanations."
)

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

_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

_AUTH_HINTS = (
    "api key", "api_key", "apikey", "permission_denied", "permission denied",
    "unauthenticated", "credential", "forbidden", "api_key_invalid",
)

_NETWORK_HINTS = (
    "timeout", "timed out", "connection", "connect", "temporarily",
    "unavailable", "deadline", "network", "ssl", "reset", "eof", "broken pipe",
)

_NETWORK_MODULES = frozenset({"httpx", "httpcore", "requests", "urllib3", "aiohttp", "socket"})

_STRING_ITEM_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')

_OBJECT_TEXT_KEYS = ("bn", "translation", "text", "bangla")

_NUMBERED_LINE_RE = re.compile(r"^\s*(?:[-*•]\s*)?(?P<num>\d{1,3})\s*[.):\-–]\s*(?P<text>.+?)\s*$")

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

_JSON_START_RE = re.compile(r"[\[{]")

_BANGLA_RE = re.compile(r"[\u0980-\u09FF]")

_RETRY_DELAY_RE = re.compile(r"retryDelay\"?\s*[:=]\s*\"?(?P<seconds>\d+(?:\.\d+)?)s", re.IGNORECASE)
