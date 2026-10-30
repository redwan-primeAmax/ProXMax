"""Gemini prompt, batching, and response-format constants."""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple

# Keep response lists small enough that paragraph-to-output alignment is reliable.
MAX_GEMINI_BATCH_SIZE = 12
DEFAULT_GEMINI_BATCH_SIZE = 12

DEFAULT_MAX_CHARS_PER_REQUEST = 0  # token budget supersedes the legacy character cap

# Retry count is in addition to the initial request.
DEFAULT_MAX_RETRIES = 100
MAX_GEMINI_RETRIES = 100
DEFAULT_RETRY_DELAY = 1.0

DURATION_TOLERANCE = 0.10

FALLBACK_GEMINI_MODELS: Tuple[str, ...] = (
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
)

SYSTEM_INSTRUCTION = (
    "You are a professional English-to-Bengali (Bangla) dubbing writer for YouTube "
    "videos. Every labeled input paragraph is a separate target chunk of at most "
    "5 seconds; it may be an incomplete phrase because speech was split at a time "
    "boundary. Translate only the words in that paragraph into natural spoken Bangla. "
    "Neighboring paragraphs may be used only to understand names or context; never "
    "move, borrow, finish, or invent their content in this paragraph. Do not include "
    "timestamps, duration labels, or explanations. "
    "Rules: "
    "1) Preserve each paragraph's integer ID exactly. Return one translation for "
    "every ID, with no missing or duplicate IDs; never merge, split, reorder, "
    "summarise, or omit paragraphs. "
    "2) Preserve every fact, number, name, opinion, and the main idea. Be concise. "
    "If the source is a fragment, keep the translation a fragment; do not invent a "
    "sentence ending or complete it from another paragraph. Add Bangla punctuation "
    "only when the source chunk itself ends a sentence. "
    "3) Keep football player names, club names, FC Mobile event names, and game terms "
    "(OVR, TOTY, UTOTY, TOTS, UTOTS, Icon, Heroes, Star Pass, Mascherano, Skill Boost, "
    "VSA, H2H, Ballon d'Or) accurate — write player names and terms in Bangla script "
    "the way Bangla speakers say them (e.g. স্টার পাস, ওভিআর, ব্যালন ডি'অর, মেসি, রোনালদো, "
    "এমবাপ্পে), but do not transliterate ordinary English verbs (translate 'giveaway' "
    "as ভাগ করে দেওয়া/উপহার, not গিভঅ্যাওয়ে). "
    "4) Preserve all numbers and their exact value. Copy every number written as "
    "digits in the input as the same digits (Bangla or ASCII digits) in the output; "
    "never drop a digit, round it, or replace it with a different value. "
    "5) Use conversational Bangla as spoken by young Bangladeshi speakers, with "
    "natural rhythm and no literary or overly formal wording. Translate idioms and "
    "forms of address by meaning; 'ladies and gentlemen' is a greeting to an audience, "
    "not a person's name. Do not mistake ordinary words for names. "
    "Use Bangla script for Bangla words; do not mix in unrelated scripts or stray characters. "
    "6) If a paragraph is already Bangla or empty, return it as is. "
    "Return only the translations in the requested structured format, preserving IDs."
)

EXPAND_INSTRUCTION = (
    "Do not lengthen or pad Bengali (Bangla) dubbing lines. Keep each translation "
    "concise, complete, and faithful to its meaning. Return only the Bangla paragraphs."
)

SHORTEN_INSTRUCTION = (
    "Each labeled paragraph contains its English source and current Bangla translation. "
    "Use the source as evidence and rewrite the Bangla more concisely only if all "
    "meaning, names, numbers, negation, and key facts remain intact. Prefer natural "
    "Bangladeshi spoken wording and Bangla script; remove filler, not information. "
    "An optional target percentage is a rough length goal, not permission to omit "
    "content; return the original Bangla unchanged if it cannot be safely shortened. "
    "Keep all source digits unchanged. Do not complete a fragment, add information, "
    "or borrow from another paragraph. Preserve every ID; do not merge or split. "
    "Return only the Bangla translation for each ID, with no explanations."
)

_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "translations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "INTEGER"},
                    "translation": {"type": "STRING"},
                },
                "required": ["id", "translation"],
            },
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

_OBJECT_TEXT_KEYS = ("id", "paragraph_id", "index", "bn", "translation", "text", "bangla")

_NUMBERED_LINE_RE = re.compile(r"^\s*(?:[-*•]\s*)?(?P<num>\d{1,3})\s*[.):\-–]\s*(?P<text>.+?)\s*$")

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

_JSON_START_RE = re.compile(r"[\[{]")

_BANGLA_RE = re.compile(r"[\u0980-\u09FF]")

_RETRY_DELAY_RE = re.compile(r"retryDelay\"?\s*[:=]\s*\"?(?P<seconds>\d+(?:\.\d+)?)s", re.IGNORECASE)
