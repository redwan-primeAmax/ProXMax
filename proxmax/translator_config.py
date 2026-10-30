"""Gemini prompt, batching, and response-format constants."""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple

DEFAULT_GEMINI_BATCH_SIZE = 25

DEFAULT_MAX_CHARS_PER_REQUEST = 12_000

DEFAULT_MAX_RETRIES = 7

DURATION_TOLERANCE = 0.10

FALLBACK_GEMINI_MODELS: Tuple[str, ...] = (
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
)

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
