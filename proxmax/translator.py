"""Compatibility facade for ProXMax translation backends and helpers."""

from __future__ import annotations

import time  # Shared module object keeps legacy time.sleep patch points working.
from pathlib import Path
from typing import Any, Dict, Optional

from .config import DEFAULT_GEMINI_MODEL
from .gemini_translator import GeminiTranslator
from .indictrans2_translator import IndicTrans2Translator
from .models import Segment
from .text_utils import DEFAULT_CHARS_PER_SECOND
from .translation_errors import (
    GeminiAuthError, TranslationError, _CapabilityError, _ContentBlockedError,
    _PartialBatchError, _RateLimiter, _ResponseFormatError, _error_message,
    _is_retryable, _retry_after_seconds, _status_code,
)
from .translation_parsing import (
    normalise_text, _strip_fences, _json_blobs, _decode_json, _salvage_string_array,
    _looks_like_bangla, parse_numbered_lines, _values_from_json, _normalise_items,
    parse_translations, salvage_translations,
)
from .translator_base import BaseTranslator
from .translator_config import (
    DEFAULT_GEMINI_BATCH_SIZE, DEFAULT_MAX_CHARS_PER_REQUEST, DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY, DURATION_TOLERANCE, FALLBACK_GEMINI_MODELS, SYSTEM_INSTRUCTION,
    EXPAND_INSTRUCTION, SHORTEN_INSTRUCTION, _RESPONSE_SCHEMA, _RETRYABLE_STATUS, _AUTH_HINTS,
    _NETWORK_HINTS, _NETWORK_MODULES, _FENCE_RE, _JSON_START_RE, _BANGLA_RE,
    _RETRY_DELAY_RE, _NUMBERED_LINE_RE, _STRING_ITEM_RE, _OBJECT_TEXT_KEYS,
)
from .utils import get_logger

logger = get_logger(__name__)

def create_translator(
    backend: str = "gemini",
    *,
    gemini_api_key: str = "",
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    gemini_batch_size: int = DEFAULT_GEMINI_BATCH_SIZE,
    gemini_max_chars_per_request: int = DEFAULT_MAX_CHARS_PER_REQUEST,
    gemini_max_input_tokens: int = 5000,
    gemini_max_retries: int = DEFAULT_MAX_RETRIES,
    gemini_retry_base_delay: float = DEFAULT_RETRY_DELAY,
    gemini_retry_max_delay: float = DEFAULT_RETRY_DELAY,
    gemini_retry_cap_delay: float = DEFAULT_RETRY_DELAY,
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
            max_input_tokens=gemini_max_input_tokens,
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
        timeout=timeout,
    )
    try:
        return translator.verify_api()
    finally:
        translator.close()
