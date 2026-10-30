"""Validation rules for :class:`proxmax.config.AppConfig`."""

from __future__ import annotations

from pathlib import Path

from .config import SUPPORTED_TTS_ENGINES, SUPPORTED_TRANSLATORS, SUPPORTED_WHISPER_MODELS
from .translator_config import MAX_GEMINI_BATCH_SIZE, MAX_GEMINI_RETRIES
from .utils import get_logger

logger = get_logger(__name__)


def validate_config(config):
    """Validate settings and return the same config, or raise ``ValueError``."""
    self = config
    if "/" not in self.whisper_model and self.whisper_model not in SUPPORTED_WHISPER_MODELS:
        raise ValueError(
            f"অজানা Whisper মডেল: '{self.whisper_model}'। "
            f"সমর্থিত: {', '.join(SUPPORTED_WHISPER_MODELS)}"
        )
    if not 5.0 <= float(self.chunk_max_seconds) <= 29.5:
        raise ValueError(
            f"chunk_max_seconds ৫–২৯.৫ এর মধ্যে হতে হবে (Whisper ৩০s-এর বেশি দেখতে "
            f"পারে না), পাওয়া গেছে: {self.chunk_max_seconds}"
        )
    if not 1.0 <= float(self.chunk_min_seconds) <= float(self.chunk_max_seconds):
        raise ValueError(
            "chunk_min_seconds অবৈধ: ১ থেকে chunk_max_seconds-এর মধ্যে হতে হবে।"
        )
    if not 0 <= int(self.carry_context_chars) <= 4000:
        raise ValueError("carry_context_chars ০–৪০০০ এর মধ্যে হতে হবে।")
    if self.seam_window_seconds <= 0:
        raise ValueError("seam_window_seconds ধনাত্মক হতে হবে।")
    if self.whisper_beam_size < 1:
        raise ValueError("whisper_beam_size অন্তত 1 হতে হবে।")

    if self.translator_backend not in SUPPORTED_TRANSLATORS:
        raise ValueError(
            f"translator_backend হতে হবে {SUPPORTED_TRANSLATORS}, পাওয়া গেছে: "
            f"'{self.translator_backend}'"
        )
    if not (self.gemini_model or "").strip():
        raise ValueError("gemini_model খালি হতে পারে না (ডিফল্ট: gemini-3.5-flash-lite)।")
    if not 1 <= self.gemini_max_retries <= MAX_GEMINI_RETRIES:
        raise ValueError(
            f"gemini_max_retries 1–{MAX_GEMINI_RETRIES} এর মধ্যে হতে হবে, "
            f"পাওয়া গেছে: {self.gemini_max_retries}"
        )
    if int(self.gemini_batch_size) < 1:
        raise ValueError("gemini_batch_size অন্তত 1 হতে হবে।")
    if int(self.gemini_batch_size) > MAX_GEMINI_BATCH_SIZE:
        logger.warning(
            "gemini_batch_size=%s নিরাপদ paragraph limit %d-এ কমানো হয়েছে।",
            self.gemini_batch_size, MAX_GEMINI_BATCH_SIZE,
        )
        self.gemini_batch_size = MAX_GEMINI_BATCH_SIZE
    if int(self.gemini_max_chars_per_request) < 0:
        raise ValueError("gemini_max_chars_per_request ০ অথবা ধনাত্মক হতে হবে।")
    if int(self.gemini_max_input_tokens) < 1000:
        raise ValueError(
            "gemini_max_input_tokens অন্তত 1000 হতে হবে, যাতে prompt নির্দেশনা ও একটি সম্পূর্ণ chunk পাঠানো যায়।"
        )
    if self.gemini_requests_per_minute < 1:
        raise ValueError("gemini_requests_per_minute অন্তত 1 হতে হবে।")
    if not 0.0 <= self.gemini_temperature <= 2.0:
        raise ValueError("gemini_temperature 0.0–2.0 এর মধ্যে হতে হবে।")
    if self.gemini_timeout < 10:
        raise ValueError("gemini_timeout অন্তত 10 সেকেন্ড হতে হবে।")
    if not 0.0 <= float(self.gemini_retry_cap_delay) <= 5.0:
        raise ValueError(
            "gemini_retry_cap_delay ০–৫ সেকেন্ডের মধ্যে হতে হবে "
            f"(পাওয়া গেছে: {self.gemini_retry_cap_delay})"
        )
    if self.gemini_retry_base_delay <= 0 or self.gemini_retry_max_delay < self.gemini_retry_base_delay:
        raise ValueError(
            "retry delay অবৈধ: gemini_retry_base_delay > 0 এবং "
            "gemini_retry_max_delay >= gemini_retry_base_delay হতে হবে।"
        )
    if not 2.0 <= float(self.tts_unit_max_seconds) <= 5.0:
        raise ValueError(
            "tts_unit_max_seconds ২–৫ এর মধ্যে হতে হবে (অনুবাদ chunk সর্বোচ্চ ৫ সেকেন্ড), "
            f"পাওয়া গেছে: {self.tts_unit_max_seconds}"
        )
    if not 40 <= int(self.tts_unit_max_chars) <= 800:
        raise ValueError(
            "tts_unit_max_chars ৪০–৮০০ এর মধ্যে হতে হবে "
            f"(পাওয়া গেছে: {self.tts_unit_max_chars})"
        )
    if not 0.0 <= float(self.fill_extension) <= 0.5:
        raise ValueError(
            "sync.fill_extension ০–০.৫ এর মধ্যে হতে হবে "
            f"(পাওয়া গেছে: {self.fill_extension})"
        )
    if not 0.0 <= float(self.tts_unit_max_gap) <= 2.0:
        raise ValueError(
            "tts_unit_max_gap ০–২ সেকেন্ডের মধ্যে হতে হবে "
            f"(পাওয়া গেছে: {self.tts_unit_max_gap})"
        )
    if not 6.0 <= float(self.translation_char_rate) <= 30.0:
        raise ValueError(
            f"translation_char_rate ৬–৩০ অক্ষর/সেকেন্ডের মধ্যে হতে হবে, "
            f"পাওয়া গেছে: {self.translation_char_rate}"
        )

    if self.tts_engine not in SUPPORTED_TTS_ENGINES:
        raise ValueError(
            f"tts_engine হতে হবে {SUPPORTED_TTS_ENGINES}, পাওয়া গেছে: '{self.tts_engine}'"
        )

    if not (0.90 <= self.min_tempo <= 1.0 < self.max_tempo <= 1.15):
        raise ValueError(
            f"tempo সীমা অবৈধ: min_tempo={self.min_tempo}, max_tempo={self.max_tempo} "
            f"(শর্ত: 0.90 <= min_tempo <= 1 < max_tempo <= 1.15)"
        )
    if self.slot_gap_ms < 0 or self.borrow_silence_ms < 0:
        raise ValueError("slot_gap_ms ও borrow_silence_ms ঋণাত্মক হতে পারবে না।")
    if self.sync_sample_rate < 8000:
        raise ValueError("sync_sample_rate অন্তত ৮০০০ হতে হবে।")
    if not 0.0 <= float(self.audio_validation_min_coverage) <= 1.0:
        raise ValueError("quality.min_text_coverage ০–১ এর মধ্যে হতে হবে।")

    if Path(self.input_dir) == Path(self.output_dir):
        raise ValueError("input_dir ও output_dir একই হতে পারবে না।")
    if self.tts_gender not in ("male", "female", "custom"):
        raise ValueError(
            "tts.gender হতে হবে male | female | custom "
            f"(পাওয়া গেছে: {self.tts_gender})"
        )
    if self.tts_gender != "custom" and not self.resolved_tts_voice.startswith("bn-"):
        raise ValueError(
            f"tts.gender={self.tts_gender} অনুযায়ী ভয়েস পাওয়া গেল না "
            f"(tts_voice={self.tts_voice!r})।"
        )
    if not self.resolved_tts_voice.startswith("bn-"):
        logger.warning(
            "TTS ভয়েস '%s' Bangla (bn-*) নয় — ডাবিংয়ের জন্য bn-BD ভয়েস সুপারিশ করা হয়।",
            self.resolved_tts_voice,
        )
    if self.translator_backend == "gemini" and not self.resolved_gemini_api_key():
        from . import config as config_module
        if not config_module._MISSING_KEY_WARNED:
            config_module._MISSING_KEY_WARNED = True
            logger.warning(
                "Gemini API key এখনো সেট হয়নি — নোটবুকের 'Gemini API key' সেলে "
                "GEMINI_API_KEY বসান, নইলে প্রোগ্রাম শুরু হতে পারবে না।"
            )
    return self
