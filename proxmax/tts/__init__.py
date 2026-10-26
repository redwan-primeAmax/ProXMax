"""TTS subpackage: built-in Bangla neural voice (edge-tts).

বিল্ট-ইন বাংলা neural ভয়েস (ডিফল্ট পুরুষ কণ্ঠ ``bn-BD-PradeepNeural``,
নারী কণ্ঠ ``bn-BD-NabanitaNeural``) ব্যবহার করে অডিও তৈরি হয় — কোনো
রেকর্ডিং, টোকেন বা আলাদা মডেল ডাউনলোড লাগে না।
"""

from __future__ import annotations

from typing import Optional

from ..config import AppConfig
from ..utils import get_logger
from .base import BaseTTSEngine, TTSError
from .edge_engine import BANGLA_VOICE_FEMALE, BANGLA_VOICE_MALE, EdgeTTSEngine

logger = get_logger(__name__)

__all__ = [
    "BaseTTSEngine",
    "TTSError",
    "EdgeTTSEngine",
    "build_tts_engine",
    "resolve_tts_engine",
    "BANGLA_VOICE_FEMALE",
    "BANGLA_VOICE_MALE",
]


def resolve_tts_engine(config: AppConfig) -> str:
    """Return the concrete engine name (``"edge"``)."""
    return "edge"


def build_tts_engine(
    config: AppConfig,
    time_budget_seconds: Optional[float] = None,
    total_audio_seconds: float = 0.0,
) -> BaseTTSEngine:
    """Configure and prepare the built-in Bangla neural TTS engine."""
    edge = EdgeTTSEngine(
        voice=config.resolved_tts_voice,
        rate=config.tts_rate,
        volume=config.tts_volume,
        pitch=config.tts_pitch,
        concurrency=config.tts_concurrency,
        max_retries=config.tts_max_retries,
        sample_rate=config.sync_sample_rate,
        trim_silence=config.tts_trim_silence,
        trim_pad_ms=config.tts_trim_pad_ms,
        gender=config.tts_gender,
    )
    edge.prepare()
    logger.info("TTS ইঞ্জিন প্রস্তুত: %s", edge.describe())
    return edge
