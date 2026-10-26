"""TTS subpackage: the built-in Bangla neural voice (edge-tts).

**কোনো ক্লোনিং নেই।** কণ্ঠের জন্য ব্যবহারকারীকে কিছু দিতে হয় না — না
রেকর্ডিং, না Hugging Face token, না লাইসেন্স সম্মতি। ডিফল্ট **পুরুষ**
বাংলা neural ভয়েস (``bn-BD-PradeepNeural``) সবসময় নিজে থেকেই কাজ করে।

ব্যবহার::

    from proxmax.tts import build_tts_engine

    engine = build_tts_engine(config)
    engine.synthesize(segments, work_dir / "tts")
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
    "BANGLA_VOICE_FEMALE",
    "BANGLA_VOICE_MALE",
]


def build_tts_engine(
    config: AppConfig,
    time_budget_seconds: Optional[float] = None,  # noqa: ARG001 - সংগতির জন্য রাখা
    total_audio_seconds: float = 0.0,  # noqa: ARG001 - সংগতির জন্য রাখা
) -> BaseTTSEngine:
    """Build and prepare the built-in Bangla voice engine.

    Args:
        config: অ্যাপ্লিকেশন কনফিগ।
        time_budget_seconds: আর ব্যবহার হয় না (সংগতি রক্ষার জন্য রাখা)।
        total_audio_seconds: আর ব্যবহার হয় না (সংগতি রক্ষার জন্য রাখা)।

    Returns:
        প্রস্তুত :class:`BaseTTSEngine` (``prepare()`` আগেই ডাকা হয়েছে)।

    Raises:
        TTSError: edge-tts ইনস্টল না থাকলে বা ভয়েস প্রস্তুত করা না গেলে।
    """
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
