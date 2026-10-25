"""TTS subpackage: IndicF5 voice cloning (primary) plus edge-tts fallback.

IndicF5 (মূল, ভয়েস ক্লোনিং) এবং edge-tts (শুধু fallback)। ব্যবহার::

    from proxmax.tts import build_tts_engine

    engine = build_tts_engine(config, profile=voice_profile)
    engine.synthesize(segments, work_dir / "tts")

``build_tts_engine`` একটাই ইঞ্জিন ফেরত দেয় — ব্যবহারকারীকে বেছে নিতে
হয় না। ডিফল্ট ও প্রধান ইঞ্জিন **IndicF5**; শুধু সেটি চালু করতে না
পারলে (মডেল ডাউনলোড ব্লকড/GPU নেই) এবং কনফিগে অনুমতি থাকলে
edge-tts-এ নামে, যাতে ভিডিওটা নষ্ট না হয়।
"""

from __future__ import annotations

from typing import Optional

from ..config import AppConfig
from ..utils import get_logger
from ..voice_profile import VoiceProfile
from .base import BaseTTSEngine, TTSError
from .edge_engine import BANGLA_VOICE_FEMALE, BANGLA_VOICE_MALE, EdgeTTSEngine
from .governor import SpeedGovernor
from .indicf5_engine import DEFAULT_MODEL_ID, IndicF5Engine

logger = get_logger(__name__)

__all__ = [
    "BaseTTSEngine",
    "TTSError",
    "IndicF5Engine",
    "EdgeTTSEngine",
    "SpeedGovernor",
    "build_tts_engine",
    "BANGLA_VOICE_FEMALE",
    "BANGLA_VOICE_MALE",
    "DEFAULT_MODEL_ID",
]


def build_tts_engine(
    config: AppConfig,
    profile: Optional[VoiceProfile] = None,
    time_budget_seconds: Optional[float] = None,
    total_audio_seconds: float = 0.0,
) -> BaseTTSEngine:
    """Configure and prepare the single TTS engine used for the run.

    Args:
        config: অ্যাপ্লিকেশন কনফিগ।
        profile: ভয়েস ক্লোনিংয়ের রেফারেন্স।
        time_budget_seconds: TTS-এর জন্য সময়ের বাজেট (governor ব্যবহার করে)।
        total_audio_seconds: মোট কত সেকেন্ড কথা হবে।

    Returns:
        প্রস্তুত :class:`BaseTTSEngine` (``prepare()`` আগেই ডাকা হয়েছে)।

    Raises:
        TTSError: কোনো ইঞ্জিনই চালু করা না গেলে।
    """
    engine_name = (config.tts_engine or "indicf5").lower()

    if engine_name == "indicf5":
        engine: BaseTTSEngine = IndicF5Engine(
            model_id=config.tts_indicf5_model,
            fp16=config.tts_fp16,
            nfe_steps=config.tts_nfe_steps,
            min_nfe_steps=config.tts_min_nfe_steps,
            sample_rate=config.sync_sample_rate,
            max_attempts=config.tts_max_attempts,
            time_budget_seconds=time_budget_seconds,
            total_audio_seconds=total_audio_seconds,
        )
        try:
            engine.prepare(profile)
            logger.info("TTS ইঞ্জিন প্রস্তুত: %s", engine.describe())
            return engine
        except Exception as exc:
            engine.close()
            if not config.tts_fallback_to_edge:
                raise TTSError(f"IndicF5 চালু করা যায়নি: {exc}") from exc
            logger.warning(
                "⚠️ IndicF5 চালু করা যায়নি (%s) — ভয়েস ক্লোনিং ছাড়া edge-tts-এ "
                "fallback করা হচ্ছে, যাতে ভিডিওটা নষ্ট না হয়।", exc,
            )

    edge = EdgeTTSEngine(
        voice=config.tts_voice,
        rate=config.tts_rate,
        volume=config.tts_volume,
        pitch=config.tts_pitch,
        concurrency=config.tts_concurrency,
        max_retries=config.tts_max_retries,
        sample_rate=config.sync_sample_rate,
    )
    edge.prepare(profile)
    logger.info("TTS ইঞ্জিন প্রস্তুত: %s", edge.describe())
    return edge
