"""TTS subpackage: built-in Bangla neural voice plus optional voice cloning.

``auto`` মোডে **কিছুই দিতে হয় না**: ইনপুট ফোল্ডারে কণ্ঠের নমুনা না
থাকলে বিল্ট-ইন বাংলা neural ভয়েস (edge-tts) ব্যবহার হয় — কোনো
Hugging Face token, লাইসেন্স সম্মতি বা রেকর্ডিং লাগে না।

ইনপুট ফোল্ডারে আপনার ১০–২০ সেকেন্ডের বাংলা কথার audio ফাইল রাখলে
(``auto`` বা ``indicf5`` মোডে) IndicF5 দিয়ে কণ্ঠ ক্লোনিং হয় — সেটি
ঐচ্ছিক, বাধ্যতামূলক নয়।

ব্যবহার::

    from proxmax.tts import build_tts_engine, resolve_tts_engine

    engine = build_tts_engine(config, profile=voice_profile)
    engine.synthesize(segments, work_dir / "tts")
"""

from __future__ import annotations

import os
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
    "resolve_tts_engine",
    "hf_token_available",
    "BANGLA_VOICE_FEMALE",
    "BANGLA_VOICE_MALE",
    "DEFAULT_MODEL_ID",
]


#: Hugging Face টোকেন যে environment variable-গুলোতে থাকতে পারে
_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")


def hf_token_available() -> bool:
    """Hugging Face access token পাওয়া যাচ্ছে কি না (IndicF5 gated মডেল)।

    টোকেন খোঁজা হয় এই ক্রমে: প্রথমে environment variable (``HF_TOKEN``
    ইত্যাদি), তারপর ``huggingface_hub``-এর সংরক্ষিত token (``login()``
    করলে সেটি সেখানেই জমা থাকে)।

    টোকেন না থাকলে কণ্ঠ ক্লোনিং **কখনোই** সম্ভব নয় — তখন অ্যাপ কিছু
    ডাউনলোডের চেষ্টা না করে সরাসরি বিল্ট-ইন বাংলা ভয়েস ব্যবহার করে,
    অর্থাৎ ব্যবহারকারী কিছু না দিলেও কাজ আগায়।

    Returns:
        ``True`` যদি কোনো টোকেন পাওয়া যায়, নাহলে ``False``।
    """
    for name in _HF_TOKEN_ENV_VARS:
        if (os.environ.get(name) or "").strip():
            return True
    try:
        from huggingface_hub import get_token  # huggingface_hub >= 0.19
    except Exception:
        return False
    try:
        return bool((get_token() or "").strip())
    except Exception:  # pragma: no cover - পুরোনো/ভাঙা install
        return False


def resolve_tts_engine(
    config: AppConfig, profile: Optional[VoiceProfile] = None
) -> str:
    """Return the concrete engine name (``"indicf5"`` or ``"edge"``).

    ``auto`` (ডিফল্ট) মানে — কণ্ঠের নমুনা থাকলে IndicF5 ক্লোনিং, না থাকলে
    বিল্ট-ইন edge ভয়েস। ব্যবহারকারীকে কিছু বেছে নিতে হয় না।

    Args:
        config: অ্যাপ্লিকেশন কনফিগ।
        profile: পাওয়া গেলে ভয়েস প্রোফাইল (নমুনা থেকে তৈরি)।

    Returns:
        ``"indicf5"`` অথবা ``"edge"``।
    """
    name = (getattr(config, "tts_engine", "auto") or "auto").strip().lower()
    if name in ("indicf5", "edge"):
        return name
    if profile is not None:
        try:
            if profile.is_valid():
                return "indicf5"
        except Exception:  # pragma: no cover - নষ্ট প্রোফাইল → বিল্ট-ইন ভয়েস
            pass
    return "edge"


def build_tts_engine(
    config: AppConfig,
    profile: Optional[VoiceProfile] = None,
    time_budget_seconds: Optional[float] = None,
    total_audio_seconds: float = 0.0,
) -> BaseTTSEngine:
    """Configure and prepare the single TTS engine used for the run.

    Args:
        config: অ্যাপ্লিকেশন কনফিগ।
        profile: কণ্ঠ ক্লোনিংয়ের রেফারেন্স (থাকলে এবং ``auto``/``indicf5``
            মোডে IndicF5 ব্যবহার হয়)।
        time_budget_seconds: TTS-এর জন্য সময়ের বাজেট (governor ব্যবহার করে)।
        total_audio_seconds: মোট কত সেকেন্ড কথা হবে।

    Returns:
        প্রস্তুত :class:`BaseTTSEngine` (``prepare()`` আগেই ডাকা হয়েছে)।

    Raises:
        TTSError: কোনো ইঞ্জিনই চালু করা না গেলে।
    """
    engine_name = resolve_tts_engine(config, profile)

    if engine_name == "indicf5" and profile is None:
        logger.info(
            "কণ্ঠ ক্লোনিংয়ের জন্য কোনো ভয়েস নমুনা নেই — বিল্ট-ইন বাংলা ভয়েস "
            "(%s) ব্যবহার হবে।", config.tts_voice,
        )
    elif engine_name == "indicf5":
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
                "⚠️ IndicF5 চালু করা যায়নি (%s) — বিল্ট-ইন বাংলা ভয়েসে "
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
