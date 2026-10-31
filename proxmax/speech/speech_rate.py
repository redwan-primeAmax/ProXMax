"""Optional TTS speech-rate diagnostic, retained for legacy callers.

The current pipeline deliberately avoids pre-synthesis duration prediction and
sends no source time range or duration hint to Gemini. It measures the generated
WAV directly during sync, so it does not run this calibration before translating.
These helpers remain available for explicit diagnostics and compatibility tests.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

from . import audio_fx
from .models import Segment
from .utils import get_logger

logger = get_logger(__name__)

#: ক্যালিব্রেশনে বলা হয় এমন ছোট বাংলা বাক্য (প্রতিদিনের কথার মতো শব্দ)
CALIBRATION_TEXTS: Sequence[str] = (
    "আজ আমরা একটি নতুন ভিডিও নিয়ে আলোচনা করব।",
    "এই কাজটি শেষ করতে আর মাত্র কয়েক মিনিট বাকি আছে।",
)

#: এই সীমার বাইরে এলে পরিমাপ বিশ্বাস করা হয় না (কল্পিত/ভাঙা audio)
_MIN_RATE = 6.0
_MAX_RATE = 24.0


def measure_rate(engine, texts: Sequence[str] = CALIBRATION_TEXTS,
                 sample_rate: int = 24000) -> Optional[float]:
    """``engine`` দিয়ে ``texts`` বলে প্রকৃত অক্ষর/সেকেন্ড মাপে।

    Args:
        engine: ``synthesize(segments, out_dir)`` সমর্থিত TTS ইঞ্জিন।
        texts: উচ্চারণ করার বাংলা বাক্য (খালি/অ-বাংলা বাদ পড়ে)।
        sample_rate: ত্রুটি হলেও পড়া যায় এমন ব্যাকআপ (`Segment` তৈরি করতে)।

    Returns:
        মাপা গতি (অক্ষর/সেকেন্ড) অথবা ``None`` — কিছু ব্যর্থ হলে কল করা
        উচিত নয় এমন কিছু করা হবে না, শুধু আগের ধরা মানই ব্যবহার হবে।
    """
    samples = [t.strip() for t in texts if (t or "").strip()]
    if not samples:
        return None

    segments = [
        Segment(index=i, start=float(i), end=float(i) + 2.0, text_bn=text)
        for i, text in enumerate(samples)
    ]
    measured_chars = sum(len(s.text_bn) for s in segments)
    measured_seconds = 0.0

    with tempfile.TemporaryDirectory(prefix="proxmax-rate-") as tmp:
        tmp_path = Path(tmp)
        try:
            engine.synthesize(segments, tmp_path)
        except Exception as exc:  # নেটওয়ার্ক/কণ্ঠ সমস্যা — ক্যালিব্রেশন বাদ
            logger.info("কণ্ঠের গতি মাপা যায়নি (%s) — অনুমানকৃত হার ব্যবহার হবে।", exc)
            return None

        for segment in segments:
            path = segment.tts_audio_path
            if not path or not Path(path).is_file():
                continue
            try:
                clip, rate = audio_fx.read_wav(path)
            except Exception:  # pragma: no cover - এইমাত্র লেখা ফাইল
                continue
            seconds = clip.size / float(rate or sample_rate)
            if seconds > 0.1:
                measured_seconds += seconds

    if measured_seconds <= 0.1 or measured_chars <= 0:
        logger.info("কণ্ঠের গতি মাপা যায়নি (কোনো অডিও নেই) — অনুমানকৃত হার ব্যবহার হবে।")
        return None

    rate = measured_chars / measured_seconds
    if not _MIN_RATE <= rate <= _MAX_RATE:
        logger.info(
            "মাপা গতি অবিশ্বাস্য (%.1f অক্ষর/সেকেন্ড) — অনুমানকৃত হার ব্যবহার হবে।", rate
        )
        return None
    logger.info(
        "🎚️ কণ্ঠের প্রকৃত গতি মাপা হলো: %.1f অক্ষর/সেকেন্ড (%.1fs অডিও, %d অক্ষর)",
        rate, measured_seconds, measured_chars,
    )
    return rate


def calibrate(engine, fallback: float, texts: Sequence[str] = CALIBRATION_TEXTS,
              sample_rate: int = 24000) -> float:
    """``measure_rate`` চেষ্টা করে; ব্যর্থ হলে ``fallback`` ফেরত দেয়।"""
    measured = measure_rate(engine, texts=texts, sample_rate=sample_rate)
    if measured is None:
        return float(fallback)
    return float(measured)


def estimate_chars(texts: List[str]) -> int:
    """কত অক্ষর মাপা হবে (লগ/টেস্টের জন্য)।"""
    return sum(len(t) for t in texts)
