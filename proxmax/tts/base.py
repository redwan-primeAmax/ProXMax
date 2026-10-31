"""TTS engine interface shared by every speech backend.

সব ইঞ্জিনকে একই তিনটি কাজ করতে হয়, তাই ঠিকানা এক: ``prepare()`` →
``synthesize()`` → ``close()``। এতে pipeline-এর কোড জানে না ভেতরে
কোন TTS ইঞ্জিন চলছে — পরীক্ষার সময় সহজে fake ইঞ্জিন বসানো যায়।
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np

from .. import audio_fx
from ..models import Segment
from ..sync_silence import trim_silence_for_overrun
from ..utils import get_logger

logger = get_logger(__name__)


class TTSError(RuntimeError):
    """Raised when speech synthesis fails beyond recovery."""


class BaseTTSEngine(ABC):
    """Common interface for Bangla speech synthesis engines."""

    #: লগে দেখানোর নাম (``edge``)
    name: str = "base"
    #: আউটপুট WAV-এর sample rate
    sample_rate: int = 24000

    def __init__(self, trim_silence: bool = True, trim_pad_ms: int = 25) -> None:
        # Kept as a legacy constructor argument. All clips are now trimmed
        # before timing, including when an older config requested False.
        if not trim_silence:
            logger.info("tts.trim_silence=false is obsolete; detected silence is always trimmed.")
        self.trim_silence = True
        self.trim_pad_ms = max(10, int(trim_pad_ms))  # never remove all speech-boundary padding

    # ------------------------------------------------------------------
    def prepare(self) -> None:
        """Engine প্রস্তুত করা।

        Raises:
            TTSError: ইঞ্জিন চালু করা না গেলে।
        """
        return None

    @abstractmethod
    def synthesize(self, segments: List[Segment], output_dir: Path | str) -> List[Segment]:
        """প্রতিটি segment-এর বাংলা অডিও তৈরি করে।

        সফল হলে ``segment.tts_audio_path`` ও ``segment.tts_duration``
        পূর্ণ হয় (WAV, mono)।

        Args:
            segments: ``text_bn`` পূর্ণ segment-গুলোর তালিকা।
            output_dir: ক্লিপগুলো কোথায় লেখা হবে।

        Returns:
            একই তালিকা (in-place হালনাগাদ)।

        Raises:
            TTSError: একটিও ক্লিপ তৈরি না হলে।
        """

    # ------------------------------------------------------------------
    def set_budget(
        self,
        time_budget_seconds: Optional[float],
        total_audio_seconds: float = 0.0,
    ) -> None:
        """প্রতি ভিডিওর জন্য সময়ের বাজেট হালনাগাদ করে (ডিফল্ট: কিছুই নয়)।"""
        return None

    def describe(self) -> str:
        """Human-readable নাম (লগ ও রিপোর্টে)।"""
        return self.name

    def close(self) -> None:
        """মডেল/রিসোর্স ছেড়ে দেওয়া (ডিফল্ট: কিছুই নয়)।"""

    # ------------------------------------------------------------------
    @staticmethod
    def usable_segments(segments: List[Segment]) -> List[Segment]:
        """যেসব segment-এ সত্যিই বাংলা টেক্সট আছে সেগুলোই ফেরত দেয়।"""
        return [s for s in segments if (s.text_bn or "").strip()]

    def _postprocess(self, audio: np.ndarray) -> np.ndarray:
        """ক্লিপের লাউডনেস ঠিক করা ও ক্লিক-মুক্ত করা।"""
        if audio.size == 0:
            return audio
        clipped = np.clip(audio, -1.0, 1.0)
        return audio_fx.fade(
            audio_fx.peak_normalize(clipped, peak_dbfs=-1.5),
            self.sample_rate, in_ms=8, out_ms=8,
        )

    def _trim_all_silence(self, audio: np.ndarray):
        """Conservatively remove detected head, internal and tail silence."""
        return trim_silence_for_overrun(
            audio, self.sample_rate, audio.size / float(self.sample_rate),
            threshold_dbfs=-70.0, pad_ms=self.trim_pad_ms,
        )

    def _finalize(self, audio: np.ndarray, seg: Optional[Segment] = None) -> np.ndarray:
        """Trim detected silence immediately, then normalize and apply short fades."""
        if audio.size == 0:
            return audio
        trimmed = self._trim_all_silence(audio)
        final = self._postprocess(trimmed.audio)
        if seg is not None:
            seg.tts_raw_duration = audio.size / float(self.sample_rate)
            seg.tts_silence_trimmed = True
            seg.tts_trimmed_head_seconds = trimmed.head_seconds
            seg.tts_trimmed_internal_seconds = trimmed.internal_seconds
            seg.tts_trimmed_tail_seconds = trimmed.tail_seconds
            seg.tts_trimmed_internal_cuts = trimmed.internal_cuts
        return final

    def _write_clip(self, seg: Segment, audio: np.ndarray, output_dir: Path) -> Path:
        """Write a trimmed TTS clip and retain audit information."""
        path = output_dir / f"seg_{seg.index:04d}.wav"
        final = self._finalize(audio, seg)
        audio_fx.write_wav(path, final, self.sample_rate)
        seg.tts_audio_path = path
        seg.tts_duration = final.size / float(self.sample_rate)
        return path
