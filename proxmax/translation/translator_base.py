"""Shared interface for translation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Sequence

from .models import Segment
from .utils import iter_progress

class BaseTranslator(ABC):
    """Common interface for all translation backends."""

    @abstractmethod
    def translate_text(self, text: str, duration: float = 0.0) -> str:
        """Translate a single English string into Bangla."""

    def translate_segments(self, segments: List[Segment]) -> List[Segment]:
        """Fill ``text_bn`` of every segment in-place (sequential default)."""
        for seg in iter_progress(segments, desc="  Translate", total=len(segments)):
            seg.text_bn = self.translate_text(seg.text_en, seg.duration)
        return segments

    def shorten_to_fit(self, segments: List[Segment], **_: Any) -> int:
        """সময়ে না আঁটা লাইন ছোট করে লেখায় (ডিফল্ট: কিছুই করে না)।"""
        return 0

    def shorten_to_durations(self, segments: List[Segment], targets: Sequence[float]) -> int:
        """Measured TTS duration অনুযায়ী text ছোট করে (ডিফল্ট: কিছুই করে না)।"""
        return 0

    def expand_to_fit(self, segments: List[Segment], targets: Sequence[float]) -> int:
        """স্লটের চেয়ে ছোট লাইন ভরাট করে লেখায় (ডিফল্ট: কিছুই করে না)।"""
        return 0

    def describe(self) -> str:
        """Human-readable backend description (log-এর জন্য)।"""
        return type(self).__name__

    def close(self) -> None:
        """Release backend resources (client, model ইত্যাদি)। ডিফল্ট: কিছুই নয়।"""
