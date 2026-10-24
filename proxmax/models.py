"""Shared data models used across the whole dubbing pipeline.

পুরো pipeline-এর প্রতিটি ধাপ (STT, Translation, TTS, Sync) এই
dataclass-গুলো দিয়েই ডেটা আদান-প্রদান করে, যাতে মডিউলগুলো পরস্পরের
উপর সরাসরি নির্ভরশীল না হয়।
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Word:
    """A single word with its own timestamps (word-level alignment)."""

    start: float
    end: float
    word: str
    probability: Optional[float] = None


@dataclass
class Segment:
    """One speech segment flowing through every stage of the pipeline.

    STT ধাপে ``text_en`` ও timestamps পূর্ণ হয়, Translation ধাপে
    ``text_bn``, TTS ধাপে ``tts_audio_path``/``tts_duration`` এবং
    Sync ধাপে ``stretched_audio_path`` পূর্ণ হয়।
    """

    index: int
    start: float  # seconds, relative to video start
    end: float  # seconds, relative to video start
    text_en: str = ""
    text_bn: str = ""
    words: List[Word] = field(default_factory=list)
    tts_audio_path: Optional[Path] = None
    tts_duration: Optional[float] = None
    stretched_audio_path: Optional[Path] = None
    stretched_duration: Optional[float] = None

    @property
    def duration(self) -> float:
        """Original (English) segment duration in seconds (never < 0.05)."""
        return max(0.05, float(self.end) - float(self.start))


@dataclass
class VideoResult:
    """Outcome of processing a single video file."""

    video_path: Path
    status: str  # "success" | "failed" | "skipped"
    output_path: Optional[Path] = None
    error: Optional[str] = None
    segment_count: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class BatchSummary:
    """Aggregated outcome of a full batch run over the input folder."""

    results: List[VideoResult] = field(default_factory=list)
    total_elapsed_seconds: float = 0.0

    @property
    def succeeded(self) -> List[VideoResult]:
        return [r for r in self.results if r.status == "success"]

    @property
    def failed(self) -> List[VideoResult]:
        return [r for r in self.results if r.status == "failed"]

    @property
    def skipped(self) -> List[VideoResult]:
        return [r for r in self.results if r.status == "skipped"]
