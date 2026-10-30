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
    #: চূড়ান্ত ট্র্যাকে **প্রকৃতপক্ষে** কত সেকেন্ড বসানো হলো (কাটা বাদ দিয়ে) —
    #: সাবটাইটেলের সময় এই মান ধরেই হিসাব হয়, তাই লাইন অডিওর সঙ্গে মেলে
    placed_duration: Optional[float] = None
    #: sync ধাপে ব্যবহৃত tempo (১.০ = স্বাভাবিক গতি)
    tempo_used: float = 1.0
    #: clip নিরাপদে বসানো যায়নি বা মেরামত দরকার হলে report-এ কারণ দেখায়
    sync_issue: Optional[str] = None
    #: ``True`` হলে এটি একাধিক segment মিলিয়ে বানানো "speech unit"
    #: (এক clip-এ পুরো বাক্য — স্বাভাবিক prosody, মাঝপথে থামা নেই)
    is_unit: bool = False
    #: unit হলে তার ভেতরের মূল segment-গুলো (সাবটাইটেলের জন্য দরকার)
    children: List["Segment"] = field(default_factory=list)

    @property
    def speech_seconds(self) -> float:
        """কত সেকেন্ডের অডিও এই segment/unit দখল করবে (জানা থাকলে)।"""
        if self.placed_duration:
            return float(self.placed_duration)
        if self.stretched_duration:
            return float(self.stretched_duration)
        if self.tts_duration:
            return float(self.tts_duration)
        return self.duration

    @property
    def duration(self) -> float:
        """Original (English) segment duration in seconds (never < 0.05)."""
        return max(0.05, float(self.end) - float(self.start))


@dataclass
class StageTimings:
    """Per-stage wall-clock timings for one video (used in the final report)."""

    extract: float = 0.0
    stt: float = 0.0
    translate: float = 0.0
    tts: float = 0.0
    sync: float = 0.0
    mux: float = 0.0
    total: float = 0.0
    audio_seconds: float = 0.0
    validation: float = 0.0

    @property
    def realtime_factor(self) -> float:
        """মোট সময় / অডিওর দৈর্ঘ্য (কম = দ্রুত)।"""
        return (self.total / self.audio_seconds) if self.audio_seconds > 0 else 0.0

    def as_text(self) -> str:
        """এক লাইনের সারসংক্ষেপ।"""
        return (
            f"extract {self.extract:.0f}s | stt {self.stt:.0f}s | translate {self.translate:.0f}s | "
            f"tts {self.tts:.0f}s | sync {self.sync:.0f}s | mux {self.mux:.0f}s | "
            f"মোট {self.total:.0f}s (RTF {self.realtime_factor:.2f})"
        )


@dataclass
class VideoResult:
    """Outcome of processing a single video file."""

    video_path: Path
    status: str  # "success" | "failed" | "skipped"
    output_path: Optional[Path] = None
    error: Optional[str] = None
    segment_count: int = 0
    elapsed_seconds: float = 0.0
    timings: Optional[StageTimings] = None
    stats: dict = field(default_factory=dict)


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
