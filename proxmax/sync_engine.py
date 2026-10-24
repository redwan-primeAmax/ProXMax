"""Audio-Video Sync Engine: time-stretch Bangla clips to the English timing.

কাজের পদ্ধতি:

1. প্রতিটি segment-এর জন্য "হাতে থাকা সময়" (slot) হিসাব করা হয় —
   নিজের duration, আর বাংলা অডিও বড় হলে পরের segment শুরু হওয়ার
   আগের ফাঁকা সময়টুকুও ব্যবহার করা যায় (ফলে কম speed-up লাগে,
   কণ্ঠ স্বাভাবিক থাকে)।
2. ffmpeg ``atempo`` ফিল্টার দিয়ে pitch না বদলে ক্লিপের গতি
   slot অনুযায়ী মেলানো হয় (সীমা: ``min_tempo`` .. ``max_tempo``)।
   একাধিক ক্লিপ সমান্তরালে (thread pool) প্রসেস হয়।
3. সব ক্লিপ মূল ভিডিওর সমান দৈর্ঘ্যের একটি নীরব (silent) ক্যানভাসের
   উপর নিজ নিজ timestamp-এ বসিয়ে চূড়ান্ত Bangla অডিও ট্র্যাক তৈরি
   হয় — তাই ভিডিওর মোট দৈর্ঘ্য বদলায় না, lip-sync মোটামুটি বজায় থাকে।
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from .models import Segment
from .utils import ensure_tool, get_logger, get_media_duration, iter_progress, run_ffmpeg

logger = get_logger(__name__)


def build_atempo_filter(tempo: float) -> str:
    """Build an ffmpeg ``atempo`` filter chain for any tempo ratio.

    একক ``atempo``-এর কার্যকর পরিসর 0.5–2.0 ধরে নিয়ে বড়/ছোট
    অনুপাতকে একাধিক ধাপে ভাঙা হয় (যেমন 2.5 → ``atempo=2.0,atempo=1.25``),
    যাতে সব ffmpeg সংস্করণেই কাজ করে।
    """
    if tempo <= 0:
        raise ValueError(f"tempo অবশ্যই ধনাত্মক হতে হবে, পাওয়া গেছে: {tempo}")
    parts: List[float] = []
    remaining = float(tempo)
    while remaining > 2.0:
        parts.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        parts.append(0.5)
        remaining /= 0.5
    parts.append(remaining)
    return ",".join(f"atempo={p:.4f}" for p in parts)


def compute_tempo(
    tts_duration: float,
    target_duration: float,
    min_tempo: float = 0.65,
    max_tempo: float = 1.90,
    slow_down_short_clips: bool = True,
) -> float:
    """Tempo ratio to fit ``tts_duration`` into ``target_duration``.

    tempo > 1 মানে দ্রুত করা (বাংলা অডিও লম্বা), tempo < 1 মানে ধীর
    করা (বাংলা অডিও ছোট)। ``slow_down_short_clips=False`` হলে ছোট
    ক্লিপ ধীর না করে স্বাভাবিক গতিতেই রাখা হয় (বাকি slot নীরব থাকে)।
    """
    if tts_duration <= 0 or target_duration <= 0:
        return 1.0
    tempo = tts_duration / target_duration
    if tempo < 1.0 and not slow_down_short_clips:
        return 1.0
    return min(max(tempo, min_tempo), max_tempo)


class SyncEngine:
    """Builds the final, full-length Bangla audio track for a video."""

    def __init__(
        self,
        min_tempo: float = 0.65,
        max_tempo: float = 1.90,
        slow_down_short_clips: bool = True,
        sample_rate: int = 24000,
        max_workers: int = 4,
    ) -> None:
        self.min_tempo = min_tempo
        self.max_tempo = max_tempo
        self.slow_down_short_clips = slow_down_short_clips
        self.sample_rate = sample_rate
        self.max_workers = max(1, max_workers)

    # ------------------------------------------------------------------
    @staticmethod
    def _target_duration(seg: Segment, next_start: Optional[float], total_duration: float) -> float:
        """How much time this segment's Bangla audio may occupy."""
        tts_duration = seg.tts_duration or 0.0
        if tts_duration <= seg.duration:
            return seg.duration  # ছোট হলে নিজের slot-ই যথেষ্ট
        # বড় হলে পরের segment শুরুর আগের ফাঁকা সময়ও ব্যবহার করা যাবে
        slot_end = next_start if next_start is not None else total_duration
        return max(seg.duration, float(slot_end) - float(seg.start))

    def _stretch_one(self, seg: Segment, target_duration: float, out_path: Path) -> Optional[Path]:
        """Time-stretch one TTS clip with ffmpeg atempo (pitch preserved)."""
        if not seg.tts_audio_path or not Path(seg.tts_audio_path).is_file():
            return None
        tts_duration = seg.tts_duration
        if not tts_duration:
            try:
                tts_duration = get_media_duration(seg.tts_audio_path)
                seg.tts_duration = tts_duration
            except Exception as exc:
                logger.warning("Segment %d-এর TTS অডিও পড়া যাচ্ছে না: %s", seg.index, exc)
                return None

        tempo = compute_tempo(
            tts_duration, target_duration,
            min_tempo=self.min_tempo, max_tempo=self.max_tempo,
            slow_down_short_clips=self.slow_down_short_clips,
        )
        filters: List[str] = []
        if abs(tempo - 1.0) >= 0.02:  # প্রায় একই গতি হলে stretch-এর দরকার নেই
            filters.append(build_atempo_filter(tempo))
        filters.append(f"aresample={self.sample_rate}")

        ffmpeg = ensure_tool("ffmpeg")
        try:
            run_ffmpeg(
                [
                    ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(seg.tts_audio_path),
                    "-filter:a", ",".join(filters),
                    "-ac", "1", "-c:a", "pcm_s16le",
                    str(out_path),
                ],
                description=f"atempo stretch (segment {seg.index})",
            )
            seg.stretched_audio_path = out_path
            seg.stretched_duration = get_media_duration(out_path)
            if seg.stretched_duration > target_duration * 1.25:
                logger.debug(
                    "Segment %d: সর্বোচ্চ tempo (%.2f)-তেও অডিও %.2fs > slot %.2fs — "
                    "কিছুটা ওভারল্যাপ হতে পারে।",
                    seg.index, self.max_tempo, seg.stretched_duration, target_duration,
                )
            return out_path
        except Exception as exc:
            logger.warning("Segment %d stretch ব্যর্থ, বাদ দেওয়া হলো: %s", seg.index, exc)
            return None

    # ------------------------------------------------------------------
    def build_final_track(
        self,
        segments: List[Segment],
        total_duration: float,
        work_dir: Path | str,
        output_wav: Path | str,
    ) -> Path:
        """Stretch every clip and assemble one full-length WAV track.

        Raises:
            RuntimeError: ব্যবহারযোগ্য কোনো TTS ক্লিপই না থাকলে।
        """
        work_dir, output_wav = Path(work_dir), Path(output_wav)
        stretch_dir = work_dir / "stretched"
        stretch_dir.mkdir(parents=True, exist_ok=True)
        output_wav.parent.mkdir(parents=True, exist_ok=True)

        usable = [s for s in sorted(segments, key=lambda s: s.start) if s.tts_audio_path]
        if not usable:
            raise RuntimeError("Sync করার মতো কোনো TTS অডিও ক্লিপ নেই।")
        if total_duration <= 0:
            total_duration = max(s.end for s in usable) + 0.5

        # --- ধাপ ১: সমান্তরালে time-stretch --------------------------------
        jobs = []
        for i, seg in enumerate(usable):
            next_start = usable[i + 1].start if i + 1 < len(usable) else None
            target = self._target_duration(seg, next_start, total_duration)
            jobs.append((seg, target, stretch_dir / f"seg_{seg.index:04d}.wav"))

        logger.info("Sync: %dটি ক্লিপ time-stretch হচ্ছে (workers=%d)…", len(jobs), self.max_workers)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            list(iter_progress(
                pool.map(lambda j: self._stretch_one(*j), jobs),
                desc="  Sync", total=len(jobs),
            ))

        stretched = [s for s in usable if s.stretched_audio_path]
        if not stretched:
            raise RuntimeError("সব ক্লিপের time-stretch ব্যর্থ হয়েছে — চূড়ান্ত ট্র্যাক বানানো যাচ্ছে না।")

        # --- ধাপ ২: নীরব ক্যানভাসে টাইমস্ট্যাম্প অনুযায়ী ক্লিপ বসানো --------
        from pydub import AudioSegment  # lazy import

        total_ms = int(total_duration * 1000) + 100
        canvas = (
            AudioSegment.silent(duration=total_ms, frame_rate=self.sample_rate)
            .set_channels(1)
            .set_sample_width(2)
        )
        for seg in stretched:
            # ফাইল হ্যান্ডল নিজে বন্ধ করি — শত শত ক্লিপ থাকলে fd ফুরিয়ে যেতে পারে
            with open(seg.stretched_audio_path, "rb") as handle:
                clip = AudioSegment.from_file(handle, format="wav")
            clip = (
                clip.set_frame_rate(self.sample_rate)
                .set_channels(1)
                .set_sample_width(2)
            )
            position_ms = int(seg.start * 1000)
            remaining_ms = len(canvas) - position_ms
            if remaining_ms <= 0:
                continue
            if len(clip) > remaining_ms:  # ভিডিও শেষের বাইরে যাওয়া অংশ কেটে দেওয়া
                clip = clip[:remaining_ms]
            canvas = canvas.overlay(clip, position=position_ms)

        canvas.export(str(output_wav), format="wav")
        logger.info(
            "চূড়ান্ত Bangla অডিও ট্র্যাক তৈরি: %s (%.1fs, %d ক্লিপ)",
            output_wav.name, total_duration, len(stretched),
        )
        return output_wav
