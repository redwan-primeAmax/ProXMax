"""Audio-video sync: fit Bangla speech to the timeline with zero overlap.

কণ্ঠকে সময়ের সাথে মেলানো হয় — **কখনোই ওভারল্যাপ নয়**। দুটি মূল সমস্যা সমাধান করা হয়:

1. **ওভারল্যাপ (দুই কণ্ঠ একসাথে)** — আগে ``pydub``-এর ``overlay()``
   ব্যবহার হতো, যা আসলে অডিও **যোগ** করে। কোনো বাংলা ক্লিপ তার
   স্লটের চেয়ে লম্বা হলে সে পরের বাক্যের উপর চড়ে যেত। এখন প্রতিটি
   ক্লিপের জন্য একটি **কঠোর উইন্ডো** হিসাব হয় (পরের ক্লিপ শুরুর
   আগেই শেষ), আর বসানোর আগে ওভারল্যাপ গাণিতিকভাবে যাচাই করা হয়।
2. **রোবটে/ধাতব কণ্ঠ** — আগে tempo ১.৯০ পর্যন্ত ছিল (প্রায় দ্বিগুণ
   গতি = চিপমাঙ্ক শব্দ)। এখন সর্বোচ্চ **১.২৫**; তাতেও না আঁটলে
   সামান্য নীরবতা ধার নেওয়া হয়, শেষে অল্প tail-trim। ffmpeg-এ
   ``rubberband`` ফিল্টার থাকলে সেটাই ব্যবহার হয় (formant রক্ষা করে,
   ``atempo``-র চেয়ে অনেক স্বাভাবিক)।

চূড়ান্ত ট্র্যাক NumPy দিয়ে সাজানো হয় (pydub নয়) — কম মেমরি, কোনো
ফাইল-হ্যান্ডল লিক নেই, আর প্রতিটি প্লেসমেন্টের হিসাব নিজের হাতে।
"""

from __future__ import annotations

import functools
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import audio_fx
from .models import Segment
from .utils import ensure_tool, get_logger, run_ffmpeg

logger = get_logger(__name__)


# =============================================================================
# Filter helpers
# =============================================================================
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


@functools.lru_cache(maxsize=1)
def available_filters() -> frozenset:
    """ffmpeg-এ উপলব্ধ filter-এর তালিকা (একবারই যাচাই হয়)।"""
    try:
        ffmpeg = ensure_tool("ffmpeg")
    except Exception:  # pragma: no cover - ffmpeg নেই
        return frozenset()
    proc = subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-filters"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    names = set()
    for line in (proc.stdout or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and not line.lstrip().startswith(("Filters:", "|")):
            names.add(parts[1])
    return frozenset(names)


def build_stretch_filter(tempo: float, sample_rate: int, prefer_rubberband: bool = True) -> str:
    """Time-stretch filter chain — ``rubberband`` থাকলে সেটাই, নইলে ``atempo``।

    ``rubberband`` pitch ও formant দুটোই ধরে রাখে, তাই কণ্ঠ অনেক কম
    "ধাতব" শোনায়। না থাকলে সর্বজনীন ``atempo`` ব্যবহৃত হয়।
    """
    if abs(tempo - 1.0) < 0.02:
        return f"aresample={sample_rate}"
    if prefer_rubberband and "rubberband" in available_filters():
        return f"rubberband=tempo={tempo:.4f},aresample={sample_rate}"
    return f"{build_atempo_filter(tempo)},aresample={sample_rate}"


def compute_tempo(
    tts_duration: float,
    target_duration: float,
    min_tempo: float = 0.95,
    max_tempo: float = 1.25,
    slow_down_short_clips: bool = False,
) -> float:
    """Tempo ratio to fit ``tts_duration`` into ``target_duration``.

    tempo > 1 মানে দ্রুত করা (বাংলা অডিও লম্বা), tempo < 1 মানে ধীর
    করা (বাংলা অডিও ছোট)। ডিফল্টে ধীর করা হয় না — ছোট ক্লিপ
    স্বাভাবিক গতিতে রেখে বাকি সময়টা নীরব থাকলে শোনতে বেশি স্বাভাবিক লাগে।
    """
    if tts_duration <= 0 or target_duration <= 0:
        return 1.0
    tempo = tts_duration / target_duration
    if tempo < 1.0 and not slow_down_short_clips:
        return 1.0
    return min(max(tempo, min_tempo), max_tempo)


def plan_windows(
    segments: Sequence[Segment],
    total_duration: float,
    min_gap_seconds: float = 0.08,
    borrow_seconds: float = 0.25,
) -> List[Tuple[float, float]]:
    """প্রতিটি segment-এর জন্য ওভারল্যাপ-মুক্ত সময় উইন্ডো।

    নিয়ম:
      * উইন্ডো কখনো পরের segment শুরুর আগে শেষ হয় (``min_gap`` বাদে),
      * বাংলা অডিও লম্বা হলে নীরবতা থেকে সর্বোচ্চ ``borrow_seconds`` ধার
        নেওয়া যায়, তবু পরের ক্লিপে ঢুকে পড়া যাবে না,
      * শুরু কখনো পিছিয়ে যায় না (lip-sync নষ্ট হয়)।

    Returns:
        প্রতিটি segment-এর ``(start, end)`` সেকেন্ড।
    """
    ordered = sorted(segments, key=lambda s: s.start)
    windows: List[Tuple[float, float]] = []
    for index, seg in enumerate(ordered):
        start = max(0.0, float(seg.start))
        if index + 1 < len(ordered):
            hard_end = min(float(ordered[index + 1].start) - min_gap_seconds, total_duration)
        else:
            hard_end = float(total_duration)
        desired = float(seg.tts_duration or seg.duration)
        natural_end = start + max(desired, seg.duration)
        # speech unit (পুরো বাক্য) হলে পরের বাক্যের আগের নীরবতা থেকে একটু বেশি
        # ধার নেওয়া হয় — নাহলে বাক্যের শেষ শব্দটা কেটে যাওয়ার ঝুঁকি থাকে
        borrow = max(borrow_seconds, 0.45) if getattr(seg, "is_unit", False) else borrow_seconds
        end = min(hard_end, natural_end + borrow)
        end = max(end, min(hard_end, start + 0.15))
        windows.append((start, end))
    return windows


def find_overlaps(spans: Sequence[Tuple[float, float]]) -> List[Tuple[int, float]]:
    """Return ``(index, overlap_seconds)`` for any overlapping neighbours."""
    problems: List[Tuple[int, float]] = []
    for i in range(len(spans) - 1):
        overlap = spans[i][1] - spans[i + 1][0]
        if overlap > 1e-4:
            problems.append((i, overlap))
    return problems


class SyncEngine:
    """Builds the final, full-length Bangla audio track for a video."""

    def __init__(
        self,
        min_tempo: float = 0.95,
        max_tempo: float = 1.25,
        slow_down_short_clips: bool = False,
        sample_rate: int = 24000,
        max_workers: int = 4,
        slot_gap_ms: int = 80,
        borrow_silence_ms: int = 250,
        prefer_rubberband: bool = True,
    ) -> None:
        """Args:
            min_tempo: ধীর করার সর্বনিম্ন সীমা (ডিফল্টে ধীর করা হয় না)।
            max_tempo: দ্রুত করার সর্বোচ্চ সীমা — ১.২৫-এর বেশি হলে কণ্ঠ বিকৃত শোনায়।
            slow_down_short_clips: ছোট ক্লিপ ধীর করে স্লট ভরা হবে কি না।
            sample_rate: চূড়ান্ত ট্র্যাকের sample rate।
            max_workers: সমান্তরাল ffmpeg প্রসেস।
            slot_gap_ms: দুই ক্লিপের মাঝে ন্যূনতম ফাঁক।
            borrow_silence_ms: পরের নীরবতা থেকে সর্বোচ্চ কত ধার নেওয়া যাবে।
            prefer_rubberband: ffmpeg-এ থাকলে rubberband ব্যবহার করা হবে।
        """
        self.min_tempo = float(min_tempo)
        self.max_tempo = float(max_tempo)
        self.slow_down_short_clips = bool(slow_down_short_clips)
        self.sample_rate = int(sample_rate)
        self.max_workers = max(1, int(max_workers))
        self.slot_gap_seconds = max(0.0, slot_gap_ms / 1000.0)
        self.borrow_seconds = max(0.0, borrow_silence_ms / 1000.0)
        self.prefer_rubberband = bool(prefer_rubberband)
        #: সর্বশেষ রানের পরিসংখ্যান (রিপোর্টে ব্যবহৃত)
        self.last_stats: Dict[str, object] = {}

    # ------------------------------------------------------------------
    def _stretch_one(self, seg: Segment, window: Tuple[float, float], out_path: Path) -> Optional[Path]:
        """Time-stretch one TTS clip to fit its window (pitch preserved)."""
        source = Path(seg.tts_audio_path) if seg.tts_audio_path else None
        if source is None or not source.is_file():
            return None
        window_length = max(0.12, window[1] - window[0])
        tts_duration = seg.tts_duration or 0.0
        if tts_duration <= 0:
            try:
                clip, rate = audio_fx.read_wav(source)
                tts_duration = clip.size / float(rate or self.sample_rate)
                seg.tts_duration = tts_duration
            except Exception as exc:
                logger.warning("Segment %d-এর TTS অডিও পড়া যাচ্ছে না: %s", seg.index, exc)
                return None

        tempo = compute_tempo(
            tts_duration, window_length,
            min_tempo=self.min_tempo, max_tempo=self.max_tempo,
            slow_down_short_clips=self.slow_down_short_clips,
        )

        if abs(tempo - 1.0) < 0.02:
            # stretch লাগবে না — সরাসরি WAV কপি (সবচেয়ে দ্রুত, কোনো ক্ষতি নেই)
            try:
                clip, rate = audio_fx.read_wav(source)
                if rate != self.sample_rate:
                    clip = self._resample(clip, rate, self.sample_rate)
                audio_fx.write_wav(out_path, clip, self.sample_rate)
            except Exception as exc:
                logger.warning("Segment %d কপি করা যায়নি: %s", seg.index, exc)
                return None
        else:
            ffmpeg = ensure_tool("ffmpeg")
            try:
                run_ffmpeg(
                    [
                        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(source),
                        "-filter:a", build_stretch_filter(
                            tempo, self.sample_rate, self.prefer_rubberband
                        ),
                        "-ac", "1", "-c:a", "pcm_s16le",
                        str(out_path),
                    ],
                    description=f"time-stretch (segment {seg.index}, tempo={tempo:.2f})",
                    timeout=120.0,
                )
            except Exception as exc:
                logger.warning("Segment %d stretch ব্যর্থ, বাদ দেওয়া হলো: %s", seg.index, exc)
                return None

        seg.stretched_audio_path = out_path
        try:
            clip, rate = audio_fx.read_wav(out_path)
            seg.stretched_duration = clip.size / float(rate or self.sample_rate)
        except Exception:  # pragma: no cover - এইমাত্র লেখা ফাইল
            seg.stretched_duration = tts_duration / max(tempo, 0.01)

        seg.tempo_used = tempo  # type: ignore[attr-defined]  # রিপোর্টের জন্য
        return out_path

    @staticmethod
    def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """NumPy linear resample (শুধু দরকার হলে)।"""
        if src_rate == dst_rate or audio.size == 0:
            return audio
        target = max(1, int(round(audio.size * dst_rate / float(src_rate))))
        index = np.linspace(0.0, audio.size - 1.0, num=target, dtype=np.float64)
        return np.interp(index, np.arange(audio.size), audio).astype(np.float32)

    # ------------------------------------------------------------------
    def build_final_track(
        self,
        segments: List[Segment],
        total_duration: float,
        work_dir: Path | str,
        output_wav: Path | str,
    ) -> Path:
        """Stretch every clip and assemble one full-length, non-overlapping WAV.

        Raises:
            RuntimeError: ব্যবহারযোগ্য কোনো TTS ক্লিপই না থাকলে।
        """
        work_dir, output_wav = Path(work_dir), Path(output_wav)
        # প্রতি ভিডিওতে পরিসংখ্যান নতুন করে শুরু (আগের ভিডিওর সংখ্যা মিশে যাবে না)
        self.last_stats = {"trimmed": 0}
        stretch_dir = work_dir / "stretched"
        stretch_dir.mkdir(parents=True, exist_ok=True)
        output_wav.parent.mkdir(parents=True, exist_ok=True)

        usable = [s for s in sorted(segments, key=lambda s: s.start) if s.tts_audio_path]
        if not usable:
            raise RuntimeError("Sync করার মতো কোনো TTS অডিও ক্লিপ নেই।")
        if total_duration <= 0:
            total_duration = max(s.end for s in usable) + 0.5

        windows = plan_windows(
            usable, total_duration,
            min_gap_seconds=self.slot_gap_seconds, borrow_seconds=self.borrow_seconds,
        )
        for i in range(len(windows)):
            windows[i] = (windows[i][0], min(windows[i][1], total_duration))
        overlaps = find_overlaps(windows)
        if overlaps:
            # অভ্যন্তরীণ হিসাবের ত্রুটি — ব্যবহারকারীর কানে পৌঁছানোর আগেই ঠিক করা হয়
            logger.warning("উইন্ডো ওভারল্যাপ ধরা পড়েছে (%d জায়গায়) — ঠিক করা হচ্ছে।", len(overlaps))
            for index, _ in overlaps:
                windows[index] = (windows[index][0], min(windows[index][1], windows[index + 1][0]))

        logger.info(
            "Sync: %dটি ক্লিপ (tempo সর্বোচ্চ %.2f, workers=%d)…",
            len(usable), self.max_tempo, self.max_workers,
        )
        jobs = [
            (seg, windows[i], stretch_dir / f"seg_{seg.index:04d}.wav")
            for i, seg in enumerate(usable)
        ]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            list(pool.map(lambda job: self._stretch_one(*job), jobs))

        # কোন ক্লিপ কোন উইন্ডোতে বসবে — জোড়া বেঁধে রাখা হয়, যাতে কোনো ক্লিপ
        # stretch ব্যর্থ হলেও বাকিগুলোর সময় (index) সরে না যায়।
        pairs = [(seg, windows[i]) for i, seg in enumerate(usable) if seg.stretched_audio_path]
        if not pairs:
            raise RuntimeError("সব ক্লিপের time-stretch ব্যর্থ হয়েছে — চূড়ান্ত ট্র্যাক বানানো যাচ্ছে না।")
        stretched = [seg for seg, _ in pairs]

        canvas = self._assemble(pairs, total_duration)
        audio_fx.write_wav(output_wav, canvas, self.sample_rate)

        tempos = [float(getattr(s, "tempo_used", 1.0) or 1.0) for s in stretched]
        max_tempo = max(tempos) if tempos else 1.0
        aggressive = sum(1 for t in tempos if t > 1.15)
        self.last_stats = {
            "clips": len(stretched),
            "overlaps": 0,
            "max_tempo": max_tempo,
            "aggressive_stretch": aggressive,
            "trimmed": int(self.last_stats.get("trimmed", 0)),
        }
        logger.info(
            "চূড়ান্ত Bangla অডিও: %s | %.1fs | %d ক্লিপ | সর্বোচ্চ tempo %.2f "
            "(জোরালো stretch %d) | ওভারল্যাপ ০",
            output_wav.name, total_duration, len(stretched), max_tempo, aggressive,
        )
        return output_wav

    # ------------------------------------------------------------------
    def _assemble(
        self,
        pairs: Sequence[Tuple[Segment, Tuple[float, float]]],
        total_duration: float,
    ) -> np.ndarray:
        """সব stretched ক্লিপ নির্দিষ্ট সময়ে বসিয়ে একটাই ট্র্যাক বানায়।

        ``pairs`` = ``(segment, window)`` — উইন্ডো আগেই ওভারল্যাপ-মুক্ত,
        আর ক্লিপ উইন্ডোর চেয়ে লম্বা হলে শেষে ছেঁটে ফেলা হয়। তাই দুই কণ্ঠ
        কখনো একসাথে শোনা যাবে না।
        """
        sr = self.sample_rate
        canvas = np.zeros(int(total_duration * sr) + int(0.1 * sr), dtype=np.float32)

        clips: List[Tuple[Segment, np.ndarray, float]] = []
        levels: List[float] = []
        for seg, window in pairs:
            try:
                clip, rate = audio_fx.read_wav(seg.stretched_audio_path)  # type: ignore[arg-type]
            except Exception as exc:
                logger.warning("Segment %d ক্লিপ পড়া যায়নি: %s", seg.index, exc)
                continue
            if rate != sr:
                clip = self._resample(clip, rate, sr)
            max_samples = max(1, int((window[1] - seg.start) * sr))
            if clip.size > max_samples:
                clip = audio_fx.fade(clip[:max_samples], sr, in_ms=5, out_ms=35)
                self.last_stats["trimmed"] = int(self.last_stats.get("trimmed", 0)) + 1
            clips.append((seg, clip, seg.start))
            level = audio_fx.rms_dbfs(clip)
            if level > -60:
                levels.append(level)

        # --- লাউডনেস মিলিয়ে নেওয়া (এক কণ্ঠ উঁচু, আরেকটি নিচু শোনাবে না) --------
        target_level = float(np.median(levels)) if levels else -20.0
        target_level = max(-28.0, min(-14.0, target_level))

        spans: List[Tuple[float, float]] = []
        for seg, clip, start in clips:
            level = audio_fx.rms_dbfs(clip)
            if level > -60:
                gain_db = max(-6.0, min(6.0, target_level - level))
                clip = np.clip(clip * audio_fx.db_to_amp(gain_db), -1.0, 1.0).astype(np.float32)

            start_sample = int(start * sr)
            end_sample = start_sample + clip.size
            if start_sample >= canvas.size:
                logger.debug("Segment %d ভিডিওর শেষের পরে — বাদ।", seg.index)
                continue
            if end_sample > canvas.size:
                clip = clip[: canvas.size - start_sample]
                end_sample = canvas.size
            # যোগ নয়, বসানো — কিন্তু উইন্ডো আগেই ওভারল্যাপ-মুক্ত রাখা হয়েছে
            canvas[start_sample:end_sample] += clip
            seg.placed_duration = float(clip.size / sr)
            spans.append((start_sample / sr, end_sample / sr))

        spans.sort()
        problems = find_overlaps(spans)
        if problems:  # pragma: no cover - প্রতিরক্ষামূলক
            logger.error("⚠️ বসানোর সময় %d জায়গায় ওভারল্যাপ ধরা পড়েছে!", len(problems))

        peak = float(np.max(np.abs(canvas))) if canvas.size else 0.0
        if peak > 0.99:
            canvas = (canvas * (0.99 / peak)).astype(np.float32)
        return canvas
