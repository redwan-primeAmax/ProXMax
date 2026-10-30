"""Audio-video sync: fit Bangla speech to the timeline with zero overlap.

কণ্ঠকে সময়ের সাথে মেলানো হয় — **কখনোই ওভারল্যাপ নয়**। দুটি মূল সমস্যা সমাধান করা হয়:

1. **ওভারল্যাপ (দুই কণ্ঠ একসাথে)** — আগে ``pydub``-এর ``overlay()``
   ব্যবহার হতো, যা আসলে অডিও **যোগ** করে। কোনো বাংলা ক্লিপ তার
   স্লটের চেয়ে লম্বা হলে সে পরের বাক্যের উপর চড়ে যেত। এখন প্রতিটি
   ক্লিপের জন্য একটি **কঠোর উইন্ডো** হিসাব হয় (পরের ক্লিপ শুরুর
   আগেই শেষ), আর বসানোর আগে ওভারল্যাপ গাণিতিকভাবে যাচাই করা হয়।
2. **রোবটে/ধাতব কণ্ঠ ও বাক্য-কাটা** — প্রথম fit-এ tempo ০.৯০×–১.২০× সীমায় থাকে; আগে
   শেষ থেকে শুরু পর্যন্ত শনাক্ত silence প্রয়োজনমতো trim হয়, তারপর measured TTS সময়কে
   window-তে fit করার চেষ্টা হয়। সর্বোচ্চ tempo মাপার পরও unfit হলে translation/TTS retry হয়;
   তবুও না আঁটলে পৃথক uncapped fallback সম্পূর্ণ clip বসায় বা স্পষ্ট error দেয়।
   ``rubberband`` থাকলে pitch/formant-সহ time-stretch হয়।

চূড়ান্ত ট্র্যাক NumPy দিয়ে সাজানো হয় (pydub নয়) — কম মেমরি, কোনো
ফাইল-হ্যান্ডল লিক নেই, আর প্রতিটি প্লেসমেন্টের হিসাব নিজের হাতে।
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import audio_fx
from .models import Segment
from .sync_assembly import SyncAssemblyMixin
from .sync_cache import PreparedClipCache
from .sync_cap_retry import retry_at_tempo_cap, suspicious_stretch_shortfall
from .utils import ensure_tool, get_logger, run_ffmpeg
from .sync_math import (
    build_atempo_filter, build_stretch_filter, compute_tempo, find_overlaps, plan_windows,
)
from .sync_silence import (
    CONSERVATIVE_SILENCE_DBFS, trim_edge_silence_for_overrun, trim_silence_for_overrun,
)

logger = get_logger(__name__)


class SyncEngine(SyncAssemblyMixin):
    """Builds the final, full-length Bangla audio track for a video."""

    def __init__(
        self,
        min_tempo: float = 0.90,
        max_tempo: float = 1.20,
        slow_down_short_clips: bool = True,
        sample_rate: int = 24000,
        max_workers: int = 4,
        slot_gap_ms: int = 80,
        borrow_silence_ms: int = 250,
        prefer_rubberband: bool = True,
        fill_extension: float = 0.12,
        silence_threshold_dbfs: float = -45.0,
        silence_pad_ms: int = 25,
    ) -> None:
        """Args:
            min_tempo: ধীর করার সর্বনিম্ন সীমা (ডিফল্ট ০.৯০×)।
            max_tempo: দ্রুত করার সর্বোচ্চ সীমা (ডিফল্ট ১.২০×)।
            slow_down_short_clips: ছোট ক্লিপ সর্বোচ্চ ০.৯০× পর্যন্ত ধীর হবে কি না।
            sample_rate: চূড়ান্ত ট্র্যাকের sample rate।
            max_workers: সমান্তরাল ffmpeg প্রসেস।
            slot_gap_ms: দুই ক্লিপের মাঝে ন্যূনতম ফাঁক।
            borrow_silence_ms: পরের নীরবতা থেকে সর্বোচ্চ কত ধার নেওয়া যাবে।
            prefer_rubberband: ffmpeg-এ থাকলে rubberband ব্যবহার করা হবে।
            fill_extension: legacy extension parameter (short clips use min_tempo).
            silence_threshold_dbfs: silence শনাক্ত করার threshold (সর্বোচ্চ −70 dBFS)।
            silence_pad_ms: শনাক্ত কথার চারপাশে যে padding অক্ষত থাকবে।
        """
        self.min_tempo = min(1.0, max(0.90, float(min_tempo)))
        self.max_tempo = max(1.0, min(1.20, float(max_tempo)))
        self.slow_down_short_clips = bool(slow_down_short_clips)
        self.sample_rate = int(sample_rate)
        self.max_workers = max(1, int(max_workers))
        self.slot_gap_seconds = max(0.0, slot_gap_ms / 1000.0)
        self.borrow_seconds = max(0.0, borrow_silence_ms / 1000.0)
        self.prefer_rubberband = bool(prefer_rubberband)
        self.fill_extension = max(0.0, min(0.5, float(fill_extension)))
        self.silence_threshold_dbfs = float(silence_threshold_dbfs)
        self.silence_pad_ms = max(0, int(silence_pad_ms))
        #: সর্বশেষ রানের পরিসংখ্যান (রিপোর্টে ব্যবহৃত)
        self.last_stats: Dict[str, object] = {}
        self._prepared_cache = PreparedClipCache()

    def _stretch_one(self, seg: Segment, window: Tuple[float, float], out_path: Path) -> Optional[Path]:
        """Trim detected silence end-first, then fit within configured tempo limits."""
        self.max_tempo = max(1.0, min(1.20, float(self.max_tempo)))
        source = Path(seg.tts_audio_path) if seg.tts_audio_path else None
        if source is None or not source.is_file():
            seg.sync_issue = "TTS audio file missing"
            return None
        try:
            clip, rate = audio_fx.read_wav(source)
        except Exception as exc:
            seg.sync_issue = f"TTS audio unreadable ({type(exc).__name__})"
            logger.warning("Segment %d-এর TTS অডিও পড়া যাচ্ছে না: %s", seg.index, exc)
            return None

        rate = int(rate or self.sample_rate)
        raw_duration = clip.size / float(rate)
        seg.tts_duration = raw_duration
        window_length = window[1] - window[0]
        if window_length <= 0:
            seg.sync_issue = "sync window has no positive duration; clip left unplaced"
            return None
        overrun = max(0.0, raw_duration - window_length)
        seg.sync_overrun_before_seconds = overrun
        seg.sync_trimmed_head_seconds = 0.0
        seg.sync_trimmed_tail_seconds = 0.0
        seg.sync_trimmed_internal_seconds = 0.0
        seg.sync_trimmed_internal_cuts = 0
        seg.sync_tempo_cap_measured = False
        if overrun > 0.5 / rate:
            logger.info(
                "Segment %d overrun: %.3fs (TTS %.3fs, window %.3fs); শেষ থেকে silence scan/trim।",
                seg.index, overrun, raw_duration, window_length,
            )
            trimmed = trim_silence_for_overrun(
                clip, rate, overrun,
                threshold_dbfs=self.silence_threshold_dbfs, pad_ms=self.silence_pad_ms,
            )
            clip = trimmed.audio
            seg.sync_trimmed_head_seconds = trimmed.head_seconds
            seg.sync_trimmed_internal_seconds = trimmed.internal_seconds
            seg.sync_trimmed_internal_cuts = trimmed.internal_cuts
            seg.sync_trimmed_tail_seconds = trimmed.tail_seconds
            if trimmed.head_seconds or trimmed.internal_seconds or trimmed.tail_seconds:
                prepared = out_path.with_name(f"{out_path.stem}_silence_trimmed.wav")
                audio_fx.write_wav(prepared, clip, rate)
                source = prepared
                logger.info(
                    "Segment %d: tail %.3fs, internal %.3fs (%d pauses), head %.3fs বাদ; অবশিষ্ট %.3fs।",
                    seg.index, trimmed.tail_seconds, trimmed.internal_seconds, trimmed.internal_cuts,
                    trimmed.head_seconds, clip.size / float(rate),
                )

        fit_duration = clip.size / float(rate)
        tempo = compute_tempo(
            fit_duration, window_length,
            min_tempo=self.min_tempo, max_tempo=self.max_tempo,
            slow_down_short_clips=self.slow_down_short_clips,
            fill_extension=self.fill_extension,
            fill_gap_seconds=self.slot_gap_seconds,
        )
        logger.debug(
            "Segment %d fit: raw=%.3fs, usable=%.3fs, target=%.3fs, tempo=%.3f×, projected=%.3fs.",
            seg.index, raw_duration, fit_duration, window_length, tempo,
            fit_duration / max(tempo, 0.01),
        )

        if tempo <= 1.0 and abs(tempo - 1.0) < 0.005:
            try:
                if rate != self.sample_rate:
                    clip = self._resample(clip, rate, self.sample_rate)
                audio_fx.write_wav(out_path, clip, self.sample_rate)
            except Exception as exc:
                seg.sync_issue = f"stretched audio copy failed ({type(exc).__name__})"
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
                seg.sync_issue = f"time-stretch failed ({type(exc).__name__})"
                logger.warning("Segment %d stretch ব্যর্থ, বাদ দেওয়া হলো: %s", seg.index, exc)
                return None

        try:
            clip, rate = audio_fx.read_wav(out_path)
            if rate <= 0 or clip.size == 0 or not np.any(clip):
                raise ValueError("empty/silent stretched waveform or invalid sample rate")
            duration = clip.size / float(rate)
            # Allow FFmpeg's normal frame rounding, but reject a suspiciously
            # short render rather than silently losing speech or exceeding the cap.
            if tempo > 1.0 and suspicious_stretch_shortfall(fit_duration, tempo, duration):
                raise ValueError("unexpectedly short stretch; possible speech loss")
            remaining_video = max(0.0, getattr(self, "_total_duration", window[1] + self.slot_gap_seconds) - window[1])
            tolerance = min(0.01, self.slot_gap_seconds / 2.0, remaining_video)
            post_head = post_tail = 0.0
            if duration > window_length + tolerance:
                # A stretch filter can itself add edge padding. Do not cut speech.
                clip, post_head, post_tail = trim_edge_silence_for_overrun(
                    clip, rate, duration - window_length,
                    threshold_dbfs=min(self.silence_threshold_dbfs, CONSERVATIVE_SILENCE_DBFS),
                    pad_ms=self.silence_pad_ms,
                )
                if post_head or post_tail:
                    audio_fx.write_wav(out_path, clip, rate)
                    seg.sync_trimmed_head_seconds += post_head
                    seg.sync_trimmed_tail_seconds += post_tail
                    duration = clip.size / float(rate)
            seg.tempo_used = tempo
            seg.sync_tempo_cap_measured = round(tempo, 4) >= round(self.max_tempo, 4)
            if duration > window_length + tolerance and not seg.sync_tempo_cap_measured:
                # FFmpeg's real output can differ from the theoretical ratio.
                # Try the allowed cap on the prepared source before any rewrite.
                cap = retry_at_tempo_cap(
                    source, out_path, seg.index, self.max_tempo, self.sample_rate,
                    self.prefer_rubberband, window_length, tolerance, duration, fit_duration,
                    self.silence_threshold_dbfs, self.silence_pad_ms, run_ffmpeg,
                )
                seg.sync_tempo_cap_measured = cap.measured
                if cap.adopted:
                    duration = cap.duration
                    seg.tempo_used = self.max_tempo
                    seg.sync_trimmed_head_seconds += cap.head_seconds - post_head
                    seg.sync_trimmed_tail_seconds += cap.tail_seconds - post_tail
            seg.stretched_duration = duration
            seg.stretched_audio_path = out_path
            return out_path
        except Exception as exc:
            # Never gate a semantic rewrite or placement on an estimated duration.
            seg.sync_issue = (
                "stretched audio unexpectedly short; possible speech loss; clip left unplaced"
                if str(exc) == "unexpectedly short stretch; possible speech loss"
                else f"stretched audio measurement failed ({type(exc).__name__})"
            )
            logger.warning("Segment %d-এর stretched audio মাপা যায়নি: %s", seg.index, exc)
            return None

    def build_final_track(
        self,
        segments: List[Segment],
        total_duration: float,
        work_dir: Path | str,
        output_wav: Path | str,
        *,
        allow_empty_intermediate: bool = False,
    ) -> Path:
        """Stretch every clip and assemble one full-length, non-overlapping WAV.

        Raises:
            RuntimeError: ব্যবহারযোগ্য কোনো TTS ক্লিপই না থাকলে।
        """
        work_dir, output_wav = Path(work_dir), Path(output_wav)
        self.max_tempo = max(1.0, min(1.20, float(self.max_tempo)))
        # প্রতি ভিডিওতে পরিসংখ্যান ও placement state নতুন করে শুরু।
        self.last_stats = {"trimmed": 0, "unplaced": 0, "sync_issues": []}
        stretch_dir = work_dir / "stretched"
        stretch_dir.mkdir(parents=True, exist_ok=True)
        output_wav.parent.mkdir(parents=True, exist_ok=True)

        usable = [s for s in sorted(segments, key=lambda s: s.start) if s.tts_audio_path]
        if not usable:
            raise RuntimeError("Sync করার মতো কোনো TTS অডিও ক্লিপ নেই।")
        for segment in usable:
            segment.stretched_audio_path = None
            segment.stretched_duration = None
            segment.placed_duration = None
            segment.placed_start = None
            segment.sync_recovery = None
            segment.sync_issue = None
            segment.tempo_used = 1.0
            segment.sync_overrun_before_seconds = 0.0
            segment.sync_trimmed_head_seconds = 0.0
            segment.sync_trimmed_tail_seconds = 0.0
            segment.sync_trimmed_internal_seconds = 0.0
            segment.sync_trimmed_internal_cuts = 0
            segment.sync_tempo_cap_measured = False
        if total_duration <= 0:
            total_duration = max(s.end for s in usable) + 0.5
        self._total_duration = float(total_duration)

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
        # Retrying a few overlong clips must not rerun FFmpeg for every other
        # unchanged clip. Exact windows, source/artifact signatures and all fit
        # settings are checked; loudness and placement are always recalculated.
        cache = self._prepared_cache
        cache.begin(stretch_dir)
        settings = (
            self.min_tempo, self.max_tempo, self.slow_down_short_clips, self.sample_rate,
            self.slot_gap_seconds, self.prefer_rubberband, self.fill_extension,
            self.silence_threshold_dbfs, self.silence_pad_ms, self._total_duration,
        )
        jobs = []
        reused = 0
        for i, seg in enumerate(usable):
            window = windows[i]
            path = stretch_dir / f"seg_{seg.index:04d}.wav"
            key = cache.key(seg, window, settings)
            if cache.restore(seg, path, key):
                reused += 1
            else:
                jobs.append((seg, window, path, key))
        if jobs:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                list(pool.map(lambda job: self._stretch_one(*job[:3]), jobs))
            for seg, window, path, key in jobs:
                cache.store(seg, path, key, settings, window)
        self.last_stats.update({"prepared_reused": reused, "prepared_computed": len(jobs)})

        head_trim = sum(s.sync_trimmed_head_seconds for s in usable)
        internal_trim = sum(s.sync_trimmed_internal_seconds for s in usable)
        tail_trim = sum(s.sync_trimmed_tail_seconds for s in usable)
        self.last_stats.update({
            "overrun_before_silence_trim_seconds": round(
                sum(s.sync_overrun_before_seconds for s in usable), 3
            ),
            "silence_trimmed_clips": sum(
                1 for s in usable if s.sync_trimmed_head_seconds
                + s.sync_trimmed_internal_seconds + s.sync_trimmed_tail_seconds > 0
            ),
            "silence_trimmed_head_seconds": round(head_trim, 3),
            "silence_trimmed_internal_seconds": round(internal_trim, 3),
            "silence_trimmed_internal_cuts": sum(s.sync_trimmed_internal_cuts for s in usable),
            "silence_trimmed_tail_seconds": round(tail_trim, 3),
            "silence_trimmed_seconds": round(head_trim + internal_trim + tail_trim, 3),
        })

        # Failed clips are reported explicitly; their source slots remain silent.
        for segment in usable:
            if segment.stretched_audio_path is None and segment.sync_issue:
                self.last_stats["unplaced"] = int(self.last_stats.get("unplaced", 0)) + 1
                self.last_stats.setdefault("sync_issues", []).append({
                    "segment_index": int(segment.index),
                    "start": round(float(segment.start), 3),
                    "reason": segment.sync_issue,
                })

        # কোন ক্লিপ কোন উইন্ডোতে বসবে — জোড়া বেঁধে রাখা হয়, যাতে কোনো ক্লিপ
        # stretch ব্যর্থ হলেও বাকিগুলোর সময় (index) সরে না যায়।
        pairs = [(seg, windows[i]) for i, seg in enumerate(usable) if seg.stretched_audio_path]
        if not pairs and not allow_empty_intermediate:
            raise RuntimeError("সব ক্লিপের time-stretch ব্যর্থ হয়েছে — চূড়ান্ত ট্র্যাক বানানো যাচ্ছে না।")
        stretched = [seg for seg, _ in pairs]

        canvas = self._assemble(pairs, total_duration)
        audio_fx.write_wav(output_wav, canvas, self.sample_rate)

        tempos = [float(getattr(s, "tempo_used", 1.0) or 1.0) for s in stretched]
        max_tempo = max(tempos) if tempos else 1.0
        aggressive = sum(1 for t in tempos if t > 1.10)
        slowed = sum(1 for t in tempos if t < 0.98)
        speech_seconds = sum(float(s.placed_duration or 0.0) for s in stretched)
        silence = max(0.0, float(total_duration) - speech_seconds)
        previous_stats = dict(self.last_stats)
        self.last_stats = {
            **previous_stats,
            "clips": len(stretched),
            "overlaps": 0,
            "max_tempo": max_tempo,
            "aggressive_stretch": aggressive,
            "slowed_to_fill": slowed,
            "trimmed": int(previous_stats.get("trimmed", 0)),
            "unplaced": int(previous_stats.get("unplaced", 0)),
            "sync_issues": list(previous_stats.get("sync_issues", [])),
            "speech_seconds": round(speech_seconds, 2),
            "silence_seconds": round(silence, 2),
            "silence_percent": round(100.0 * silence / total_duration, 1) if total_duration else 0.0,
        }
        logger.info(
            "চূড়ান্ত Bangla অডিও: %s | %.1fs | %d ক্লিপ | সর্বোচ্চ tempo %.2f "
            "(জোরালো stretch %d, স্লট ভরাতে ধীর %d) | নীরবতা %.1fs (%.1f%%) | ওভারল্যাপ ০",
            output_wav.name, total_duration, len(stretched), max_tempo, aggressive, slowed,
            silence, self.last_stats["silence_percent"],
        )
        logger.info(
            "Silence-fit: আগে মোট overrun %.3fs; tail %.3fs, internal %.3fs, head %.3fs; %d clip; unplaced %d.",
            self.last_stats["overrun_before_silence_trim_seconds"],
            self.last_stats["silence_trimmed_tail_seconds"],
            self.last_stats["silence_trimmed_internal_seconds"],
            self.last_stats["silence_trimmed_head_seconds"],
            self.last_stats["silence_trimmed_clips"], self.last_stats["unplaced"],
        )
        return output_wav
