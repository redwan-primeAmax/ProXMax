"""Audio array resampling and non-overlapping clip assembly."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from . import audio_fx
from .models import Segment
from .utils import get_logger

logger = get_logger(__name__)


class SyncAssemblyMixin:
    """Methods shared by the timeline fitter to resample and place clips."""

    @staticmethod
    def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """NumPy linear resample (শুধু দরকার হলে)।"""
        if src_rate == dst_rate or audio.size == 0:
            return audio
        target = max(1, int(round(audio.size * dst_rate / float(src_rate))))
        index = np.linspace(0.0, audio.size - 1.0, num=target, dtype=np.float64)
        return np.interp(index, np.arange(audio.size), audio).astype(np.float32)

    def _assemble(
        self,
        pairs: Sequence[Tuple[Segment, Tuple[float, float]]],
        total_duration: float,
    ) -> np.ndarray:
        """সব stretched ক্লিপ নির্দিষ্ট সময়ে বসিয়ে একটাই ট্র্যাক বানায়।

        ``pairs`` = ``(segment, window)`` — উইন্ডো আগেই ওভারল্যাপ-মুক্ত।
        কোনো clip window-তে না আঁটলে সেটি কেটে দেওয়ার বদলে unplaced issue
        হিসেবে নথিভুক্ত হয়, তাই sentence tail নীরবে বাদ যায় না।
        """
        sr = self.sample_rate
        # The picture plan already rounds each edit to a whole video frame.
        # Do not append extra silence beyond the planned picture timeline.
        canvas = np.zeros(max(1, int(round(total_duration * sr))), dtype=np.float32)

        clips: List[Tuple[Segment, np.ndarray, float]] = []
        levels: List[float] = []
        for seg, window in pairs:
            try:
                clip, rate = audio_fx.read_wav(seg.stretched_audio_path)  # type: ignore[arg-type]
                if rate <= 0 or clip.size == 0 or not np.any(clip):
                    raise ValueError("empty or silent stretched clip")
            except Exception as exc:
                seg.sync_issue = f"stretched clip unreadable ({type(exc).__name__})"
                self.last_stats["unplaced"] = int(self.last_stats.get("unplaced", 0)) + 1
                self.last_stats.setdefault("sync_issues", []).append({
                    "segment_index": int(seg.index),
                    "start": round(float(seg.start), 3),
                    "reason": seg.sync_issue,
                })
                logger.warning("Segment %d ক্লিপ পড়া যায়নি: %s", seg.index, exc)
                continue
            if rate != sr:
                clip = self._resample(clip, rate, sr)
            start = max(0.0, seg.actual_start)
            # Round absolute boundaries, not the duration independently; that
            # avoids a one-sample overlap when neighbouring windows touch.
            max_samples = max(1, int(round(window[1] * sr)) - int(round(start * sr)))
            # The frame planner has already reserved enough picture for the
            # *measured* waveform. If the file changed or a boundary disagrees,
            # fail instead of spilling into another window or cropping speech.
            if clip.size > max_samples:
                overrun = (clip.size - max_samples) / float(sr)
                seg.sync_issue = (
                    f"measured clip exceeds its planned video window by {overrun:.3f}s; not truncated"
                )
                self.last_stats["unplaced"] = int(self.last_stats.get("unplaced", 0)) + 1
                self.last_stats.setdefault("sync_issues", []).append({
                    "segment_index": int(seg.index),
                    "start": round(float(seg.start), 3),
                    "overrun_seconds": round(overrun, 3),
                    "overrun_before_silence_trim_seconds": round(seg.sync_overrun_before_seconds, 3),
                    "trimmed_head_seconds": round(seg.sync_trimmed_head_seconds, 3),
                    "trimmed_internal_seconds": round(seg.sync_trimmed_internal_seconds, 3),
                    "trimmed_internal_cuts": seg.sync_trimmed_internal_cuts,
                    "trimmed_tail_seconds": round(seg.sync_trimmed_tail_seconds, 3),
                    "tempo_used": round(float(seg.tempo_used), 3),
                    "reason": "measured speech exceeds planned video window; left unplaced",
                })
                logger.error(
                    "🛑 Segment %d: measured speech exceeds planned picture by %.3fs; বাক্য কাটা হয়নি।",
                    seg.index, overrun,
                )
                continue
            clips.append((seg, clip, start))
            level = audio_fx.rms_dbfs(clip)
            if level > -60:
                levels.append(level)

        # --- লাউডনেস মিলিয়ে নেওয়া (এক কণ্ঠ উঁচু, আরেকটি নিচু শোনাবে না) --------
        target_level = float(np.median(levels)) if levels else -20.0
        target_level = max(-28.0, min(-14.0, target_level))

        spans: List[Tuple[int, int]] = []
        for seg, clip, start in clips:
            level = audio_fx.rms_dbfs(clip)
            if level > -60:
                gain_db = max(-6.0, min(6.0, target_level - level))
                clip = np.clip(clip * audio_fx.db_to_amp(gain_db), -1.0, 1.0).astype(np.float32)

            start_sample = int(round(start * sr))
            end_sample = start_sample + clip.size
            if start_sample >= canvas.size:
                seg.sync_issue = "clip starts beyond final track bounds"
                self.last_stats["unplaced"] = int(self.last_stats.get("unplaced", 0)) + 1
                self.last_stats.setdefault("sync_issues", []).append({
                    "segment_index": int(seg.index),
                    "start": round(float(seg.start), 3),
                    "reason": seg.sync_issue,
                })
                logger.warning("Segment %d ভিডিওর শেষের পরে শুরু — unplaced হিসেবে নথিভুক্ত।", seg.index)
                continue
            if end_sample > canvas.size:
                seg.sync_issue = "clip exceeds final track bounds; not truncated"
                self.last_stats["unplaced"] = int(self.last_stats.get("unplaced", 0)) + 1
                self.last_stats.setdefault("sync_issues", []).append({
                    "segment_index": int(seg.index),
                    "start": round(float(seg.start), 3),
                    "overrun_seconds": round((end_sample - canvas.size) / float(sr), 3),
                    "reason": "clip extends beyond final track bounds",
                })
                continue
            # যোগ নয়, বসানো — কিন্তু উইন্ডো আগেই ওভারল্যাপ-মুক্ত রাখা হয়েছে
            canvas[start_sample:end_sample] += clip
            seg.placed_start = start_sample / float(sr)
            seg.placed_duration = float(clip.size / sr)
            spans.append((start_sample, end_sample))

        spans.sort()
        # Compare integer samples: even a one-sample overlap is forbidden.
        problems = sum(left[1] > right[0] for left, right in zip(spans, spans[1:]))
        if problems:
            raise RuntimeError(f"Sync produced {problems} overlapping speech clips")

        peak = float(np.max(np.abs(canvas))) if canvas.size else 0.0
        if peak > 0.99:
            canvas = (canvas * (0.99 / peak)).astype(np.float32)
        return canvas
