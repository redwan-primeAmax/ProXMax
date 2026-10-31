"""Prepare speech in parallel and build a non-overlapping edited-video timeline.

All detected TTS silence is removed first. Speech is never sped up: short
speech may be slowed down to 0.90x, while long speech gets repeated video-tail
frames. Short video windows may be sped up to 1.15x, leaving any remaining gap.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import audio_fx
from .models import Segment
from .sync_assembly import SyncAssemblyMixin
from .sync_cache import PreparedClipCache
from .sync_math import build_atempo_filter, compute_tempo, find_overlaps, plan_windows
from .sync_prepare import prepare_clip
from .utils import get_logger
from .video_timeline import VideoPlan, build_video_plan, parse_fps, source_windows

logger = get_logger(__name__)


class SyncEngine(SyncAssemblyMixin):
    """Prepare and schedule all speech, plus a frame-accurate picture edit plan."""

    def __init__(
        self, min_tempo: float = 0.90, max_tempo: float = 1.0,
        slow_down_short_clips: bool = True, sample_rate: int = 24000,
        max_workers: int = 4, slot_gap_ms: int = 80, borrow_silence_ms: int = 250,
        prefer_rubberband: bool = True, fill_extension: float = 0.12,
        silence_threshold_dbfs: float = -70.0, silence_pad_ms: int = 25,
        video_max_speed: float = 1.15,
    ) -> None:
        # Legacy speedup/disable flags are deliberately ignored; callers cannot
        # accidentally re-enable the old audio-speedup or no-slowdown paths.
        self.min_tempo = min(1.0, max(0.90, float(min_tempo)))
        self.max_tempo = 1.0
        self.slow_down_short_clips = True
        self.sample_rate = int(sample_rate)
        self.max_workers = max(1, min(16, int(max_workers)))
        self.slot_gap_seconds = max(0.0, float(slot_gap_ms) / 1000.0)
        self.borrow_seconds = max(0.0, float(borrow_silence_ms) / 1000.0)
        self.prefer_rubberband = bool(prefer_rubberband)
        self.fill_extension = float(fill_extension)  # obsolete, retained for API compatibility
        self.silence_threshold_dbfs = min(-70.0, float(silence_threshold_dbfs))
        self.silence_pad_ms = max(10, int(silence_pad_ms))  # preserve speech-boundary padding
        self.video_max_speed = min(1.15, max(1.0, float(video_max_speed)))
        self.last_stats: Dict[str, object] = {}
        self.video_plan: Optional[VideoPlan] = None
        self._prepared_cache = PreparedClipCache()

    def _stretch_one(self, seg: Segment, window: Tuple[float, float], out_path: Path) -> Optional[Path]:
        return prepare_clip(self, seg, window, out_path)

    def build_final_track(
        self, segments: List[Segment], total_duration: float, work_dir: Path | str,
        output_wav: Path | str, *, allow_empty_intermediate: bool = False,
        video_fps: Fraction | str | float = Fraction(25, 1),
    ) -> Path:
        """Prepare every voice clip, schedule it and produce the edited timeline.

        With ``allow_empty_intermediate``, a failed preparation yields a silent
        temporary track and explicit unplaced stats so invalid TTS can be retried.
        Such an intermediate track must never be muxed or reported as success.
        """
        if not math.isfinite(float(total_duration)) or total_duration <= 0 or self.sample_rate <= 0:
            raise ValueError("A positive video duration and sample rate are required")
        fps = parse_fps(video_fps)
        ordered = [s for s in sorted(segments, key=lambda s: (s.start, s.index))
                   if s.tts_audio_path or (s.text_en or "").strip() or (s.text_bn or "").strip() or s.words]
        if not ordered:
            raise RuntimeError("Sync করার মতো কোনো TTS অডিও ক্লিপ নেই।")
        if len({s.index for s in ordered}) != len(ordered):
            raise RuntimeError("Duplicate segment indexes prevent safe audio placement")
        if any(not math.isfinite(float(s.start)) or not math.isfinite(float(s.end)) for s in ordered):
            raise ValueError("Non-finite source timestamp")
        self.max_tempo = 1.0
        self.video_plan = None
        self.last_stats = {"trimmed": 0, "unplaced": 0, "overlaps": 0, "sync_issues": []}
        stretch_dir = Path(work_dir) / "stretched"
        stretch_dir.mkdir(parents=True, exist_ok=True)
        output_wav = Path(output_wav)
        output_wav.parent.mkdir(parents=True, exist_ok=True)

        total_frames, windows = source_windows(
            ordered, float(total_duration), fps, min_gap=self.slot_gap_seconds, borrow=self.borrow_seconds,
        )
        seconds = [(float(Fraction(a, 1) / fps), float(Fraction(b, 1) / fps)) for a, b in windows]
        for seg in ordered:
            seg.stretched_audio_path = None
            seg.stretched_duration = seg.placed_duration = seg.placed_start = None
            seg.sync_issue = seg.sync_recovery = None
            seg.sync_tempo_cap_measured = False
            seg.tempo_used = 1.0
            seg.video_window_start = seg.video_window_end = None
            seg.video_source_start = seg.video_source_end = None
            seg.video_speed, seg.video_added_seconds = 1.0, 0.0

        cache = self._prepared_cache
        cache.begin(stretch_dir)
        settings = (self.min_tempo, self.sample_rate, self.prefer_rubberband,
                    self.silence_threshold_dbfs, self.silence_pad_ms, str(fps))
        jobs = []
        reused = 0
        for seg, window in zip(ordered, seconds):
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

        failed = [s for s in ordered if not s.stretched_audio_path or s.stretched_duration is None]
        if failed:
            self.last_stats["unplaced"] = len(failed)
            self.last_stats["sync_issues"] = [
                {"segment_index": int(s.index), "start": round(float(s.start), 3),
                 "reason": s.sync_issue or "TTS preparation failed"} for s in failed
            ]
            if not allow_empty_intermediate:
                raise RuntimeError(f"Cannot prepare all TTS clips: {[s.index for s in failed]}")
            audio_fx.write_wav(output_wav, np.zeros(max(1, round(total_duration * self.sample_rate)), np.float32),
                               self.sample_rate)
            return output_wav

        self.video_plan = build_video_plan(ordered, windows, total_frames, fps, self.sample_rate,
                                           self.video_max_speed)
        pairs = [(s, (float(s.video_window_start), float(s.video_window_end))) for s in ordered]
        canvas = self._assemble(pairs, self.video_plan.duration)
        missing = [s.index for s in ordered if s.placed_duration is None]
        spans = [(s.actual_start, s.actual_start + float(s.placed_duration or 0.0)) for s in ordered]
        if missing or self.last_stats["unplaced"] or find_overlaps(spans):
            raise RuntimeError(f"Could not safely place every audio segment; missing={missing}")
        if any(s.tempo_used > 1.0 + 1e-9 or s.tempo_used < 0.90 - 1e-9 for s in ordered):
            raise RuntimeError("Audio tempo violated the 0.90x–1.00x safety policy")
        audio_fx.write_wav(output_wav, canvas, self.sample_rate)
        self._record_stats(ordered)
        return output_wav

    def _record_stats(self, segments: List[Segment]) -> None:
        assert self.video_plan is not None
        def total(tts_name: str, sync_name: str) -> float:
            return sum(float(getattr(s, tts_name) or 0.0) + float(getattr(s, sync_name) or 0.0)
                       for s in segments)
        head = total("tts_trimmed_head_seconds", "sync_trimmed_head_seconds")
        inside = total("tts_trimmed_internal_seconds", "sync_trimmed_internal_seconds")
        tail = total("tts_trimmed_tail_seconds", "sync_trimmed_tail_seconds")
        removed = head + inside + tail
        speech = sum(float(s.placed_duration or 0.0) for s in segments)
        duration = self.video_plan.duration
        shifted = [{"segment_index": s.index, "seconds": round(s.actual_start - s.start, 3)} for s in segments
                   if abs(s.actual_start - s.start) > 0.001]
        added = sum(float(s.video_added_seconds) for s in segments)
        saved = sum((p.source_frames - p.output_frames) / float(self.video_plan.fps)
                    for p in self.video_plan.pieces if p.kind == "speed")
        self.last_stats.update({
            "policy": "video-tail-v1", "clips": len(segments), "unplaced": 0, "overlaps": 0,
            "max_tempo": max((s.tempo_used for s in segments), default=1.0),
            "min_tempo": min((s.tempo_used for s in segments), default=1.0),
            "slowed_to_fill": sum(s.tempo_used < 0.999999 for s in segments),
            "speech_seconds": round(speech, 3), "silence_seconds": round(max(0.0, duration - speech), 3),
            "silence_percent": round(100 * max(0.0, duration - speech) / duration, 1),
            "silence_trimmed_head_seconds": round(head, 3),
            "silence_trimmed_internal_seconds": round(inside, 3),
            "silence_trimmed_tail_seconds": round(tail, 3), "silence_trimmed_seconds": round(removed, 3),
            "silence_trimmed_internal_cuts": sum(s.tts_trimmed_internal_cuts + s.sync_trimmed_internal_cuts for s in segments),
            "silence_trimmed_clips": sum(bool(s.tts_trimmed_head_seconds + s.tts_trimmed_internal_seconds
                                              + s.tts_trimmed_tail_seconds + s.sync_trimmed_head_seconds
                                              + s.sync_trimmed_internal_seconds + s.sync_trimmed_tail_seconds)
                                         for s in segments),
            "overrun_before_silence_trim_seconds": round(sum(s.sync_overrun_before_seconds for s in segments), 3),
            "video_added_seconds": round(added, 6), "video_saved_seconds": round(saved, 6),
            "video_extension_seconds": max(0.0, duration - self.video_plan.source_duration),
            "video_max_speed": max((s.video_speed for s in segments), default=1.0),
            "extended_windows": sum(s.video_added_seconds > 0 for s in segments),
            "sped_up_windows": sum(s.video_speed > 1 for s in segments),
            "shifted_segments": shifted, "timeline_duration_seconds": duration,
            "video_fps": str(self.video_plan.fps), "video_pieces": len(self.video_plan.pieces),
        })
        logger.info("Sync: %d speech clips, no speed-up; %d windows repeated, %d sped up; output %.3fs.",
                    len(segments), self.last_stats["extended_windows"], self.last_stats["sped_up_windows"], duration)
