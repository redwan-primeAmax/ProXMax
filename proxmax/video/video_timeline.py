"""Frame-based picture edits and sample-based speech placement.

The source picture is covered once, in order. A long dub adds a repetition of
its window's tail; a short dub may shorten only that window, by at most 1.15x.
No generated speech is accelerated or intentionally cut. Output is constant
frame rate, so picture edits are rounded to whole frames (audio remains sample
accurate); additions round up to ensure every audio sample is covered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from .models import Segment
from .sync_math import plan_windows


@dataclass(frozen=True)
class VideoPiece:
    """A source frame range rendered to exactly ``output_frames`` frames."""

    start: int
    end: int
    output_frames: int
    kind: str = "normal"  # normal, speed, repeat

    @property
    def source_frames(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class VideoPlan:
    fps: Fraction
    source_frames: int
    pieces: tuple[VideoPiece, ...]
    output_frames: int

    @property
    def duration(self) -> float:
        return float(Fraction(self.output_frames, 1) / self.fps)

    @property
    def source_duration(self) -> float:
        return float(Fraction(self.source_frames, 1) / self.fps)

    @property
    def changed(self) -> bool:
        return any(p.kind == "repeat" or p.source_frames != p.output_frames for p in self.pieces)


def parse_fps(value: Fraction | str | float | int) -> Fraction:
    try:
        fps = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, ZeroDivisionError, TypeError) as exc:
        raise ValueError(f"Invalid video frame rate: {value!r}") from exc
    if fps <= 0 or fps > 300:
        raise ValueError(f"Unsupported video frame rate: {value!r}")
    return fps


def source_windows(segments: Sequence[Segment], duration: float, fps: Fraction, *,
                   min_gap: float, borrow: float) -> tuple[int, list[tuple[int, int]]]:
    """Return non-overlapping source windows snapped to the picture frame grid."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Video duration must be finite and positive")
    fps = parse_fps(fps)
    total_frames = max(1, round(Fraction(str(duration)) * fps))
    raw = plan_windows(segments, duration, min_gap_seconds=min_gap, borrow_seconds=borrow)
    starts = [min(total_frames, max(0, round(Fraction(str(start)) * fps))) for start, _ in raw]
    windows: list[tuple[int, int]] = []
    previous_end = 0
    for i, (_, end) in enumerate(raw):
        start = max(previous_end, starts[i])
        limit = starts[i + 1] if i + 1 < len(starts) else total_frames
        end_frame = round(Fraction(str(max(0.0, end))) * fps)
        end_frame = max(start, min(total_frames, limit, end_frame))
        windows.append((start, end_frame))
        previous_end = end_frame
    return total_frames, windows


def build_video_plan(segments: Sequence[Segment], windows: Sequence[tuple[int, int]],
                     source_frames: int, fps: Fraction, sample_rate: int,
                     max_video_speed: float = 1.15) -> VideoPlan:
    """Make all video edits and set speech placements on the resulting timeline.

    Source windows and segments must be in chronological order. Each audio clip
    is measured after silence removal/slowdown. Every source frame is covered
    once by normal or sped-up pieces; extra pieces repeat only the window tail.
    """
    fps = parse_fps(fps)
    if len(segments) != len(windows) or source_frames <= 0 or sample_rate <= 0:
        raise ValueError("Invalid segments, windows, picture length or sample rate")
    cap = Fraction(str(max_video_speed))
    if cap < 1 or cap > Fraction(23, 20):
        raise ValueError("Video speed must be between 1.0x and 1.15x")
    pieces: list[VideoPiece] = []
    source_cursor = 0
    out_cursor = 0

    def append(start: int, end: int, count: int, kind: str) -> None:
        nonlocal out_cursor
        if start < 0 or end > source_frames or end <= start or count <= 0:
            raise ValueError(f"Invalid picture piece: {start}:{end} -> {count}")
        if kind == "normal" and pieces and pieces[-1].kind == "normal" and pieces[-1].end == start:
            prev = pieces[-1]
            pieces[-1] = VideoPiece(prev.start, end, prev.output_frames + count, "normal")
        else:
            pieces.append(VideoPiece(start, end, count, kind))
        out_cursor += count

    def repeat_tail(start: int, end: int, count: int) -> str | None:
        # Empty/single-frame windows cannot contain moving footage. When a
        # multi-frame repeat is needed, use the nearest available frames; an
        # input containing just one picture can only repeat that one picture.
        recovery = None
        if end - start < 2 and count > 1 and source_frames >= 2:
            recovery = "tiny source window: repeated nearest available moving picture frames"
            if end >= 2:
                start = max(0, end - max(count, 2))
            else:
                start, end = 0, min(source_frames, max(count, 2))
        elif end <= start:
            recovery = "empty source window: repeated nearest available picture frames"
            if end > 0:
                start = max(0, end - count)
            else:
                end = min(source_frames, count)
        if source_frames == 1 and count > 1:
            recovery = "source has only one picture frame; repeated motion is impossible"
        length = end - start
        if length <= 0:
            raise ValueError("No picture frames available to repeat")
        whole, remainder = divmod(count, length)
        for _ in range(whole):
            append(start, end, length, "repeat")
        if remainder:
            append(end - remainder, end, remainder, "repeat")
        return recovery

    for seg, (start, end) in zip(segments, windows):
        if start < source_cursor or start > end or end > source_frames:
            raise ValueError("Source windows overlap or exceed picture duration")
        if start > source_cursor:
            append(source_cursor, start, start - source_cursor, "normal")
        window_output_start = out_cursor
        start_sample = round(Fraction(out_cursor * sample_rate, 1) / fps)
        duration = float(seg.stretched_duration or 0.0)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Segment {seg.index} has no measurable prepared audio")
        samples = max(1, round(duration * sample_rate))
        required_end = math.ceil(Fraction((start_sample + samples), sample_rate) * fps)
        needed = max(1, required_end - out_cursor)
        source_count = end - start
        extra = max(0, needed - source_count)
        if source_count:
            minimum_count = math.ceil(Fraction(source_count, 1) / cap)
            output_count = min(source_count, max(needed, minimum_count))
            kind = "speed" if output_count < source_count else "normal"
            append(start, end, output_count, kind)
            speed = source_count / output_count
        else:
            output_count, speed = 0, 1.0
        recovery = repeat_tail(start, end, extra) if extra else None
        seg.placed_start = start_sample / float(sample_rate)
        seg.video_source_start = float(Fraction(start, 1) / fps)
        seg.video_source_end = float(Fraction(end, 1) / fps)
        seg.video_window_start = float(Fraction(window_output_start, 1) / fps)
        seg.video_window_end = float(Fraction(out_cursor, 1) / fps)
        seg.video_speed = speed
        seg.video_added_seconds = float(Fraction(extra, 1) / fps)
        if recovery:
            seg.sync_recovery = "; ".join(filter(None, (seg.sync_recovery, recovery)))
        source_cursor = end
    if source_cursor < source_frames:
        append(source_cursor, source_frames, source_frames - source_cursor, "normal")
    if not pieces:
        raise ValueError("No picture pieces were produced")
    return VideoPlan(fps=fps, source_frames=source_frames, pieces=tuple(pieces), output_frames=out_cursor)
