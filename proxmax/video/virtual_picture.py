"""Verified CFR source-tail mapping without an intermediate whole-video encode."""

from __future__ import annotations

import json
import re
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Callable

from .utils import FFmpegError, ensure_tool
from .video_timeline import VideoPiece, VideoPlan

# Bound metadata scanning/memory; unknown or large sources use the safe path.
_MAX_PACKETS = 200_000


def is_constant_frame_grid(source: Path, fps: Fraction, frames: int) -> bool:
    """Verify every packet's PTS/duration and the stream's exact CFR time base.

    Merely matching avg_frame_rate/r_frame_rate is not enough for VFR. If any
    packet, timestamp, duration, count or tool result is ambiguous, return
    False and let the caller retain the established materialized-source path.
    """
    if frames <= 0 or frames > _MAX_PACKETS:
        return False
    try:
        probe = ensure_tool("ffprobe")
        stream_proc = subprocess.run(
            [probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=time_base,avg_frame_rate,r_frame_rate", "-of", "json", str(source)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30, check=True,
        )
        stream = json.loads(stream_proc.stdout)["streams"][0]
        time_base = Fraction(stream["time_base"])
        if Fraction(stream["avg_frame_rate"]) != fps or Fraction(stream["r_frame_rate"]) != fps:
            return False
        packet_proc = subprocess.run(
            [probe, "-v", "error", "-select_streams", "v:0", "-read_intervals", f"%+#{_MAX_PACKETS + 1}",
             "-show_entries", "packet=pts,duration", "-of", "csv=p=0", str(source)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60, check=True,
        )
        rows = [line.strip().split(",") for line in packet_proc.stdout.splitlines() if line.strip()]
        if len(rows) != frames or any(len(row) != 2 for row in rows):
            return False
        packets = [(int(row[0]), int(row[1])) for row in rows]
        duration = packets[0][1]
        if duration <= 0 or duration * time_base != 1 / fps:
            return False
        if any(packet_duration != duration for _pts, packet_duration in packets):
            return False
        points = sorted(pts for pts, _duration in packets)
        return all(b - a == duration for a, b in zip(points, points[1:]))
    except (FFmpegError, OSError, ValueError, KeyError, IndexError, TypeError,
            subprocess.SubprocessError, ZeroDivisionError):
        return False


def mapped_spans(piece: VideoPiece, source_map: VideoPlan) -> list[tuple[int, int]]:
    """Map one virtual frame interval to ordered, unsped physical ranges."""
    cursor, spans = 0, []
    for part in source_map.pieces:
        if part.source_frames != part.output_frames:
            raise ValueError("Virtual source map must not retime its source")
        left, right = max(piece.start, cursor), min(piece.end, cursor + part.output_frames)
        if left < right:
            spans.append((part.start + left - cursor, part.start + right - cursor))
        cursor += part.output_frames
        if cursor >= piece.end:
            break
    if sum(end - start for start, end in spans) != piece.source_frames:
        raise FFmpegError("Virtual source map does not cover the planned picture piece")
    return spans


def render_virtual_piece(source: Path, piece: VideoPiece, fps: Fraction, out: Path,
                         encoder: str, source_map: VideoPlan, *,
                         run_ffmpeg: Callable, frame_count: Callable) -> None:
    """Concatenate mapped frames first, then apply the original piece's speed."""
    if piece.source_frames <= 0 or piece.output_frames <= 0:
        raise ValueError("Picture piece must contain frames")
    if Fraction(piece.source_frames, piece.output_frames) > Fraction(23, 20):
        raise ValueError("Picture speed exceeds the 1.15x cap")
    spans = mapped_spans(piece, source_map)
    fps_text = f"{fps.numerator}/{fps.denominator}"
    command = [ensure_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "info",
               "-filter_complex_threads", "1"]
    filters = []
    for index, (start, end) in enumerate(spans):
        seek, count = float(Fraction(start, 1) / fps), end - start
        command += ["-ss", f"{seek:.9f}", "-threads", "1", "-i", str(source)]
        filters.append(f"[{index}:v]setpts=PTS-STARTPTS,fps=fps={fps_text}:start_time=0,"
                       f"trim=end_frame={count},setpts=PTS-STARTPTS,split=2[s{index}][check{index}]")
        # Emit just the last expected source frame to a diagnostic sink. This
        # prevents the look-ahead below from hiding an under-decoded span.
        filters.append(f"[check{index}]select='eq(n,{count - 1})',showinfo@tailcheck{index},nullsink")
        if index == len(spans) - 1:
            filters.append(f"[s{index}]tpad=stop=1:stop_mode=clone[v{index}]")
        else:
            filters.append(f"[s{index}]null[v{index}]")
    labels = "".join(f"[v{i}]" for i in range(len(spans)))
    # Match the original piece renderer's fps time base before applying its
    # speed multiplier; microsecond concat PTS round differently on speedups.
    clock = f"settb=expr={fps.denominator}/{fps.numerator},setpts=N"
    if len(spans) == 1:
        combined = f"[v0]{clock}"
    else:
        combined = f"{labels}concat=n={len(spans)}:v=1:a=0,{clock}"
    filters.append(
        # The final input supplies one look-ahead frame to prevent concat's
        # EOF dropping its last frame in fps; trim excludes the look-ahead.
        f"{combined},setpts=PTS-STARTPTS,"
        f"setpts=PTS*{piece.output_frames}/{piece.source_frames},"
        f"fps=fps={fps_text}:start_time=0,trim=end_frame={piece.output_frames},"
        "setpts=PTS-STARTPTS,pad=ceil(iw/2)*2:ceil(ih/2)*2[out]"
    )
    codec = (["-c:v", "h264_nvenc", "-preset", "fast", "-cq", "23"] if encoder == "h264_nvenc"
             else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"])
    command += ["-filter_complex", ";".join(filters), "-map", "[out]", "-an",
                "-frames:v", str(piece.output_frames), *codec, "-threads", "2", "-bf", "0",
                "-pix_fmt", "yuv420p", "-r", fps_text, str(out)]
    result = run_ffmpeg(command, description="video piece with source tail", timeout=1800.0)
    log = result.stderr or ""
    for index in range(len(spans)):
        if not re.search(rf"\[showinfo@tailcheck{index}\s+@[^\]]+\]\s+n:\s*0\b", log):
            raise FFmpegError("Cannot verify every decoded virtual-tail source span")
    actual = frame_count(out)
    if actual != piece.output_frames:
        raise FFmpegError(f"Virtual-tail piece expected {piece.output_frames} frames, got {actual}")
