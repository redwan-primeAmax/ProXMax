"""Render frame-based video edits with bounded parallel FFmpeg workers.

Small adjacent pieces are encoded together, while repeated identical ranges
remain reusable. Per-group frame counts and the final timeline are verified;
if grouping fails, the original one-piece-at-a-time renderer is the fallback.
"""

from __future__ import annotations

import contextvars
import json
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from fractions import Fraction
from pathlib import Path
from typing import Optional

from .utils import FFmpegError, ensure_tool, get_logger, run_ffmpeg, run_ffmpeg_with_progress
from .reporting_runtime import record_event
from .video_timeline import VideoPiece, VideoPlan
from .virtual_picture import render_virtual_piece

logger = get_logger(__name__)


def _frame_count(path: Path) -> int:
    """Read MP4 frame count, falling back to a lightweight packet count."""
    ffprobe = ensure_tool("ffprobe")
    for count_packets in (False, True):
        command = [ffprobe, "-v", "error"]
        if count_packets:
            command.append("-count_packets")
        command += ["-select_streams", "v:0", "-show_entries", "stream=nb_frames,nb_read_packets",
                    "-of", "json", str(path)]
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, encoding="utf-8", errors="replace", timeout=120, check=False)
        try:
            stream = json.loads(proc.stdout)["streams"][0]
            value = stream.get("nb_read_packets") if count_packets else stream.get("nb_frames")
            count = int(value)
            if proc.returncode == 0 and count > 0:
                return count
        except (ValueError, KeyError, IndexError, TypeError):
            pass
    raise FFmpegError(f"Cannot verify video frame count: {path.name}")


def _render_piece(source: Path, piece: VideoPiece, fps: Fraction, out: Path, encoder: str) -> None:
    if piece.source_frames <= 0 or piece.output_frames <= 0:
        raise ValueError("Picture piece must contain frames")
    if Fraction(piece.source_frames, piece.output_frames) > Fraction(23, 20):
        raise ValueError("Picture speed exceeds the 1.15x cap")
    start = float(Fraction(piece.start, 1) / fps)
    fps_text = f"{fps.numerator}/{fps.denominator}"
    # First normalize the input frame grid, then map the selected source frames
    # to precisely the planned output frame count. No audio is processed here.
    filters = (
        f"setpts=PTS-STARTPTS,fps=fps={fps_text}:start_time=0,"
        f"trim=end_frame={piece.source_frames},setpts=PTS-STARTPTS,"
        f"setpts=PTS*{piece.output_frames}/{piece.source_frames},"
        f"fps=fps={fps_text}:start_time=0,trim=end_frame={piece.output_frames},"
        "setpts=PTS-STARTPTS,pad=ceil(iw/2)*2:ceil(ih/2)*2"
    )
    if encoder == "h264_nvenc":
        codec = ["-c:v", "h264_nvenc", "-preset", "fast", "-cq", "23"]
    else:
        codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    run_ffmpeg(
        [ensure_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
         "-ss", f"{start:.9f}", "-i", str(source), "-an", "-vf", filters,
         "-frames:v", str(piece.output_frames), *codec, "-threads", "2", "-bf", "0",
         "-pix_fmt", "yuv420p", "-r", fps_text, str(out)],
        description=f"video frames {piece.start}–{piece.end} ({piece.kind})", timeout=1800.0,
    )
    actual = _frame_count(out)
    if actual != piece.output_frames:
        raise FFmpegError(
            f"Video piece {piece.start}:{piece.end} expected {piece.output_frames} frames, got {actual}"
        )


# At most four independently sought inputs per FFmpeg process. Additional
# pieces cover at most 60 planned output frames; decoders may buffer more.
_GROUP_SIZE = 4
_EXTRA_FRAMES = 60


def _piece_key(piece: VideoPiece) -> tuple[int, int, int]:
    return piece.start, piece.end, piece.output_frames


def _group_pieces(pieces: tuple[VideoPiece, ...], *, grouped: bool = True,
                  physical_end: Optional[int] = None) -> list[tuple[VideoPiece, ...]]:
    """Pack adjacent unique pieces; duplicates and virtual-tail pieces remain separate."""
    counts = Counter(_piece_key(piece) for piece in pieces)
    groups: list[tuple[VideoPiece, ...]] = []
    current: list[VideoPiece] = []
    extra_frames = 0

    def flush() -> None:
        nonlocal extra_frames
        if current:
            groups.append(tuple(current))
            current.clear()
            extra_frames = 0

    for piece in pieces:
        if not grouped or counts[_piece_key(piece)] > 1 or (physical_end is not None and piece.end > physical_end):
            flush()
            groups.append((piece,))
            continue
        if current and (len(current) >= _GROUP_SIZE or extra_frames + piece.output_frames > _EXTRA_FRAMES):
            flush()
        if current:
            extra_frames += piece.output_frames
        current.append(piece)
    flush()
    return groups


def _group_key(group: tuple[VideoPiece, ...]) -> tuple[tuple[int, int, int], ...]:
    return tuple(_piece_key(piece) for piece in group)


def _render_group(source: Path, group: tuple[VideoPiece, ...], fps: Fraction,
                  out: Path, encoder: str) -> None:
    """Render bounded, independently sought pieces into one checked CFR file."""
    if len(group) == 1:
        _render_piece(source, group[0], fps, out, encoder)
        return
    fps_text = f"{fps.numerator}/{fps.denominator}"
    command = [ensure_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
               "-filter_complex_threads", "1"]
    filters = []
    for index, piece in enumerate(group):
        if piece.source_frames <= 0 or piece.output_frames <= 0:
            raise ValueError("Picture piece must contain frames")
        if Fraction(piece.source_frames, piece.output_frames) > Fraction(23, 20):
            raise ValueError("Picture speed exceeds the 1.15x cap")
        start = float(Fraction(piece.start, 1) / fps)
        # Separate inputs preserve the established seek and VFR normalization
        # semantics. Limit decoder threads and filtergraph size for memory.
        command += ["-ss", f"{start:.9f}", "-threads", "1", "-i", str(source)]
        filters.append(
            f"[{index}:v]setpts=PTS-STARTPTS,fps=fps={fps_text}:start_time=0,"
            f"trim=end_frame={piece.source_frames},setpts=PTS-STARTPTS,"
            f"setpts=PTS*{piece.output_frames}/{piece.source_frames},"
            f"fps=fps={fps_text}:start_time=0,trim=end_frame={piece.output_frames},"
            f"setpts=PTS-STARTPTS[v{index}]"
        )
    labels = "".join(f"[v{i}]" for i in range(len(group)))
    # concat uses a microsecond time base; force frame-index-based CFR PTS so
    # fractional rates cannot accumulate rounding drift at group boundaries.
    filters.append(f"{labels}concat=n={len(group)}:v=1:a=0,"
                   f"setpts=N/(({fps_text})*TB),pad=ceil(iw/2)*2:ceil(ih/2)*2[out]")
    codec = (["-c:v", "h264_nvenc", "-preset", "fast", "-cq", "23"] if encoder == "h264_nvenc"
             else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"])
    expected = sum(piece.output_frames for piece in group)
    command += ["-filter_complex", ";".join(filters), "-map", "[out]", "-an",
                "-frames:v", str(expected), *codec, "-threads", "2", "-bf", "0",
                "-pix_fmt", "yuv420p", "-r", fps_text, str(out)]
    run_ffmpeg(command, description=f"video group ({len(group)} pieces)", timeout=1800.0)
    actual = _frame_count(out)
    if actual != expected:
        raise FFmpegError(f"Video group expected {expected} frames, got {actual}")


def _run_groups(source: Path, fps: Fraction, unique: dict, files: dict,
                encoder: str, limit: int, source_map: Optional[VideoPlan] = None) -> None:
    """Keep only worker-count jobs in flight, so an encoder failure stops early."""
    entries = iter(unique.items())
    with ThreadPoolExecutor(max_workers=limit) as pool:
        pending = set()

        def render_one(key, group) -> None:
            started = time.perf_counter()
            details = {"count": len(group), "encoder": encoder,
                       "source_start_frame": group[0].start, "source_end_frame": group[-1].end}
            record_event("video_group_started", **details)
            error = None
            try:
                if source_map is not None and group[0].end > source_map.source_frames:
                    if len(group) != 1:
                        raise ValueError("A virtual-tail piece must be rendered separately")
                    render_virtual_piece(source, group[0], fps, files[key], encoder, source_map,
                                         run_ffmpeg=run_ffmpeg, frame_count=_frame_count)
                else:
                    _render_group(source, group, fps, files[key], encoder)
            except BaseException as exc:
                error = exc
                raise
            finally:
                record_event("video_group_finished", **details, elapsed_seconds=time.perf_counter() - started,
                             status="failed" if error else "success",
                             error_type=type(error).__name__ if error else None)

        def submit_one() -> bool:
            item = next(entries, None)
            if item is None:
                return False
            key, group = item
            context = contextvars.copy_context()
            future = pool.submit(context.run, render_one, key, group)
            pending.add(future)
            return True

        for _ in range(limit):
            if not submit_one():
                break
        try:
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                # Do not enqueue anything else until every completed result is checked.
                for future in finished:
                    future.result()
                for _ in finished:
                    submit_one()
        except BaseException:
            for future in pending:
                future.cancel()
            raise


def normalize_video_start(source: Path, output: Path, fps: Fraction, blank_frames: int,
                          *, use_nvenc: bool = False, diagnostics: Optional[dict] = None) -> Path:
    """Preserve a late-starting picture's offset with black, not shifted frames.

    The source has no picture before its first frame; adding blank frames in
    that interval keeps later video aligned with the extracted audio timeline.
    """
    if blank_frames <= 0:
        return source
    if source.resolve() == output.resolve():
        raise ValueError("Normalized output must not overwrite its source video")
    expected = _frame_count(source) + blank_frames
    fps_text = f"{fps.numerator}/{fps.denominator}"
    filters = (f"setpts=PTS-STARTPTS,fps=fps={fps_text}:start_time=0,"
               f"tpad=start={blank_frames}:start_mode=add:color=black,"
               "pad=ceil(iw/2)*2:ceil(ih/2)*2")
    attempts = ["h264_nvenc", "libx264"] if use_nvenc else ["libx264"]
    last_error: Exception | None = None
    details = diagnostics if diagnostics is not None else {}
    details.clear()
    details.update({"blank_frames": blank_frames, "attempts": []})
    for encoder in attempts:
        started = time.perf_counter()
        entry = {"encoder": encoder}
        details["attempts"].append(entry)
        try:
            codec = (["-c:v", "h264_nvenc", "-preset", "fast", "-cq", "23"] if encoder == "h264_nvenc"
                     else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"])
            run_ffmpeg([ensure_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(source), "-an", "-vf", filters, "-frames:v", str(expected),
                        *codec, "-threads", "2", "-bf", "0", "-pix_fmt", "yuv420p",
                        "-r", fps_text, str(output)],
                       description="preserve late video start", timeout=1800.0)
            actual = _frame_count(output)
            if actual != expected:
                raise FFmpegError(f"Start-normalized video expected {expected} frames, got {actual}")
            entry.update({"status": "success", "seconds": round(time.perf_counter() - started, 3)})
            details["encoder"] = encoder
            return output
        except Exception as exc:
            last_error = exc
            entry.update({"status": "failed", "error_type": type(exc).__name__,
                          "seconds": round(time.perf_counter() - started, 3)})
            output.unlink(missing_ok=True)
            logger.warning("Video start normalization via %s failed: %s", encoder, str(exc)[:300])
    raise FFmpegError(f"Cannot preserve picture start offset: {last_error}") from last_error


def render_video_plan(source: Path | str, plan: VideoPlan, output: Path | str, *,
                      audio: Optional[Path | str] = None, audio_bitrate: str = "160k",
                      workers: int = 2, use_nvenc: bool = False,
                      diagnostics: Optional[dict] = None,
                      source_map: Optional[VideoPlan] = None) -> Path:
    """Render and verify all picture edits, falling back safely when necessary.

    Repeated identical ranges reuse a file. Encoder or grouped-filter failures
    retry consistently with CPU and, if necessary, individual pieces. No
    unverified output is returned. Diagnostics contain timings/counts only.
    """
    source, output = Path(source), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not source.is_file() or not plan.pieces or plan.output_frames <= 0:
        raise FFmpegError("Missing source video or empty video plan")
    if source.resolve() == output.resolve():
        raise ValueError("Edited output must not overwrite its source video")
    if sum(piece.output_frames for piece in plan.pieces) != plan.output_frames:
        raise ValueError("Video plan frame counts do not add up")
    for piece in plan.pieces:
        if piece.start < 0 or piece.end > plan.source_frames or piece.end <= piece.start:
            raise ValueError("Video plan contains an invalid source range")
    if source_map is not None:
        if (source_map.fps != plan.fps or source_map.source_frames <= 0
                or source_map.output_frames < plan.source_frames
                or sum(p.output_frames for p in source_map.pieces) != source_map.output_frames
                or any(p.start < 0 or p.end > source_map.source_frames or p.end <= p.start
                       or p.source_frames != p.output_frames for p in source_map.pieces)):
            raise ValueError("Invalid virtual source picture map")

    details = diagnostics if diagnostics is not None else {}
    details.clear()
    details.update({"planned_pieces": len(plan.pieces), "group_size_limit": _GROUP_SIZE,
                    "extra_group_frames_limit": _EXTRA_FRAMES, "attempts": []})
    if source_map is not None:
        details["virtual_source_tail_frames"] = max(0, plan.source_frames - source_map.source_frames)
    attempts = ["h264_nvenc", "libx264"] if use_nvenc else ["libx264"]
    last_error: Exception | None = None
    with tempfile.TemporaryDirectory(prefix="video_edits_", dir=str(output.parent)) as temp:
        root = Path(temp)
        for grouped in (True, False):
            groups = _group_pieces(plan.pieces, grouped=grouped,
                                   physical_end=source_map.source_frames if source_map is not None else None)
            unique = {_group_key(group): group for group in groups}
            files = {key: root / f"piece_{index:06d}.mp4" for index, key in enumerate(unique)}
            for encoder in attempts:
                limit = min(2 if encoder == "h264_nvenc" else 4, max(1, int(workers)))
                entry = {"encoder": encoder, "strategy": "grouped" if grouped else "individual",
                         "unique_groups": len(unique), "workers": limit}
                details["attempts"].append(entry)
                started = time.perf_counter()
                try:
                    for path in root.glob("piece_*.mp4"):
                        path.unlink(missing_ok=True)
                    output.unlink(missing_ok=True)
                    logger.info("Rendering %d video groups / %d pieces (%s, workers=%d)…",
                                len(unique), len(plan.pieces), encoder, limit)
                    _run_groups(source, plan.fps, unique, files, encoder, limit, source_map=source_map)
                    entry["render_seconds"] = round(time.perf_counter() - started, 3)
                    manifest = root / "pieces.txt"
                    manifest.write_text("".join(
                        f"file '{files[_group_key(group)].name}'\n" for group in groups
                    ), encoding="utf-8")
                    cmd = [ensure_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
                           "-f", "concat", "-safe", "1", "-i", str(manifest)]
                    if audio is not None:
                        cmd += ["-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac",
                                "-b:a", audio_bitrate, "-ar", "44100"]
                    else:
                        cmd += ["-map", "0:v:0", "-an"]
                    cmd += ["-c:v", "copy", "-movflags", "+faststart", str(output)]
                    join_started = time.perf_counter()
                    run_ffmpeg_with_progress(cmd, description="Join edited video pieces",
                                             duration_seconds=plan.duration, stall_timeout=120.0)
                    actual = _frame_count(output)
                    if actual != plan.output_frames:
                        raise FFmpegError(f"Edited video expected {plan.output_frames} frames, got {actual}")
                    entry.update({"status": "success", "join_seconds": round(time.perf_counter() - join_started, 3)})
                    details.update({"encoder": encoder, "strategy": entry["strategy"],
                                    "unique_groups": len(unique), "workers": limit,
                                    "fallbacks": len(details["attempts"]) - 1})
                    return output
                except Exception as exc:
                    last_error = exc
                    entry.update({"status": "failed", "error_type": type(exc).__name__,
                                  "elapsed_seconds": round(time.perf_counter() - started, 3)})
                    output.unlink(missing_ok=True)
                    logger.warning("Video %s encoding via %s failed: %s",
                                   entry["strategy"], encoder, str(exc)[:300])
            # If there were no multi-piece groups, individual is identical and
            # must not repeat all the same failed encoder attempts.
            if all(len(group) == 1 for group in groups):
                break
    raise FFmpegError(f"All video render methods failed: {last_error}") from last_error
