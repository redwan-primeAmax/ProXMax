"""Render frame-based video edits with bounded parallel FFmpeg workers.

Each unique source range is encoded once, including tails referenced multiple
times. A concat manifest joins the pieces without a second video encode. This
avoids a huge, whole-video split filtergraph and keeps memory bounded per job.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Optional

from .utils import FFmpegError, ensure_tool, get_logger, run_ffmpeg, run_ffmpeg_with_progress
from .video_timeline import VideoPiece, VideoPlan

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


def normalize_video_start(source: Path, output: Path, fps: Fraction, blank_frames: int,
                          *, use_nvenc: bool = False) -> Path:
    """Preserve a late-starting picture's offset with black, not shifted frames.

    The source has no picture before its first frame; adding blank frames in
    that interval keeps later video aligned with the extracted audio timeline.
    """
    if blank_frames <= 0:
        return source
    expected = _frame_count(source) + blank_frames
    fps_text = f"{fps.numerator}/{fps.denominator}"
    filters = (f"setpts=PTS-STARTPTS,fps=fps={fps_text}:start_time=0,"
               f"tpad=start={blank_frames}:start_mode=add:color=black,"
               "pad=ceil(iw/2)*2:ceil(ih/2)*2")
    attempts = ["h264_nvenc", "libx264"] if use_nvenc else ["libx264"]
    last_error: Exception | None = None
    for encoder in attempts:
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
            return output
        except Exception as exc:
            last_error = exc
            output.unlink(missing_ok=True)
            logger.warning("Video start normalization via %s failed: %s", encoder, str(exc)[:300])
    raise FFmpegError(f"Cannot preserve picture start offset: {last_error}") from last_error


def render_video_plan(source: Path | str, plan: VideoPlan, output: Path | str, *,
                      audio: Optional[Path | str] = None, audio_bitrate: str = "160k",
                      workers: int = 2, use_nvenc: bool = False) -> Path:
    """Create the actual edited picture and optionally mux a complete dub track.

    Repeated identical source ranges reuse their encoded piece. If the GPU
    encoder is unavailable or fails, all pieces are retried with libx264 so the
    concat stream always has one consistent codec and timing configuration.
    """
    source, output = Path(source), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not source.is_file() or not plan.pieces or plan.output_frames <= 0:
        raise FFmpegError("Missing source video or empty video plan")
    if sum(piece.output_frames for piece in plan.pieces) != plan.output_frames:
        raise ValueError("Video plan frame counts do not add up")
    for p in plan.pieces:
        if p.start < 0 or p.end > plan.source_frames or p.end <= p.start:
            raise ValueError("Video plan contains an invalid source range")

    unique: dict[tuple[int, int, int], VideoPiece] = {}
    for piece in plan.pieces:
        unique.setdefault((piece.start, piece.end, piece.output_frames), piece)
    with tempfile.TemporaryDirectory(prefix="video_edits_", dir=str(output.parent)) as temp:
        root = Path(temp)
        files = {key: root / f"piece_{index:06d}.mp4" for index, key in enumerate(unique)}
        attempts = ["h264_nvenc", "libx264"] if use_nvenc else ["libx264"]
        last_error: Exception | None = None
        for encoder in attempts:
            try:
                for path in files.values():
                    path.unlink(missing_ok=True)
                limit = min(2 if encoder == "h264_nvenc" else 4, max(1, int(workers)))
                logger.info("Rendering %d unique video pieces (%s, workers=%d)…", len(unique), encoder, limit)
                with ThreadPoolExecutor(max_workers=limit) as pool:
                    futures = [pool.submit(_render_piece, source, piece, plan.fps, files[key], encoder)
                               for key, piece in unique.items()]
                    for future in futures:
                        future.result()
                break
            except Exception as exc:
                last_error = exc
                logger.warning("Video piece encoding via %s failed: %s", encoder, str(exc)[:300])
        else:
            raise FFmpegError(f"All video encoders failed: {last_error}") from last_error

        manifest = root / "pieces.txt"
        manifest.write_text("".join(
            f"file '{files[(p.start, p.end, p.output_frames)].name}'\n" for p in plan.pieces
        ), encoding="utf-8")
        cmd = [ensure_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
               "-f", "concat", "-safe", "1", "-i", str(manifest)]
        if audio is not None:
            cmd += ["-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac",
                    "-b:a", audio_bitrate, "-ar", "44100"]
        else:
            cmd += ["-map", "0:v:0", "-an"]
        cmd += ["-c:v", "copy", "-movflags", "+faststart", str(output)]
        try:
            run_ffmpeg_with_progress(cmd, description="Join edited video pieces",
                                     duration_seconds=plan.duration, stall_timeout=120.0)
            actual = _frame_count(output)
            if actual != plan.output_frames:
                raise FFmpegError(f"Edited video expected {plan.output_frames} frames, got {actual}")
        except Exception:
            output.unlink(missing_ok=True)
            raise
    return output
