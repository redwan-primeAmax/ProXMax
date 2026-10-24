"""Common helpers: logging, ffmpeg/ffprobe wrappers, SRT writing.

ffmpeg চালানো, মিডিয়ার duration বের করা, লগিং সেটআপ — এমন সব
সাধারণ কাজ এখানে রাখা হয়েছে যাতে বাকি মডিউলগুলো পরিষ্কার থাকে।
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .models import Segment

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


class FFmpegError(RuntimeError):
    """Raised when an ffmpeg/ffprobe command fails."""


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once (used by the CLI entry point)."""
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format=_LOG_FORMAT,
        datefmt="%H:%M:%S",
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def ensure_tool(name: str) -> str:
    """Return the full path of a CLI tool or raise a helpful error.

    Args:
        name: Executable name, e.g. ``"ffmpeg"``.

    Raises:
        FFmpegError: টুলটি PATH-এ পাওয়া না গেলে (কীভাবে ইনস্টল করতে
            হবে সেই নির্দেশনাসহ)।
    """
    found = shutil.which(name)
    if not found:
        raise FFmpegError(
            f"'{name}' পাওয়া যায়নি (not found in PATH). "
            f"Google Colab-এ সাধারণত আগে থেকেই থাকে; না থাকলে চালান: "
            f"apt-get update -qq && apt-get install -y -qq ffmpeg"
        )
    return found


def run_ffmpeg(args: Sequence[str], description: str = "ffmpeg") -> subprocess.CompletedProcess:
    """Run an ffmpeg command, raising :class:`FFmpegError` on failure.

    Args:
        args: Full argument list starting with ``"ffmpeg"``.
        description: মানব-পাঠযোগ্য নাম, error message-এ ব্যবহার হয়।

    Returns:
        The completed process (stdout/stderr captured as text).
    """
    logger = get_logger(__name__)
    logger.debug("Running: %s", " ".join(str(a) for a in args))
    proc = subprocess.run(
        [str(a) for a in args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-1500:]
        raise FFmpegError(f"{description} failed (exit={proc.returncode}):\n{tail}")
    return proc


def get_media_duration(media_path: Path | str) -> float:
    """Return media duration in seconds.

    প্রথমে ``ffprobe`` দিয়ে চেষ্টা করা হয়; ffprobe না থাকলে
    ``ffmpeg -i``-এর আউটপুট পার্স করে duration বের করা হয় (fallback)।

    Raises:
        FFmpegError: ফাইলটি পড়া না গেলে বা duration জানা না গেলে।
    """
    path = str(media_path)
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        proc = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace", check=False,
        )
        if proc.returncode == 0:
            try:
                return float(proc.stdout.strip())
            except (TypeError, ValueError):
                pass  # fall through to the ffmpeg fallback below

    # --- fallback: parse "Duration: 00:01:23.45" from `ffmpeg -i` ----------
    ffmpeg = ensure_tool("ffmpeg")
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr or "")
    if not match:
        raise FFmpegError(f"Could not determine duration of: {path}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def format_duration(seconds: float) -> str:
    """Format seconds as ``MM:SS`` (or ``HH:MM:SS`` when >= 1 hour)."""
    total = int(round(max(0.0, seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp: ``00:00:01,230``."""
    if seconds < 0:
        seconds = 0.0
    ms_total = int(round(seconds * 1000))
    ms = ms_total % 1000
    total_s = ms_total // 1000
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: Iterable[Segment], path: Path | str, language: str = "bn") -> Path:
    """Write segments to an ``.srt`` subtitle file.

    Args:
        segments: Segments carrying timestamps and text.
        path: Destination ``.srt`` path.
        language: ``"en"`` for original English text, ``"bn"`` for Bangla.
    """
    lines: List[str] = []
    counter = 0
    for seg in segments:
        text = (seg.text_en if language == "en" else seg.text_bn or "").strip()
        if not text:
            continue
        counter += 1
        lines.append(
            f"{counter}\n"
            f"{format_srt_timestamp(seg.start)} --> {format_srt_timestamp(seg.end)}\n"
            f"{text}\n"
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def iter_progress(iterable, desc: str = "", total: Optional[int] = None):
    """Iterate with a tqdm progress bar when tqdm is available.

    tqdm না থাকলে সাধারণ iterable হিসেবেই ফেরত দেয় — কোড কোথাও
    আটকায় না।
    """
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - tqdm is a pinned dependency
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit="seg", leave=False)


def output_filename_for(video_path: Path | str) -> str:
    """``lecture1.mp4`` → ``lecture1_bn.mp4`` (সবসময় .mp4 আউটপুট)।"""
    return f"{Path(video_path).stem}_bn.mp4"
