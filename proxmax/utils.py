"""Common helpers: logging, ffmpeg/ffprobe wrappers, SRT writing.

ffmpeg চালানো, মিডিয়ার duration বের করা, লগিং সেটআপ — এমন সব
সাধারণ কাজ এখানে রাখা হয়েছে যাতে বাকি মডিউলগুলো পরিষ্কার থাকে।
"""

from __future__ import annotations

import gc
import logging
import queue
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

from .models import Segment

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

#: ffmpeg-এর stdin সবসময় বন্ধ রাখা হয়। নোটবুকে (``!python main.py``)
#: stdin একটি TTY/pipe হয়ে থাকে — ffmpeg সেখানে কী-ইনপুটের জন্য অপেক্ষা
#: করে **ঝুলে যেতে পারে**। ``-nostdin`` + ``DEVNULL`` দুটোই দেওয়া হয়।
_FFMPEG_STDIN = subprocess.DEVNULL

#: অগ্রগতি না দেখিয়ে এতক্ষণ কাটালে ধরে নেওয়া হয় ffmpeg ঝুলে গেছে (সেকেন্ড)
DEFAULT_STALL_TIMEOUT = 90.0


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


def _close_streams(proc: subprocess.Popen) -> None:
    """পাইপ বন্ধ করে (``ResourceWarning`` এড়াতে, Windows-এ ফাইল লক ছাড়াতে)।"""
    for stream in (proc.stdout, proc.stderr, proc.stdin):
        try:
            if stream is not None:
                stream.close()
        except Exception:  # ইতিমধ্যে বন্ধ/অপঠিত — কিছু করার নেই
            pass


def ffmpeg_args(args: Sequence[str]) -> List[str]:
    """Normalize an ffmpeg command: ``-nostdin`` যোগ করে, সব অংশ ``str`` করে।

    ``-nostdin`` ছাড়া ffmpeg কখনো কখনো stdin থেকে কমান্ডের অপেক্ষায়
    আটকে যায় — নোটবুকে ঠিক এটাই "কাজ করতে করতে ঝুলে যাওয়া"-র একটি
    সাধারণ কারণ।
    """
    parts = [str(a) for a in args]
    if parts and "-nostdin" not in parts:
        parts = [parts[0], "-nostdin"] + parts[1:]
    return parts


def run_ffmpeg(
    args: Sequence[str],
    description: str = "ffmpeg",
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess:
    """Run an ffmpeg command, raising :class:`FFmpegError` on failure.

    Args:
        args: Full argument list starting with ``"ffmpeg"``.
        description: মানব-পাঠযোগ্য নাম, error message-এ ব্যবহার হয়।
        timeout: সর্বোচ্চ সময় (সেকেন্ড); ``None`` = সীমা নেই।

    Returns:
        The completed process (stdout/stderr captured as text).
    """
    logger = get_logger(__name__)
    parts = ffmpeg_args(args)
    logger.debug("Running: %s", " ".join(parts))
    try:
        proc = subprocess.run(
            parts,
            stdin=_FFMPEG_STDIN,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(
            f"{description} {timeout:.0f}s-এর মধ্যে শেষ হয়নি — ফাইলটি পড়া যাচ্ছে না "
            f"বা ffmpeg ঝুলে গেছে।"
        ) from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-1500:]
        raise FFmpegError(f"{description} failed (exit={proc.returncode}):\n{tail}")
    return proc


def run_ffmpeg_with_progress(
    args: Sequence[str],
    description: str,
    duration_seconds: Optional[float] = None,
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
    report_every_seconds: float = 10.0,
    on_progress: Optional[Callable[[float], None]] = None,
) -> subprocess.CompletedProcess:
    """ffmpeg চালায় এবং **লাইভ অগ্রগতি** লগ করে (ঝুলে গেলে স্পষ্ট error দেয়)।

    দীর্ঘ extraction/decode-এর সময় ব্যবহারকারী যেন "কিছুই হচ্ছে না" ভেবে
    বিভ্রান্ত না হন, তাই অগ্রগতি নিয়মিত (এবং কেবল **পরিবর্তিত হলে**) লগ
    করা হয়। ``stall_timeout`` সেকেন্ড ধরে কোনো অগ্রগতিই না থাকলে প্রসেসটি
    kill করা হয় এবং বোধগম্য :class:`FFmpegError` ছোড়া হয় — ফলে পরের
    ধাপ (যেমন software decoding) চেষ্টা করা যায়।

    Args:
        args: ffmpeg কমান্ড (``-nostdin`` নিজে থেকেই যোগ হয়)।
        description: লগ/error-এ দেখানোর নাম।
        duration_seconds: মোট দৈর্ঘ্য (জানা থাকলে শতাংশ দেখানো হয়)।
        stall_timeout: কত সেকেন্ড অগ্রগতি না হলে ঝুলে গেছে ধরা হবে।
        report_every_seconds: অন্তত এত সেকেন্ড পরপর লগ।
        on_progress: ঐচ্ছিক callback — শতাংশ (জানা না থাকলে ``-1``)।

    Returns:
        ``subprocess.CompletedProcess`` (stdout/stderr text)।
    """
    logger = get_logger(__name__)
    parts = ffmpeg_args(args)
    if "-progress" not in parts:
        parts = parts[:2] + ["-progress", "pipe:1", "-nostats"] + parts[2:]
    logger.debug("Running: %s", " ".join(parts))

    proc = subprocess.Popen(
        parts,
        stdin=_FFMPEG_STDIN,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None and proc.stderr is not None

    stderr_tail: deque = deque(maxlen=80)
    lines: "queue.Queue[Optional[str]]" = queue.Queue()

    def pump_stdout() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    def pump_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            text = line.rstrip()
            if text:
                stderr_tail.append(text)

    for target in (pump_stdout, pump_stderr):
        threading.Thread(target=target, daemon=True).start()

    started = time.monotonic()
    last_activity = started
    last_log = 0.0
    last_percent = -1.0
    out_seconds = 0.0

    while True:
        try:
            line = lines.get(timeout=0.5)
        except queue.Empty:
            line = ""
        if line is None:                     # stdout শেষ — প্রসেস বন্ধ হচ্ছে
            break
        now = time.monotonic()
        if now - last_activity > stall_timeout:
            proc.kill()
            proc.wait()
            tail = "\n".join(list(stderr_tail)[-5:])
            raise FFmpegError(
                f"{description}: {stall_timeout:.0f} সেকেন্ড ধরে কোনো অগ্রগতি নেই — "
                f"প্রসেসটি থামানো হলো (ফাইলটি পড়া যাচ্ছে না বা codec সমর্থিত নয়)।"
                + (f"\n{tail}" if tail else "")
            )
        if not line:
            continue
        last_activity = now
        text = line.strip()
        if text.startswith(("out_time_us=", "out_time_ms=")):
            try:
                out_seconds = max(out_seconds, float(text.split("=", 1)[1]) / 1_000_000.0)
            except (TypeError, ValueError):
                continue
        elif not text.startswith(("progress=", "speed=", "frame=", "fps=", "bitrate=",
                                  "total_size=", "out_time=", "dup_frames=",
                                  "drop_frames=")):
            continue

        percent = (
            min(100.0, max(0.0, out_seconds / duration_seconds * 100.0))
            if duration_seconds and duration_seconds > 0
            else -1.0
        )
        if on_progress is not None:
            on_progress(percent)
        changed = percent < 0 or percent - last_percent >= 5.0
        if now - last_log >= report_every_seconds and changed:
            if percent >= 0:
                logger.info(
                    "  %s: %d%% (%s / %s) — চলছে…",
                    description, int(percent),
                    format_duration(out_seconds), format_duration(duration_seconds or 0),
                )
            else:
                logger.info("  %s: %s প্রসেস হয়েছে — চলছে…", description,
                            format_duration(out_seconds))
            last_log, last_percent = now, percent

    proc.wait()
    _close_streams(proc)
    if proc.returncode != 0:
        tail = "\n".join(list(stderr_tail)).strip()[-1500:]
        raise FFmpegError(f"{description} failed (exit={proc.returncode}):\n{tail}")
    if duration_seconds:
        logger.info(
            "  %s: ১০০%% (%s) — শেষ।", description, format_duration(duration_seconds)
        )
    return subprocess.CompletedProcess(parts, proc.returncode, stdout="",
                                       stderr="\n".join(stderr_tail))


def capture_bytes(
    args: Sequence[str],
    description: str = "ffmpeg",
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
    timeout: Optional[float] = None,
) -> tuple[bytes, str, int]:
    """চালায় এবং কাঁচা (binary) stdout ফেরত দেয় — ঝুলে গেলে থামিয়ে দেয়।

    ``subprocess.run`` সরাসরি ব্যবহার করলে পাইপ ভরে গেলে প্রসেস আটকে
    যেতে পারে; এখানে stdout আলাদা থ্রেডে পড়া হয়, তাই কখনো deadlock হয় না
    এবং ``stall_timeout`` পার হলে প্রসেস kill করা হয়।

    Returns:
        ``(stdout_bytes, stderr_tail, returncode)``।
    """
    logger = get_logger(__name__)
    parts = ffmpeg_args(args)
    logger.debug("Running: %s", " ".join(parts))
    proc = subprocess.Popen(
        parts,
        stdin=_FFMPEG_STDIN,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None and proc.stderr is not None

    chunks: List[bytes] = []
    last_activity = [time.monotonic()]

    def read_stdout() -> None:
        while True:
            chunk = proc.stdout.read(65536)
            if not chunk:
                break
            chunks.append(chunk)
            last_activity[0] = time.monotonic()

    stderr_tail: deque = deque(maxlen=40)

    def read_stderr() -> None:
        for raw in proc.stderr:
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                stderr_tail.append(line)
                last_activity[0] = time.monotonic()

    threads = [threading.Thread(target=read_stdout, daemon=True),
               threading.Thread(target=read_stderr, daemon=True)]
    for thread in threads:
        thread.start()

    started = time.monotonic()
    while proc.poll() is None:
        time.sleep(0.2)
        now = time.monotonic()
        idle = now - last_activity[0]
        if idle > stall_timeout or (timeout is not None and now - started > timeout):
            proc.kill()
            proc.wait()
            reason = (
                f"{stall_timeout:.0f} সেকেন্ড ধরে কোনো অগ্রগতি নেই"
                if idle > stall_timeout
                else f"{timeout:.0f}s সময়সীমা পেরিয়ে গেছে"
            )
            tail = "\n".join(list(stderr_tail)[-5:])
            raise FFmpegError(
                f"{description}: {reason} — প্রসেসটি থামানো হলো।"
                + (f"\n{tail}" if tail else "")
            )

    for thread in threads:
        thread.join(timeout=5.0)
    _close_streams(proc)
    return b"".join(chunks), "\n".join(stderr_tail), proc.returncode


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
            stdin=_FFMPEG_STDIN,
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
        ffmpeg_args([ffmpeg, "-hide_banner", "-i", path]),
        stdin=_FFMPEG_STDIN,
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


def iter_progress(
    iterable,
    desc: str = "",
    total: Optional[int] = None,
    unit: str = "seg",
):
    """Iterate with a tqdm progress bar when tqdm is available.

    tqdm না থাকলে সাধারণ iterable হিসেবেই ফেরত দেয় — কোড কোথাও
    আটকায় না।

    Args:
        iterable: যার উপর progress দেখানো হবে।
        desc: progress bar-এর নাম।
        total: মোট item সংখ্যা।
        unit: এককের নাম (যেমন ``"seg"`` বা ``"batch"``)।
    """
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - tqdm is a pinned dependency
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit=unit, leave=False)


def release_cuda_memory() -> None:
    """Force garbage collection and empty the CUDA cache (best effort).

    Whisper/CTranslate2 বা torch-এর মডেল object মেমরি থেকে সরে গেলে GPU
    (VRAM) ছেড়ে দেয়; তার পরও allocator-এর cache আটকে থাকতে পারে।
    কাজ শেষে এই ফাংশনটি ডাকলে Colab-এ GPU ব্যবহার দ্রুত কমে যায়।
    ``torch`` ইনস্টল না থাকলেও নিরাপদে কাজ করে।
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # torch নেই / CUDA নেই — কিছু করার দরকার নেই
        pass


def output_filename_for(video_path: Path | str) -> str:
    """``lecture1.mp4`` → ``lecture1_bn.mp4`` (সবসময় .mp4 আউটপুট)।"""
    return f"{Path(video_path).stem}_bn.mp4"
