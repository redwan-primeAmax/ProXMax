"""Common logging, duration formatting, and SRT helpers.

FFmpeg process wrappers are implemented in :mod:`proxmax.ffmpeg_utils` and
re-exported here to preserve the established import path.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

from .models import Segment
from .ffmpeg_utils import (
    DEFAULT_STALL_TIMEOUT, FFmpegError, _FFMPEG_STDIN, _close_streams, capture_bytes,
    ensure_tool, ffmpeg_args, get_media_duration, run_ffmpeg, run_ffmpeg_with_progress,
)

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


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
