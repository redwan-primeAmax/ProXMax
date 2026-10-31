"""Input discovery: scan the input folder and build the processing queue.

input ফোল্ডারের সব ভিডিও ফাইল (.mp4, .mkv, .mov, .avi) নাম অনুযায়ী
সাজিয়ে একটা queue তৈরি করে, যাতে একটার পর একটা (sequentially)
প্রসেস করা যায়।
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .utils import get_logger

logger = get_logger(__name__)

VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".mov", ".avi"})


def discover_videos(input_dir: Path | str) -> List[Path]:
    """Return all processable video files in ``input_dir``, sorted by name.

    লুকানো ফাইল (``.`` দিয়ে শুরু), সাময়িক ফাইল (``~``, ``.part``) ও
    শূন্য-আকারের ফাইল বাদ দেওয়া হয়।

    Raises:
        FileNotFoundError: input ফোল্ডারটাই না থাকলে (Drive mount হয়েছে
            কিনা এবং পাথ ঠিক আছে কিনা — সেই হিন্টসহ)।
    """
    folder = Path(input_dir).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Input ফোল্ডার পাওয়া যায়নি: {folder}\n"
            f"→ Google Drive mount হয়েছে কিনা যাচাই করুন, এবং পাথের বানান দেখুন "
            f"('Video' বড় হাতের V, 'input' ছোট হাতের)।"
        )

    videos: List[Path] = []
    for entry in folder.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if name.startswith((".", "~")) or name.endswith((".part", ".tmp", ".download")):
            logger.debug("Skipping hidden/temp file: %s", name)
            continue
        if entry.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        try:
            if entry.stat().st_size == 0:
                logger.warning("শূন্য-আকারের (0 byte) ফাইল বাদ দেওয়া হলো: %s", name)
                continue
        except OSError as exc:  # e.g. Drive file momentarily unavailable
            logger.warning("ফাইল পড়া যাচ্ছে না, বাদ দেওয়া হলো: %s (%s)", name, exc)
            continue
        videos.append(entry)

    videos.sort(key=lambda p: p.name.lower())
    logger.info("Input ফোল্ডারে %dটি ভিডিও পাওয়া গেছে: %s", len(videos), folder)
    for i, video in enumerate(videos, start=1):
        logger.info("  queue[%d] %s", i, video.name)
    return videos
