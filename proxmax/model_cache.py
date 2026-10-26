"""Whisper মডেল আগেই নামিয়ে রাখা (এবং প্রসেসিংয়ের সময় আর নামানো নয়)।

সমস্যা
------
ব্যবহারকারীর অভিযোগ: রিকোয়ারমেন্ট ইনস্টল শেষ হওয়ার পরও **ভিডিও প্রসেসিংয়ের
ধাপে** Whisper `large-v3` (≈৩ GB) আবার ডাউনলোড হতে দেখা যায় — ফলে ৩০–৬০
সেকেন্ড নষ্ট হয়, আর প্রতিটি নতুন Colab সেশনে সেটাই আবার হয়।

সমাধান
------
1. **আগেই নামানো** — নোটবুকের সেটআপ ধাপে :func:`prefetch_whisper` ডাকা হয়;
   মডেল আগে থেকে থাকলে কিছুই ডাউনলোড হয় না (শুধু "আগেই আছে" দেখায়)।
2. **ডাউনলোড হবে না, শুধু লোড** — cache সম্পূর্ণ থাকলে STT
   ``local_files_only=True`` দিয়ে মডেল খোলে, অর্থাৎ প্রসেসিংয়ের সময়
   নেটওয়ার্ক ছোঁয়াও হয় না (:func:`model_is_cached`)।
3. **CUDA যাচাই** — GPU-র libs ঠিক আছে কি না দেখে দরকার হলেই ভারী
   ``nvidia-*`` প্যাকেজ ইনস্টল করা হয় (:func:`cuda_runtime_ok`), ফলে
   সাধারণ Colab-এ ~৭০০ MB বাঁচে ও ইনস্টল দ্রুত হয়।
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Tuple

from .utils import get_logger

logger = get_logger(__name__)

#: faster-whisper মডেলের নাম → Hugging Face রিপো আইডি
HF_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}

#: মডেল চালাতে যা যা লাগে — সব থাকলে cache "সম্পূর্ণ" ধরা হয়
_REQUIRED_FILES = ("model.bin", "config.json", "tokenizer.json")
_REQUIRED_ANY = ("vocabulary.txt", "vocabulary.json", "preprocessor_config.json")


def hf_cache_root() -> Path:
    """Hugging Face cache-এর রুট (``HF_HOME`` মানে সম্মান করা হয়)।"""
    home = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if home:
        path = Path(home).expanduser()
        return path / "hub" if path.name != "hub" else path
    return Path.home() / ".cache" / "huggingface" / "hub"


def repo_id_for(model_size: str) -> Optional[str]:
    """Whisper মডেলের নাম থেকে HF রিপো আইডি (অজানা হলে ``None``)।"""
    name = (model_size or "").strip()
    if name in HF_REPOS:
        return HF_REPOS[name]
    if "/" in name:  # সরাসরি repo id দিলেও চলে
        return name
    return None


def _snapshot_dir(model_size: str) -> Optional[Path]:
    """cache-এ থাকা snapshot ফোল্ডার (স্ট্যাটাস-ভিত্তিক, নেটওয়ার্ক ছাড়া)।"""
    repo = repo_id_for(model_size)
    if not repo:
        return None
    folder = hf_cache_root() / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not folder.is_dir():
        return None
    candidates = [p for p in folder.iterdir() if p.is_dir()]
    for candidate in candidates:
        if not all((candidate / name).is_file() for name in _REQUIRED_FILES):
            continue
        if any((candidate / name).is_file() for name in _REQUIRED_ANY):
            return candidate
    return None


def model_is_cached(model_size: str) -> bool:
    """মডেলের ওজন আগেই সম্পূর্ণ নামানো আছে কি না (নেটওয়ার্ক ছোঁয় না)।

    Args:
        model_size: Whisper মডেলের নাম (``large-v3`` ইত্যাদি)।

    Returns:
        ``True`` হলে প্রসেসিংয়ের সময় আর কিছু ডাউনলোড হবে না।
    """
    return _snapshot_dir(model_size) is not None


def cached_model_path(model_size: str) -> Optional[str]:
    """cache করা স্ন্যাপশটের পাথ (থাকলে), নাহলে ``None``।

    STT এই পাথটা ``WhisperModel``-কে দেয়, যাতে সে নেটওয়ার্ক একেবারেই না ছোঁয়।
    """
    snapshot = _snapshot_dir(model_size)
    return str(snapshot) if snapshot is not None else None


def prefetch_whisper(model_size: str = "large-v3", show_progress: bool = True) -> Tuple[bool, float]:
    """Whisper মডেল আগেই নামিয়ে রাখে (আগে থেকে থাকলে কিছুই করে না)।

    Args:
        model_size: Whisper মডেলের নাম (``large-v3`` / ``medium`` ইত্যাদি)।
        show_progress: ডাউনলোডের অগ্রগতি দেখাবে কি না।

    Returns:
        ``(ok, seconds)`` — মডেল ব্যবহারযোগ্য কি না এবং কত সময় লাগল।
    """
    started = time.perf_counter()
    if model_is_cached(model_size):
        logger.info(
            "✅ Whisper %s আগেই নামানো আছে (%s) — পুনরায় ডাউনলোড হবে না।",
            model_size, hf_cache_root(),
        )
        return True, 0.0

    repo = repo_id_for(model_size)
    if not repo:
        logger.warning(
            "Whisper মডেল '%s'-এর ডাউনলোড-লিংক অজানা — প্রথম রানে নিজেই নামবে।",
            model_size,
        )
        return False, 0.0

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - requirements-এ pin করা
        logger.warning("huggingface_hub নেই (%s) — মডেল প্রথম রানে নামবে।", exc)
        return False, 0.0

    logger.info(
        "⬇️ Whisper %s মডেল আগেই নামানো হচ্ছে (≈%s) — একবারই লাগবে, "
        "পরে প্রসেসিংয়ের সময় আর অপেক্ষা করতে হবে না…",
        model_size, "৩ GB" if model_size.startswith("large") else "০.৫–১.৫ GB",
    )
    try:
        path = snapshot_download(
            repo_id=repo,
            allow_patterns=["*.bin", "*.json", "*.txt", "*.md"],
            tqdm_class=None if not show_progress else None,
        )
    except Exception as exc:  # নেটওয়ার্ক সমস্যা → প্রথম রানে নিজেই চেষ্টা করবে
        logger.warning("মডেল আগেই নামানো যায়নি (%s) — প্রথম রানে নামবে।", exc)
        return False, time.perf_counter() - started
    elapsed = time.perf_counter() - started
    logger.info("✅ Whisper %s প্রস্তুত (%.0fs): %s", model_size, elapsed, path)
    return True, elapsed


def cuda_runtime_ok() -> bool:
    """CTranslate2 আসলে GPU দেখতে পায় কি না (CUDA libs লোড হয় কি না)।

    ``nvidia-*`` প্যাকেজ (≈৭০০ MB) দরকার কি না তা এতেই ঠিক হয় — Colab-এ
    সাধারণত torch-এর সাথে আসা libs যথেষ্ট, তাই আলাদা ইনস্টল লাগে না।
    """
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception as exc:  # pragma: no cover - install না থাকলে
        logger.debug("CUDA যাচাই করা যায়নি: %s", exc)
        return False
