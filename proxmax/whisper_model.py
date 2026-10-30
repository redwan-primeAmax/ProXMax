"""Whisper cache, CUDA-library, and device-resolution helpers."""

from __future__ import annotations

import ctypes
import glob
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

from .utils import get_logger

logger = get_logger(__name__)
_CUDA_LIBS_READY = False

def _default_whisper_cache_dir() -> Optional[str]:
    """Return an explicit cache directory if configured via environment."""
    env_dir = (os.environ.get("PROXMAX_WHISPER_CACHE") or "").strip()
    if env_dir:
        Path(env_dir).mkdir(parents=True, exist_ok=True)
        return env_dir
    return None

def _resolve_cached_model_path(
    model_size: str,
    cache_dir: Optional[str | Path] = None,
) -> Optional[str]:
    """Return local directory path for ``model_size`` if already cached, else ``None``.

    এটি নেটওয়ার্ক ছাড়াই (`local_files_only=True`) লোকাল ক্যাশ চেক করে, যাতে
    ভিডিও প্রসেসিংয়ের সময় Hugging Face থেকে পুনরায় ডাউনলোড না হয়।
    """
    candidate = Path(model_size)
    if candidate.is_dir() and (candidate / "model.bin").is_file():
        return str(candidate.resolve())

    env_path = (os.environ.get("PROXMAX_WHISPER_MODEL_PATH") or "").strip()
    env_name = (os.environ.get("PROXMAX_WHISPER_MODEL_NAME") or "").strip()
    if env_path and (
        not env_name
        or env_name == model_size
        or Path(env_name).name.endswith(model_size)
        or Path(env_path).name.endswith(model_size)
    ):
        p = Path(env_path)
        if p.is_dir() and (p / "model.bin").is_file():
            return str(p.resolve())

    resolved_cache = str(cache_dir) if cache_dir else _default_whisper_cache_dir()
    try:
        from faster_whisper.utils import download_model

        local_dir = download_model(
            model_size,
            local_files_only=True,
            cache_dir=resolved_cache,
        )
        if local_dir and Path(local_dir).is_dir():
            return str(Path(local_dir).resolve())
    except Exception:
        pass
    return None

def ensure_whisper_model(
    model_size: str = "large-v3",
    cache_dir: Optional[str | Path] = None,
) -> str:
    """Pre-download and cache the faster-whisper model once during setup.

    ডিপেন্ডেন্সি ইনস্টল করার ধাপে এটি কল করলে মডেলটি লোকাল ডিস্কে ডাউনলোড হয়ে
    ``PROXMAX_WHISPER_MODEL_PATH``-এ সেট হয়ে থাকে। ফলে পরে ভিডিও ডাবিংয়ের সময়
    Whisper সরাসরি লোকাল ফোল্ডার থেকে লোড হয় — আর কখনো ডাউনলোড হয় না।
    """
    cached = _resolve_cached_model_path(model_size, cache_dir=cache_dir)
    if cached:
        os.environ["PROXMAX_WHISPER_MODEL_PATH"] = cached
        os.environ["PROXMAX_WHISPER_MODEL_NAME"] = model_size
        logger.info("Whisper মডেল আগে থেকেই ক্যাশে আছে: %s", cached)
        return cached

    resolved_cache = str(cache_dir) if cache_dir else _default_whisper_cache_dir()
    from faster_whisper.utils import download_model

    logger.info("Whisper '%s' মডেল ডাউনলোড ও ক্যাশ করা হচ্ছে...", model_size)
    started = time.perf_counter()
    local_dir = download_model(
        model_size,
        local_files_only=False,
        cache_dir=resolved_cache,
    )
    resolved = str(Path(local_dir).resolve())
    os.environ["PROXMAX_WHISPER_MODEL_PATH"] = resolved
    os.environ["PROXMAX_WHISPER_MODEL_NAME"] = model_size
    logger.info(
        "Whisper '%s' ক্যাশে সংরক্ষিত হয়েছে (%.1fs): %s",
        model_size,
        time.perf_counter() - started,
        resolved,
    )
    return resolved

def _ensure_cuda_libraries() -> None:
    """Make pip-installed cuBLAS/cuDNN shared libraries discoverable."""
    global _CUDA_LIBS_READY
    if _CUDA_LIBS_READY:
        return
    _CUDA_LIBS_READY = True

    try:
        import site
        import sysconfig

        search_roots = set(site.getsitepackages()) | {sysconfig.get_paths().get("purelib", "")}
    except Exception:  # pragma: no cover
        return

    lib_dirs: List[str] = []
    for root in search_roots:
        if not root:
            continue
        lib_dirs += glob.glob(os.path.join(root, "nvidia", "*", "lib"))
        torch_lib = os.path.join(root, "torch", "lib")
        if os.path.isdir(torch_lib):
            lib_dirs.append(torch_lib)
    lib_dirs += [d for d in ("/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu") if os.path.isdir(d)]

    if not lib_dirs:
        return

    existing = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    merged = list(dict.fromkeys(lib_dirs + existing))  # order-preserving dedupe
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)

    for pattern in ("libcublasLt.so*", "libcublas.so*", "libcudnn.so*", "libcudnn_ops*.so*"):
        for lib_dir in lib_dirs:
            for candidate in sorted(glob.glob(os.path.join(lib_dir, pattern))):
                try:
                    ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    continue
                break

def _resolve_device(requested: str, compute_type: str) -> Tuple[str, str]:
    """Decide the actual device/compute-type pair to run on."""
    requested = (requested or "auto").lower()
    if requested == "cpu":
        return "cpu", ("int8" if compute_type == "float16" else compute_type)

    cuda_available = False
    try:
        import ctranslate2

        cuda_available = ctranslate2.get_cuda_device_count() > 0
    except Exception as exc:  # pragma: no cover
        logger.debug("CUDA যাচাই করা যায়নি: %s", exc)

    if cuda_available:
        return "cuda", compute_type

    logger.warning(
        "⚠️ CUDA GPU পাওয়া যায়নি — CPU (int8) দিয়ে চলবে, যা অনেক ধীর হবে। "
        "Colab-এ Runtime → Change runtime type → T4 GPU নির্বাচন করেছেন তো?"
    )
    return "cpu", "int8"
