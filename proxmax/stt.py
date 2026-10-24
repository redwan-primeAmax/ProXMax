"""Speech-to-Text with faster-whisper (CTranslate2 backend, GPU).

English অডিও → timestamp-সহ English টেক্সট। মূল অপটিমাইজেশনসমূহ:

* CUDA GPU + ``float16`` compute (CPU-তে নিজে থেকে ``int8`` fallback)
* ``BatchedInferencePipeline`` — একাধিক 30s chunk একসঙ্গে প্রসেস হয়
* VAD filter — নীরব অংশ আগেই বাদ, ফলে দ্রুত ও hallucination কম
* Word-level timestamps — পরের sync ধাপের জন্য
* ভাষা ফিক্সড ``en`` — language-detection-এর সময় বাঁচে

মডেল প্রথমবার রান করার সময় faster-whisper নিজে থেকেই Hugging Face
থেকে ডাউনলোড করে নেয়; আলাদা কোনো download স্ক্রিপ্টের দরকার নেই।
"""

from __future__ import annotations

import ctypes
import glob
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

from .config import SOURCE_LANGUAGE
from .models import Segment, Word
from .utils import format_duration, get_logger, release_cuda_memory

logger = get_logger(__name__)

_CUDA_LIBS_READY = False


def _ensure_cuda_libraries() -> None:
    """Make pip-installed cuBLAS/cuDNN shared libraries discoverable.

    Google Colab-এ CTranslate2 মাঝে মাঝে ``libcudnn_ops.so.9`` খুঁজে
    পায় না (``Library not loaded`` error)। pip-এর ``nvidia-*``
    প্যাকেজ ও ``torch/lib`` ফোল্ডার খুঁজে লাইব্রেরিগুলো আগেভাগে
    লোড করে রাখলে এই সমস্যা হয় না। ব্যর্থ হলেও চুপচাপ এগিয়ে যাওয়া
    হয় — সিস্টেমে CUDA ঠিকমতো থাকলে এমনিতেই কাজ করবে।
    """
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

    # Pre-load the libraries CTranslate2 needs, with global visibility.
    for pattern in ("libcublasLt.so*", "libcublas.so*", "libcudnn.so*", "libcudnn_ops*.so*"):
        for lib_dir in lib_dirs:
            for candidate in sorted(glob.glob(os.path.join(lib_dir, pattern))):
                try:
                    ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    continue
                break


def _resolve_device(requested: str, compute_type: str) -> Tuple[str, str]:
    """Decide the actual device/compute-type pair to run on.

    GPU না থাকলে (যেমন ভুল করে CPU runtime নেওয়া হলে) crash না করে
    সতর্কবার্তাসহ CPU ``int8``-তে fallback করা হয়।
    """
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


class SpeechToText:
    """faster-whisper wrapper producing timestamped :class:`Segment`s.

    মডেল একবার লোড হয়ে মেমরিতে থাকে — ব্যাচের সব ভিডিওর জন্য একই
    instance পুনরায় ব্যবহার হয় (প্রতিবার লোড করার সময় বাঁচে)।
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "float16",
        batch_size: int = 8,
        beam_size: int = 5,
        vad_filter: bool = True,
        vad_min_silence_duration_ms: int = 500,
    ) -> None:
        self.model_size = model_size
        self.requested_device = device
        self.requested_compute_type = compute_type
        self.batch_size = max(1, batch_size)
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.vad_min_silence_duration_ms = vad_min_silence_duration_ms
        self._model = None            # WhisperModel (lazy)
        self._batched = None          # BatchedInferencePipeline (lazy)
        self.device: Optional[str] = None
        self.compute_type: Optional[str] = None

    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load the Whisper model on first use (auto-downloads weights)."""
        if self._model is not None:
            return
        _ensure_cuda_libraries()
        self.device, self.compute_type = _resolve_device(
            self.requested_device, self.requested_compute_type
        )
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        logger.info(
            "Whisper মডেল লোড হচ্ছে: %s | device=%s | compute=%s | batch=%d "
            "(প্রথমবার মডেল ডাউনলোড হতে কিছু সময় লাগতে পারে)…",
            self.model_size, self.device, self.compute_type, self.batch_size,
        )
        started = time.perf_counter()
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=(os.cpu_count() or 4) if self.device == "cpu" else 0,
        )
        if self.batch_size > 1:
            self._batched = BatchedInferencePipeline(model=self._model)
        logger.info("Whisper মডেল প্রস্তুত (লোড: %.1fs)।", time.perf_counter() - started)

    # ------------------------------------------------------------------
    def release(self) -> None:
        """Unload the Whisper model and free the GPU (VRAM).

        এক ব্যাচের কাজ শেষ হলে ডাকা হয়, যাতে Colab-এ GPU মেমরি আটকে না
        থাকে। পরে আবার ``transcribe()`` ডাকলে মডেলটি নিজে থেকেই
        পুনরায় লোড হয়।
        """
        had_model = self._model is not None
        self._batched = None
        self._model = None
        self.device = None
        self.compute_type = None
        if had_model:
            release_cuda_memory()
            logger.info("Whisper মডেল আনলোড হয়েছে — GPU মেমরি (VRAM) খালি করা হলো।")
        else:
            release_cuda_memory()

    # ------------------------------------------------------------------
    def transcribe(self, audio_path: Path | str) -> List[Segment]:
        """Transcribe an audio file into word-timestamped segments.

        Raises:
            ValueError: অডিওতে কোনো কথা (speech) পাওয়া না গেলে।
        """
        self._load()
        audio_path = Path(audio_path)
        kwargs = dict(
            language=SOURCE_LANGUAGE,          # ফিক্সড English → detection-এর সময় বাঁচে
            task="transcribe",
            beam_size=self.beam_size,
            word_timestamps=True,              # sync ধাপের জন্য জরুরি
            without_timestamps=False,
            vad_filter=self.vad_filter,
            vad_parameters={
                "min_silence_duration_ms": self.vad_min_silence_duration_ms,
                "speech_pad_ms": 200,
            },
            # আগের segment-এর টেক্সট শর্ত হিসেবে না নিলে repetition-loop কম হয়
            condition_on_previous_text=False,
            # word-timestamp থাকলে দীর্ঘ নীরবতায় hallucinated segment বাদ যায়
            hallucination_silence_threshold=2.0,
        )

        started = time.perf_counter()
        if self._batched is not None:
            try:
                raw_segments, info = self._batched.transcribe(
                    str(audio_path), batch_size=self.batch_size, **kwargs
                )
                segments = self._collect(raw_segments, info)
            except Exception as exc:
                # VRAM শেষ হয়ে গেলে (OOM) অন্তত non-batched পথে কাজটা শেষ করা যায়
                logger.warning(
                    "Batched transcription ব্যর্থ (%s) — সাধারণ (non-batched) "
                    "পদ্ধতিতে আবার চেষ্টা হচ্ছে…", exc,
                )
                raw_segments, info = self._model.transcribe(str(audio_path), **kwargs)
                segments = self._collect(raw_segments, info)
        else:
            raw_segments, info = self._model.transcribe(str(audio_path), **kwargs)
            segments = self._collect(raw_segments, info)

        elapsed = time.perf_counter() - started
        if not segments:
            raise ValueError(
                "অডিওতে কোনো English speech পাওয়া যায়নি। ভিডিওটিতে কথা আছে তো?"
            )
        speed = (info.duration / elapsed) if elapsed > 0 else 0.0
        logger.info(
            "Transcription শেষ: %d segment | অডিও %s | সময় %.1fs (%.1f× realtime) | "
            "language=%s (p=%.2f)",
            len(segments), format_duration(info.duration), elapsed, speed,
            info.language, info.language_probability,
        )
        return segments

    # ------------------------------------------------------------------
    @staticmethod
    def _collect(raw_segments, info) -> List[Segment]:
        """Materialise faster-whisper's lazy generator into Segments."""
        segments: List[Segment] = []
        try:
            from tqdm.auto import tqdm

            progress = tqdm(
                total=max(1, int(round(info.duration))),
                desc="  STT", unit="s", leave=False,
            )
        except Exception:  # pragma: no cover
            progress = None

        last_end = 0
        try:
            for raw in raw_segments:
                text = (raw.text or "").strip()
                if progress is not None:
                    progress.update(max(0, int(raw.end) - last_end))
                    last_end = int(raw.end)
                if not text or raw.end <= raw.start:
                    continue
                words = [
                    Word(
                        start=float(w.start),
                        end=float(w.end),
                        word=w.word,
                        probability=getattr(w, "probability", None),
                    )
                    for w in (raw.words or [])
                ]
                segments.append(
                    Segment(
                        index=len(segments),
                        start=float(raw.start),
                        end=float(raw.end),
                        text_en=text,
                        words=words,
                    )
                )
        finally:
            if progress is not None:
                progress.close()
        return segments
