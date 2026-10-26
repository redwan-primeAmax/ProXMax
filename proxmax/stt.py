"""Speech-to-Text: silence-aware, context-carrying faster-whisper pipeline.

কেন আগের পদ্ধতি যথেষ্ট ছিল না
-------------------------------
Whisper একবারে ৩০ সেকেন্ডের বেশি দেখতে পারে না, তাই কাটা বাধ্যতামূলক।
আগে ``BatchedInferencePipeline`` ব্যবহার হতো, যেখানে faster-whisper
**নিজেই** কাটে — শেষ নীরবতা না পেলে শব্দের মাঝখানে জোর করে। ফলে
"XYZ" শব্দটা "XY" + "Z" হয়ে যেত, আর প্রতিটি টুকরো আগের প্রসঙ্গ ভুলে
যাওয়ায় পাংচুয়েশন ও বাক্যের ভঙ্গি হারিয়ে যেত।

এখন যা করা হয়
--------------
1. পুরো অডিও **একবারই** decode হয় (Drive/FUSE-এ বারবার পড়া বন্ধ)।
2. :mod:`proxmax.chunking` VAD/energy দিয়ে নীরবতার সীমানা খুঁজে
   ১৮–২৮ সেকেন্ডের chunk বানায় — কাটা পড়ে নীরবতার মাঝখানে, শব্দের মাঝে নয়।
3. প্রতিটি chunk আলাদাভাবে ডিকোড হয় **আগের টেক্সটের শেষ অংশ
   ``initial_prompt`` হিসেবে দিয়ে** — তাই পুরো ভিডিও জুড়ে প্রসঙ্গ,
   পাংচুয়েশন আর নামের বানান টানা থাকে (⚠️ ব্যাচড মোডে এটি সম্ভব নয়,
   কারণ সেখানে ``condition_on_previous_text=False`` হার্ড-কোডেড)।
4. একটুও জোর করে কাটা পড়লে **seam repair**: সীমানার ±২.৫ সেকেন্ড আবার
   ডিকোড করে শব্দ জোড়া লাগানো হয় — ভাঙা শব্দ ঠিক হয়ে যায়।
5. শব্দগুলো পূর্ণ বাক্যে সাজানো হয় (:mod:`proxmax.segmenter`), যাতে
   TTS স্বাভাবিক শ্বাস-বিরতি পায়।
6. ``hotwords.txt`` থেকে FC Mobile টার্ম, ইভেন্ট ও খেলোয়াড়ের নাম লোড করে
   Whisper-কে প্রম্পট দেওয়া হয় এবং ট্রান্সক্রিপ্টের সাধারণ ভুল বানান ঠিক করা হয়।

নোটবুকের সেটআপ ধাপে :func:`ensure_whisper_model` কল করলে মডেলটি আগেভাগেই
লোকাল ক্যাশে ডাউনলোড হয়ে থাকে — ফলে ভিডিও প্রসেসিংয়ের সময় আর ডাউনলোড হয় না।
"""

from __future__ import annotations

import ctypes
import glob
import os
import re
import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .audio_fx import decode_audio
from .chunking import SpeechChunk, plan_chunks
from .config import SOURCE_LANGUAGE
from .hotwords import (
    load_hotwords,
    repair_transcript_hotwords,
    select_whisper_hotwords,
)
from .models import Segment, Word
from .segmenter import build_sentences, join_words
from .utils import format_duration, get_logger, release_cuda_memory

logger = get_logger(__name__)

_CUDA_LIBS_READY = False
#: STT-এর বিশ্লেষণ sample rate (Whisper-এর পছন্দ)
STT_SAMPLE_RATE = 16000


class STTError(RuntimeError):
    """Raised when speech-to-text fails."""


# =============================================================================
# Whisper model pre-download & local cache resolution
# =============================================================================
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


# =============================================================================
# CUDA / device helpers
# =============================================================================
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


# =============================================================================
# Seam helpers (pure functions — unit-testable)
# =============================================================================
def find_suspicious_boundaries(
    words: Sequence[Word], chunks: Sequence[SpeechChunk], tolerance: float = 0.12
) -> List[float]:
    """Chunk সীমানাগুলোর মধ্যে যেগুলোতে শব্দ কাটা পড়ার আশঙ্কা আছে।"""
    boundaries: List[float] = []
    for chunk in chunks[1:]:
        b = chunk.start
        straddle = any(w.start < b - 0.02 and w.end > b + 0.02 for w in words)
        left = [w for w in words if w.end <= b + 0.02]
        right = [w for w in words if w.start >= b - 0.02]
        near_left = bool(left) and (b - left[-1].end) <= tolerance
        near_right = bool(right) and (right[0].start - b) <= tolerance
        if straddle or near_left or near_right:
            boundaries.append(b)
    return boundaries


def merge_windows(
    boundaries: Sequence[float], half_width: float, max_seconds: float
) -> List[Tuple[float, float]]:
    """সীমানাগুলোকে একে অন্যের সাথে মিলিয়ে (ওভারল্যাপ-মুক্ত) উইন্ডো বানায়।"""
    windows: List[Tuple[float, float]] = []
    for b in sorted(boundaries):
        start = max(0.0, b - half_width)
        end = max(0.0, b + half_width)
        if end - start > max_seconds:  # নিরাপত্তা: এক উইন্ডো খুব বড় নয়
            end = start + max_seconds
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append((start, end))
    return windows


def replace_in_window(
    words: Sequence[Word], replacement: Sequence[Word], start: float, end: float
) -> List[Word]:
    """উইন্ডোর ভেতরের শব্দ বদলে দেয় (বাইরের শব্দ অপরিবর্তিত থাকে)।"""
    kept = [
        w for w in words
        if not (start <= (w.start + w.end) / 2.0 <= end)
    ]
    inside = [
        w for w in replacement
        if start - 0.35 <= (w.start + w.end) / 2.0 <= end + 0.35
    ]
    combined = kept + inside
    combined.sort(key=lambda w: (w.start, w.end))
    return combined


def _tail_context(text: str, limit: int) -> str:
    """Prompt carry-এর জন্য শেষ কয়েক অক্ষর (সম্ভব হলে বাক্যের শুরু থেকে)।"""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    tail = text[-limit:]
    for mark in (". ", "! ", "? ", "। "):
        idx = tail.find(mark)
        if 0 <= idx < len(tail) // 2:
            return tail[idx + len(mark):]
    return tail[tail.find(" ") + 1:] if " " in tail else tail


# =============================================================================
# SpeechToText
# =============================================================================
class SpeechToText:
    """faster-whisper wrapper producing sentence-level :class:`Segment`s."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "float16",
        beam_size: int = 5,
        vad_min_silence_ms: int = 300,
        chunk_min_seconds: float = 18.0,
        chunk_max_seconds: float = 28.0,
        chunk_pad_ms: int = 120,
        carry_context_chars: int = 220,
        condition_on_previous_text: bool = True,
        seam_repair: bool = True,
        seam_window_seconds: float = 2.5,
        max_seam_repairs: int = 60,
        no_speech_threshold: float = 0.6,
        compression_ratio_threshold: float = 2.4,
        log_prob_threshold: float = -1.0,
        hotwords: str = "",
        hotwords_file: Optional[str | Path] = None,
    ) -> None:
        self.model_size = model_size
        self.requested_device = device
        self.requested_compute_type = compute_type
        self.beam_size = max(1, int(beam_size))
        self.vad_min_silence_ms = int(vad_min_silence_ms)
        self.chunk_min_seconds = float(chunk_min_seconds)
        self.chunk_max_seconds = float(chunk_max_seconds)
        self.chunk_pad_seconds = max(0.0, chunk_pad_ms / 1000.0)
        self.carry_context_chars = max(0, int(carry_context_chars))
        self.condition_on_previous_text = bool(condition_on_previous_text)
        self.seam_repair_enabled = bool(seam_repair)
        self.seam_window_seconds = max(0.5, float(seam_window_seconds))
        self.max_seam_repairs = max(0, int(max_seam_repairs))
        self.no_speech_threshold = float(no_speech_threshold)
        self.hotwords_file = hotwords_file
        if hotwords_file is not None:
            self.hotwords_list: List[str] = load_hotwords(
                hotwords_file, extra_hotwords=hotwords
            )
            self.hotwords = select_whisper_hotwords(self.hotwords_list)
        else:
            self.hotwords_list = []
            self.hotwords = (hotwords or "").strip()
        #: প্রথম chunk-এ শুধু একবার দেওয়া প্রসঙ্গ (ভিডিওর নাম/বিষয়)
        self.context_hint = ""
        self.compression_ratio_threshold = float(compression_ratio_threshold)
        self.log_prob_threshold = float(log_prob_threshold)

        self._model = None
        self.device: Optional[str] = None
        self.compute_type: Optional[str] = None
        #: সর্বশেষ ``transcribe()``-এর পরিসংখ্যান (রিপোর্টে ব্যবহৃত)
        self.last_stats: dict = {}

    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load the Whisper model on first use (using local cache if available)."""
        if self._model is not None:
            return
        _ensure_cuda_libraries()
        self.device, self.compute_type = _resolve_device(
            self.requested_device, self.requested_compute_type
        )
        from faster_whisper import WhisperModel

        model_source = _resolve_cached_model_path(self.model_size)
        if model_source:
            logger.info(
                "Whisper মডেল ক্যাশ থেকে লোড হচ্ছে (কোনো ডাউনলোড লাগবে না): %s | device=%s | compute=%s",
                model_source, self.device, self.compute_type,
            )
        else:
            logger.info(
                "Whisper মডেল ক্যাশে পাওয়া যায়নি — একবারের জন্য ডাউনলোড ও ক্যাশ করা হচ্ছে: %s",
                self.model_size,
            )
            try:
                model_source = ensure_whisper_model(self.model_size)
            except Exception:
                model_source = self.model_size

        started = time.perf_counter()
        stop_heartbeat = threading.Event()

        def _heartbeat() -> None:
            while not stop_heartbeat.wait(30.0):
                logger.info(
                    "  … Whisper মডেল লোড এখনো চলছে (%.0fs) — অপেক্ষা করুন।",
                    time.perf_counter() - started,
                )

        heartbeat = threading.Thread(target=_heartbeat, daemon=True)
        heartbeat.start()
        try:
            is_local_dir = Path(model_source).is_dir()
            self._model = WhisperModel(
                model_source,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=(os.cpu_count() or 4) if self.device == "cpu" else 0,
                local_files_only=is_local_dir,
            )
        except TypeError:
            # পুরোনো faster-whisper সংস্করণ যেখানে local_files_only নেই
            self._model = WhisperModel(
                model_source,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=(os.cpu_count() or 4) if self.device == "cpu" else 0,
            )
        finally:
            stop_heartbeat.set()
        logger.info("Whisper মডেল প্রস্তুত (লোড: %.1fs)।", time.perf_counter() - started)

    def release(self) -> None:
        """Unload the model and free GPU (VRAM)."""
        had_model = self._model is not None
        self._model = None
        self.device = None
        self.compute_type = None
        release_cuda_memory()
        if had_model:
            logger.info("Whisper মডেল আনলোড হয়েছে — GPU মেমরি (VRAM) খালি করা হলো।")

    # Alias for callers using unload()
    unload = release

    # ------------------------------------------------------------------
    # Model-facing primitive (tests override this)
    # ------------------------------------------------------------------
    def _decode_words(
        self,
        samples,
        offset: float = 0.0,
        initial_prompt: Optional[str] = None,
        language: str = SOURCE_LANGUAGE,
    ) -> Tuple[List[Word], str]:
        """Run Whisper on one in-memory chunk and return (words, text)."""
        if self._model is None:  # pragma: no cover - transcribe() আগে _load() ডাকে
            self._load()
        options = dict(
            language=language,
            task="transcribe",
            beam_size=self.beam_size,
            word_timestamps=True,          # sync ও segmenter-এর জন্য জরুরি
            without_timestamps=False,
            vad_filter=False,              # আমরা নিজেরাই VAD করে কাটছি
            condition_on_previous_text=self.condition_on_previous_text,
            initial_prompt=initial_prompt or None,
            no_speech_threshold=self.no_speech_threshold,
            compression_ratio_threshold=self.compression_ratio_threshold,
            log_prob_threshold=self.log_prob_threshold,
            hallucination_silence_threshold=2.0,
        )
        if self.hotwords:
            # পুরোনো faster-whisper-এ ``hotwords`` নেই — TypeError হলে বাদ দেওয়া হয়
            try:
                segments, _info = self._model.transcribe(
                    samples, hotwords=self.hotwords, **options
                )
            except TypeError:  # pragma: no cover - পুরোনো সংস্করণ
                logger.info(
                    "এই faster-whisper সংস্করণে hotwords নেই — সাধারণ ডিকোড চলবে।"
                )
                self.hotwords = ""
                segments, _info = self._model.transcribe(samples, **options)
        else:
            segments, _info = self._model.transcribe(samples, **options)
        words: List[Word] = []
        texts: List[str] = []
        for segment in segments:
            text = (segment.text or "").strip()
            if text:
                texts.append(text)
            for word in (segment.words or []):
                if not (word.word or "").strip():
                    continue
                words.append(
                    Word(
                        start=float(offset) + float(word.start),
                        end=float(offset) + float(word.end),
                        word=word.word,
                        probability=getattr(word, "probability", None),
                    )
                )
        return words, " ".join(texts)

    # ------------------------------------------------------------------
    def set_context_hint(self, text: str, max_chars: int = 200) -> None:
        """প্রথম chunk-এ দেওয়ার জন্য সংক্ষিপ্ত প্রসঙ্গ (ভিডিওর নাম) বসায়।"""
        cleaned = re.sub(r"[_\-]+", " ", str(text or ""))
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"\b(mp4|mkv|mov|webm|avi|flv|wav|mp3|m4a)\b", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
        self.context_hint = cleaned[:max_chars]

    # ------------------------------------------------------------------
    def transcribe(self, audio_path: Path | str) -> List[Segment]:
        """Transcribe a media file into full-sentence :class:`Segment`s."""
        audio = decode_audio(audio_path, STT_SAMPLE_RATE)
        return self.transcribe_audio(audio, STT_SAMPLE_RATE)

    def transcribe_audio(self, audio, sample_rate: int = STT_SAMPLE_RATE) -> List[Segment]:
        """Transcribe an in-memory waveform (already decoded at ``sample_rate``)."""
        self._load()
        started = time.perf_counter()
        total_seconds = audio.size / float(sample_rate) if sample_rate else 0.0

        chunks, method = plan_chunks(
            audio,
            sample_rate=sample_rate,
            min_seconds=self.chunk_min_seconds,
            max_seconds=self.chunk_max_seconds,
            pad_seconds=self.chunk_pad_seconds,
            min_silence_ms=self.vad_min_silence_ms,
        )
        if not chunks:
            raise ValueError("অডিওতে কোনো English speech পাওয়া যায়নি। ভিডিওটিতে কথা আছে তো?")

        words: List[Word] = []
        carried = ""
        per_chunk_text: List[str] = []
        total_chunks = len(chunks)
        logger.info(
            "Whisper ডিকোড শুরু: %dটি chunk | মোট অডিও %s | device=%s",
            total_chunks, format_duration(total_seconds), self.device,
        )
        first_prompt = self.context_hint or None
        for index, chunk in enumerate(chunks, start=1):
            start_sample = int(chunk.start * sample_rate)
            end_sample = min(audio.size, int(chunk.end * sample_rate))
            if end_sample <= start_sample:
                continue
            logger.info(
                "  STT %d/%d — %s–%s (%s)",
                index, total_chunks, format_duration(chunk.start),
                format_duration(chunk.end), format_duration(chunk.duration),
            )
            chunk_words, text = self._decode_words(
                audio[start_sample:end_sample],
                offset=chunk.start,
                initial_prompt=carried or (first_prompt if index == 1 else None),
            )
            if chunk_words:
                words.extend(chunk_words)
            if text:
                per_chunk_text.append(text)
                if self.carry_context_chars:
                    carried = _tail_context(
                        f"{carried} {text}".strip(), self.carry_context_chars
                    )

        if not words:
            raise ValueError("অডিওতে কোনো English speech পাওয়া যায়নি। ভিডিওটিতে কথা আছে তো?")

        repaired = 0
        if self.seam_repair_enabled and len(chunks) > 1:
            words, repaired = self._repair_seams(words, audio, sample_rate, chunks)

        segments = build_sentences(words)
        if not segments:
            raise ValueError("শব্দ পাওয়া গেছে কিন্তু কোনো পূর্ণ বাক্য তৈরি করা যায়নি।")

        if self.hotwords_list:
            for seg in segments:
                seg.text_en = repair_transcript_hotwords(seg.text_en, self.hotwords_list)

        elapsed = time.perf_counter() - started
        self.last_stats = {
            "audio_seconds": total_seconds,
            "elapsed_seconds": elapsed,
            "rtf": (elapsed / total_seconds) if total_seconds else 0.0,
            "chunks": len(chunks),
            "forced_chunks": sum(1 for c in chunks if c.forced),
            "regions_method": method,
            "words": len(words),
            "sentences": len(segments),
            "seam_repairs": repaired,
            "device": self.device,
        }
        logger.info(
            "Transcription শেষ: %d বাক্য (%d শব্দ) | অডিও %s | সময় %.1fs (%.1f× realtime) | "
            "chunk=%d, জোর করে কাটা=%d, seam repair=%d",
            len(segments), len(words), format_duration(total_seconds), elapsed,
            (total_seconds / elapsed) if elapsed else 0.0,
            len(chunks), self.last_stats["forced_chunks"], repaired,
        )
        return segments

    # ------------------------------------------------------------------
    def _repair_seams(
        self,
        words: List[Word],
        audio,
        sample_rate: int,
        chunks: Sequence[SpeechChunk],
    ) -> Tuple[List[Word], int]:
        """Boundary-তে ভাঙা শব্দগুলো ছোট উইন্ডো আবার ডিকোড করে ঠিক করে।"""
        boundaries = find_suspicious_boundaries(words, chunks)
        if not boundaries:
            return words, 0
        windows = merge_windows(
            boundaries, self.seam_window_seconds, self.chunk_max_seconds
        )[: self.max_seam_repairs]
        logger.info(
            "Seam repair: %dটি সীমানায় শব্দ কাটা পড়তে পারে — %dটি উইন্ডো আবার ডিকোড হচ্ছে…",
            len(boundaries), len(windows),
        )

        repaired_words = list(words)
        repaired = 0
        for start, end in windows:
            start_sample = max(0, int(start * sample_rate))
            end_sample = min(audio.size, int(end * sample_rate))
            if end_sample - start_sample < sample_rate * 0.2:
                continue
            prompt_words = [w for w in repaired_words if w.end <= start]
            prompt = _tail_context(join_words(prompt_words[-40:]), 200) or None
            try:
                new_words, _text = self._decode_words(
                    audio[start_sample:end_sample], offset=start, initial_prompt=prompt
                )
            except Exception as exc:  # মেরামত ব্যর্থ হলেও মূল ফলাফল থাক
                logger.warning("Seam repair ব্যর্থ (%.1f–%.1fs): %s", start, end, exc)
                continue
            if not new_words:
                continue
            repaired_words = replace_in_window(repaired_words, new_words, start, end)
            repaired += 1
        return repaired_words, repaired


# Backwards-compatible alias
WhisperTranscriber = SpeechToText
