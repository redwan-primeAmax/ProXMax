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

মডেল প্রথমবার নিজে থেকেই Hugging Face থেকে ডাউনলোড হয় — আলাদা
ডাউনলোড স্ক্রিপ্টের দরকার নেই।
"""

from __future__ import annotations

import ctypes
import glob
import os
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .audio_fx import decode_audio
from .chunking import SpeechChunk, plan_chunks
from .config import SOURCE_LANGUAGE
from .models import Word
from .segmenter import build_sentences, join_words
from .utils import format_duration, get_logger, release_cuda_memory

logger = get_logger(__name__)

_CUDA_LIBS_READY = False
#: STT-এর বিশ্লেষণ sample rate (Whisper-এর পছন্দ)
STT_SAMPLE_RATE = 16000


# =============================================================================
# CUDA / device helpers
# =============================================================================
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
    """Chunk সীমানাগুলোর মধ্যে যেগুলোতে শব্দ কাটা পড়ার আশঙ্কা আছে।

    কাটা পড়ার লক্ষণ:
      * কোনো শব্দ সীমানার দুই পাশে বিস্তৃত (straddle), অথবা
      * বাঁ দিকের শেষ শব্দ হুবহু সীমানায় শেষ, অথবা
      * ডান দিকের প্রথম শব্দ হুবহু সীমানায় শুরু।
    """
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
    ) -> None:
        """Args:
            model_size: Whisper মডেল (``large-v3`` / ``medium``)।
            device: ``auto`` | ``cuda`` | ``cpu``।
            compute_type: GPU-তে ``float16``; CPU-তে নিজেই ``int8`` হয়।
            beam_size: Beam search প্রস্থ (৫ ভালো ভারসাম্য)।
            vad_min_silence_ms: VAD-এর ন্যূনতম নীরবতা (কাটার জায়গা)।
            chunk_min_seconds: চাঙ্কের কাঙ্ক্ষিত ন্যূনতম দৈর্ঘ্য।
            chunk_max_seconds: চাঙ্কের সর্বোচ্চ দৈর্ঘ্য (৩০s-এর নিচে)।
            chunk_pad_ms: চাঙ্কের দুই পাশে context padding।
            carry_context_chars: ``initial_prompt``-এ কত অক্ষর প্রসঙ্গ নেওয়া হবে।
            condition_on_previous_text: Whisper-এর ভেতরের প্রসঙ্গ-সূত্র চালু।
            seam_repair: সীমানায় শব্দ কাটা পড়লে ঠিক করা হবে কি না।
            seam_window_seconds: seam repair-এ সীমানার দুই পাশে কত সেকেন্ড।
            max_seam_repairs: সর্বোচ্চ কতটি সীমানা মেরামত করা হবে (সময় নিয়ন্ত্রণে)।
            no_speech_threshold: Whisper-এর নীরবতা থ্রেশহোল্ড।
            compression_ratio_threshold: পুনরাবৃত্তি-লুপ আটকানোর থ্রেশহোল্ড।
            log_prob_threshold: কম আত্মবিশ্বাসের segment বাদ দেওয়ার সীমা।
        """
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
        self.compression_ratio_threshold = float(compression_ratio_threshold)
        self.log_prob_threshold = float(log_prob_threshold)

        self._model = None
        self.device: Optional[str] = None
        self.compute_type: Optional[str] = None
        #: সর্বশেষ ``transcribe()``-এর পরিসংখ্যান (রিপোর্টে ব্যবহৃত)
        self.last_stats: dict = {}

    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load the Whisper model on first use (auto-downloads weights)."""
        if self._model is not None:
            return
        _ensure_cuda_libraries()
        self.device, self.compute_type = _resolve_device(
            self.requested_device, self.requested_compute_type
        )
        from faster_whisper import WhisperModel

        logger.info(
            "Whisper মডেল লোড হচ্ছে: %s | device=%s | compute=%s "
            "(প্রথমবার ডাউনলোড হতে সময় লাগবে)…",
            self.model_size, self.device, self.compute_type,
        )
        started = time.perf_counter()
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=(os.cpu_count() or 4) if self.device == "cpu" else 0,
        )
        logger.info("Whisper মডেল প্রস্তুত (লোড: %.1fs)।", time.perf_counter() - started)

    def release(self) -> None:
        """Unload the model and free GPU (VRAM).

        এক ভিডিও বা পুরো ব্যাচের কাজ শেষে ডাকা হয়। পরে আবার
        ``transcribe()`` ডাকলে মডেল নিজে থেকেই পুনরায় লোড হবে।
        """
        had_model = self._model is not None
        self._model = None
        self.device = None
        self.compute_type = None
        release_cuda_memory()
        if had_model:
            logger.info("Whisper মডেল আনলোড হয়েছে — GPU মেমরি (VRAM) খালি করা হলো।")

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
        """Run Whisper on one in-memory chunk and return (words, text).

        এটিই একমাত্র জায়গা যেখানে মডেল স্পর্শ করা হয় — তাই টেস্টে এটি
        override করে পুরো পাইপলাইন পরীক্ষা করা যায়।

        Args:
            samples: ``float32`` mono array (সেকেন্ডে নয়, sample-এ)।
            offset: এই অডিওটুকু পুরো ফাইলের কোন সেকেন্ডে শুরু, তা যোগ হয়।
            initial_prompt: আগের টেক্সটের প্রসঙ্গ।
            language: ``"en"`` (ভিডিও) বা ``"bn"`` (রেফারেন্স নমুনা)।

        Returns:
            ``(words, text)`` — শব্দগুলোর সময় absolute।
        """
        if self._model is None:  # pragma: no cover - transcribe() আগে _load() ডাকে
            self._load()
        segments, _info = self._model.transcribe(
            samples,
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
    def transcribe(self, audio_path: Path | str) -> List[Segment]:
        """Transcribe a media file into full-sentence :class:`Segment`s.

        Raises:
            ValueError: অডিওতে কোনো কথা (speech) পাওয়া না গেলে।
        """
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
        for chunk in chunks:
            start_sample = int(chunk.start * sample_rate)
            end_sample = min(audio.size, int(chunk.end * sample_rate))
            if end_sample <= start_sample:
                continue
            chunk_words, text = self._decode_words(
                audio[start_sample:end_sample],
                offset=chunk.start,
                initial_prompt=carried or None,
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

    # ------------------------------------------------------------------
    def transcribe_reference(self, audio_path: Path | str, language: str = "bn") -> str:
        """Transcribe the voice sample (``ref_text`` for IndicF5).

        Args:
            audio_path: ভয়েস রেফারেন্স WAV।
            language: নমুনার ভাষা (ডিফল্ট বাংলা)।

        Returns:
            পরিষ্কার এক লাইন টেক্সট (কিছু না পেলে খালি স্ট্রিং)।
        """
        self._load()
        audio = decode_audio(audio_path, STT_SAMPLE_RATE)
        chunks, _ = plan_chunks(
            audio, STT_SAMPLE_RATE,
            min_seconds=self.chunk_min_seconds, max_seconds=self.chunk_max_seconds,
            pad_seconds=0.0, min_silence_ms=self.vad_min_silence_ms,
        )
        texts: List[str] = []
        carried = ""
        for chunk in chunks or [SpeechChunk(0.0, audio.size / STT_SAMPLE_RATE)]:
            start = int(chunk.start * STT_SAMPLE_RATE)
            end = min(audio.size, int(chunk.end * STT_SAMPLE_RATE))
            if end <= start:
                continue
            _words, text = self._decode_words(
                audio[start:end], offset=chunk.start,
                initial_prompt=carried or None, language=language,
            )
            if text:
                texts.append(text)
                carried = _tail_context(" ".join(texts), self.carry_context_chars)
        result = " ".join(texts).strip()
        if not result:
            logger.warning("ভয়েস রেফারেন্সে কোনো কথা চেনা যায়নি (ভাষা=%s)।", language)
        return result
