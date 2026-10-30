"""Speech-to-Text orchestration using silence-aware, context-carrying chunks."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .audio_fx import decode_audio
from .chunking import SpeechChunk, plan_chunks
from .hotwords import (
    load_hotwords, parse_hotwords_text, repair_transcript_hotwords,
    select_whisper_hotwords,
)
from .models import Segment, Word
from .segmenter import build_sentences, join_words
from .stt_inference import STT_SAMPLE_RATE, WhisperModelMixin
from .stt_seams import (
    _tail_context, find_suspicious_boundaries, merge_windows, replace_in_window,
)
from .utils import format_duration, get_logger
from .whisper_model import (
    _default_whisper_cache_dir, _ensure_cuda_libraries, _resolve_cached_model_path,
    _resolve_device, ensure_whisper_model,
)

logger = get_logger(__name__)

class STTError(RuntimeError):
    """Raised when speech-to-text fails."""


class SpeechToText(WhisperModelMixin):
    """faster-whisper wrapper producing sentence-level :class:`Segment`s."""

    def __init__(
        self,
        model_size: str = "distil-large-v3",
        device: str = "auto",
        compute_type: str = "float16",
        beam_size: int = 5,
        vad_min_silence_ms: int = 300,
        sentence_max_seconds: float = 5.0,
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
        self.sentence_max_seconds = max(0.1, float(sentence_max_seconds))
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
            logger.info(
                "Whisper hotwords লোড: %dটি local term/name | প্রাথমিক context-aware prompt %d অক্ষর",
                len(self.hotwords_list), len(self.hotwords),
            )
        else:
            self.hotwords_list = parse_hotwords_text(hotwords)
            self.hotwords = select_whisper_hotwords(self.hotwords_list)
        self._supports_hotwords = True
        self.last_hotwords_prompt = ""
        self._prompt_fallback_stats = {"without_hotwords": 0, "without_context": 0}
        #: প্রথম chunk-এ শুধু একবার দেওয়া প্রসঙ্গ (ভিডিওর নাম/বিষয়)
        self.context_hint = ""
        self.compression_ratio_threshold = float(compression_ratio_threshold)
        self.log_prob_threshold = float(log_prob_threshold)

        self._model = None
        self.device: Optional[str] = None
        self.compute_type: Optional[str] = None
        #: সর্বশেষ ``transcribe()``-এর পরিসংখ্যান (রিপোর্টে ব্যবহৃত)
        self.last_stats: dict = {}

    def transcribe(self, audio_path: Path | str) -> List[Segment]:
        """Transcribe a media file into full-sentence :class:`Segment`s."""
        audio = decode_audio(audio_path, STT_SAMPLE_RATE)
        return self.transcribe_audio(audio, STT_SAMPLE_RATE)

    def transcribe_audio(self, audio, sample_rate: int = STT_SAMPLE_RATE) -> List[Segment]:
        """Transcribe an in-memory waveform (already decoded at ``sample_rate``)."""
        self._load()
        started = time.perf_counter()
        self._prompt_fallback_stats = {"without_hotwords": 0, "without_context": 0}
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

        before_dedup = len(words)
        words = self._deduplicate_words(words)
        duplicates_removed = before_dedup - len(words)
        repaired = 0
        if self.seam_repair_enabled and len(chunks) > 1:
            words, repaired = self._repair_seams(words, audio, sample_rate, chunks)
            before_dedup = len(words)
            words = self._deduplicate_words(words)
            duplicates_removed += before_dedup - len(words)

        segments = build_sentences(words, max_seconds=self.sentence_max_seconds)
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
            "overlapping_duplicate_words_removed": duplicates_removed,
            "decoder_retries_without_hotwords": self._prompt_fallback_stats["without_hotwords"],
            "decoder_retries_without_context": self._prompt_fallback_stats["without_context"],
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
            candidate = replace_in_window(repaired_words, new_words, start, end)
            if candidate != repaired_words:
                repaired_words = candidate
                repaired += 1
        return repaired_words, repaired


# Backwards-compatible alias
WhisperTranscriber = SpeechToText
