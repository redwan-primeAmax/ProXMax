"""Model-loading and recognition methods shared by the public STT class."""

from __future__ import annotations

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
    load_hotwords, parse_hotwords_text, repair_transcript_hotwords,
    select_whisper_hotwords,
)
from .models import Segment, Word
from .utils import get_logger, release_cuda_memory
from .whisper_model import (
    _ensure_cuda_libraries, _resolve_cached_model_path, _resolve_device,
    ensure_whisper_model,
)

logger = get_logger(__name__)
STT_SAMPLE_RATE = 16000


class WhisperModelMixin:
    """Whisper model lifecycle and multilingual word-recognition methods."""

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
        hotwords_prompt = self.hotwords if language == SOURCE_LANGUAGE else ""
        if self.hotwords_list and language == SOURCE_LANGUAGE:
            hotwords_prompt = select_whisper_hotwords(
                self.hotwords_list,
                context=initial_prompt or self.context_hint,
            )
        self.last_hotwords_prompt = hotwords_prompt

        prompt = initial_prompt or None
        attempts: List[Tuple[Optional[str], str]] = []
        if hotwords_prompt and self._supports_hotwords:
            attempts.append((prompt, hotwords_prompt))
        attempts.append((prompt, ""))
        if prompt is not None:
            attempts.append((None, ""))

        for attempt_index, (prompt_text, attempt_hotwords) in enumerate(attempts):
            call_options = dict(options)
            call_options["initial_prompt"] = prompt_text
            try:
                if attempt_hotwords:
                    try:
                        segments, _info = self._model.transcribe(
                            samples, hotwords=attempt_hotwords, **call_options
                        )
                    except TypeError as exc:  # pragma: no cover - older faster-whisper
                        if "hotword" not in str(exc).lower():
                            raise
                        logger.info(
                            "এই faster-whisper সংস্করণে hotwords নেই — সাধারণ ডিকোড চলবে।"
                        )
                        self._supports_hotwords = False
                        self.hotwords = ""
                        continue
                else:
                    segments, _info = self._model.transcribe(samples, **call_options)

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
                self.last_hotwords_prompt = attempt_hotwords
                return words, " ".join(texts)
            except ValueError as exc:
                if "maximum decoding length must be > 0" not in str(exc).lower():
                    raise
                if attempt_index + 1 >= len(attempts):
                    raise
                next_prompt, next_hotwords = attempts[attempt_index + 1]
                fallback_stats = getattr(self, "_prompt_fallback_stats", None)
                if attempt_hotwords and not next_hotwords:
                    if isinstance(fallback_stats, dict):
                        fallback_stats["without_hotwords"] = int(
                            fallback_stats.get("without_hotwords", 0)
                        ) + 1
                    logger.warning(
                        "Whisper decoder prompt window পূর্ণ হয়ে গেছে — এই chunk hotwords ছাড়া retry হবে।"
                    )
                elif prompt_text is not None and next_prompt is None:
                    if isinstance(fallback_stats, dict):
                        fallback_stats["without_context"] = int(
                            fallback_stats.get("without_context", 0)
                        ) + 1
                    logger.warning(
                        "Whisper decoder prompt এখনো বড় — এই chunk hotwords ও context ছাড়া retry হবে।"
                    )

        raise RuntimeError("Whisper decode কোনো fallback ফলাফল ফেরত দেয়নি।")

    def set_context_hint(self, text: str, max_chars: int = 200) -> None:
        """প্রথম chunk-এ দেওয়ার জন্য সংক্ষিপ্ত প্রসঙ্গ (ভিডিওর নাম) বসায়।"""
        cleaned = re.sub(r"[_\-]+", " ", str(text or ""))
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"\b(mp4|mkv|mov|webm|avi|flv|wav|mp3|m4a)\b", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
        self.context_hint = cleaned[:max_chars]

    def transcribe_words_in_language(
        self, audio_path: Path | str, language: str = "bn"
    ) -> List[Word]:
        """Decode speech words in a requested language for post-mux QA.

        This reuses the already-loaded Whisper model where possible. Hotwords
        and English context hints are deliberately disabled for non-source
        language checks.
        """
        if language not in {"en", "bn"}:
            raise ValueError(f"Unsupported validation language: {language}")
        audio = decode_audio(audio_path, STT_SAMPLE_RATE)
        self._load()
        chunks, _method = plan_chunks(
            audio,
            sample_rate=STT_SAMPLE_RATE,
            min_seconds=self.chunk_min_seconds,
            max_seconds=self.chunk_max_seconds,
            pad_seconds=self.chunk_pad_seconds,
            min_silence_ms=self.vad_min_silence_ms,
        )
        words: List[Word] = []
        for chunk in chunks:
            left = max(0, int(chunk.start * STT_SAMPLE_RATE))
            right = min(audio.size, int(chunk.end * STT_SAMPLE_RATE))
            if right <= left:
                continue
            decoded, _text = self._decode_words(
                audio[left:right], offset=chunk.start, language=language
            )
            words.extend(decoded)
        return self._deduplicate_words(words)

    @staticmethod
    def _deduplicate_words(words: Sequence[Word]) -> List[Word]:
        """Remove only the same token occupying essentially the same time span.

        Nearby repeated words ("very, very") are not duplicates merely because
        they are close together. Shared logic covers source STT and QA chunks.
        """
        ordered = sorted(words, key=lambda word: (word.start, word.end))
        result: List[Word] = []
        for word in ordered:
            token = re.sub(r"[^\w\u0980-\u09FF]+", "", word.word.casefold())
            match = None
            for index in range(len(result) - 1, -1, -1):
                previous = result[index]
                if word.start - previous.start > 2.0:
                    break
                if not token or token != re.sub(r"[^\w\u0980-\u09FF]+", "", previous.word.casefold()):
                    continue
                overlap = max(0.0, min(previous.end, word.end) - max(previous.start, word.start))
                durations = (max(0.0, previous.end - previous.start), max(0.0, word.end - word.start))
                shorter, longer = min(durations), max(durations)
                starts_match = abs(previous.start - word.start) < 0.015
                same_zero_span = shorter == 0.0 and starts_match and abs(previous.end - word.end) < 0.015
                same_start_overlap = shorter > 0 and starts_match and overlap / shorter >= 0.8
                similar_spans = longer > 0 and overlap / longer >= 0.8
                if same_zero_span or same_start_overlap or similar_spans:
                    match = index
                    break
            if match is None:
                result.append(word)
            elif (word.probability or 0.0) > (result[match].probability or 0.0):
                result[match] = word
        return sorted(result, key=lambda word: (word.start, word.end))
