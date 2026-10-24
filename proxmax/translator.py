"""Translation: English → Bangla.

দুটি backend আছে, ``config.yaml`` থেকে বেছে নেওয়া যায়:

* ``google`` (ডিফল্ট) — ``deep-translator`` দিয়ে Google Translate।
  দ্রুত, ফ্রি, কোনো মডেল ডাউনলোড লাগে না। একাধিক segment সমান্তরালে
  (thread pool) translate হয় এবং পুনরাবৃত্ত লাইন cache থেকে আসে।
* ``indictrans2`` — AI4Bharat-এর IndicTrans2 লোকাল মডেল। English→Bangla-তে
  বিশেষভাবে প্রশিক্ষিত, সম্পূর্ণ offline চলতে পারে; তবে আগে
  ``pip install -r requirements-indictrans2.txt`` চালাতে হয়।

Segment-by-segment translate করা হয়, ফলে প্রতিটি segment-এর
timestamp হুবহু বজায় থাকে।
"""

from __future__ import annotations

import random
import re
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from .config import SOURCE_LANGUAGE, TARGET_LANGUAGE
from .models import Segment
from .utils import get_logger, iter_progress

logger = get_logger(__name__)

# Google Translate একবারে ~5000 অক্ষরের বেশি নেয় না; নিরাপদ সীমা ধরে ভাঙা হয়
_MAX_CHARS_PER_REQUEST = 4500
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


class TranslationError(RuntimeError):
    """Raised when translation fails beyond recovery."""


def split_long_text(text: str, max_chars: int = _MAX_CHARS_PER_REQUEST) -> List[str]:
    """Split long text into <= ``max_chars`` chunks on sentence boundaries."""
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: List[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        if len(sentence) > max_chars:  # অত্যন্ত লম্বা একক বাক্য — মাঝপথে ভাঙতে হবে
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars)
            )
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


class BaseTranslator(ABC):
    """Common interface for all translation backends."""

    @abstractmethod
    def translate_text(self, text: str) -> str:
        """Translate a single English string into Bangla."""

    def translate_segments(self, segments: List[Segment]) -> List[Segment]:
        """Fill ``text_bn`` of every segment in-place (sequential default)."""
        for seg in iter_progress(segments, desc="  Translate", total=len(segments)):
            seg.text_bn = self.translate_text(seg.text_en)
        return segments


# =============================================================================
# Backend 1 (default): Google Translate via deep-translator
# =============================================================================
class GoogleBanglaTranslator(BaseTranslator):
    """Google Translate (``deep-translator``) with retries + concurrency."""

    def __init__(self, max_workers: int = 5, max_retries: int = 3) -> None:
        self.max_workers = max(1, max_workers)
        self.max_retries = max(1, max_retries)
        self._thread_local = threading.local()
        self._cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()

    def _translator(self):
        """One ``GoogleTranslator`` instance per thread (thread-safety)।"""
        instance = getattr(self._thread_local, "instance", None)
        if instance is None:
            from deep_translator import GoogleTranslator

            instance = GoogleTranslator(source=SOURCE_LANGUAGE, target=TARGET_LANGUAGE)
            self._thread_local.instance = instance
        return instance

    def translate_text(self, text: str) -> str:
        text = " ".join((text or "").split())
        if not text:
            return ""
        with self._cache_lock:
            cached = self._cache.get(text)
        if cached is not None:
            return cached

        translated_parts: List[str] = []
        for chunk in split_long_text(text):
            translated_parts.append(self._translate_chunk_with_retry(chunk))
        result = " ".join(p.strip() for p in translated_parts if p and p.strip())

        with self._cache_lock:
            self._cache[text] = result
        return result

    def _translate_chunk_with_retry(self, chunk: str) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                result = self._translator().translate(chunk)
                if result and result.strip():
                    return result.strip()
                raise TranslationError("খালি অনুবাদ ফিরেছে (empty response)")
            except Exception as exc:  # নেটওয়ার্ক ঝামেলা / rate-limit — retry যোগ্য
                last_error = exc
                if attempt < self.max_retries:
                    delay = (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
                    logger.debug("অনুবাদে সমস্যা (%s) — %.1fs পরে retry…", exc, delay)
                    time.sleep(delay)
        raise TranslationError(f"অনুবাদ ব্যর্থ হয়েছে: {last_error}")

    def translate_segments(self, segments: List[Segment]) -> List[Segment]:
        """Translate all segments concurrently, preserving their order.

        কোনো একটি segment শেষ পর্যন্ত ব্যর্থ হলে সেটির ``text_bn`` খালি
        থাকে (পরে বাদ যাবে) — বাকি ভিডিওর কাজ থামে না। তবে অর্ধেকের বেশি
        ব্যর্থ হলে ধরে নেওয়া হয় নেটওয়ার্ক/সার্ভিস ডাউন এবং error ওঠে।
        """
        if not segments:
            return segments
        started = time.perf_counter()
        failures = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self.translate_text, seg.text_en): seg for seg in segments}
            for future in iter_progress(
                as_completed(futures), desc="  Translate", total=len(segments)
            ):
                seg = futures[future]
                try:
                    seg.text_bn = future.result()
                except Exception as exc:
                    failures += 1
                    seg.text_bn = ""
                    logger.warning(
                        "Segment %d-এর অনুবাদ হয়নি: %s | EN: %.60s…",
                        seg.index, exc, seg.text_en,
                    )

        if failures > len(segments) / 2:
            raise TranslationError(
                f"{len(segments)}টি segment-এর মধ্যে {failures}টির অনুবাদ ব্যর্থ। "
                f"ইন্টারনেট সংযোগ / Google Translate-এর উপলব্ধতা যাচাই করুন।"
            )
        logger.info(
            "অনুবাদ শেষ: %d/%d segment সফল (%.1fs, cache-এ %dটি লাইন)।",
            len(segments) - failures, len(segments),
            time.perf_counter() - started, len(self._cache),
        )
        return segments


# =============================================================================
# Backend 2 (optional): IndicTrans2 (AI4Bharat) — offline, En→Bn specialist
# =============================================================================
class IndicTrans2Translator(BaseTranslator):
    """IndicTrans2 local model — trained specifically for Indic languages.

    প্রয়োজন: ``pip install -r requirements-indictrans2.txt``
    (transformers + IndicTransToolkit; Colab-এ torch আগে থেকেই থাকে)।
    মডেল (~1B) প্রথমবারে Hugging Face থেকে ডাউনলোড হয়।
    """

    _SRC_TAG = "eng_Latn"
    _TGT_TAG = "ben_Beng"

    def __init__(
        self,
        model_name: str = "ai4bharat/indictrans2-en-indic-1B",
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise TranslationError(
                "IndicTrans2 backend-এর জন্য প্রয়োজনীয় প্যাকেজ নেই। আগে চালান:\n"
                "    pip install -r requirements-indictrans2.txt"
            ) from exc

        # IndicProcessor ট্যাগ/placeholder ঠিকভাবে বসায় — ছাড়া অনুবাদের মান খারাপ হয়
        IndicProcessor = None
        for import_path in (
            "IndicTransToolkit.processor",
            "IndicTransToolkit.IndicTransToolkit.processor",
        ):
            try:
                module = __import__(import_path, fromlist=["IndicProcessor"])
                IndicProcessor = module.IndicProcessor
                break
            except Exception:
                continue
        if IndicProcessor is None:
            raise TranslationError(
                "IndicTransToolkit পাওয়া যায়নি। ইনস্টল করুন:\n"
                "    pip install -r requirements-indictrans2.txt"
            )

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "IndicTrans2 লোড হচ্ছে: %s | device=%s (প্রথমবার ডাউনলোডে সময় লাগবে)…",
            self.model_name, self._device,
        )
        self._processor = IndicProcessor(inference=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
        ).to(self._device)
        self._model.eval()
        logger.info("IndicTrans2 প্রস্তুত।")

    def translate_text(self, text: str) -> str:
        text = " ".join((text or "").split())
        return self._translate_batch([text])[0] if text else ""

    def translate_segments(self, segments: List[Segment]) -> List[Segment]:
        if not segments:
            return segments
        started = time.perf_counter()
        texts = [seg.text_en for seg in segments]
        for i in iter_progress(
            range(0, len(texts), self.batch_size),
            desc="  Translate(IndicTrans2)",
            total=(len(texts) + self.batch_size - 1) // self.batch_size,
        ):
            batch = texts[i : i + self.batch_size]
            for seg, translated in zip(segments[i : i + self.batch_size], self._translate_batch(batch)):
                seg.text_bn = translated
        logger.info(
            "IndicTrans2 অনুবাদ শেষ: %d segment (%.1fs)।",
            len(segments), time.perf_counter() - started,
        )
        return segments

    def _translate_batch(self, texts: List[str]) -> List[str]:
        self._load()
        import torch

        processed = self._processor.preprocess_batch(
            texts, src_lang=self._SRC_TAG, tgt_lang=self._TGT_TAG
        )
        encoded = self._tokenizer(
            processed, truncation=True, padding="longest",
            return_tensors="pt", return_attention_mask=True,
        ).to(self._device)
        with torch.no_grad():
            generated = self._model.generate(
                **encoded,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )
        with self._tokenizer.as_target_tokenizer():
            decoded = self._tokenizer.batch_decode(
                generated.detach().cpu().tolist(),
                skip_special_tokens=True, clean_up_tokenization_spaces=True,
            )
        return self._processor.postprocess_batch(decoded, lang=self._TGT_TAG)


# =============================================================================
def create_translator(backend: str, **kwargs) -> BaseTranslator:
    """Factory: config-এর backend নাম থেকে translator তৈরি করে।"""
    backend = (backend or "google").lower()
    if backend == "google":
        return GoogleBanglaTranslator(
            max_workers=kwargs.get("translation_max_workers", 5),
            max_retries=kwargs.get("translation_max_retries", 3),
        )
    if backend == "indictrans2":
        return IndicTrans2Translator(
            model_name=kwargs.get("indictrans2_model", "ai4bharat/indictrans2-en-indic-1B"),
        )
    raise ValueError(f"অজানা translator backend: '{backend}' (সমর্থিত: google, indictrans2)")
