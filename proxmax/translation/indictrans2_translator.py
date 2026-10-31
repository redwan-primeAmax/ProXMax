"""Optional offline IndicTrans2 English-to-Bengali translator."""

from __future__ import annotations

import time
from typing import List

from .models import Segment
from .text_utils import normalise_bangla
from .translation_errors import TranslationError
from .translation_parsing import normalise_text
from .translator_base import BaseTranslator
from .utils import get_logger, iter_progress

logger = get_logger(__name__)

class IndicTrans2Translator(BaseTranslator):
    """IndicTrans2 local model — trained specifically for Indic languages."""

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

    def translate_text(self, text: str, duration: float = 0.0) -> str:
        original = normalise_text(text)
        return normalise_bangla(self._translate_batch([original])[0]) if original else ""

    def translate_segments(self, segments: List[Segment]) -> List[Segment]:
        if not segments:
            return segments
        started = time.perf_counter()
        texts = [seg.text_en for seg in segments]
        for i in iter_progress(
            range(0, len(texts), self.batch_size),
            desc="  Translate(IndicTrans2)",
            total=(len(texts) + self.batch_size - 1) // self.batch_size,
            unit="batch",
        ):
            batch = texts[i : i + self.batch_size]
            for seg, translated in zip(segments[i : i + self.batch_size], self._translate_batch(batch)):
                seg.text_bn = normalise_bangla(translated)
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

    def describe(self) -> str:
        return f"indictrans2:{self.model_name}"

    def close(self) -> None:
        """Unload the local model and free GPU/CPU memory."""
        from .utils import release_cuda_memory

        self._model = None
        self._tokenizer = None
        self._processor = None
        release_cuda_memory()

    unload = close
