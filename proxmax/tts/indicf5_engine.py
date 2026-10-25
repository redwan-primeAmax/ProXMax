"""IndicF5 voice-cloning TTS engine — your own voice speaking Bangla.

আপনার নিজের গলায় বাংলা ডাবিং। AI4Bharat-এর **IndicF5** (MIT লাইসেন্স, ~৩৩৭M প্যারামিটার, 24 kHz) একটি
zero-shot voice-cloning TTS। এটি ১৪১৭ ঘণ্টা Indic ডেটায় প্রশিক্ষিত এবং
বাংলা সমর্থন করে। আমাদের কাছে যা লাগে:

* ``ref_audio_path`` — ব্যবহারকারীর পরিষ্কার বাংলা নমুনা (VoiceProfile)
* ``ref_text`` — ওই নমুনায় ঠিক কী বলা হয়েছে
* ``text`` — যে বাংলা বাক্যটি বলাতে হবে

এই ইঞ্জিনই "রোবট গলা" সমস্যার মূল সমাধান: কণ্ঠ, সুর ও শ্বাস-বিরতি
ব্যবহারকারীর নিজের নমুনা থেকেই আসে, তাই সিন্থেটিক ফ্ল্যাট ভাব থাকে না।

গতি নিয়ন্ত্রণ :class:`~proxmax.tts.governor.SpeedGovernor` করে — বাজেট
ছাড়ালে ``nfe`` ধাপ কমে।
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from ..models import Segment
from ..text_utils import prepare_for_speech
from ..utils import get_logger, release_cuda_memory
from ..voice_profile import VoiceProfile
from .base import BaseTTSEngine, TTSError
from .governor import SpeedGovernor

logger = get_logger(__name__)

#: Hugging Face-এ মডেলের আইডি (প্রথমবার স্বয়ংক্রিয় ডাউনলোড হবে)
DEFAULT_MODEL_ID = "ai4bharat/IndicF5"


def _to_numpy(output: Any) -> np.ndarray:
    """Model output (torch tensor / list / ndarray) → mono float32 array."""
    if hasattr(output, "detach"):
        output = output.detach().to("cpu").float().numpy()
    audio = np.asarray(output, dtype=np.float32)
    audio = np.squeeze(audio)
    if audio.ndim > 1:
        # (channels, samples) বা (samples, channels) — ছোট অক্ষটি channel
        audio = audio.mean(axis=0) if audio.shape[0] < audio.shape[1] else audio.mean(axis=1)
    return np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """NumPy-ভিত্তিক হালকা resample (শুধু দরকার হলে ব্যবহৃত হয়)।"""
    if src_rate == dst_rate or audio.size == 0:
        return audio
    duration = audio.size / float(src_rate)
    target_size = max(1, int(round(duration * dst_rate)))
    src_index = np.linspace(0.0, audio.size - 1.0, num=target_size, dtype=np.float64)
    return np.interp(src_index, np.arange(audio.size), audio).astype(np.float32)


class IndicF5Engine(BaseTTSEngine):
    """Voice-cloning Bangla TTS through AI4Bharat's IndicF5."""

    name = "indicf5"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        fp16: bool = True,
        nfe_steps: int = 16,
        min_nfe_steps: int = 8,
        sample_rate: int = 24000,
        max_attempts: int = 2,
        time_budget_seconds: Optional[float] = None,
        total_audio_seconds: float = 0.0,
    ) -> None:
        """Args:
            model_id: Hugging Face মডেল আইডি।
            fp16: CUDA-তে half precision ব্যবহার করা হবে (দ্রুত, কম VRAM)।
            nfe_steps: diffusion ধাপ (বেশি = ভালো মান, ধীর)।
            min_nfe_steps: governor সর্বনিম্ন যে ধাপে নামতে পারে।
            sample_rate: আউটপুট sample rate।
            max_attempts: একটি বাক্যের জন্য সর্বোচ্চ চেষ্টা।
            time_budget_seconds: TTS-এর জন্য সময়ের বাজেট।
            total_audio_seconds: মোট কত সেকেন্ড কথা হবে (governor-এর অনুমান)।
        """
        super().__init__()
        self.model_id = model_id
        self.fp16 = bool(fp16)
        self.sample_rate = int(sample_rate)
        self.max_attempts = max(1, int(max_attempts))
        self._model: Any = None
        self._torch: Any = None
        self._device = "cpu"
        self._model_rate = self.sample_rate
        self.governor = SpeedGovernor(
            nfe_steps=nfe_steps,
            min_nfe_steps=min_nfe_steps,
            time_budget_seconds=time_budget_seconds,
            total_audio_seconds=total_audio_seconds,
        )

    # ------------------------------------------------------------------
    def prepare(self, profile: Optional[VoiceProfile] = None) -> None:
        """ভয়েস প্রোফাইল যাচাই করে মডেল লোড করে।

        Raises:
            TTSError: রেফারেন্স না থাকলে বা মডেল লোড না হলে।
        """
        super().prepare(profile)
        if profile is None or not profile.is_valid():
            raise TTSError(
                "IndicF5-এর জন্য ভয়েস রেফারেন্স দরকার। ইনপুট ফোল্ডারে "
                "আপনার ১০–২০ সেকেন্ডের পরিষ্কার বাংলা কথা সম্বলিত একটি "
                "অডিও ফাইল (.wav/.mp3/.m4a) রাখুন "
                "(অথবা reference/ ফোল্ডারে)।"
            )
        self._load()

    def _load(self) -> None:
        """Load the model once (first run auto-downloads ~2 GB weights)."""
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - Colab-এ থাকে
            raise TTSError(
                "IndicF5-এর জন্য transformers/torch দরকার। ইনস্টল করুন:\n"
                "    pip install -r requirements-indicf5.txt"
            ) from exc

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if (self.fp16 and self._device == "cuda") else torch.float32
        logger.info(
            "IndicF5 লোড হচ্ছে: %s | device=%s | dtype=%s "
            "(প্রথমবার ~২GB ডাউনলোড হতে সময় লাগবে)…",
            self.model_id, self._device, dtype,
        )
        started = time.perf_counter()
        try:
            self._model = AutoModel.from_pretrained(
                self.model_id, trust_remote_code=True, torch_dtype=dtype
            )
        except (TypeError, ValueError) as exc:
            logger.warning("fp16 লোড ব্যর্থ (%s) — default precision-এ চেষ্টা হচ্ছে।", exc)
            self._model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        except Exception as exc:  # ডাউনলোড/নেটওয়ার্ক/gated-অনুমতি সমস্যা
            text = f"{type(exc).__name__}: {exc}"
            lowered = text.lower()
            if any(word in lowered for word in ("gated", "401", "403", "unauthorized",
                                                "forbidden", "access to model")):
                raise TTSError(
                    "IndicF5 মডেলটি gated — আগে অনুমতি নিতে হবে:\n"
                    "  1) https://huggingface.co/ai4bharat/IndicF5 খুলে "
                    "\"Agree and access repository\" চাপুন\n"
                    "  2) Colab-এ Hugging Face token দিয়ে লগইন করুন "
                    "(নোটবুকের ধাপ ৩)\n"
                    f"  মূল ত্রুটি: {text}"
                ) from exc
            raise TTSError(f"IndicF5 মডেল লোড করা যায়নি: {text}") from exc

        try:
            self._model = self._model.to(self._device)
        except Exception as exc:  # pragma: no cover - CPU-তে থেকে গেলেও চলবে
            logger.warning("মডেল %s-এ নেওয়া যায়নি (%s) — CPU-তে চলবে।", self._device, exc)
            self._device = "cpu"
        if hasattr(self._model, "eval"):
            self._model.eval()

        for attr in ("sampling_rate", "sample_rate"):
            value = getattr(self._model, attr, None)
            if isinstance(value, (int, float)) and int(value) > 8000:
                self._model_rate = int(value)
                break
        logger.info(
            "IndicF5 প্রস্তুত (লোড: %.1fs | model sample rate=%d)।",
            time.perf_counter() - started, self._model_rate,
        )

    # ------------------------------------------------------------------
    def _generate(self, text: str) -> np.ndarray:
        """একটি বাক্যের অডিও তৈরি করে (retry ছাড়া)।"""
        if self._model is None:  # pragma: no cover - prepare() আগে ডাকা হয়
            self._load()
        assert self.profile is not None  # noqa: S101 - prepare() নিশ্চিত করেছে
        torch = self._torch
        kwargs = {
            "ref_audio_path": str(self.profile.audio_path),
            "ref_text": self.profile.text,
        }
        nfe = self.governor.current_nfe
        with torch.inference_mode():
            try:
                output = self._model(text, nfe_step=nfe, **kwargs)
            except TypeError:
                # কিছু সংস্করণে nfe_step সাপোর্ট করে না — ডিফল্টে চালানো হয়
                output = self._model(text, **kwargs)
        audio = _to_numpy(output)
        if audio.size == 0:
            raise TTSError("IndicF5 খালি অডিও ফিরিয়েছে")
        return _resample_linear(audio, self._model_rate, self.sample_rate)

    # ------------------------------------------------------------------
    def synthesize(self, segments: List[Segment], output_dir: Path | str) -> List[Segment]:
        """প্রতিটি segment-এর জন্য ক্লোন করা বাংলা কণ্ঠ তৈরি করে।"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        todo = self.usable_segments(segments)
        if not todo:
            raise TTSError("TTS-এর জন্য কোনো বাংলা টেক্সট নেই (সব segment-এর text_bn খালি)।")

        logger.info(
            "IndicF5 TTS শুরু: %d segment | voice=আপনার ক্লোন | nfe=%d | device=%s",
            len(todo), self.governor.current_nfe, self._device,
        )
        failures = 0
        started_all = time.perf_counter()

        for position, seg in enumerate(todo, start=1):
            text = prepare_for_speech(seg.text_bn)
            if not text:
                seg.tts_audio_path = None
                continue
            last_error: Optional[BaseException] = None
            for attempt in range(1, self.max_attempts + 1):
                started = time.perf_counter()
                try:
                    audio = self._generate(text)
                    elapsed = time.perf_counter() - started
                    self._write_clip(seg, audio, output_dir)
                    self.governor.observe(seg.tts_duration or 0.0, elapsed)
                    logger.debug(
                        "  TTS %d/%d: %.2fs অডিও (%.2fs সময়, nfe=%d)",
                        position, len(todo), seg.tts_duration or 0.0, elapsed,
                        self.governor.current_nfe,
                    )
                    last_error = None
                    break
                except Exception as exc:  # GPU OOM / মডেল ত্রুটি
                    last_error = exc
                    logger.warning(
                        "Segment %d TTS ব্যর্থ (চেষ্টা %d/%d): %s",
                        seg.index, attempt, self.max_attempts, exc,
                    )
                    if attempt < self.max_attempts:
                        release_cuda_memory()
            if last_error is not None:
                failures += 1
                seg.tts_audio_path = None
                seg.tts_duration = None

        done = [s for s in todo if s.tts_audio_path]
        if not done:
            raise TTSError(
                "একটি segment-এরও অডিও তৈরি হয়নি। GPU মেমরি (VRAM) ও IndicF5 "
                "মডেলের অবস্থা যাচাই করুন।"
            )
        logger.info(
            "IndicF5 TTS শেষ: %d/%d segment | %s | মোট %.1fs",
            len(done), len(todo), self.governor.summary(),
            time.perf_counter() - started_all,
        )
        if failures:
            logger.warning("%dটি segment-এর কণ্ঠ তৈরি হয়নি — সেগুলো বাদ যাবে।", failures)
        return segments

    # ------------------------------------------------------------------
    def set_budget(
        self,
        time_budget_seconds: Optional[float],
        total_audio_seconds: float = 0.0,
    ) -> None:
        """নতুন ভিডিওর কথা-দৈর্ঘ্য অনুযায়ী governor-এর বাজেট বদলায়।"""
        self.governor.set_budget(time_budget_seconds, total_audio_seconds)

    def describe(self) -> str:
        return f"indicf5:{self.model_id.split('/')[-1]}"

    def close(self) -> None:
        """Unload the model so Colab's GPU memory is released."""
        had_model = self._model is not None
        self._model = None
        self._torch = None
        if had_model:
            release_cuda_memory()
            logger.info("IndicF5 মডেল আনলোড — GPU মেমরি ছেড়ে দেওয়া হলো।")
