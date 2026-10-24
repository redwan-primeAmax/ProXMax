"""Application configuration: defaults, YAML loading, validation.

সব সেটিংস (Drive-এর পাথ, Whisper মডেল, TTS ভয়েস, tempo সীমা ইত্যাদি)
একটাই ``AppConfig`` dataclass-এ থাকে। ``config.yaml`` থেকে লোড হয়,
চাইলে CLI আর্গুমেন্ট দিয়েও override করা যায়।
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from .utils import get_logger

logger = get_logger(__name__)

#: missing API key-এর warning যেন বারবার না ছাপে (validate() একাধিকবার ডাকা হয়)
_MISSING_KEY_WARNED = False

# ---------------------------------------------------------------------------
# Constants (প্রজেক্টটি শুধু English → Bangla, তাই ভাষা-জোড়া ফিক্সড)
# ---------------------------------------------------------------------------
SOURCE_LANGUAGE = "en"
TARGET_LANGUAGE = "bn"

DEFAULT_INPUT_DIR = "/content/drive/MyDrive/Video/input"
DEFAULT_OUTPUT_DIR = "/content/drive/MyDrive/Video/output"

SUPPORTED_WHISPER_MODELS = (
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v1", "large-v2", "large-v3",
    "distil-large-v2", "distil-large-v3",
)

SUPPORTED_TRANSLATORS = ("gemini", "indictrans2")

#: ডিফল্ট অনুবাদ মডেল (ব্যবহারকারীর নির্দেশ অনুযায়ী) — নোটবুক/CLI দিয়ে বদলানো যায়
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-light"

# edge-tts এর native Bangla (Bangladesh) ভয়েস
BANGLA_VOICES = {
    "female": "bn-BD-NabanitaNeural",
    "male": "bn-BD-PradeepNeural",
}


@dataclass
class AppConfig:
    """All tunable settings for the dubbing pipeline."""

    # --- paths -------------------------------------------------------------
    input_dir: str = DEFAULT_INPUT_DIR
    output_dir: str = DEFAULT_OUTPUT_DIR
    # "auto" = /content/proxmax_work on Colab (দ্রুত লোকাল SSD), অন্যথায় ./proxmax_work
    work_dir: str = "auto"

    # --- speech-to-text (faster-whisper) ------------------------------------
    whisper_model: str = "large-v3"        # best accuracy; speed দরকার হলে "medium"
    whisper_device: str = "auto"           # auto = CUDA থাকলে GPU, না থাকলে CPU
    whisper_compute_type: str = "float16"  # GPU-এর জন্য; CPU fallback-এ int8 হয়
    whisper_batch_size: int = 8            # T4 (16GB) + large-v3 এর জন্য নিরাপদ মান
    whisper_beam_size: int = 5
    vad_filter: bool = True
    vad_min_silence_duration_ms: int = 500

    # --- translation: Google Gemini API (ডিফল্ট) -------------------------------
    translator_backend: str = "gemini"      # "gemini" | "indictrans2"
    gemini_api_key: str = ""                # খালি হলে GEMINI_API_KEY / GOOGLE_API_KEY env
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_batch_size: int = 25             # এক request-এ কত segment
    gemini_max_chars_per_request: int = 12000
    gemini_max_retries: int = 7             # সর্বোচ্চ ৭ বার চেষ্টা
    gemini_retry_base_delay: float = 4.0    # প্রথম retry-এর আগে অপেক্ষা (সেকেন্ড)
    gemini_retry_max_delay: float = 90.0    # এক retry-তে সর্বোচ্চ অপেক্ষা (সেকেন্ড)
    gemini_requests_per_minute: int = 15    # rate limit (RPM)
    gemini_temperature: float = 0.2         # কম = বেশি নির্ভুল অনুবাদ
    gemini_timeout: int = 120               # এক request-এর timeout (সেকেন্ড)
    gemini_max_output_tokens: int = 8192
    indictrans2_model: str = "ai4bharat/indictrans2-en-indic-1B"

    # --- text-to-speech (edge-tts) -------------------------------------------
    tts_voice: str = BANGLA_VOICES["female"]
    tts_rate: str = "+0%"                  # যেমন "+10%" দ্রুত, "-10%" ধীরে
    tts_volume: str = "+0%"
    tts_pitch: str = "+0Hz"
    tts_concurrency: int = 5               # একসঙ্গে কয়টি segment generate হবে
    tts_max_retries: int = 3

    # --- audio-video sync ------------------------------------------------------
    min_tempo: float = 0.65                # এর চেয়ে বেশি ধীর করা হবে না
    max_tempo: float = 1.90                # এর চেয়ে বেশি দ্রুত করা হবে না
    slow_down_short_clips: bool = True     # ছোট ক্লিপ সামান্য ধীর করে slot পূরণ
    sync_workers: int = 4                  # সমান্তরাল ffmpeg atempo কাজ
    sync_sample_rate: int = 24000          # edge-tts এর native rate (মেমরি বাঁচায়)

    # --- runtime ------------------------------------------------------------------
    release_gpu_when_done: bool = True     # কাজ শেষে Whisper মডেল আনলোড → VRAM খালি

    # --- output ------------------------------------------------------------------
    audio_bitrate: str = "160k"
    use_hwaccel: bool = True               # ffmpeg hwaccel / NVENC চেষ্টা করা হবে
    skip_existing: bool = True             # output-এ থাকলে আবার প্রসেস নয় (resume)
    save_srt: bool = True                  # English + Bangla .srt সেভ করা হবে
    keep_temp_files: bool = False          # True হলে debug-এর জন্য মধ্যবর্তী ফাইল থাকে
    log_level: str = "INFO"

    # ---------------------------------------------------------------------------
    @property
    def resolved_work_dir(self) -> Path:
        """Resolve ``work_dir`` ("auto" হলে Colab-এর লোকাল ডিস্ক ব্যবহার হয়)।

        Google Drive (FUSE) ধীরগতির, তাই সব সাময়িক ফাইল Colab-এর লোকাল
        SSD-তে রাখা হয়; শুধু চূড়ান্ত ফলাফল Drive-এ কপি হয়।
        """
        if self.work_dir and self.work_dir != "auto":
            return Path(self.work_dir).expanduser()
        if Path("/content").is_dir():  # Google Colab
            return Path("/content/proxmax_work")
        return Path.cwd() / "proxmax_work"

    # ---------------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, yaml_path: Path | str | None) -> "AppConfig":
        """Load configuration from a YAML file (missing file → defaults)."""
        cfg = cls()
        if yaml_path is None:
            return cfg
        path = Path(yaml_path)
        if not path.is_file():
            logger.warning("Config ফাইল পাওয়া যায়নি (%s) — default সেটিংস ব্যবহৃত হচ্ছে।", path)
            return cfg
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "PyYAML ইনস্টল নেই। চালান: pip install -r requirements.txt"
            ) from exc

        data: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg._apply_mapping(data)
        logger.info("Config লোড হয়েছে: %s", path)
        return cfg

    def _apply_mapping(self, data: Mapping[str, Any]) -> None:
        """Apply a (possibly sectioned) mapping onto this config.

        ``config.yaml``-এর section গঠন:
        paths / stt / translation / gemini / tts / sync / runtime / output —
        পাশাপাশি সরাসরি flat key দিলেও কাজ করবে।
        """
        flat: Dict[str, Any] = {}
        # সেকশনভিত্তিক ছোট alias নাম — লক্ষণীয়: "max_retries" translation ও
        # tts দুই সেকশনেই আছে, তাই alias অবশ্যই সেকশন অনুযায়ী আলাদা হতে হবে
        section_aliases: Dict[str, Dict[str, str]] = {
            "stt": {
                "model": "whisper_model", "device": "whisper_device",
                "compute_type": "whisper_compute_type", "batch_size": "whisper_batch_size",
                "beam_size": "whisper_beam_size",
            },
            "translation": {
                "backend": "translator_backend",
                # পুরনো/সরল config-এর সুবিধার্থে: max_retries এখানেও দেওয়া যায়
                "max_retries": "gemini_max_retries",
            },
            "gemini": {
                "api_key": "gemini_api_key",
                "model": "gemini_model",
                "batch_size": "gemini_batch_size",
                "max_chars_per_request": "gemini_max_chars_per_request",
                "max_retries": "gemini_max_retries",
                "retry_base_delay": "gemini_retry_base_delay",
                "retry_max_delay": "gemini_retry_max_delay",
                "requests_per_minute": "gemini_requests_per_minute",
                "temperature": "gemini_temperature",
                "timeout": "gemini_timeout",
                "max_output_tokens": "gemini_max_output_tokens",
            },
            "runtime": {"release_gpu_when_done": "release_gpu_when_done"},
            "tts": {
                "voice": "tts_voice", "rate": "tts_rate", "volume": "tts_volume",
                "pitch": "tts_pitch", "concurrency": "tts_concurrency",
                "max_retries": "tts_max_retries",
            },
            "output": {"bitrate": "audio_bitrate"},
        }
        sections = ("paths", "stt", "translation", "gemini", "tts", "sync", "runtime", "output")
        known = set(vars(self).keys())

        for section in sections:
            values = data.get(section)
            aliases = section_aliases.get(section, {})
            if isinstance(values, Mapping):
                for k, v in values.items():
                    target = aliases.get(k, k)
                    if target in known:
                        flat[target] = v
        for k, v in data.items():  # সরাসরি flat (canonical) key দিলেও কাজ করবে
            if k in known and not isinstance(v, Mapping):
                flat[k] = v

        for key, value in flat.items():
            if value is not None:
                setattr(self, key, value)

    def resolved_gemini_api_key(self) -> str:
        """Return the Gemini API key from config or environment (may be empty)."""
        key = (self.gemini_api_key or "").strip()
        if key:
            return key
        import os

        return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()

    def apply_overrides(self, overrides: Mapping[str, Any]) -> "AppConfig":
        """Apply CLI overrides (``None`` মান উপেক্ষা করা হয়)।"""
        for key, value in overrides.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
        return self

    # ---------------------------------------------------------------------------
    def validate(self) -> "AppConfig":
        """Validate settings; raises ``ValueError`` with a clear message."""
        if "/" not in self.whisper_model and self.whisper_model not in SUPPORTED_WHISPER_MODELS:
            raise ValueError(
                f"অজানা Whisper মডেল: '{self.whisper_model}'। "
                f"সমর্থিত: {', '.join(SUPPORTED_WHISPER_MODELS)}"
            )
        if self.translator_backend not in SUPPORTED_TRANSLATORS:
            raise ValueError(
                f"translator_backend হতে হবে {SUPPORTED_TRANSLATORS}, পাওয়া গেছে: "
                f"'{self.translator_backend}' (Google Translate আর সমর্থিত নয় — "
                f"Gemini API ব্যবহার করুন)"
            )
        if not (self.gemini_model or "").strip():
            raise ValueError("gemini_model খালি হতে পারে না (ডিফল্ট: gemini-3.5-flash-light)।")
        if not 1 <= self.gemini_max_retries <= 10:
            raise ValueError(
                f"gemini_max_retries 1–10 এর মধ্যে হতে হবে, পাওয়া গেছে: {self.gemini_max_retries}"
            )
        if self.gemini_batch_size < 1:
            raise ValueError("gemini_batch_size অন্তত 1 হতে হবে।")
        if self.gemini_max_chars_per_request < 500:
            raise ValueError("gemini_max_chars_per_request অন্তত 500 হতে হবে।")
        if self.gemini_requests_per_minute < 1:
            raise ValueError("gemini_requests_per_minute অন্তত 1 হতে হবে।")
        if not 0.0 <= self.gemini_temperature <= 2.0:
            raise ValueError("gemini_temperature 0.0–2.0 এর মধ্যে হতে হবে।")
        if self.gemini_timeout < 10:
            raise ValueError("gemini_timeout অন্তত 10 সেকেন্ড হতে হবে।")
        if self.gemini_retry_base_delay <= 0 or self.gemini_retry_max_delay < self.gemini_retry_base_delay:
            raise ValueError(
                "retry delay অবৈধ: gemini_retry_base_delay > 0 এবং "
                "gemini_retry_max_delay >= gemini_retry_base_delay হতে হবে।"
            )
        if self.translator_backend == "gemini" and not self.resolved_gemini_api_key():
            global _MISSING_KEY_WARNED
            if not _MISSING_KEY_WARNED:
                _MISSING_KEY_WARNED = True
                logger.warning(
                    "Gemini API key এখনো সেট হয়নি — নোটবুকের 'Gemini কনফিগারেশন' সেলে "
                    "GEMINI_API_KEY বসান, নইলে প্রোগ্রাম শুরু হতে পারবে না।"
                )
        if self.whisper_batch_size < 1:
            raise ValueError("whisper_batch_size অন্তত 1 হতে হবে।")
        if not (0.1 <= self.min_tempo < 1.0 < self.max_tempo <= 4.0):
            raise ValueError(
                f"tempo সীমা অবৈধ: min_tempo={self.min_tempo}, max_tempo={self.max_tempo} "
                f"(শর্ত: 0.1 <= min_tempo < 1 < max_tempo <= 4)"
            )
        if Path(self.input_dir) == Path(self.output_dir):
            raise ValueError("input_dir ও output_dir একই হতে পারবে না।")
        if not self.tts_voice.startswith("bn-"):
            logger.warning(
                "TTS ভয়েস '%s' Bangla (bn-*) নয় — ডাবিংয়ের জন্য bn-BD ভয়েস সুপারিশ করা হয়।",
                self.tts_voice,
            )
        return self
