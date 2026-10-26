"""Application configuration: defaults, YAML loading, validation.

সব সেটিংস (Drive-এর পাথ, Whisper মডেল, chunking, hotwords, Gemini, TTS,
sync ইত্যাদি) একটাই ``AppConfig`` dataclass-এ থাকে। ``config.yaml`` থেকে
লোড হয়, চাইলে CLI আর্গুমেন্ট দিয়েও override করা যায়।
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .hotwords import load_hotwords
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
SUPPORTED_TTS_ENGINES = ("edge", "auto")

#: ডিফল্ট অনুবাদ মডেল — নোটবুক/CLI দিয়ে বদলানো যায়
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

#: বিল্ট-ইন Bangla neural ভয়েস (কোনো নমুনা/টোকেন ছাড়াই কাজ করে)
BANGLA_VOICES = {
    "male": "bn-BD-PradeepNeural",
    "female": "bn-BD-NabanitaNeural",
}
#: ডিফল্ট কণ্ঠের লিঙ্গ (পুরুষ)
DEFAULT_TTS_GENDER = "male"
#: ডিফল্ট কণ্ঠ (পুরুষ)
DEFAULT_TTS_VOICE = BANGLA_VOICES[DEFAULT_TTS_GENDER]


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
    whisper_beam_size: int = 5
    #: নীরবতা শনাক্তকরণের সংবেদনশীলতা (এই নীরবতাগুলোতেই কাটা হয়)
    vad_min_silence_ms: int = 300
    #: চাঙ্ক ১৮–২৮s — Whisper-এর ৩০s সীমার ভেতরে, কিন্তু শব্দের মাঝে নয়
    chunk_min_seconds: float = 18.0
    chunk_max_seconds: float = 28.0
    chunk_pad_ms: int = 120
    #: আগের টেক্সটের শেষ কত অক্ষর prompt হিসেবে পরের চাঙ্কে যাবে (পাংচুয়েশন/প্রসঙ্গ)
    carry_context_chars: int = 220
    condition_on_previous_text: bool = True
    #: সীমানায় শব্দ কাটা পড়লে ছোট উইন্ডো আবার ডিকোড করে ঠিক করা
    seam_repair: bool = True
    seam_window_seconds: float = 2.5
    max_seam_repairs: int = 60
    no_speech_threshold: float = 0.6
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    #: TTS শুরু করার আগে Whisper আনলোড করা (VRAM বাঁচে)
    unload_stt_before_tts: bool = False
    #: অতিরিক্ত ভিডিও-নির্দিষ্ট নাম/শব্দ (কমা দিয়ে)
    stt_hotwords: str = ""
    #: আলাদা hotwords ফাইল (ডিফল্ট: রিপোর hotwords.txt)
    stt_hotwords_file: str = "hotwords.txt"

    # --- translation: Google Gemini API (ডিফল্ট) -------------------------------
    translator_backend: str = "gemini"      # "gemini" | "indictrans2"
    gemini_api_key: str = ""                # খালি হলে GEMINI_API_KEY / GOOGLE_API_KEY env
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_batch_size: int = 25             # এক request-এ কত লাইন
    gemini_max_chars_per_request: int = 12000
    gemini_max_retries: int = 7             # সর্বোচ্চ ৭ বার চেষ্টা
    gemini_retry_base_delay: float = 0.5
    gemini_retry_max_delay: float = 2.0
    gemini_retry_cap_delay: float = 2.0
    gemini_requests_per_minute: int = 15
    gemini_temperature: float = 0.2
    gemini_timeout: int = 120
    gemini_max_output_tokens: int = 8192
    #: বাংলা কথার গতি (অক্ষর/সেকেন্ড) — ক্যালিব্রেশন না হলে এই মানই ধরা হয়
    translation_char_rate: float = 12.0
    #: শুরুতেই আসল কণ্ঠ দিয়ে ২টি বাক্য বলে প্রকৃত গতি মাপা (≈১–২s খরচ)
    translation_calibrate_speech_rate: bool = True
    #: সময়ে না আঁটা লাইন Gemini দিয়ে ছোট করে লেখানো
    translation_shorten_pass: bool = True
    #: স্লটের চেয়ে অনেক ছোট হয়ে যাওয়া লাইন আরও ভরাট করে লেখানো
    translation_expand_pass: bool = True
    #: কত সেকেন্ডের বেশি ফাঁক থাকলে লাইনটি ভরাট করে লেখানো হবে
    translation_expand_min_gap: float = 0.9
    indictrans2_model: str = "ai4bharat/indictrans2-en-indic-1B"

    # --- text-to-speech (বিল্ট-ইন edge-tts ভয়েস) -----------------------------
    tts_engine: str = "edge"
    #: কণ্ঠের লিঙ্গ — ``male`` (ডিফল্ট) | ``female`` | ``custom``
    tts_gender: str = DEFAULT_TTS_GENDER
    #: নির্দিষ্ট ভয়েস (খালি = ``gender`` অনুযায়ী স্বয়ংক্রিয়)
    tts_voice: str = ""
    tts_rate: str = "+0%"
    tts_volume: str = "+0%"
    tts_pitch: str = "+0Hz"
    tts_concurrency: int = 5
    tts_max_retries: int = 3
    #: TTS ক্লিপের শুরু/শেষের নীরবতা ছেঁটে ফেলা
    tts_trim_silence: bool = True
    tts_trim_pad_ms: int = 25
    #: এক clip-এ কয় সেকেন্ড পর্যন্ত কথা (বাক্য জোড়া লাগিয়ে)
    tts_unit_max_seconds: float = 12.0
    #: এক clip-এ সর্বোচ্চ অক্ষর
    tts_unit_max_chars: int = 240
    #: এই সেকেন্ডের কম বিরতি থাকলে দুই segment এক clip-এ পড়া হয়
    tts_unit_max_gap: float = 0.6

    # --- audio-video sync ----------------------------------------------------
    min_tempo: float = 0.95
    fill_extension: float = 0.12
    max_tempo: float = 1.25                 # এর বেশি দ্রুত করলে কণ্ঠ বিকৃত শোনায়
    slow_down_short_clips: bool = False
    slot_gap_ms: int = 80
    borrow_silence_ms: int = 250
    prefer_rubberband: bool = True
    sync_workers: int = 4
    sync_sample_rate: int = 24000

    # --- runtime -------------------------------------------------------------
    release_gpu_when_done: bool = True      # কাজ শেষে মডেল আনলোড → VRAM খালি
    enable_overlap: bool = True             # অনুবাদ ও TTS একসাথে (সময় বাঁচে)
    extract_stall_seconds: float = 90.0

    # --- output --------------------------------------------------------------
    audio_bitrate: str = "160k"
    use_hwaccel: bool = True
    skip_existing: bool = True
    save_srt: bool = True
    keep_temp_files: bool = False
    log_level: str = "INFO"

    # ---------------------------------------------------------------------------
    @property
    def resolved_work_dir(self) -> Path:
        """Resolve ``work_dir`` ("auto" হলে Colab-এর লোকাল ডিস্ক ব্যবহার হয়)।"""
        if self.work_dir and self.work_dir != "auto":
            return Path(self.work_dir).expanduser()
        if Path("/content").is_dir():  # Google Colab
            return Path("/content/proxmax_work")
        return Path.cwd() / "proxmax_work"

    @property
    def resolved_tts_voice(self) -> str:
        """ব্যবহার হবে এমন বিল্ট-ইন ভয়েস — ``gender`` অনুযায়ী।"""
        gender = (self.tts_gender or "").strip().lower()
        voice = (self.tts_voice or "").strip()
        if gender == "custom":
            return voice
        if gender in BANGLA_VOICES:
            return BANGLA_VOICES[gender]
        return voice or BANGLA_VOICES[DEFAULT_TTS_GENDER]

    def resolved_hotwords_list(self) -> List[str]:
        """``hotwords.txt`` এবং ``stt_hotwords`` মিলিয়ে সম্পূর্ণ hotword তালিকা।"""
        return load_hotwords(self.stt_hotwords_file, extra_hotwords=self.stt_hotwords)

    def resolved_gemini_api_key(self) -> str:
        """Return the Gemini API key from config or environment (may be empty)."""
        key = (self.gemini_api_key or "").strip()
        if key:
            return key
        return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()

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
        """Apply a (possibly sectioned) mapping onto this config."""
        flat: Dict[str, Any] = {}
        section_aliases: Dict[str, Dict[str, str]] = {
            "paths": {"input": "input_dir", "output": "output_dir", "work": "work_dir"},
            "stt": {
                "model": "whisper_model", "device": "whisper_device",
                "compute_type": "whisper_compute_type", "beam_size": "whisper_beam_size",
                "vad_min_silence_ms": "vad_min_silence_ms",
                "chunk_min_seconds": "chunk_min_seconds",
                "chunk_max_seconds": "chunk_max_seconds",
                "chunk_pad_ms": "chunk_pad_ms",
                "carry_context_chars": "carry_context_chars",
                "seam_repair": "seam_repair",
                "seam_window_seconds": "seam_window_seconds",
                "max_seam_repairs": "max_seam_repairs",
                "unload_before_tts": "unload_stt_before_tts",
                "hotwords": "stt_hotwords",
                "hotwords_file": "stt_hotwords_file",
            },
            "translation": {
                "backend": "translator_backend",
                "max_retries": "gemini_max_retries",
                "char_rate": "translation_char_rate",
                "calibrate_speech_rate": "translation_calibrate_speech_rate",
                "shorten_pass": "translation_shorten_pass",
                "expand_pass": "translation_expand_pass",
                "expand_min_gap": "translation_expand_min_gap",
                "indictrans2_model": "indictrans2_model",
            },
            "gemini": {
                "api_key": "gemini_api_key",
                "model": "gemini_model",
                "batch_size": "gemini_batch_size",
                "max_chars_per_request": "gemini_max_chars_per_request",
                "max_retries": "gemini_max_retries",
                "retry_base_delay": "gemini_retry_base_delay",
                "retry_max_delay": "gemini_retry_max_delay",
                "retry_cap_delay": "gemini_retry_cap_delay",
                "requests_per_minute": "gemini_requests_per_minute",
                "temperature": "gemini_temperature",
                "timeout": "gemini_timeout",
                "max_output_tokens": "gemini_max_output_tokens",
            },
            "tts": {
                "engine": "tts_engine",
                "gender": "tts_gender",
                "voice": "tts_voice", "rate": "tts_rate", "volume": "tts_volume",
                "trim_silence": "tts_trim_silence", "trim_pad_ms": "tts_trim_pad_ms",
                "pitch": "tts_pitch", "concurrency": "tts_concurrency",
                "max_retries": "tts_max_retries",
                "unit_max_seconds": "tts_unit_max_seconds",
                "unit_max_chars": "tts_unit_max_chars",
                "unit_max_gap": "tts_unit_max_gap",
            },
            "sync": {
                "min_tempo": "min_tempo", "max_tempo": "max_tempo",
                "slow_down_short_clips": "slow_down_short_clips",
                "slot_gap_ms": "slot_gap_ms",
                "borrow_silence_ms": "borrow_silence_ms",
                "fill_extension": "fill_extension",
                "prefer_rubberband": "prefer_rubberband",
                "workers": "sync_workers",
                "sample_rate": "sync_sample_rate",
            },
            "runtime": {
                "release_gpu_when_done": "release_gpu_when_done",
                "enable_overlap": "enable_overlap",
                "extract_stall_seconds": "extract_stall_seconds",
            },
            "output": {
                "bitrate": "audio_bitrate", "hwaccel": "use_hwaccel",
                "skip_existing": "skip_existing", "save_srt": "save_srt",
                "keep_temp_files": "keep_temp_files", "log_level": "log_level",
            },
        }
        sections = (
            "paths", "stt", "translation", "gemini",
            "tts", "sync", "runtime", "output",
        )
        known = set(vars(self).keys())

        for section in sections:
            values = data.get(section)
            aliases = section_aliases.get(section, {})
            if isinstance(values, Mapping):
                for k, v in values.items():
                    target = aliases.get(k, k)
                    if target in known:
                        flat[target] = v
                    else:
                        logger.debug("অজানা config key উপেক্ষা করা হলো: %s.%s", section, k)
        for k, v in data.items():
            if k in known and not isinstance(v, Mapping):
                flat[k] = v

        for key, value in flat.items():
            if value is not None:
                setattr(self, key, value)

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
        if not 5.0 <= float(self.chunk_max_seconds) <= 29.5:
            raise ValueError(
                f"chunk_max_seconds ৫–২৯.৫ এর মধ্যে হতে হবে (Whisper ৩০s-এর বেশি দেখতে "
                f"পারে না), পাওয়া গেছে: {self.chunk_max_seconds}"
            )
        if not 1.0 <= float(self.chunk_min_seconds) <= float(self.chunk_max_seconds):
            raise ValueError(
                "chunk_min_seconds অবৈধ: ১ থেকে chunk_max_seconds-এর মধ্যে হতে হবে।"
            )
        if not 0 <= int(self.carry_context_chars) <= 4000:
            raise ValueError("carry_context_chars ০–৪০০০ এর মধ্যে হতে হবে।")
        if self.seam_window_seconds <= 0:
            raise ValueError("seam_window_seconds ধনাত্মক হতে হবে।")
        if self.whisper_beam_size < 1:
            raise ValueError("whisper_beam_size অন্তত 1 হতে হবে।")

        if self.translator_backend not in SUPPORTED_TRANSLATORS:
            raise ValueError(
                f"translator_backend হতে হবে {SUPPORTED_TRANSLATORS}, পাওয়া গেছে: "
                f"'{self.translator_backend}'"
            )
        if not (self.gemini_model or "").strip():
            raise ValueError("gemini_model খালি হতে পারে না (ডিফল্ট: gemini-3.5-flash-lite)।")
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
        if not 0.0 <= float(self.gemini_retry_cap_delay) <= 5.0:
            raise ValueError(
                "gemini_retry_cap_delay ০–৫ সেকেন্ডের মধ্যে হতে হবে "
                f"(পাওয়া গেছে: {self.gemini_retry_cap_delay})"
            )
        if self.gemini_retry_base_delay <= 0 or self.gemini_retry_max_delay < self.gemini_retry_base_delay:
            raise ValueError(
                "retry delay অবৈধ: gemini_retry_base_delay > 0 এবং "
                "gemini_retry_max_delay >= gemini_retry_base_delay হতে হবে।"
            )
        if not 1.0 <= float(self.tts_unit_max_seconds) <= 25.0:
            raise ValueError(
                "tts_unit_max_seconds ১–২৫ এর মধ্যে হতে হবে "
                f"(পাওয়া গেছে: {self.tts_unit_max_seconds})"
            )
        if not 40 <= int(self.tts_unit_max_chars) <= 800:
            raise ValueError(
                "tts_unit_max_chars ৪০–৮০০ এর মধ্যে হতে হবে "
                f"(পাওয়া গেছে: {self.tts_unit_max_chars})"
            )
        if not 0.0 <= float(self.fill_extension) <= 0.5:
            raise ValueError(
                "sync.fill_extension ০–০.৫ এর মধ্যে হতে হবে "
                f"(পাওয়া গেছে: {self.fill_extension})"
            )
        if not 0.0 <= float(self.tts_unit_max_gap) <= 2.0:
            raise ValueError(
                "tts_unit_max_gap ০–২ সেকেন্ডের মধ্যে হতে হবে "
                f"(পাওয়া গেছে: {self.tts_unit_max_gap})"
            )
        if not 6.0 <= float(self.translation_char_rate) <= 30.0:
            raise ValueError(
                f"translation_char_rate ৬–৩০ অক্ষর/সেকেন্ডের মধ্যে হতে হবে, "
                f"পাওয়া গেছে: {self.translation_char_rate}"
            )

        if self.tts_engine not in SUPPORTED_TTS_ENGINES:
            raise ValueError(
                f"tts_engine হতে হবে {SUPPORTED_TTS_ENGINES}, পাওয়া গেছে: '{self.tts_engine}'"
            )

        if not (0.5 <= self.min_tempo <= 1.0 < self.max_tempo <= 2.0):
            raise ValueError(
                f"tempo সীমা অবৈধ: min_tempo={self.min_tempo}, max_tempo={self.max_tempo} "
                f"(শর্ত: 0.5 <= min_tempo <= 1 < max_tempo <= 2)"
            )
        if self.slot_gap_ms < 0 or self.borrow_silence_ms < 0:
            raise ValueError("slot_gap_ms ও borrow_silence_ms ঋণাত্মক হতে পারবে না।")
        if self.sync_sample_rate < 8000:
            raise ValueError("sync_sample_rate অন্তত ৮০০০ হতে হবে।")

        if Path(self.input_dir) == Path(self.output_dir):
            raise ValueError("input_dir ও output_dir একই হতে পারবে না।")
        if self.tts_gender not in ("male", "female", "custom"):
            raise ValueError(
                "tts.gender হতে হবে male | female | custom "
                f"(পাওয়া গেছে: {self.tts_gender})"
            )
        if self.tts_gender != "custom" and not self.resolved_tts_voice.startswith("bn-"):
            raise ValueError(
                f"tts.gender={self.tts_gender} অনুযায়ী ভয়েস পাওয়া গেল না "
                f"(tts_voice={self.tts_voice!r})।"
            )
        if not self.resolved_tts_voice.startswith("bn-"):
            logger.warning(
                "TTS ভয়েস '%s' Bangla (bn-*) নয় — ডাবিংয়ের জন্য bn-BD ভয়েস সুপারিশ করা হয়।",
                self.resolved_tts_voice,
            )
        if self.translator_backend == "gemini" and not self.resolved_gemini_api_key():
            global _MISSING_KEY_WARNED
            if not _MISSING_KEY_WARNED:
                _MISSING_KEY_WARNED = True
                logger.warning(
                    "Gemini API key এখনো সেট হয়নি — নোটবুকের 'Gemini API key' সেলে "
                    "GEMINI_API_KEY বসান, নইলে প্রোগ্রাম শুরু হতে পারবে না।"
                )
        return self
