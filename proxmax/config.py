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
from .translator_config import (
    DEFAULT_GEMINI_BATCH_SIZE, DEFAULT_MAX_RETRIES, DEFAULT_RETRY_DELAY,
)
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
#: Parallel Edge TTS requests; can be reduced if the service throttles.
DEFAULT_TTS_CONCURRENCY = 12


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
    #: ছোট Gemini batch paragraph-to-output alignment নিরাপদ রাখে
    gemini_batch_size: int = DEFAULT_GEMINI_BATCH_SIZE
    gemini_max_chars_per_request: int = 0    # 0 = legacy character cap বন্ধ; input-token cap ব্যবহার
    gemini_max_input_tokens: int = 5000     # প্রতি Gemini request-এর conservative input-token cap
    gemini_max_retries: int = DEFAULT_MAX_RETRIES  # প্রথম অনুরোধের পর সর্বোচ্চ retry
    gemini_retry_base_delay: float = DEFAULT_RETRY_DELAY
    gemini_retry_max_delay: float = DEFAULT_RETRY_DELAY
    gemini_retry_cap_delay: float = DEFAULT_RETRY_DELAY
    gemini_requests_per_minute: int = 15
    gemini_temperature: float = 0.2
    gemini_timeout: int = 120
    gemini_max_output_tokens: int = 8192
    #: Legacy local speech-rate setting; no duration hint is sent to Gemini
    translation_char_rate: float = 12.0
    #: Legacy compatibility flags; pre-TTS prediction/expansion is intentionally disabled
    translation_calibrate_speech_rate: bool = False
    translation_shorten_pass: bool = False
    translation_expand_pass: bool = False
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
    tts_concurrency: int = DEFAULT_TTS_CONCURRENCY
    tts_max_retries: int = 3
    #: পূর্ণ TTS edge trim বন্ধ; overrun হলে sync-এ শেষ থেকে প্রয়োজনীয় silence বাদ যায়
    tts_trim_silence: bool = False
    tts_trim_pad_ms: int = 25
    #: এক clip-এ কয় সেকেন্ড পর্যন্ত কথা (বাক্য জোড়া লাগিয়ে)
    tts_unit_max_seconds: float = 5.0
    #: এক clip-এ সর্বোচ্চ অক্ষর
    tts_unit_max_chars: int = 240
    #: এই সেকেন্ডের কম বিরতি থাকলে দুই segment এক clip-এ পড়া হয়
    tts_unit_max_gap: float = 0.6

    # --- audio-video sync ----------------------------------------------------
    min_tempo: float = 0.90                # সর্বোচ্চ ১০% ধীর
    fill_extension: float = 0.12
    max_tempo: float = 1.20                 # সর্বোচ্চ ২০% দ্রুত
    slow_down_short_clips: bool = True
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
    save_report: bool = True
    keep_temp_files: bool = False
    log_level: str = "INFO"

    # --- post-mux speech quality checks -------------------------------------
    #: ঐচ্ছিক full-video Whisper pass; ডিফল্টে বন্ধ, কারণ এটি ধীর
    validate_dub_audio: bool = False
    audio_validation_min_coverage: float = 0.58

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
                "max_input_tokens": "gemini_max_input_tokens",
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
                "skip_existing": "skip_existing", "save_report": "save_report",
                "keep_temp_files": "keep_temp_files", "log_level": "log_level",
            },
            "quality": {
                "validate_dub_audio": "validate_dub_audio",
                "min_text_coverage": "audio_validation_min_coverage",
            },
        }
        sections = (
            "paths", "stt", "translation", "gemini",
            "tts", "sync", "runtime", "output", "quality",
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
        from .config_validation import validate_config

        return validate_config(self)
