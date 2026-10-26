"""Application configuration: defaults, YAML loading, validation.

সব সেটিংস (Drive-এর পাথ, Whisper মডেল, hotwords ফাইল, chunking, Gemini,
বিল্ট-ইন বাংলা কণ্ঠ, sync ইত্যাদি) একটাই ``AppConfig`` dataclass-এ থাকে। ``config.yaml`` থেকে
লোড হয়, চাইলে CLI আর্গুমেন্ট দিয়েও override করা যায়।

নীতিগুলো:

* **গুণমান আগে, কিন্তু একটাই পথ** — ব্যবহারকারীকে accuracy mode বাছতে
  হয় না; ডিফল্ট মানগুলোই সর্বোচ্চ মানের সেটিং।
* **নীরবতায় কাটা** — STT chunk সীমা এমনভাবে রাখা হয়েছে যাতে Whisper
  কখনো শব্দের মাঝখানে না কাটে।
* **কণ্ঠ যেন রোবট না লাগে** — tempo সীমা ১.২৫, ক্লিপের দুই প্রান্তের
  নীরবতা ছাঁটা, স্লট ভরাতে সামান্য ধীর করা।
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from .hotwords import load_hotword_file, split_hotwords
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
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

#: রিপোর রুট (``hotwords.txt`` ইত্যাদি ডিফল্ট ফাইল খুঁজতে লাগে)
REPO_ROOT = Path(__file__).resolve().parents[1]
#: ডিফল্ট hotwords ফাইল — ফুটবল/FC Mobile-এর নাম-শব্দের তালিকা
DEFAULT_HOTWORDS_FILE = "hotwords.txt"

#: বিল্ট-ইন Bangla neural ভয়েস (কোনো নমুনা/টোকেন ছাড়াই কাজ করে)
#: ডিফল্ট **পুরুষ** কণ্ঠ (Pradeep) — ব্যবহারকারীর নির্দেশ; নারী কণ্ঠ চাইলে
#: ``tts.voice: bn-BD-NabanitaNeural`` দিতে হবে।
BANGLA_VOICES = {
    "male": "bn-BD-PradeepNeural",
    "female": "bn-BD-NabanitaNeural",
}
#: ডিফল্ট কণ্ঠের লিঙ্গ (ব্যবহারকারীর নির্দেশ: পুরুষ)
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
    #: TTS শুরুর আগে Whisper আনলোড করা (VRAM বাঁচে, কণ্ঠ তৈরি দ্রুত হয়)
    unload_stt_before_tts: bool = True
    #: হাতে লেখা নাম/শব্দ (কমা দিয়ে) — খালি থাকলে ``hotwords.txt`` ফাইল ব্যবহৃত হয়
    #: যেমন ``"FC Mobile, Ballon d'Or, OVR, star pass, Yashin"``। খালি = বন্ধ।
    stt_hotwords: str = ""
    #: নাম-শব্দের আলাদা ফাইল (রিপো রুট থেকে আপেক্ষিক বা পূর্ণ পাথ)।
    #: ফাইলে এক লাইনে একটি নাম, ``#`` দিয়ে মন্তব্য। ব্যবহারকারীকে কিছু টাইপ করতে হয় না।
    stt_hotwords_file: str = DEFAULT_HOTWORDS_FILE
    #: Whisper-এর prompt সীমা (২২৪ টোকেন) মেনে hotwords কত অক্ষর পর্যন্ত যাবে
    stt_hotwords_max_chars: int = 480

    # --- audio preprocessing (0 GPU) -----------------------------------------
    audio_preprocess: bool = True
    audio_highpass_hz: int = 70
    #: EBU R128 loudness target (নিচু/উঁচু ভিডিওর জন্য একই রকম লেভেল)
    audio_loudnorm_i: float = -16.0

    # --- translation: Google Gemini API (ডিফল্ট) -------------------------------
    translator_backend: str = "gemini"      # "gemini" | "indictrans2"
    gemini_api_key: str = ""                # খালি হলে GEMINI_API_KEY / GOOGLE_API_KEY env
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_batch_size: int = 25             # এক request-এ কত লাইন
    gemini_max_chars_per_request: int = 12000
    gemini_max_retries: int = 7             # সর্বোচ্চ ৭ বার চেষ্টা
    #: retry-এর আগে অপেক্ষা — ব্যবহারকারীর নির্দেশ: সর্বোচ্চ ২ সেকেন্ড,
    #: দ্বিগুণ করে বাড়ানো যাবে না (ফরম্যাট ভুল হলে অপেক্ষাই নেই)
    gemini_retry_base_delay: float = 0.5
    gemini_retry_max_delay: float = 2.0
    gemini_retry_cap_delay: float = 2.0
    gemini_requests_per_minute: int = 15
    gemini_temperature: float = 0.2
    gemini_timeout: int = 120
    gemini_max_output_tokens: int = 8192
    #: বাংলা কথার গতি (অক্ষর/সেকেন্ড) — ক্যালিব্রেশন না হলে এই মানই ধরা হয়।
    #: বাস্তব বাংলা neural কণ্ঠ এর চেয়ে ধীরে বলে (≈১২), তাই ১৪ দিলে অনুবাদ
    #: লম্বা হয়ে sync-এ ১.২৫× দ্রুত করতে হতো → কণ্ঠ তাড়াহুড়ো শোনাত।
    translation_char_rate: float = 12.0
    #: শুরুতেই আসল কণ্ঠ দিয়ে ২টি বাক্য বলে প্রকৃত গতি মাপা (≈১–২s খরচ)
    translation_calibrate_speech_rate: bool = True
    #: সময়ে না আঁটা লাইন Gemini দিয়ে ছোট করে লেখানো
    translation_shorten_pass: bool = True
    #: স্লটের চেয়ে অনেক ছোট হয়ে যাওয়া লাইন (বিরাট ফাঁক) আরও ভরাট করে লেখানো
    translation_expand_pass: bool = True
    #: কত সেকেন্ডের বেশি ফাঁক থাকলে লাইনটি ভরাট করে লেখানো হবে
    translation_expand_min_gap: float = 0.9
    indictrans2_model: str = "ai4bharat/indictrans2-en-indic-1B"

    # --- text-to-speech (বিল্ট-ইন বাংলা neural ভয়েস) -------------------------
    # কোনো ক্লোনিং নেই — কিছু দিতে হয় না, কিছু ইনস্টল করতে হয় না।
    #: কণ্ঠের লিঙ্গ — ``male`` (ডিফল্ট) | ``female`` | ``custom``।
    #: ``voice`` খালি থাকলে এখান থেকেই ভয়েস বাছা হয়; তাই পুরোনো/config-এ ভুল
    #: ভয়েস থাকলেও পুরুষ কণ্ঠই ব্যবহার হবে (ব্যবহারকারীর নির্দেশ)।
    tts_gender: str = DEFAULT_TTS_GENDER
    #: নির্দিষ্ট ভয়েস (খালি = ``gender`` অনুযায়ী স্বয়ংক্রিয়)
    tts_voice: str = ""
    tts_rate: str = "+0%"
    tts_volume: str = "+0%"
    tts_pitch: str = "+0Hz"
    tts_concurrency: int = 5
    tts_max_retries: int = 3
    #: TTS ক্লিপের শুরু/শেষের নীরবতা ছেঁটে ফেলা (edge-tts প্রায় ০.৫–০.৮s
    #: নীরবতা দেয়; না ছাঁটলে প্রতিটি বাক্যের পর অযথা বিরতি শোনায়)
    tts_trim_silence: bool = True
    tts_trim_pad_ms: int = 25
    #: এক clip-এ কয় সেকেন্ড পর্যন্ত কথা (বাক্য জোড়া লাগিয়ে) — বড় হলে
    #: কণ্ঠ আরও স্বাভাবিক, কিন্তু sync-এ নড়াচড়ার জায়গা কমে
    tts_unit_max_seconds: float = 12.0
    #: এক clip-এ সর্বোচ্চ অক্ষর (খুব লম্বা হলে নেটওয়ার্ক/মডেল সমস্যা করে)
    tts_unit_max_chars: int = 240
    #: এই সেকেন্ডের কম বিরতি থাকলে দুই segment এক clip-এ পড়া হয়
    tts_unit_max_gap: float = 0.6

    # --- audio-video sync ----------------------------------------------------
    min_tempo: float = 0.95
    #: ক্লিপ স্লটের চেয়ে ছোট হলে সর্বোচ্চ কতটা ধীর করে স্লট ভরা হবে
    #: (০.১২ = ১২% ধীর → কথার মাঝে অযথা ফাঁক কমে)
    fill_extension: float = 0.12                 # ডিফল্টে ধীর করা হয় না
    max_tempo: float = 1.25                 # এর বেশি দ্রুত করলে কণ্ঠ বিকৃত শোনায়
    slow_down_short_clips: bool = False     # ছোট ক্লিপ ধীর না করে নীরবতা রাখা হয়
    slot_gap_ms: int = 80                   # দুই ক্লিপের মাঝে ন্যূনতম ফাঁক
    borrow_silence_ms: int = 250            # পরের নীরবতা থেকে সর্বোচ্চ ধার
    prefer_rubberband: bool = True          # থাকলে rubberband (formant রক্ষা করে)
    sync_workers: int = 4
    sync_sample_rate: int = 24000           # বাংলা neural ভয়েসের native rate

    # --- runtime -------------------------------------------------------------
    release_gpu_when_done: bool = True      # কাজ শেষে মডেল আনলোড → VRAM খালি
    enable_overlap: bool = True             # অনুবাদ ও TTS একসাথে (সময় বাঁচে)
    #: ffmpeg এত সেকেন্ড অগ্রগতি না দেখালে ঝুলে গেছে ধরে থামিয়ে দেওয়া হয়
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
        """ব্যবহার হবে এমন বিল্ট-ইন ভয়েস — ``gender`` অনুযায়ী।

        ``tts.gender``-ই প্রধান: পুরোনো config-এ ভুল ভয়েস (যেমন নারী কণ্ঠ)
        থাকলেও পুরুষ ডিফল্টই ব্যবহার হবে, যদি না ব্যবহারকারী স্পষ্টভাবে
        ``gender: custom`` + ``voice: <name>`` দেন (``--voice`` দিলে main.py
        নিজেই ``custom`` করে দেয়)।
        """
        gender = (self.tts_gender or "").strip().lower()
        voice = (self.tts_voice or "").strip()
        if gender == "custom":
            return voice
        if gender in BANGLA_VOICES:
            return BANGLA_VOICES[gender]
        return voice or BANGLA_VOICES[DEFAULT_TTS_GENDER]

    # ------------------------------------------------------------------
    @property
    def hotwords_path(self) -> Path | None:
        """hotwords ফাইলের পূর্ণ পাথ (না পেলে ``None``)।"""
        raw = (self.stt_hotwords_file or "").strip()
        if not raw:
            return None
        path = Path(raw).expanduser()
        if path.is_absolute():
            return path if path.is_file() else None
        for base in (Path.cwd(), REPO_ROOT):
            candidate = base / path
            if candidate.is_file():
                return candidate
        return None

    def hotword_entries(self) -> list[str]:
        """hotwords-এর সব এন্ট্রি (কোনো কাটছাঁট ছাড়া)।

        অগ্রাধিকার: ``stt.hotwords`` (হাতে লেখা) → ``hotwords.txt`` ফাইল →
        কিছুই নয়। ফাইলে এক লাইনে একটি নাম/শব্দ, ``#`` দিয়ে মন্তব্য;
        কমা দিয়ে একাধিকও লেখা যায়।

        Returns:
            ডুপ্লিকেট-মুক্ত নামের তালিকা (ফাইল/লেখার ক্রমে)।
        """
        explicit = (self.stt_hotwords or "").strip()
        entries = split_hotwords(explicit) if explicit else []
        if not entries:
            path = self.hotwords_path
            if path is not None:
                try:
                    entries = load_hotword_file(path)
                except Exception as exc:  # ফাইল পড়া না গেলেও ডাবিং চলবে
                    logger.warning("hotwords ফাইল পড়া যায়নি (%s): %s", path, exc)
                    entries = []
        return entries

    def resolved_hotwords(self) -> str:
        """Whisper-এ দেওয়ার জন্য পুরো নাম-শব্দের তালিকা (এক স্ট্রিং)।

        Whisper-এর prompt সীমা ২২৪ টোকেন, তাই :class:`~proxmax.stt.SpeechToText`
        এই তালিকাকে ছোট ছোট ভাগে ভেঙে প্রতি chunk-এ আলাদা ভাগ দেয়
        (:func:`proxmax.hotwords.build_hotword_slices`)। এতে ফাইলের প্রতিটি
        নামই একবার প্রাইম হয় — কোনো নাম বাদ পড়ে না।

        Returns:
            কমা দিয়ে আলাদা করা শব্দের স্ট্রিং (খালি হতে পারে)।
        """
        return ", ".join(self.hotword_entries())

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
        """Apply a (possibly sectioned) mapping onto this config.

        ``config.yaml``-এর section গঠন:
        paths / stt / audio / translation / gemini / voice / tts / sync /
        runtime / output — পাশাপাশি সরাসরি flat key দিলেও কাজ করবে।
        """
        flat: Dict[str, Any] = {}
        # সেকশনভিত্তিক ছোট alias — একই নাম দুই সেকশনে থাকলে সংঘর্ষ এড়াতে
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
                "hotwords_max_chars": "stt_hotwords_max_chars",
            },
            "audio": {
                "preprocess": "audio_preprocess",
                "highpass_hz": "audio_highpass_hz",
                "loudnorm_i": "audio_loudnorm_i",
            },
            "translation": {
                "backend": "translator_backend",
                "max_retries": "gemini_max_retries",   # পুরনো config-এর সঙ্গে সামঞ্জস্য
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
                # পুরোনো config-এ থাকা ক্লোনিং কী-গুলো আর ব্যবহার হয় না
                "engine": "_ignored_tts_engine",
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
            "paths", "stt", "audio", "translation", "gemini",
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
        for k, v in data.items():  # সরাসরি flat (canonical) key দিলেও কাজ করবে
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
                f"'{self.translator_backend}' (Google Translate আর সমর্থিত নয় — "
                f"Gemini API ব্যবহার করুন)"
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

        if not 100 <= int(self.stt_hotwords_max_chars) <= 2000:
            raise ValueError(
                "stt.hotwords_max_chars ১০০–২০০০ অক্ষরের মধ্যে হতে হবে "
                f"(পাওয়া গেছে: {self.stt_hotwords_max_chars})"
            )
        if (self.stt_hotwords_file or "").strip() and self.hotwords_path is None:
            logger.warning(
                "hotwords ফাইল পাওয়া যায়নি: %s — শুধু stt.hotwords ব্যবহার হবে।",
                self.stt_hotwords_file,
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
                    "Gemini API key এখনো সেট হয়নি — নোটবুকের ধাপ ৬-এর সেলে "
                    "GEMINI_API_KEY বসান, নইলে প্রোগ্রাম শুরু হতে পারবে না।"
                )
        return self
