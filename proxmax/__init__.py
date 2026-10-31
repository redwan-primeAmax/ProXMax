"""ProXMax — English-to-Bangla video dubbing package.

English ভিডিও থেকে স্বয়ংক্রিয়ভাবে Bangla ডাবিং তৈরি করার সম্পূর্ণ
pipeline এই প্যাকেজে রয়েছে: নীরবতা-সচেতন chunking → faster-whisper STT
(hotwords.txt সহ) → token-budgeted, ≤৫s Gemini chunk অনুবাদ → বিল্ট-ইন বাংলা neural
ভয়েস (edge-tts) → non-overlap sync → mux।
"""

from __future__ import annotations

from pathlib import Path
from ._module_groups import extend_package_path

# Physical domain folders share the legacy package namespace; see
# _module_groups for backwards-compatible grouped aliases and module identity.
extend_package_path(__path__, Path(__file__).resolve().parent)

from .config import AppConfig
from .hotwords import load_hotwords, resolve_hotwords_path
from .models import BatchSummary, Segment, StageTimings, VideoResult, Word
from .pipeline import DubbingPipeline, run_Single_or_batch
from .stt import ensure_whisper_model
from .translator import verify_gemini_api

__version__ = "3.1.0"
__all__ = [
    "AppConfig",
    "DubbingPipeline",
    "run_Single_or_batch",
    "Segment",
    "Word",
    "VideoResult",
    "BatchSummary",
    "StageTimings",
    "ensure_whisper_model",
    "verify_gemini_api",
    "load_hotwords",
    "resolve_hotwords_path",
    "__version__",
]
