"""ProXMax — English-to-Bangla video dubbing package.

English ভিডিও থেকে **আপনার নিজের কণ্ঠে** Bangla ডাবিং তৈরি করার সম্পূর্ণ
pipeline এই প্যাকেজে রয়েছে: নীরবতা-সচেতন chunking → faster-whisper STT →
duration-aware Gemini অনুবাদ → IndicF5 ভয়েস ক্লোনিং → non-overlap sync → mux।
"""

from __future__ import annotations

__version__ = "3.0.0"
__all__ = ["__version__"]
