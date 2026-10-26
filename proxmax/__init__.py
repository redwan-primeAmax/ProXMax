"""ProXMax — English-to-Bangla video dubbing package.

English ভিডিও থেকে **বাংলা কণ্ঠে** ডাবিং তৈরি করার সম্পূর্ণ pipeline এই
প্যাকেজে রয়েছে: নীরবতা-সচেতন chunking → faster-whisper STT (hotwords সহ) →
duration-aware Gemini অনুবাদ → বিল্ট-ইন বাংলা neural ভয়েস → non-overlap
sync → mux → SRT।
"""

from __future__ import annotations

__version__ = "3.0.0"
__all__ = ["__version__"]
