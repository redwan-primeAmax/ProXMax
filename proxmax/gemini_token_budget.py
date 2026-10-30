"""Fast local prompt-token budgeting for Gemini translation batches.

Gemini's exact tokenizer is service-side. This conservative CPU-only estimate
keeps whole paragraphs below a user-selected per-request budget without adding
an extra token-count API round trip.
"""

from __future__ import annotations

import math
import re

_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")


def estimate_gemini_input_tokens(text: str) -> int:
    """Return a conservative, fast estimate for a complete Gemini prompt.

    ASCII text is estimated at roughly four characters per token, with a
    minimum token per word and punctuation mark. Each non-ASCII code point is
    counted as a token. A 15% safety margin plus fixed request overhead covers
    model-specific tokenization and structured-output metadata.
    """
    value = str(text or "")
    if not value:
        return 64

    ascii_chars = 0
    non_ascii_chars = 0
    ascii_punctuation = 0
    for char in value:
        if ord(char) < 128:
            ascii_chars += 1
            if not char.isspace() and not char.isalnum() and char != "_":
                ascii_punctuation += 1
        else:
            non_ascii_chars += 1

    words = len(_ASCII_WORD_RE.findall(value))
    ascii_tokens = max(math.ceil(ascii_chars / 4.0), words + ascii_punctuation)
    estimated = ascii_tokens + non_ascii_chars
    return math.ceil(estimated * 1.15) + 64
