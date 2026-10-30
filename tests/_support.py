"""Shared smoke-test imports and lightweight fake collaborators."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proxmax import audio_fx                                         # noqa: E402
from proxmax.audio_quality import assess_bengali_coverage             # noqa: E402
from proxmax.chunking import (                                       # noqa: E402
    SpeechChunk,
    energy_regions,
    plan_chunks,
    regions_to_chunks,
)
from proxmax.config import (                                          # noqa: E402
    DEFAULT_GEMINI_MODEL,
    AppConfig,
)
from proxmax.discovery import discover_videos                          # noqa: E402
from proxmax.models import Segment, StageTimings, Word                  # noqa: E402
from proxmax.reporting import write_detailed_report                    # noqa: E402
from proxmax.segmenter import build_sentences                          # noqa: E402
from proxmax.stt import (                                              # noqa: E402
    SpeechToText,
    _tail_context,
    find_suspicious_boundaries,
    merge_windows,
    replace_in_window,
)
from proxmax.sync_engine import (                                      # noqa: E402
    SyncEngine,
    build_atempo_filter,
    compute_tempo,
    find_overlaps,
    plan_windows,
)
from proxmax.sync_silence import trim_edge_silence_for_overrun        # noqa: E402
from proxmax.text_utils import (                                       # noqa: E402
    estimate_speech_seconds,
    normalise_bangla,
    prepare_for_speech,
    split_sentences,
    verbalize_numbers,
)
from proxmax.translator import (                                       # noqa: E402
    GeminiAuthError,
    GeminiTranslator,
    TranslationError,
    create_translator,
    parse_numbered_lines,
    parse_translations,
    salvage_translations,
    verify_gemini_api,
)
from proxmax.tts import (                                              # noqa: E402
    TTSError,
    EdgeTTSEngine,
    resolve_tts_engine,
)
from proxmax.utils import (                                            # noqa: E402
    format_srt_timestamp,
    output_filename_for,
    release_cuda_memory,
)

FFMPEG = shutil.which("ffmpeg")

def _speech_with_silence(
    pattern: list[tuple[float, float]], sample_rate: int = 16000
) -> np.ndarray:
    """Build a waveform from ``[(speech_seconds, silence_seconds), ...]``."""
    pieces = []
    for speech, gap in pattern:
        t = np.arange(int(speech * sample_rate)) / sample_rate
        pieces.append((0.25 * np.sin(2 * np.pi * 180 * t)).astype(np.float32))
        pieces.append(np.zeros(int(gap * sample_rate), dtype=np.float32))
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

class _FakeResponse:
    def __init__(self, text, block_reason=None, finish_reason=None):
        self._text = text
        self.prompt_feedback = type("FB", (), {"block_reason": block_reason})()
        if finish_reason is not None:
            self.candidates = [
                type("C", (), {"finish_reason": type("F", (), {"name": finish_reason})()})()
            ]
        else:
            self.candidates = []

    @property
    def text(self):
        if self._text is None:
            raise RuntimeError("no candidates")
        return self._text

class _FakeClient:
    def __init__(self, responder):
        self.responder = responder
        self.calls = []
        self.closed = 0
        self.models = self

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        answer = self.responder(contents, len(self.calls))
        return answer if isinstance(answer, _FakeResponse) else _FakeResponse(answer)

    def close(self):
        self.closed += 1

def _lines_from_contents(contents: str) -> list:
    """Extract separate text-only Gemini paragraphs from a request."""
    marker = contents.find("Input paragraphs:")
    if marker >= 0:
        payload = contents[marker + len("Input paragraphs:"):].strip()
        return [
            match.group("text").strip()
            for match in re.finditer(
                r"(?:^|\n)Paragraph\s+\d+\s*:\s*\n(?P<text>.*?)(?=\n\s*\nParagraph\s+\d+\s*:|\Z)",
                payload,
                flags=re.DOTALL,
            )
            if match.group("text").strip()
        ]
    payload = contents.split("Input lines:", 1)[-1]
    lines = []
    for line in payload.splitlines():
        match = re.match(r"^\s*(?:\d+[.)]\s*)?(?P<text>.+)$", line)
        if match and match.group("text").strip():
            lines.append(match.group("text").strip())
    return lines

def _make_translator(client, **kwargs):
    defaults = dict(
        api_key="test-key", model="gemini-test", batch_size=25,
        max_retries=3, retry_base_delay=0.01, retry_max_delay=0.02,
        requests_per_minute=6000, client=client,
    )
    defaults.update(kwargs)
    return GeminiTranslator(**defaults)

def _segments(texts):
    return [
        Segment(index=i, start=i * 2.0, end=i * 2.0 + 1.8, text_en=text)
        for i, text in enumerate(texts)
    ]

class _RecordingTranslator:
    def __init__(self, fail_on_group=None):
        self.translated = []
        self.shorten_calls = 0
        self.fail_on_group = fail_on_group
        self._group_no = 0

    def translate_segments(self, segments):
        self._group_no += 1
        if self.fail_on_group is not None and self._group_no == self.fail_on_group:
            raise RuntimeError("fake translation failure")
        for seg in segments:
            seg.text_bn = f"টি{seg.index}"
        self.translated.append([s.index for s in segments])
        return segments

    def shorten_to_fit(self, segments, **_):
        self.shorten_calls += 1
        for seg in segments:
            if seg.index >= 0 and (seg.text_bn or "").strip():
                seg.text_bn += "s"      # ছোট করা হয়েছে — TTS-এ এটাই যাবে
        return 0

    def describe(self):
        return "recording"

    def close(self):
        pass

class _RecordingTTS:
    def __init__(self):
        self.calls = []
        self.closed = False

    def set_budget(self, *_a, **_k):
        pass

    def synthesize(self, segments, output_dir):
        self.calls.append([(s.index, s.text_bn) for s in segments])
        return segments

    def describe(self):
        return "recording-tts"

    def close(self):
        self.closed = True

class _FakeSTT:
    def __init__(self):
        self.released = False
        self.last_stats = {"chunks": 1, "words": 5}

    def transcribe(self, audio_path):
        return [
            Segment(index=0, start=0.3, end=2.3, text_en="Hello world."),
            Segment(index=1, start=2.8, end=5.4, text_en="This is a test video."),
        ]

    def transcribe_reference(self, path, language="bn"):  # pragma: no cover
        return "পরীক্ষা"

    def release(self):
        self.released = True

class _FakeTranslator:
    def __init__(self):
        self.closed = False
        self.shorten_calls = 0

    def translate_segments(self, segments):
        for seg in segments:
            seg.text_bn = "হ্যালো বিশ্ব।" if seg.index == 0 else "এটি একটি পরীক্ষামূলক ভিডিও।"
        return segments

    def shorten_to_fit(self, segments, **_):
        self.shorten_calls += 1
        return 0

    def describe(self):
        return "fake:translator"

    def close(self):
        self.closed = True

class _FakeTTS:
    name = "fake-tts"

    def __init__(self):
        self.closed = False

    def set_budget(self, *_args, **_kwargs):
        return None

    def synthesize(self, segments, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for seg in segments:
            path = output_dir / f"seg_{seg.index:04d}.wav"
            # ইচ্ছাকৃতভাবে একটু লম্বা → atempo লাগবে
            dur = max(0.8, seg.duration * 1.4)
            t = np.arange(int(dur * 24000)) / 24000.0
            audio_fx.write_wav(path, 0.3 * np.sin(2 * np.pi * 330 * t), 24000)
            seg.tts_audio_path = path
            seg.tts_duration = dur
        return segments

    def describe(self):
        return "fake:tts"

    def close(self):
        self.closed = True

__all__ = [name for name in globals() if not name.startswith("__")]
__all__ += [
    "_speech_with_silence", "_FakeResponse", "_FakeClient", "_lines_from_contents",
    "_make_translator", "_segments", "_RecordingTranslator", "_RecordingTTS",
    "_FakeSTT", "_FakeTranslator", "_FakeTTS",
]
