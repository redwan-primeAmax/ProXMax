"""Silence-aware chunk planning for Whisper — never cut in the middle of a word.

Whisper-কে আর শব্দের মাঝখানে কাটতে দেওয়া হয় না। কেন এই মডিউল:

* Whisper একবারে ৩০ সেকেন্ডের বেশি অডিও দেখতে পারে না। therefore বাধ্যতামূলক
  কাটা লাগে। কিন্তু faster-whisper নিজে কাটে **শেষ ভালো নীরবতা** খুঁজে;
  না পেলে ``max_speech_duration_s`` সীমায় **জোর করে** কেটে দেয় — ফলে
  একটা শব্দ দুই টুকরো হয়ে যায় ("XYZ" → "XY" + "Z")।
* এখানে আগে VAD (বা fallback-এ energy) দিয়ে নীরবতার সীমানা বের করা হয়,
  তারপর সেই নীরবতার **মাঝখানে** কাটা হয়। তাই কোনো শব্দ কাটা পড়া
  গাণিতিকভাবেই প্রায় অসম্ভব।
* প্রতিটি chunk-এর সাথে ``forced`` ফ্ল্যাগ থাকে — কোথাও জোর করে কাটা
  পড়লে STT ধাপে seam repair করা হয়।

নীরবতা খোঁজায় Silero VAD (faster-whisper-এর সাথে আসে) ব্যবহার হয়;
সেটি না থাকলে (যেমন onnxruntime নেই) RMS-ভিত্তিক নিজস্ব ডিটেক্টর।
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .utils import get_logger

logger = get_logger(__name__)

#: VAD/energy থেকে পাওয়া speech অঞ্চল — ``(start_seconds, end_seconds)``
Region = Tuple[float, float]


@dataclass(frozen=True)
class SpeechChunk:
    """One portion of audio handed to Whisper as a single unit."""

    start: float
    end: float
    forced: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


# =============================================================================
# Speech region detection
# =============================================================================
def energy_regions(
    audio: np.ndarray,
    sample_rate: int = 16000,
    threshold_dbfs: float = -42.0,
    frame_ms: int = 20,
    min_speech_ms: int = 120,
    min_silence_ms: int = 300,
) -> List[Region]:
    """RMS-ভিত্তিক speech অঞ্চল বের করে (VAD-এর হালকা বিকল্প)।

    Args:
        audio: Mono float32 waveform।
        sample_rate: Sample rate।
        threshold_dbfs: এর চেয়ে জোরে ফ্রেমকে কথা ধরা হয়।
        frame_ms: বিশ্লেষণের ফ্রেম দৈর্ঘ্য।
        min_speech_ms: এর চেয়ে ছোট "কথা" বাদ (ক্লিক/নয়েজ)।
        min_silence_ms: এর কম বিরতি দিয়ে কথা আলাদা ধরা হয় না।

    Returns:
        সময় অনুযায়ী সাজানো ``(start, end)`` সেকেন্ডের তালিকা।
    """
    if audio.size == 0:
        return []
    frame = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = int(math.ceil(audio.size / frame))
    padded = np.zeros(n_frames * frame, dtype=np.float32)
    padded[: audio.size] = audio
    frames = padded.reshape(n_frames, frame)
    level = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    loud = level > (10.0 ** (threshold_dbfs / 20.0))
    if not bool(loud.any()):
        return []

    min_silence_frames = max(1, int(min_silence_ms / frame_ms))
    regions: List[Region] = []
    start_idx = None
    silent_run = 0
    for i, is_loud in enumerate(loud):
        if is_loud:
            if start_idx is None:
                start_idx = i
            silent_run = 0
        elif start_idx is not None:
            silent_run += 1
            if silent_run >= min_silence_frames:
                end_idx = i - silent_run + 1
                regions.append((start_idx * frame / sample_rate, end_idx * frame / sample_rate))
                start_idx = None
                silent_run = 0
    if start_idx is not None:
        regions.append((start_idx * frame / sample_rate, min(audio.size, n_frames * frame) / sample_rate))

    min_speech = min_speech_ms / 1000.0
    return [(s, e) for s, e in regions if e - s >= min_speech]


def vad_regions(
    audio: np.ndarray,
    sample_rate: int = 16000,
    threshold: float = 0.5,
    min_silence_ms: int = 300,
) -> List[Region]:
    """Silero VAD দিয়ে speech অঞ্চল বের করে (faster-whisper-এর বান্ডল)।

    ``max_speech_duration_s=inf`` দেওয়া হয় — অর্থাৎ VAD নিজে কখনো
    শব্দের মাঝখানে কাটবে না; কাটার সিদ্ধান্ত শুধু
    :func:`regions_to_chunks` নেয়।
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(
        threshold=threshold,
        min_silence_duration_ms=int(min_silence_ms),
        min_speech_duration_ms=100,
        speech_pad_ms=0,
        max_speech_duration_s=float("inf"),
    )
    stamps = get_speech_timestamps(audio, options, sampling_rate=sample_rate)
    return [(s["start"] / sample_rate, s["end"] / sample_rate) for s in stamps]


def detect_regions(
    audio: np.ndarray,
    sample_rate: int = 16000,
    min_silence_ms: int = 300,
    vad_threshold: float = 0.5,
) -> Tuple[List[Region], str]:
    """VAD চেষ্টা করে; ব্যর্থ হলে energy-ভিত্তিক ডিটেক্টরে নামে।

    Returns:
        ``(regions, method)`` — ``method`` হলো ``"vad"`` বা ``"energy"``।
    """
    try:
        regions = vad_regions(audio, sample_rate, threshold=vad_threshold,
                              min_silence_ms=min_silence_ms)
        if regions:
            return regions, "vad"
        logger.warning("VAD কোনো কথা পায়নি — energy ডিটেক্টরে যাওয়া হচ্ছে।")
    except Exception as exc:  # VAD মডেল/onnxruntime নেই
        logger.warning("VAD চালানো যায়নি (%s) — energy ডিটেক্টর ব্যবহার হচ্ছে।", exc)
    return energy_regions(audio, sample_rate, min_silence_ms=min_silence_ms), "energy"


# =============================================================================
# Region → chunk plan
# =============================================================================
def regions_to_chunks(
    regions: Sequence[Region],
    total_seconds: float,
    min_seconds: float = 18.0,
    max_seconds: float = 28.0,
    pad_seconds: float = 0.12,
) -> List[SpeechChunk]:
    """কথার অঞ্চলগুলোকে ১৮–২৮ সেকেন্ডের chunk-এ জোড়া লাগায়।

    নিয়ম:
      1. পরের অঞ্চল যোগ করলে ``max_seconds`` ছাড়ালে সেখানেই chunk শেষ —
         অর্থাৎ কাটা পড়ে **দুই অঞ্চলের মাঝের নীরবতায়**।
      2. একটা একক অঞ্চলই ``max_seconds``-এর চেয়ে লম্বা হলে (কেউ ৩০s
         টানা কথা বললে) শুধু তখনই জোর করে কাটা হয়, এবং ``forced=True``
         সেট হয় যাতে seam repair চালু হয়।
      3. দুই পাশে সামান্য pad যোগ হয় — কিন্তু প্রতিবেশী chunk বা পরের
         কথা-অঞ্চলের উপর চড়ে যায় না।

    Args:
        regions: ``detect_regions``-এর ফলাফল (সেকেন্ডে)।
        total_seconds: পুরো অডিওর দৈর্ঘ্য।
        min_seconds: চেষ্টা করার ন্যূনতম chunk দৈর্ঘ্য (স্ট্যাটিস্টিকসে ব্যবহৃত)।
        max_seconds: সর্বোচ্চ chunk দৈর্ঘ্য (৩০s-এর নিচে রাখা বাধ্যতামূলক)।
        pad_seconds: দুই পাশে context padding।

    Returns:
        সময় অনুযায়ী সাজানো :class:`SpeechChunk`-এর তালিকা (ওভারল্যাপ-মুক্ত)।
    """
    max_seconds = max(2.0, min(float(max_seconds), 29.5))
    min_seconds = max(1.0, min(float(min_seconds), max_seconds))
    cleaned = [(max(0.0, float(s)), min(float(total_seconds), float(e))) for s, e in regions]
    cleaned = [(s, e) for s, e in cleaned if e - s > 0.05]
    if not cleaned:
        return []

    chunks: List[SpeechChunk] = []
    i = 0
    while i < len(cleaned):
        start = cleaned[i][0]
        end = cleaned[i][1]
        last = i

        # একক অঞ্চলই খুব লম্বা → বাধ্যতামূলক ভাগ (forced=True)
        if end - start > max_seconds:
            piece_start = start
            while end - piece_start > max_seconds:
                chunks.append(SpeechChunk(piece_start, piece_start + max_seconds, forced=True))
                piece_start += max_seconds
            if end - piece_start > 0.05:
                chunks.append(SpeechChunk(piece_start, end, forced=True))
            i += 1
            continue

        # পরের অঞ্চলগুলো যোগ করা যায় যতক্ষণ max_seconds-এ আঁটে
        while last + 1 < len(cleaned) and cleaned[last + 1][1] - start <= max_seconds:
            last += 1
            end = cleaned[last][1]

        # একটাই ছোট অঞ্চল হলে বাদ দেব না — ছোট chunk ঠিক আছে
        chunks.append(SpeechChunk(start, end, forced=False))
        i = last + 1

    if not chunks:  # pragma: no cover - উপরের লজিক এখানে পৌঁছায় না
        return []

    # --- padding ভেতরে যোগ করা হয়, প্রতিবেশীকে স্পর্শ না করে ----------------
    padded: List[SpeechChunk] = []
    for idx, chunk in enumerate(chunks):
        prev_end = chunks[idx - 1].end if idx > 0 else 0.0
        next_start = chunks[idx + 1].start if idx + 1 < len(chunks) else float(total_seconds)
        head_room = max(0.0, (chunk.start - prev_end) / 2.0)
        tail_room = max(0.0, (next_start - chunk.end) / 2.0)
        new_start = max(prev_end, chunk.start - min(pad_seconds, head_room))
        new_end = min(next_start, chunk.end + min(pad_seconds, tail_room))
        padded.append(SpeechChunk(new_start, new_end, forced=chunk.forced))

    # সংখ্যালঘু ছোট chunk জোড়া লাগিয়ে ফেলা যায় (min_seconds-এর অনেক নিচে হলে)
    merged: List[SpeechChunk] = []
    for chunk in padded:
        if merged and (chunk.end - merged[-1].start) <= max_seconds and (
            merged[-1].duration < min_seconds * 0.5
        ):
            prev = merged[-1]
            merged[-1] = SpeechChunk(prev.start, chunk.end, forced=prev.forced or chunk.forced)
        else:
            merged.append(chunk)

    return [c for c in merged if c.duration > 0.2]


def plan_chunks(
    audio: np.ndarray,
    sample_rate: int = 16000,
    min_seconds: float = 18.0,
    max_seconds: float = 28.0,
    pad_seconds: float = 0.12,
    min_silence_ms: int = 300,
    vad_threshold: float = 0.5,
) -> Tuple[List[SpeechChunk], str]:
    """Detect speech regions and turn them into a Whisper chunk plan.

    Returns:
        ``(chunks, method)``। অডিওতে কথা না থাকলে খালি তালিকা।
    """
    regions, method = detect_regions(
        audio, sample_rate, min_silence_ms=min_silence_ms, vad_threshold=vad_threshold
    )
    total = audio.size / float(sample_rate) if sample_rate else 0.0
    chunks = regions_to_chunks(
        regions, total, min_seconds=min_seconds, max_seconds=max_seconds, pad_seconds=pad_seconds
    )
    if not chunks and total > 0:
        # VAD/energy কিছু না পেলেও অডিও আছে (খুব নিচু কথা?) — নিরাপদে
        # max_seconds-এ ভাগ করা হয়, আর forced=True রাখা হয় যাতে seam repair চলে।
        logger.warning(
            "কোনো speech region পাওয়া যায়নি — %.0fs অডিও %.0fs করে ভাগ করা হচ্ছে "
            "(seam repair চালু থাকবে)।", total, max_seconds,
        )
        pieces: List[SpeechChunk] = []
        start = 0.0
        while start < total:
            end = min(total, start + max_seconds)
            if end - start > 0.05:
                pieces.append(SpeechChunk(start, end, forced=True))
            start = end
        return pieces, method
    logger.info(
        "Chunk plan: %dটি chunk (%s ডিটেক্টর) | মোট %.1fs | জোর করে কাটা: %d",
        len(chunks), method, total, sum(1 for c in chunks if c.forced),
    )
    return chunks, method
