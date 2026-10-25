"""Voice reference profile built from the speaker's own recording.

আপনার গলার নমুনা থেকে TTS-র জন্য প্রস্তুত প্রোফাইল তৈরি হয়। IndicF5 হলো **zero-shot voice cloning** মডেল — এটি তিনটি জিনিস চায়:
১. যে টেক্সট বলতে হবে, ২. রেফারেন্স অডিও, ৩. রেফারেন্স অডিওতে ঠিক কী
বলা হয়েছে তার ট্রান্সক্রিপ্ট। এই তিনটির মধ্যে ২ ও ৩ **হুবহু মিললে**
ক্লোনিংয়ের মান অনেক বেড়ে যায় — ভুল ``ref_text`` হলে কণ্ঠ এলোমেলো হয়।

এই মডিউল তাই করে:

1. ইনপুট ফোল্ডারেই রাখা নমুনা খুঁজে বের করে (``.wav .mp3 .m4a .flac
   .ogg .aac``) — আলাদা কনফিগ লাগে না। চাইলে ``reference/`` সাবফোল্ডারেও।
2. নমুনা ২৪ kHz mono-তে আনে, peak-normalize করে।
3. কথা/নীরবতা আলাদা করে **সবচেয়ে পরিষ্কার ধারাবাহিক কথা-অঞ্চল** বেছে
   নেয় (টার্গেট ~১৬ সেকেন্ড) — শুরু ও শেষ সর্বদা নীরবতায়, তাই কোনো
   শব্দ মাঝখানে কাটা পড়ে না।
4. ভেতরের লম্বা বিরতিগুলো (আপনি যেখানে থেমেছিলেন) স্বাভাবিক রাখতে
   সর্বোচ্চ ৮০০ ms-এ সীমিত করে — শ্বাস-প্রশ্বাসের ছন্দ থাকে, কিন্তু
   নমুনা অযথা লম্বা/ফাঁকা হয় না।
5. **ওই চূড়ান্ত অডিওটাই** Whisper দিয়ে ট্রান্সক্রাইব করে ``ref_text``
   বানায় — তাই অডিও ও টেক্সট কখনো মিলবে না, এমনটা হয় না।
6. ফলাফল cache করে (ফাইলের hash দেখে) — প্রতি ভিডিওতে আবার হিসাব নয়।
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from . import audio_fx
from .chunking import Region, detect_regions
from .utils import get_logger

logger = get_logger(__name__)

REFERENCE_AUDIO_EXTENSIONS = frozenset(
    {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".wma"}
)
REFERENCE_DIR_NAMES = ("reference", "voice", "voice_reference", "ref")

#: রেফারেন্স তৈরির সময় ব্যবহৃত অডিওর sample rate (IndicF5-এর native)
REFERENCE_SAMPLE_RATE = 24000
#: বিশ্লেষণের জন্য হালকা sample rate
ANALYSIS_SAMPLE_RATE = 16000

#: এক অঞ্চলের দুই পাশে যতটুকু margin রাখা হয়
_REGION_MARGIN_MS = 60
#: ভেতরের বিরতি সর্বোচ্চ এই পরিমাণ রাখা হয়
_DEFAULT_MAX_PAUSE_MS = 800


# =============================================================================
# Discovery
# =============================================================================
def discover_reference_audio(
    input_dir: Path | str,
    explicit: Optional[Path | str] = None,
) -> Optional[Path]:
    """Find the voice-sample audio file in the input folder.

    ক্রম: ``explicit`` পাথ → ``<input>/reference/`` (বা সমগোত্রীয় ফোল্ডার)
    → ইনপুট ফোল্ডারের গোড়ায় থাকা অডিও ফাইল। একাধিক থাকলে সবচেয়ে বড়টি।

    Args:
        input_dir: ভিডিওর ইনপুট ফোল্ডার।
        explicit: ব্যবহারকারীর দেওয়া নির্দিষ্ট পাথ (যদি থাকে)।

    Returns:
        পাওয়া গেলে ফাইলের পাথ, নাহলে ``None``।
    """
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path
        logger.warning("বলা হয়েছে এমন রেফারেন্স অডিও পাওয়া যায়নি: %s", path)

    root = Path(input_dir).expanduser()
    candidates: List[Path] = []

    for name in REFERENCE_DIR_NAMES:
        folder = root / name
        if folder.is_dir():
            candidates.extend(
                p for p in sorted(folder.iterdir())
                if p.is_file() and p.suffix.lower() in REFERENCE_AUDIO_EXTENSIONS
            )

    if root.is_dir():
        candidates.extend(
            p for p in sorted(root.iterdir())
            if p.is_file()
            and p.suffix.lower() in REFERENCE_AUDIO_EXTENSIONS
            and not p.name.startswith((".", "~"))
        )

    if not candidates:
        return None
    # সবচেয়ে বড় ফাইলটি সাধারণত সবচেয়ে দীর্ঘ/ভালো নমুনা
    best = max(candidates, key=lambda p: p.stat().st_size)
    logger.info("ভয়েস রেফারেন্স পাওয়া গেছে: %s", best)
    return best


# =============================================================================
# Profile
# =============================================================================
@dataclass
class VoiceProfile:
    """Ready-to-use voice reference for the TTS engine."""

    audio_path: Path
    text: str = ""
    duration: float = 0.0
    source_path: Optional[Path] = None
    window_start: float = 0.0
    window_end: float = 0.0
    speech_seconds: float = 0.0
    created_at: float = field(default_factory=time.time)

    def is_valid(self) -> bool:
        """প্রোফাইলটি ব্যবহারযোগ্য কি না।"""
        try:
            return bool(self.text.strip()) and self.duration > 0.5 and self.audio_path.is_file()
        except OSError:
            return False

    def summary(self) -> str:
        """এক লাইনের human-readable বর্ণনা (লগের জন্য)।"""
        return (
            f"{self.audio_path.name} | {self.duration:.1f}s | "
            f"টেক্সট {len(self.text)} অক্ষর | উৎস: "
            f"{self.source_path.name if self.source_path else '-'}"
        )

    def to_json(self) -> dict:
        data = asdict(self)
        data["audio_path"] = str(self.audio_path)
        data["source_path"] = str(self.source_path) if self.source_path else None
        return data

    @classmethod
    def from_json(cls, data: dict) -> "VoiceProfile":
        return cls(
            audio_path=Path(data["audio_path"]),
            text=data.get("text", ""),
            duration=float(data.get("duration", 0.0)),
            source_path=Path(data["source_path"]) if data.get("source_path") else None,
            window_start=float(data.get("window_start", 0.0)),
            window_end=float(data.get("window_end", 0.0)),
            speech_seconds=float(data.get("speech_seconds", 0.0)),
            created_at=float(data.get("created_at", 0.0)),
        )


# =============================================================================
# Window selection (pure function — testable without audio models)
# =============================================================================
def select_reference_window(
    regions: Sequence[Region],
    target_seconds: float = 16.0,
    min_seconds: float = 5.0,
    max_seconds: float = 40.0,
) -> Optional[Tuple[int, int, float]]:
    """Pick the best contiguous run of speech regions for the reference.

    "সেরা" মানে — মোট কথা ``target_seconds``-এর যত কাছে, আর নীরবতার
    অনুপাত যত কম।

    Args:
        regions: ``(start, end)`` সেকেন্ডের ক্রমবর্ধমান তালিকা।
        target_seconds: কাঙ্ক্ষিত কথা-দৈর্ঘ্য (১০–২০s সবচেয়ে ভালো)।
        min_seconds: এর কম কথা হলে রেফারেন্স দুর্বল।
        max_seconds: উইন্ডো এর চেয়ে বড় হবে না।

    Returns:
        ``(first_region_index, last_region_index, speech_seconds)``
        অথবা কিছু না মিললে ``None``।
    """
    if not regions:
        return None
    limit = max(target_seconds * 1.8, target_seconds + 2.0)
    limit = min(limit, max(max_seconds, target_seconds))

    best: Optional[Tuple[float, int, int, float]] = None
    for i in range(len(regions)):
        speech = 0.0
        last = i
        while last < len(regions):
            span_start = regions[i][0]
            span_end = regions[last][1]
            if span_end - span_start > limit:
                break
            speech += regions[last][1] - regions[last][0]
            if speech >= target_seconds:
                break
            last += 1
        if last >= len(regions):
            last = len(regions) - 1
        speech = sum(regions[k][1] - regions[k][0] for k in range(i, last + 1))
        if speech < min_seconds:
            continue
        span = regions[last][1] - regions[i][0]
        if span > limit or span <= 0:
            continue
        density = speech / span
        # আগে টার্গেট-দৈর্ঘ্যের নৈকট্য, তারপর কম নীরবতা
        score = (abs(speech - target_seconds) - density * 1.5, -speech)
        if best is None or score < best[0]:
            best = (score[0], i, last, speech)  # type: ignore[assignment]
    if best is None:
        return None
    _, first, last, speech = best
    return first, last, speech


def _hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _hash_params(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


# =============================================================================
# Builder
# =============================================================================
class VoiceProfileBuilder:
    """Builds (and caches) a :class:`VoiceProfile` from a raw recording."""

    def __init__(
        self,
        cache_dir: Path | str,
        target_seconds: float = 16.0,
        min_seconds: float = 5.0,
        max_seconds: float = 40.0,
        max_internal_pause_ms: int = _DEFAULT_MAX_PAUSE_MS,
        min_silence_ms: int = 300,
        vad_threshold: float = 0.5,
        language: str = "bn",
        use_cache: bool = True,
        highpass_hz: int = 0,
        loudnorm_i: float = 0.0,
    ) -> None:
        """Args:
            cache_dir: প্রোফাইল কোথায় রাখা হবে (``work/voice_profile``)।
            target_seconds: রেফারেন্সের কাঙ্ক্ষিত কথা-দৈর্ঘ্য।
            min_seconds: সর্বনিম্ন গ্রহণযোগ্য কথা-দৈর্ঘ্য।
            max_seconds: সর্বোচ্চ অনুমোদিত নমুনা দৈর্ঘ্য।
            max_internal_pause_ms: বিরতি এর বেশি হলে ছেঁটে আনা হয়।
            min_silence_ms: VAD-এর ন্যূনতম নীরবতা।
            vad_threshold: VAD সংবেদনশীলতা।
            language: রেফারেন্সের ভাষা (Bangla = ``"bn"``)।
            use_cache: False হলে প্রতিবার নতুন করে বানাবে।
            highpass_hz: নমুনার কম-কম্পাঙ্কের গুঞ্জন (hum/rumble) সরাতে
                ffmpeg high-pass-এর কাটঅফ (০ = বন্ধ) — Demucs-এর হালকা বিকল্প।
            loudnorm_i: EBU R128 loudness target (০ = বন্ধ)। লেভেল অসমান হলে
                ক্লোনিংয়ের মান কমে, তাই হালকা normalization দরকার।
        """
        self.cache_dir = Path(cache_dir)
        self.target_seconds = float(target_seconds)
        self.min_seconds = float(min_seconds)
        self.max_seconds = float(max_seconds)
        self.max_internal_pause_ms = int(max(100, max_internal_pause_ms))
        self.min_silence_ms = int(min_silence_ms)
        self.vad_threshold = float(vad_threshold)
        self.language = language
        self.use_cache = use_cache
        self.highpass_hz = max(0, int(highpass_hz))
        self.loudnorm_i = float(loudnorm_i)

    # ------------------------------------------------------------------
    def _params(self) -> dict:
        return {
            "target": self.target_seconds,
            "min": self.min_seconds,
            "max": self.max_seconds,
            "pause": self.max_internal_pause_ms,
            "silence": self.min_silence_ms,
            "vad": self.vad_threshold,
            "sr": REFERENCE_SAMPLE_RATE,
            "lang": self.language,
            "hp": self.highpass_hz,
            "ln": round(self.loudnorm_i, 2),
            "v": 3,
        }

    def _decode_filters(self) -> Optional[str]:
        """নমুনা পরিষ্কারের ffmpeg filter (high-pass + হালকা loudness)।"""
        parts = []
        if self.highpass_hz > 0:
            parts.append(f"highpass=f={self.highpass_hz}")
        if self.loudnorm_i < 0:
            parts.append(f"loudnorm=I={self.loudnorm_i:.1f}:TP=-1.5:LRA=11")
        return ",".join(parts) or None

    # ------------------------------------------------------------------
    def build(
        self,
        source: Path | str,
        transcriber: Optional[Callable[[Path], str]] = None,
        force: bool = False,
    ) -> VoiceProfile:
        """Create (or load) the voice profile from ``source``.

        Args:
            source: ব্যবহারকারীর দেওয়া কাঁচা রেকর্ডিং।
            transcriber: ``Path -> str`` ফাংশন, যা রেফারেন্সের টেক্সট বানায়।
                না দিলে ``text`` খালি থাকবে (ক্লোনিংয়ের মান কমবে)।
            force: cache উপেক্ষা করে নতুন করে বানানো।

        Raises:
            FileNotFoundError: নমুনা ফাইল না থাকলে।
            ValueError: নমুনা খুব ছোট/খালি হলে বা তাতে কথা না থাকলে।
        """
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(f"ভয়েস রেফারেন্স ফাইল পাওয়া যায়নি: {source}")

        fingerprint = _hash_file(source)
        folder = self.cache_dir / f"{fingerprint[:16]}_{_hash_params(self._params())}"
        ref_path = folder / "reference.wav"
        meta_path = folder / "profile.json"

        if self.use_cache and not force and ref_path.is_file() and meta_path.is_file():
            try:
                profile = VoiceProfile.from_json(json.loads(meta_path.read_text(encoding="utf-8")))
                if profile.audio_path.is_file() and profile.duration > 0:
                    logger.info("ক্যাশ করা ভয়েস প্রোফাইল ব্যবহার হচ্ছে: %s", profile.summary())
                    return profile
            except Exception as exc:  # নষ্ট cache → আবার বানানো হবে
                logger.warning("ভয়েস প্রোফাইল cache পড়া যায়নি (%s) — নতুন করে তৈরি হচ্ছে।", exc)

        full = audio_fx.decode_audio(
            source, REFERENCE_SAMPLE_RATE, filters=self._decode_filters()
        )
        full_duration = full.size / REFERENCE_SAMPLE_RATE
        if full_duration < self.min_seconds:
            raise ValueError(
                f"ভয়েস নমুনা খুব ছোট ({full_duration:.1f}s) — অন্তত {self.min_seconds:.0f} "
                f"সেকেন্ড পরিষ্কার বাংলা কথা দরকার।"
            )
        if full_duration > self.max_seconds * 4:
            logger.warning(
                "নমুনা বেশ লম্বা (%.1fs) — এর মধ্য থেকে সর্বোত্তম %0.f সেকেন্ড নেওয়া হবে।",
                full_duration, self.target_seconds,
            )

        analysis = audio_fx.decode_audio(source, ANALYSIS_SAMPLE_RATE)
        regions, method = detect_regions(
            analysis, ANALYSIS_SAMPLE_RATE,
            min_silence_ms=self.min_silence_ms, vad_threshold=self.vad_threshold,
        )
        total_speech = sum(e - s for s, e in regions)
        if not regions or total_speech < self.min_seconds * 0.6:
            raise ValueError(
                "ভয়েস নমুনায় যথেষ্ট কথা পাওয়া যায়নি (অনেকটা নীরবতা/আওয়াজ?)। "
                "শান্ত ঘরে ১০–২০ সেকেন্ড পরিষ্কার বাংলা কথা রেকর্ড করুন।"
            )

        window = select_reference_window(
            regions, target_seconds=self.target_seconds,
            min_seconds=min(self.min_seconds, total_speech), max_seconds=self.max_seconds,
        )
        if window is None:
            first, last = 0, len(regions) - 1
            speech = total_speech
        else:
            first, last, speech = window

        reference = self._render_reference(full, regions[first:last + 1], method)
        folder.mkdir(parents=True, exist_ok=True)
        audio_fx.write_wav(ref_path, reference, REFERENCE_SAMPLE_RATE)
        duration = reference.size / REFERENCE_SAMPLE_RATE

        text = ""
        if transcriber is not None:
            try:
                text = (transcriber(ref_path) or "").strip()
            except Exception as exc:
                logger.warning("রেফারেন্স টেক্সট তৈরি করা যায়নি: %s", exc)
        if not text:
            logger.warning(
                "রেফারেন্সের টেক্সট পাওয়া যায়নি — ক্লোনিংয়ের মান কমতে পারে। "
                "নমুনাটি পরিষ্কার ও স্পষ্ট হতে হবে।"
            )

        profile = VoiceProfile(
            audio_path=ref_path,
            text=text,
            duration=duration,
            source_path=source,
            window_start=float(regions[first][0]),
            window_end=float(regions[last][1]),
            speech_seconds=float(speech),
        )
        meta_path.write_text(
            json.dumps(profile.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            "ভয়েস প্রোফাইল তৈরি: %s | ডিটেক্টর=%s | কথা %.1fs (উইন্ডো %.1f–%.1fs)",
            profile.summary(), method, speech, profile.window_start, profile.window_end,
        )
        return profile

    # ------------------------------------------------------------------
    def _render_reference(
        self,
        audio: np.ndarray,
        regions: Sequence[Region],
        method: str = "vad",
    ) -> np.ndarray:
        """বেছে নেওয়া অঞ্চলগুলো জোড়া লাগিয়ে পরিষ্কার রেফারেন্স বানায়।

        অঞ্চলের বাইরের অংশ বাদ যায় (শুরু/শেষ নীরবতায়), আর অঞ্চলের
        মাঝের বিরতি ``max_internal_pause_ms``-এ সীমিত হয় — যাতে ব্যবহারকারীর
        স্বাভাবিক থামা/শ্বাস নেওয়ার ধরনটা থাকে, কিন্তু অযথা ফাঁক না থাকে।
        """
        sr = REFERENCE_SAMPLE_RATE
        margin = int(sr * _REGION_MARGIN_MS / 1000)
        max_pause = int(sr * self.max_internal_pause_ms / 1000)
        pieces: List[np.ndarray] = []
        prev_end_sample: Optional[int] = None

        for start_s, end_s in regions:
            start = max(0, int(start_s * sr) - margin)
            end = min(audio.size, int(end_s * sr) + margin)
            if end <= start:
                continue
            if prev_end_sample is not None and pieces:
                gap = max(0, start - prev_end_sample)
                if gap > max_pause:
                    pieces.append(audio_fx.silence((gap - max_pause) / sr, sr))
            pieces.append(audio[start:end])
            prev_end_sample = end

        if not pieces:  # pragma: no cover - regions খালি হলে আগেই error হয়েছে
            return audio_fx.trim_silence(audio, sr)

        combined = np.concatenate(pieces).astype(np.float32)
        combined = audio_fx.fade(combined, sr, in_ms=10, out_ms=60)
        return audio_fx.peak_normalize(combined, peak_dbfs=-1.0)
