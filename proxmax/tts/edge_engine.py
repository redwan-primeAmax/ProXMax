"""Built-in Bangla neural voice via edge-tts.

Microsoft edge-tts-এর ``bn-BD`` neural ভয়েস (ডিফল্ট পুরুষ কণ্ঠ
``bn-BD-PradeepNeural``, নারী কণ্ঠ ``bn-BD-NabanitaNeural``)।
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time
from pathlib import Path
from typing import List, Optional

from .. import audio_fx
from ..config import DEFAULT_TTS_CONCURRENCY, safe_tts_rate
from ..models import Segment
from ..reporting_runtime import record_event
from ..text_utils import prepare_for_speech
from ..utils import FFmpegError, ensure_tool, get_logger, run_ffmpeg
from .base import BaseTTSEngine, TTSError

logger = get_logger(__name__)

BANGLA_VOICE_FEMALE = "bn-BD-NabanitaNeural"
BANGLA_VOICE_MALE = "bn-BD-PradeepNeural"
_EDGE_RECOVERY_CONCURRENCY = 5

#: একই প্রসেসে বারবার ভয়েস-তালিকা নামানো এড়াতে একবারই রাখা হয়
_VOICE_LIST: dict = {}


class EdgeTTSEngine(BaseTTSEngine):
    """Concurrent Bangla synthesis through edge-tts."""

    name = "edge"

    def __init__(
        self,
        voice: str = BANGLA_VOICE_MALE,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        concurrency: int = DEFAULT_TTS_CONCURRENCY,
        max_retries: int = 3,
        sample_rate: int = 24000,
        trim_silence: bool = True,
        trim_pad_ms: int = 25,
        gender: str = "male",
        keep_raw: bool = False,
    ) -> None:
        super().__init__(trim_silence=trim_silence, trim_pad_ms=trim_pad_ms)
        self.voice = voice
        #: ব্যর্থ ভয়েস হলে এই লিঙ্গের বিকল্পই খোঁজা হবে (নারী কণ্ঠ কখনো জোর করে বসবে না)
        self.gender = (gender or "male").lower()
        self.rate = safe_tts_rate(rate)
        self.volume = volume
        self.pitch = pitch
        self.concurrency = max(1, int(concurrency))
        self._active_concurrency = self.concurrency
        self.max_retries = max(1, max_retries)
        self.sample_rate = sample_rate
        self.keep_raw = bool(keep_raw)

    # ------------------------------------------------------------------
    def prepare(self) -> None:
        """edge-tts ইনস্টল আছে কি না যাচাই করে।"""
        super().prepare()
        try:
            import edge_tts  # noqa: F401
        except ImportError as exc:
            raise TTSError(
                "edge-tts ইনস্টল নেই। চালান: pip install -r requirements.txt"
            ) from exc

    def set_budget(
        self, time_budget_seconds: Optional[float], total_audio_seconds: float = 0.0,
    ) -> None:
        """Try the configured concurrency again at the start of each video."""
        self._active_concurrency = self.concurrency

    # ------------------------------------------------------------------
    def synthesize(self, segments: List[Segment], output_dir: Path | str) -> List[Segment]:
        """প্রতিটি segment-এর জন্য বাংলা ক্লিপ তৈরি করে (একসাথে কয়েকটি)।"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        todo = self.usable_segments(segments)
        if not todo:
            raise TTSError("TTS-এর জন্য কোনো বাংলা টেক্সট নেই (সব segment-এর text_bn খালি)।")

        self.voice = self._resolve_voice()
        logger.info(
            "edge-tts (বিল্ট-ইন বাংলা ভয়েস) শুরু: %d segment | voice=%s | concurrency=%d",
            len(todo), self.voice, self._active_concurrency,
        )
        started = time.perf_counter()
        self._run_async(self._synthesize_all(todo, output_dir))

        failed = [s for s in todo if not s.tts_audio_path]
        if failed and self._active_concurrency > _EDGE_RECOVERY_CONCURRENCY:
            logger.warning(
                "⚠️ %dটি Edge TTS segment ব্যর্থ — সার্ভিসে চাপ কমাতে concurrency=%d দিয়ে আবার চেষ্টা।",
                len(failed), _EDGE_RECOVERY_CONCURRENCY,
            )
            self._run_async(
                self._synthesize_all(failed, output_dir, concurrency=_EDGE_RECOVERY_CONCURRENCY)
            )
            if len(failed) >= max(2, (len(todo) + 4) // 5):
                self._active_concurrency = _EDGE_RECOVERY_CONCURRENCY
                logger.warning("বেশি segment ব্যর্থ হওয়ায় এই ভিডিওতে TTS concurrency=%d থাকবে।",
                               self._active_concurrency)

        done = [s for s in todo if s.tts_audio_path]
        if not done:
            raise TTSError(
                "একটি segment-এরও অডিও তৈরি হয়নি। ইন্টারনেট সংযোগ ও edge-tts "
                "সার্ভিসের উপলব্ধতা যাচাই করুন।"
            )
        if len(done) < len(todo):
            logger.warning("⚠️ %dটি segment-এর TTS অডিও তৈরি হয়নি।", len(todo) - len(done))
        logger.info(
            "edge-tts শেষ: %d/%d segment (%.1fs)", len(done), len(todo),
            time.perf_counter() - started,
        )
        return segments

    # ------------------------------------------------------------------
    @staticmethod
    def _run_async(coro):
        """Notebook-এর চলমান event loop-এও নিরাপদে coroutine চালায় (ফলাফল ফেরত দেয়)।"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        box: dict = {}

        def _runner() -> None:
            try:
                box["value"] = asyncio.run(coro)
            except BaseException as exc:  # calling thread-এ re-raise হবে
                box["error"] = exc

        context = contextvars.copy_context()
        thread = threading.Thread(target=lambda: context.run(_runner), daemon=True)
        thread.start()
        thread.join()
        if "error" in box:
            raise box["error"]
        return box.get("value")

    # ------------------------------------------------------------------
    def _resolve_voice(self) -> str:
        """কনফিগের ভয়েস সত্যিই আছে কি না দেখে নেয়; না থাকলে **সমান লিঙ্গের**
        বিকল্প বেছে নেয়। নারী কণ্ঠে নীরবে নেমে যাওয়া কখনোই হবে না
        (ব্যবহারকারীর শর্ত: কণ্ঠ অবশ্যই পুরুষ)।

        Returns:
            ব্যবহারযোগ্য ভয়েসের নাম (সমস্যা হলে কনফিগের ভয়েসই ফেরত যায়)।
        """
        if _VOICE_LIST.get("voices") is None:
            try:
                import edge_tts

                _VOICE_LIST["voices"] = self._run_async(edge_tts.list_voices())
            except Exception as exc:  # নেটওয়ার্ক সমস্যা — কনফিগের ভয়েসেই চেষ্টা
                logger.debug("edge-tts ভয়েস-তালিকা আনা যায়নি (%s) — কনফিগের ভয়েসই ব্যবহার হবে।", exc)
                _VOICE_LIST["voices"] = []
        voices = _VOICE_LIST["voices"] or []
        if not voices:
            return self.voice
        names = {v.get("ShortName") for v in voices if v.get("ShortName")}
        if self.voice in names:
            return self.voice

        wanted = "Male" if self.gender in {"male", "custom", "m"} else "Female"
        bangla = [v for v in voices if str(v.get("Locale", "")).startswith("bn-")]
        same_gender = [v for v in bangla if str(v.get("Gender", "")) == wanted]
        if same_gender:
            logger.warning(
                "⚠️ কনফিগের ভয়েস «%s» edge-tts-এ নেই — সমান লিঙ্গের ভয়েস «%s» নেওয়া হলো।",
                self.voice, same_gender[0]["ShortName"],
            )
            return same_gender[0]["ShortName"]
        available = ", ".join(sorted(v.get("ShortName", "?") for v in bangla)) or "কোনো bn-* ভয়েস নেই"
        raise TTSError(
            f"«{self.voice}» ভয়েসটি edge-tts-এ পাওয়া যায়নি, আর {wanted.lower()} "
            f"বাংলা ভয়েসও নেই। উপলব্ধ বাংলা ভয়েস: {available}. "
            "`config.yaml`-এ `tts.gender` বা `tts.voice` ঠিক করুন।"
        )

    async def _synthesize_all(
        self, segments: List[Segment], output_dir: Path, *, concurrency: int | None = None,
    ) -> None:
        """Keep a bounded pool busy across all segments, without creating one task per clip."""
        jobs = iter(segments)
        completed = 0
        last_log = time.perf_counter()
        active = self._active_concurrency if concurrency is None else int(concurrency)
        workers = min(len(segments), max(1, active))

        async def worker() -> None:
            nonlocal completed, last_log
            for seg in jobs:
                await self._synthesize_one(seg, output_dir)
                completed += 1
                now = time.perf_counter()
                if completed < len(segments) and (completed % 20 == 0 or now - last_log >= 15.0):
                    logger.info("edge-tts অগ্রগতি: %d/%d segment প্রক্রিয়া হয়েছে", completed, len(segments))
                    last_log = now

        await asyncio.gather(*(worker() for _ in range(workers)))

    async def _synthesize_one(self, seg: Segment, output_dir: Path) -> None:
        import edge_tts

        text = prepare_for_speech(seg.text_bn)
        mp3_path = output_dir / f"seg_{seg.index:04d}.mp3"
        wav_path = output_dir / f"seg_{seg.index:04d}.wav"
        last_error: BaseException | None = None
        started = time.perf_counter()
        record_event("tts_clip_started", segment=seg.index)

        for attempt in range(1, self.max_retries + 1):
            try:
                communicate = edge_tts.Communicate(
                    text, self.voice,
                    rate=self.rate, volume=self.volume, pitch=self.pitch,
                )
                await communicate.save(str(mp3_path))
                if not mp3_path.is_file() or mp3_path.stat().st_size == 0:
                    raise TTSError("edge-tts খালি অডিও ফেরত দিয়েছে (NoAudioReceived)")
                duration = await asyncio.to_thread(self._convert_and_finalize, mp3_path, wav_path, seg)
                seg.tts_audio_path = wav_path
                seg.tts_duration = duration
                record_event("tts_clip_finished", segment=seg.index, attempt=attempt,
                             elapsed_seconds=time.perf_counter() - started, clip_seconds=duration, status="success")
                return
            except Exception as exc:
                last_error = exc
                delay = 1.0 * (2 ** (attempt - 1)) if attempt < self.max_retries else 0.0
                record_event("tts_attempt_failed", segment=seg.index, attempt=attempt,
                             max_attempts=self.max_retries, delay_seconds=delay, error_type=type(exc).__name__)
                if attempt < self.max_retries:
                    await asyncio.sleep(delay)
        record_event("tts_clip_finished", segment=seg.index, attempt=self.max_retries,
                     elapsed_seconds=time.perf_counter() - started, status="failed",
                     error_type=type(last_error).__name__ if last_error else None)
        logger.warning("Segment %d-এর edge-tts ব্যর্থ: %s", seg.index, last_error)
        seg.tts_audio_path = None
        seg.tts_duration = None

    def _convert_and_finalize(self, mp3_path: Path, wav_path: Path,
                              seg: Optional[Segment] = None) -> float:
        """Convert, postprocess and measure a clip outside the network event loop."""
        self._to_wav(mp3_path, wav_path)
        clip, rate = audio_fx.read_wav(wav_path)
        final = self._finalize(clip, seg)
        actual_rate = rate or self.sample_rate
        audio_fx.write_wav(wav_path, final, actual_rate)
        return final.size / float(actual_rate)

    def _to_wav(self, mp3_path: Path, wav_path: Path) -> Path:
        """MP3 → mono WAV (sync ইঞ্জিন শুধু WAV পড়ে)।"""
        try:
            run_ffmpeg(
                [
                    ensure_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(mp3_path),
                    "-ac", "1", "-ar", str(self.sample_rate),
                    "-c:a", "pcm_s16le", str(wav_path),
                ],
                description="edge-tts mp3→wav",
                timeout=120.0,
            )
        except FFmpegError as exc:
            raise TTSError(f"edge-tts অডিও রূপান্তর ব্যর্থ: {exc}") from exc
        finally:
            if not getattr(self, "keep_raw", False):
                mp3_path.unlink(missing_ok=True)
        return wav_path

    @staticmethod
    def _wav_seconds(path: Path) -> float:
        from ..audio_fx import read_wav

        audio, rate = read_wav(path)
        return audio.size / float(rate or 1)

    def describe(self) -> str:
        return f"edge:{self.voice}"
