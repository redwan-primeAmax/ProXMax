"""Built-in Bangla neural voice via edge-tts.

Microsoft edge-tts-এর ``bn-BD`` neural ভয়েস (ডিফল্ট পুরুষ কণ্ঠ
``bn-BD-PradeepNeural``, নারী কণ্ঠ ``bn-BD-NabanitaNeural``)।
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import List

from .. import audio_fx
from ..models import Segment
from ..text_utils import prepare_for_speech
from ..utils import FFmpegError, ensure_tool, get_logger, run_ffmpeg
from .base import BaseTTSEngine, TTSError

logger = get_logger(__name__)

BANGLA_VOICE_FEMALE = "bn-BD-NabanitaNeural"
BANGLA_VOICE_MALE = "bn-BD-PradeepNeural"

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
        concurrency: int = 5,
        max_retries: int = 3,
        sample_rate: int = 24000,
        trim_silence: bool = True,
        trim_pad_ms: int = 25,
        gender: str = "male",
    ) -> None:
        super().__init__(trim_silence=trim_silence, trim_pad_ms=trim_pad_ms)
        self.voice = voice
        #: ব্যর্থ ভয়েস হলে এই লিঙ্গের বিকল্পই খোঁজা হবে (নারী কণ্ঠ কখনো জোর করে বসবে না)
        self.gender = (gender or "male").lower()
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        self.concurrency = max(1, concurrency)
        self.max_retries = max(1, max_retries)
        self.sample_rate = sample_rate

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
            len(todo), self.voice, self.concurrency,
        )
        started = time.perf_counter()
        self._run_async(self._synthesize_all(todo, output_dir))

        done = [s for s in todo if s.tts_audio_path]
        if not done:
            raise TTSError(
                "একটি segment-এরও অডিও তৈরি হয়নি। ইন্টারনেট সংযোগ ও edge-tts "
                "সার্ভিসের উপলব্ধতা যাচাই করুন।"
            )
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

        thread = threading.Thread(target=_runner, daemon=True)
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

    async def _synthesize_all(self, segments: List[Segment], output_dir: Path) -> None:
        semaphore = asyncio.Semaphore(self.concurrency)

        async def run_one(seg: Segment) -> None:
            async with semaphore:
                await self._synthesize_one(seg, output_dir)

        await asyncio.gather(*(run_one(seg) for seg in segments))

    async def _synthesize_one(self, seg: Segment, output_dir: Path) -> None:
        import edge_tts

        text = prepare_for_speech(seg.text_bn)
        mp3_path = output_dir / f"seg_{seg.index:04d}.mp3"
        wav_path = output_dir / f"seg_{seg.index:04d}.wav"
        last_error: BaseException | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                communicate = edge_tts.Communicate(
                    text, self.voice,
                    rate=self.rate, volume=self.volume, pitch=self.pitch,
                )
                await communicate.save(str(mp3_path))
                if not mp3_path.is_file() or mp3_path.stat().st_size == 0:
                    raise TTSError("edge-tts খালি অডিও ফেরত দিয়েছে (NoAudioReceived)")
                await asyncio.to_thread(self._to_wav, mp3_path, wav_path)
                # edge-tts দুই প্রান্তে নীরবতা দেয় — ছেঁটে দৈর্ঘ্য আবার মাপা হয়
                clip, rate = audio_fx.read_wav(wav_path)
                final = self._finalize(clip)
                audio_fx.write_wav(wav_path, final, rate or self.sample_rate)
                seg.tts_audio_path = wav_path
                seg.tts_duration = final.size / float(rate or self.sample_rate)
                return
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(1.0 * (2 ** (attempt - 1)))
        logger.warning("Segment %d-এর edge-tts ব্যর্থ: %s", seg.index, last_error)
        seg.tts_audio_path = None
        seg.tts_duration = None

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
            mp3_path.unlink(missing_ok=True)
        return wav_path

    @staticmethod
    def _wav_seconds(path: Path) -> float:
        from ..audio_fx import read_wav

        audio, rate = read_wav(path)
        return audio.size / float(rate or 1)

    def describe(self) -> str:
        return f"edge:{self.voice}"
