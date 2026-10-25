"""Built-in Bangla neural voice via edge-tts (no sample, token or license needed).

Microsoft edge-tts-এর ``bn-BD`` neural ভয়েস। ``tts.engine: auto`` (ডিফল্ট)
এবং কণ্ঠের নমুনা না থাকলে **এটিই ব্যবহৃত হয়** — ব্যবহারকারীকে কিছু দিতে
হয় না। ভয়েস ক্লোনিং (IndicF5) চালু করা না গেলেও এখানেই ফিরে আসা হয়,
যাতে ভিডিওটা নষ্ট না হয়।

কনফিগে ভয়েস বদলানো যায়: ``tts.voice`` (``bn-BD-NabanitaNeural`` /
``bn-BD-PradeepNeural``), আর ক্লোনিং-ব্যর্থতায় ফিরে আসা বন্ধ করতে
``tts → fallback_to_edge: false``।
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import List

from ..models import Segment
from ..text_utils import prepare_for_speech
from ..utils import FFmpegError, ensure_tool, get_logger, run_ffmpeg
from ..voice_profile import VoiceProfile
from .base import BaseTTSEngine, TTSError

logger = get_logger(__name__)

BANGLA_VOICE_FEMALE = "bn-BD-NabanitaNeural"
BANGLA_VOICE_MALE = "bn-BD-PradeepNeural"


class EdgeTTSEngine(BaseTTSEngine):
    """Concurrent Bangla synthesis through edge-tts (no voice cloning)."""

    name = "edge"

    def __init__(
        self,
        voice: str = BANGLA_VOICE_FEMALE,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        concurrency: int = 5,
        max_retries: int = 3,
        sample_rate: int = 24000,
    ) -> None:
        super().__init__()
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        self.concurrency = max(1, concurrency)
        self.max_retries = max(1, max_retries)
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------
    def prepare(self, profile: VoiceProfile | None = None) -> None:
        """edge-tts-এর জন্য রেফারেন্স লাগে না — শুধু নেটওয়ার্ক দরকার।"""
        super().prepare(profile)
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
    def _run_async(coro) -> None:
        """Notebook-এর চলমান event loop-এও নিরাপদে coroutine চালায়।"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coro)
            return

        error_box: dict = {}

        def _runner() -> None:
            try:
                asyncio.run(coro)
            except BaseException as exc:  # calling thread-এ re-raise হবে
                error_box["error"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if "error" in error_box:
            raise error_box["error"]

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
                seg.tts_audio_path = wav_path
                seg.tts_duration = wav_path.stat().st_size and self._wav_seconds(wav_path)
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
