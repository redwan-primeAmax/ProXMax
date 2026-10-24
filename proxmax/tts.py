"""Text-to-Speech: Bangla speech synthesis with edge-tts.

Microsoft Edge-এর neural ভয়েস ব্যবহার করে প্রতিটি অনূদিত segment-এর
জন্য আলাদা অডিও ক্লিপ তৈরি হয়। ``asyncio`` + semaphore দিয়ে একাধিক
segment একসঙ্গে (concurrently) generate হয় — sequential-এর তুলনায়
কয়েকগুণ দ্রুত। ভয়েস প্রথমবার রান করার সময় edge-tts নিজে থেকেই
সার্ভার থেকে অডিও নিয়ে আসে; লোকাল কোনো মডেল ডাউনলোড লাগে না।

Native Bangla (Bangladesh) ভয়েস:
    * ``bn-BD-NabanitaNeural`` — নারী কণ্ঠ (ডিফল্ট)
    * ``bn-BD-PradeepNeural``  — পুরুষ কণ্ঠ
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import List

from .models import Segment
from .utils import get_logger, get_media_duration

logger = get_logger(__name__)

BANGLA_VOICE_FEMALE = "bn-BD-NabanitaNeural"
BANGLA_VOICE_MALE = "bn-BD-PradeepNeural"


class TTSError(RuntimeError):
    """Raised when speech synthesis fails beyond recovery."""


class BanglaTTS:
    """Concurrent Bangla TTS over a list of translated segments."""

    def __init__(
        self,
        voice: str = BANGLA_VOICE_FEMALE,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        concurrency: int = 5,
        max_retries: int = 3,
    ) -> None:
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        self.concurrency = max(1, concurrency)
        self.max_retries = max(1, max_retries)

    # ------------------------------------------------------------------
    def synthesize_segments(self, segments: List[Segment], output_dir: Path | str) -> List[Segment]:
        """Generate one MP3 per segment (fills ``tts_audio_path``/duration).

        যেসব segment-এর ``text_bn`` খালি, সেগুলো বাদ যাবে। সব segment
        ব্যর্থ হলে :class:`TTSError` উঠবে; আংশিক ব্যর্থতায় বাকিগুলো
        নিয়ে কাজ এগিয়ে যাবে।
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        todo = [s for s in segments if (s.text_bn or "").strip()]
        if not todo:
            raise TTSError("TTS-এর জন্য কোনো বাংলা টেক্সট নেই (সব segment-এর text_bn খালি)।")

        logger.info(
            "TTS শুরু: %d segment | voice=%s | concurrency=%d …",
            len(todo), self.voice, self.concurrency,
        )
        started = time.perf_counter()
        self._run_async(self._synthesize_all(todo, output_dir))

        done = [s for s in todo if s.tts_audio_path and Path(s.tts_audio_path).is_file()]
        if not done:
            raise TTSError(
                "একটি segment-এরও অডিও তৈরি হয়নি। ইন্টারনেট সংযোগ ও edge-tts "
                "সার্ভিসের উপলব্ধতা যাচাই করুন।"
            )
        logger.info(
            "TTS শেষ: %d/%d segment-এর অডিও তৈরি (%.1fs)।",
            len(done), len(todo), time.perf_counter() - started,
        )
        return segments

    # ------------------------------------------------------------------
    @staticmethod
    def _run_async(coro) -> None:
        """Run a coroutine to completion — even inside Jupyter/Colab.

        Notebook kernel-এ ইতিমধ্যে একটা event loop চলতে থাকে, যেখানে
        ``asyncio.run()`` সরাসরি চলে না; সেক্ষেত্রে আলাদা thread-এ
        নতুন loop চালানো হয়।
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coro)
            return

        error_box: dict = {}

        def _runner() -> None:
            try:
                asyncio.run(coro)
            except BaseException as exc:  # re-raised in the calling thread
                error_box["error"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if "error" in error_box:
            raise error_box["error"]

    async def _synthesize_all(self, segments: List[Segment], output_dir: Path) -> None:
        import edge_tts  # lazy import: প্যাকেজ import হালকা থাকে

        semaphore = asyncio.Semaphore(self.concurrency)
        try:
            from tqdm.auto import tqdm

            progress = tqdm(total=len(segments), desc="  TTS", unit="seg", leave=False)
        except Exception:  # pragma: no cover
            progress = None

        async def run_one(seg: Segment) -> None:
            try:
                await self._synthesize_one(edge_tts, semaphore, seg, output_dir)
            finally:
                if progress is not None:
                    progress.update(1)

        try:
            await asyncio.gather(*(run_one(seg) for seg in segments))
        finally:
            if progress is not None:
                progress.close()

    async def _synthesize_one(self, edge_tts, semaphore: asyncio.Semaphore,
                              seg: Segment, output_dir: Path) -> None:
        text = " ".join(seg.text_bn.split())
        if len(text) > 800:
            logger.warning(
                "Segment %d-এর বাংলা টেক্সট অস্বাভাবিক লম্বা (%d অক্ষর) — "
                "TTS অডিও মূল সময়ে আঁটতে সমস্যা হতে পারে।", seg.index, len(text),
            )
        path = output_dir / f"seg_{seg.index:04d}.mp3"
        last_error: Exception | None = None

        async with semaphore:
            for attempt in range(1, self.max_retries + 1):
                try:
                    communicate = edge_tts.Communicate(
                        text, self.voice,
                        rate=self.rate, volume=self.volume, pitch=self.pitch,
                    )
                    await communicate.save(str(path))
                    if not path.is_file() or path.stat().st_size == 0:
                        raise TTSError("edge-tts খালি অডিও ফেরত দিয়েছে (NoAudioReceived)")
                    seg.tts_audio_path = path
                    # ffprobe একটি blocking subprocess — thread-এ চালানো হয়
                    seg.tts_duration = await asyncio.to_thread(get_media_duration, path)
                    return
                except Exception as exc:
                    last_error = exc
                    if attempt < self.max_retries:
                        await asyncio.sleep(1.0 * (2 ** (attempt - 1)))
        logger.warning("Segment %d-এর TTS ব্যর্থ: %s", seg.index, last_error)
        seg.tts_audio_path = None
        seg.tts_duration = None
