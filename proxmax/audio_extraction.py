"""Audio extraction: pull a 16 kHz mono WAV track out of a video.

ffmpeg দিয়ে ভিডিও থেকে অডিও আলাদা করে Whisper-এর পছন্দের ফরম্যাটে
(16 kHz, mono, PCM WAV) রূপান্তর করা হয় — এতে STT দ্রুত ও নির্ভুল হয়।

কাজটা যেন "ঝুলে না যায়", সেজন্য তিনটি সুরক্ষা আছে:

* ``-nostdin`` + stdin বন্ধ — ffmpeg কখনো কী-ইনপুটের অপেক্ষায় আটকায় না,
* ``-progress`` থেকে **লাইভ শতাংশ** লগ হয় (১০ সেকেন্ড পরপর, কেবল বদলালে),
* কোনো অগ্রগতি ছাড়াই দীর্ঘ সময় কাটলে প্রসেস kill হয়ে software
  decoding দিয়ে আবার চেষ্টা করা হয় (hardware decoding-এ আটকে যাওয়া
  Colab-এ বেশ সাধারণ)।
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .utils import (
    FFmpegError,
    ensure_tool,
    get_logger,
    get_media_duration,
    run_ffmpeg_with_progress,
)

logger = get_logger(__name__)


class AudioExtractor:
    """Extracts Whisper-ready WAV audio from video files via ffmpeg."""

    def __init__(
        self,
        use_hwaccel: bool = False,
        sample_rate: int = 16000,
        stall_timeout: float = 90.0,
    ) -> None:
        """
        Args:
            use_hwaccel: ``-hwaccel auto`` ব্যবহার করা হবে কি না। ডিফল্ট
                ``False`` — ভিডিও stream বাদ দেওয়া হয় (``-vn``), তাই
                hardware decoding-এ কোনো লাভ নেই, কেবল আটকে যাওয়ার ঝুঁকি।
            sample_rate: Output sample rate (Whisper-এর জন্য 16000)।
            stall_timeout: এত সেকেন্ড অগ্রগতি না হলে ঝুলে গেছে ধরা হবে।
        """
        self.use_hwaccel = use_hwaccel
        self.sample_rate = sample_rate
        self.stall_timeout = float(stall_timeout)
        #: সর্বশেষ extraction-এ জানা ভিডিওর দৈর্ঘ্য (সেকেন্ড) — না জানলে ``None``
        self.last_duration: Optional[float] = None

    def extract(
        self,
        video_path: Path | str,
        output_wav: Path | str,
        duration_seconds: Optional[float] = None,
    ) -> Path:
        """Extract audio from ``video_path`` into ``output_wav``.

        Args:
            video_path: ইনপুট ভিডিও।
            output_wav: আউটপুট WAV পাথ (প্যারেন্ট ফোল্ডার নিজে তৈরি হয়)।
            duration_seconds: আগেই জানা থাকলে দিন — দুবার probe করতে হয় না।

        Returns:
            তৈরি হওয়া WAV ফাইলের :class:`Path`।

        Raises:
            FFmpegError: ভিডিওটি corrupt হলে, এতে কোনো অডিও stream না
                থাকলে, বা ffmpeg বারবার ঝুলে গেলে।
        """
        video_path, output_wav = Path(video_path), Path(output_wav)
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = ensure_tool("ffmpeg")

        # আগেই duration জেনে নেওয়া হয় — progress শতাংশ দেখাতে কাজে লাগে
        duration = (
            float(duration_seconds)
            if duration_seconds and duration_seconds > 0
            else self._probe_duration(video_path)
        )
        self.last_duration = duration

        def build_cmd(hwaccel: bool) -> list[str]:
            cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
            if hwaccel:
                cmd += ["-hwaccel", "auto"]
            cmd += [
                "-i", str(video_path),
                "-vn",                       # ভিডিও stream বাদ
                "-ac", "1",                  # mono
                "-ar", str(self.sample_rate),
                "-c:a", "pcm_s16le",
                str(output_wav),
            ]
            return cmd

        attempts = [self.use_hwaccel]
        if self.use_hwaccel:
            attempts.append(False)           # ব্যর্থ/ঝুলে গেলে software পথ

        last_error: Optional[Exception] = None
        for index, hwaccel in enumerate(attempts):
            label = "অডিও extract (hwaccel)" if hwaccel else "অডিও extract"
            logger.info(
                "  %s শুরু — %s%s", label,
                (f"{duration:.0f}s ভিডিও" if duration else video_path.name),
                "… (hardware decoding-এ আটকে গেলে নিজেই software-এ নামবে)"
                if hwaccel else "…",
            )
            try:
                run_ffmpeg_with_progress(
                    build_cmd(hwaccel),
                    description=label,
                    duration_seconds=duration,
                    stall_timeout=self.stall_timeout,
                )
                last_error = None
                break
            except FFmpegError as exc:
                last_error = exc
                if index == len(attempts) - 1:
                    break
                logger.warning(
                    "hwaccel দিয়ে extraction ব্যর্থ/ঝুলে গেছে (%s) — "
                    "software decoding দিয়ে আবার চেষ্টা হচ্ছে…", exc,
                )

        if last_error is not None:
            raise last_error

        # WAV header মাত্র 44 byte; এর চেয়ে বড় না হলে কার্যত কোনো অডিও নেই
        if not output_wav.is_file() or output_wav.stat().st_size <= 44:
            raise FFmpegError(
                f"এই ভিডিও থেকে কোনো অডিও পাওয়া যায়নি (অডিও stream নেই বা ফাইল corrupt): "
                f"{video_path.name}"
            )
        size_mb = output_wav.stat().st_size / (1024 * 1024)
        logger.info("অডিও extract হয়েছে: %s (%.1f MB)", output_wav.name, size_mb)
        return output_wav

    @staticmethod
    def _probe_duration(video_path: Path) -> Optional[float]:
        """duration জানার চেষ্টা (না পারলে ``None`` — progress শুধু সময় দেখাবে)।"""
        try:
            duration = get_media_duration(video_path)
        except Exception as exc:
            logger.debug("duration জানা যায়নি (%s): %s", video_path.name, exc)
            return None
        if duration and duration > 0:
            return float(duration)
        return None
