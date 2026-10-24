"""Audio extraction: pull a 16 kHz mono WAV track out of a video.

ffmpeg দিয়ে ভিডিও থেকে অডিও আলাদা করে Whisper-এর পছন্দের ফরম্যাটে
(16 kHz, mono, PCM WAV) রূপান্তর করা হয় — এতে STT দ্রুত ও নির্ভুল হয়।
"""

from __future__ import annotations

from pathlib import Path

from .utils import FFmpegError, ensure_tool, get_logger, run_ffmpeg

logger = get_logger(__name__)


class AudioExtractor:
    """Extracts Whisper-ready WAV audio from video files via ffmpeg."""

    def __init__(self, use_hwaccel: bool = True, sample_rate: int = 16000) -> None:
        """
        Args:
            use_hwaccel: ``-hwaccel auto`` চেষ্টা করা হবে; ব্যর্থ হলে
                স্বয়ংক্রিয়ভাবে software decoding-এ fallback হবে।
            sample_rate: Output sample rate (Whisper-এর জন্য 16000)।
        """
        self.use_hwaccel = use_hwaccel
        self.sample_rate = sample_rate

    def extract(self, video_path: Path | str, output_wav: Path | str) -> Path:
        """Extract audio from ``video_path`` into ``output_wav``.

        Raises:
            FFmpegError: ভিডিওটি corrupt হলে বা এতে কোনো অডিও stream
                না থাকলে।
        """
        video_path, output_wav = Path(video_path), Path(output_wav)
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = ensure_tool("ffmpeg")

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

        try:
            run_ffmpeg(build_cmd(self.use_hwaccel), description="Audio extraction")
        except FFmpegError:
            if not self.use_hwaccel:
                raise
            # কিছু ffmpeg build-এ hwaccel ব্যর্থ হতে পারে — software পথে আবার চেষ্টা
            logger.warning("hwaccel দিয়ে extraction ব্যর্থ; software decoding দিয়ে আবার চেষ্টা হচ্ছে…")
            run_ffmpeg(build_cmd(False), description="Audio extraction (software)")

        # WAV header মাত্র 44 byte; এর চেয়ে বড় না হলে কার্যত কোনো অডিও নেই
        if not output_wav.is_file() or output_wav.stat().st_size <= 44:
            raise FFmpegError(
                f"এই ভিডিও থেকে কোনো অডিও পাওয়া যায়নি (অডিও stream নেই বা ফাইল corrupt): "
                f"{video_path.name}"
            )
        logger.info("অডিও extract হয়েছে: %s", output_wav.name)
        return output_wav
