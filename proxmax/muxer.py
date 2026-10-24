"""Final muxing: original video stream + new Bangla audio → MP4.

কৌশল: ভিডিও stream ``-c:v copy`` দিয়ে হুবহু কপি করা হয় (re-encode
নেই → প্রচুর সময় বাঁচে, quality একদম একই থাকে) এবং নতুন Bangla
অডিও AAC হিসেবে বসানো হয়। কোনো কোডেক MP4 কন্টেইনারে copy করা না
গেলে (যেমন .mkv-তে VP9) স্বয়ংক্রিয়ভাবে re-encode fallback চলে —
আগে NVENC (GPU) চেষ্টা, তারপর libx264।
"""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path

from .utils import FFmpegError, ensure_tool, get_logger, get_media_duration, run_ffmpeg

logger = get_logger(__name__)


@functools.lru_cache(maxsize=1)
def _available_encoders() -> frozenset:
    """ffmpeg-এ উপলব্ধ সব encoder-এর নাম (একবারই যাচাই হয়)।"""
    ffmpeg = ensure_tool("ffmpeg")
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    encoders = set()
    for line in (proc.stdout or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0][0] in "VASD" and "=" not in parts[0]:
            encoders.add(parts[1])
    return frozenset(encoders)


class VideoMuxer:
    """Muxes the original picture with the dubbed Bangla audio track."""

    def __init__(self, audio_bitrate: str = "160k", use_hwaccel: bool = True) -> None:
        self.audio_bitrate = audio_bitrate
        self.use_hwaccel = use_hwaccel

    def mux(self, video_path: Path | str, audio_wav: Path | str, output_path: Path | str) -> Path:
        """Create the final dubbed MP4.

        Raises:
            FFmpegError: copy ও re-encode — দুটো পথই ব্যর্থ হলে।
        """
        video_path, audio_wav, output_path = Path(video_path), Path(audio_wav), Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = ensure_tool("ffmpeg")

        common = [
            "-map", "0:v:0",                 # মূল ভিডিওর picture
            "-map", "1:a:0",                 # নতুন Bangla অডিও
            "-c:a", "aac", "-b:a", self.audio_bitrate, "-ar", "44100",
            "-movflags", "+faststart",       # ওয়েব/প্লেয়ারে দ্রুত শুরু
        ]
        attempts = [{"label": "video copy (re-encode ছাড়া)", "video_args": ["-c:v", "copy"]}]
        if self.use_hwaccel and "h264_nvenc" in _available_encoders():
            attempts.append({
                "label": "NVENC (GPU) re-encode",
                "video_args": ["-c:v", "h264_nvenc", "-preset", "fast", "-cq", "23",
                               "-pix_fmt", "yuv420p"],
            })
        attempts.append({
            "label": "libx264 re-encode",
            "video_args": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                           "-pix_fmt", "yuv420p"],
        })

        last_error: Exception | None = None
        for attempt in attempts:
            cmd = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(video_path),
                "-i", str(audio_wav),
                *attempt["video_args"], *common,
                str(output_path),
            ]
            try:
                logger.info("Muxing চলছে (%s)…", attempt["label"])
                run_ffmpeg(cmd, description=f"Muxing [{attempt['label']}]")
                self._verify(output_path)
                logger.info("Muxing সম্পন্ন: %s", output_path.name)
                return output_path
            except Exception as exc:
                last_error = exc
                logger.warning("Muxing পদ্ধতি '%s' ব্যর্থ: %s", attempt["label"], str(exc)[:300])
                output_path.unlink(missing_ok=True)
        raise FFmpegError(f"সব muxing পদ্ধতি ব্যর্থ হয়েছে। শেষ error: {last_error}")

    @staticmethod
    def _verify(output_path: Path) -> None:
        """Ensure the produced file exists and has a readable duration."""
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise FFmpegError(f"Muxing-এর আউটপুট তৈরি হয়নি: {output_path}")
        duration = get_media_duration(output_path)  # raises if unreadable
        if duration <= 0:
            raise FFmpegError(f"Muxing-এর আউটপুট অবৈধ (duration=0): {output_path}")
