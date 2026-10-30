"""Final muxing: original video stream + new Bangla audio → MP4.

কৌশল: ভিডিও stream ``-c:v copy`` দিয়ে হুবহু কপি করা হয় (re-encode
নেই → প্রচুর সময় বাঁচে, quality একদম একই থাকে) এবং নতুন Bangla
অডিও AAC হিসেবে বসানো হয়। কোনো কোডেক MP4 কন্টেইনারে copy করা না
গেলে (যেমন .mkv-তে VP9) বা শেষ frame বাড়ানো দরকার হলে re-encode চলে —
আগে NVENC (GPU) চেষ্টা, তারপর libx264।
"""

from __future__ import annotations

import functools
import json
import math
import subprocess
from pathlib import Path

from .utils import (
    FFmpegError,
    ensure_tool,
    get_logger,
    get_media_duration,
    run_ffmpeg_with_progress,
)

logger = get_logger(__name__)


@functools.lru_cache(maxsize=1)
def _available_encoders() -> frozenset:
    """ffmpeg-এ উপলব্ধ সব encoder-এর নাম (একবারই যাচাই হয়)।"""
    ffmpeg = ensure_tool("ffmpeg")
    proc = subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-encoders"],
        stdin=subprocess.DEVNULL,
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

    def mux(self, video_path: Path | str, audio_wav: Path | str, output_path: Path | str,
            *, required_video_duration: float | None = None) -> Path:
        """Create the final dubbed MP4.

        Raises:
            FFmpegError: copy ও re-encode — দুটো পথই ব্যর্থ হলে।
        """
        video_path, audio_wav, output_path = Path(video_path), Path(audio_wav), Path(output_path)
        if required_video_duration is not None and (
            not math.isfinite(float(required_video_duration)) or float(required_video_duration) < 0
        ):
            raise ValueError("required_video_duration must be finite and non-negative")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = ensure_tool("ffmpeg")

        common = [
            "-map", "0:v:0",                 # মূল ভিডিওর picture
            "-map", "1:a:0",                 # নতুন Bangla অডিও
            "-c:a", "aac", "-b:a", self.audio_bitrate, "-ar", "44100",
            "-movflags", "+faststart",       # ওয়েব/প্লেয়ারে দ্রুত শুরু
        ]
        # Ordinary output remains the fast, lossless stream-copy path. If the
        # scheduler extends speech past the video, a real held-last-frame video
        # stream is required; an audio-only container extension is not enough.
        extension = 0.0
        if required_video_duration is not None:
            try:
                source_video_duration = self._video_stream_duration(video_path)
            except FFmpegError:
                # Some source containers do not expose a stream duration. The
                # copied MP4 will be checked below; use this only as an estimate.
                source_video_duration = self._duration_hint(video_path) or 0.0
            extension = max(0.0, float(required_video_duration) - source_video_duration)
        attempts = [] if extension > 0 else [
            {"label": "video copy (re-encode ছাড়া)", "video_args": ["-c:v", "copy"]}
        ]
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
        verify_retries = 0
        for attempt in attempts:
            cmd = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(video_path),
                "-i", str(audio_wav),
                *(["-vf", f"tpad=stop_mode=clone:stop_duration={extension + 0.25:.6f}"] if extension > 0 else []),
                *attempt["video_args"], *common,
                str(output_path),
            ]
            try:
                logger.info("Muxing চলছে (%s)…", attempt["label"])
                run_ffmpeg_with_progress(
                    cmd,
                    description=f"Muxing [{attempt['label']}]",
                    duration_seconds=max(required_video_duration or 0.0, self._duration_hint(video_path) or 0.0) or None,
                    stall_timeout=120.0,
                )
                self._verify(output_path)
                if required_video_duration is not None:
                    measured_video = self._video_stream_duration(output_path)
                    if measured_video + 1e-4 < float(required_video_duration):
                        # If an unknown source duration caused an optimistic copy,
                        # subsequent re-encode attempts add actual cloned frames.
                        extension += float(required_video_duration) - measured_video
                        if attempt is attempts[-1] and verify_retries < 2:
                            attempts.append(dict(attempt))
                            verify_retries += 1
                        raise FFmpegError(f"Picture stream is too short: {measured_video:.3f}s")
                logger.info("Muxing সম্পন্ন: %s", output_path.name)
                return output_path
            except Exception as exc:
                last_error = exc
                logger.warning("Muxing পদ্ধতি '%s' ব্যর্থ: %s", attempt["label"], str(exc)[:300])
                output_path.unlink(missing_ok=True)
        raise FFmpegError(f"সব muxing পদ্ধতি ব্যর্থ হয়েছে। শেষ error: {last_error}")

    @staticmethod
    def _video_stream_duration(path: Path) -> float:
        """Measure the picture stream, not just the (possibly audio-longer) container."""
        ffprobe = ensure_tool("ffprobe")
        try:
            proc = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=start_time,duration", "-of", "json", str(path)],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", timeout=30, check=False,
            )
            stream = json.loads(proc.stdout or "{}")["streams"][0]
            duration = float(stream["duration"])
            start = float(stream.get("start_time", 0))
            end = start + duration
            if proc.returncode or not all(map(math.isfinite, (start, duration, end))) or duration <= 0 or end <= 0:
                raise ValueError("invalid video stream timing")
            return end
        except (OSError, ValueError, KeyError, IndexError, TypeError, subprocess.TimeoutExpired) as exc:
            raise FFmpegError(f"Cannot verify picture-stream duration for {path.name}: {exc}") from exc

    @staticmethod
    def _duration_hint(video_path: Path):
        """Muxing progress-এর জন্য duration (না পেলে ``None``)।"""
        try:
            duration = get_media_duration(video_path)
        except Exception:
            return None
        return duration if duration and duration > 0 else None

    @staticmethod
    def _verify(output_path: Path) -> None:
        """Ensure the produced file exists and has a readable duration."""
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise FFmpegError(f"Muxing-এর আউটপুট তৈরি হয়নি: {output_path}")
        duration = get_media_duration(output_path)  # raises if unreadable
        if duration <= 0:
            raise FFmpegError(f"Muxing-এর আউটপুট অবৈধ (duration=0): {output_path}")
