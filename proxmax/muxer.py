"""Mux dubbed audio with frame-accurate repeated/retimed picture windows."""

from __future__ import annotations

import functools
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

from .models import Segment
from .utils import FFmpegError, ensure_tool, get_logger, get_media_duration, run_ffmpeg_with_progress
from .video_render import _frame_count, normalize_video_start, render_video_plan
from .video_timeline import VideoPlan, build_video_plan, parse_fps

logger = get_logger(__name__)


@functools.lru_cache(maxsize=1)
def _available_encoders() -> frozenset:
    proc = subprocess.run(
        [ensure_tool("ffmpeg"), "-nostdin", "-hide_banner", "-encoders"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", check=False,
    )
    encoders = set()
    for line in (proc.stdout or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0][0] in "VASD" and "=" not in parts[0]:
            encoders.add(parts[1])
    return frozenset(encoders)


class VideoMuxer:
    """Render optional per-window video edits, then mux the Bangla audio."""

    def __init__(self, audio_bitrate: str = "160k", use_hwaccel: bool = True,
                 video_workers: int = 2) -> None:
        self.audio_bitrate = audio_bitrate
        self.use_hwaccel = bool(use_hwaccel)
        self.video_workers = max(1, min(4, int(video_workers)))

    def _gpu_encoder(self) -> bool:
        return self.use_hwaccel and "h264_nvenc" in _available_encoders()

    @staticmethod
    def video_frame_rate(path: Path | str) -> Fraction:
        proc = subprocess.run(
            [ensure_tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=avg_frame_rate,r_frame_rate", "-of", "json", str(path)],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", timeout=30, check=False,
        )
        try:
            stream = json.loads(proc.stdout or "{}")["streams"][0]
            for key in ("avg_frame_rate", "r_frame_rate"):
                try:
                    return parse_fps(stream.get(key, "0/0"))
                except ValueError:
                    continue
        except (ValueError, KeyError, IndexError, TypeError):
            pass
        raise FFmpegError(f"Cannot determine usable video frame rate: {Path(path).name}")

    @staticmethod
    def _video_start_offset(path: Path) -> float:
        """Video start relative to the container timeline used by extraction."""
        proc = subprocess.run(
            [ensure_tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=start_time:format=start_time", "-of", "json", str(path)],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", timeout=30, check=False,
        )
        try:
            data = json.loads(proc.stdout or "{}")
            stream = float(data["streams"][0].get("start_time", 0))
            container = float(data.get("format", {}).get("start_time", 0))
            if proc.returncode or not math.isfinite(stream) or not math.isfinite(container):
                raise ValueError("invalid start timing")
            return max(0.0, stream - container)
        except (ValueError, KeyError, IndexError, TypeError):
            logger.warning("Cannot determine video start offset; assuming zero for %s.", path.name)
            return 0.0

    def picture_timeline_duration(self, source: Path | str, fps: Fraction) -> float:
        """Actual picture frame coverage including any late-start interval."""
        path = Path(source)
        offset_frames = max(0, round(self._video_start_offset(path) * float(fps)))
        return (_frame_count(path) + offset_frames) / float(fps)

    def _tail_plan(self, fps: Fraction, source_duration: float, required_duration: float) -> VideoPlan:
        source_frames = max(1, round(source_duration * float(fps)))
        dummy = Segment(0, 0.0, source_duration, stretched_duration=required_duration)
        return build_video_plan([dummy], [(0, source_frames)], source_frames, fps, 24000, 1.0)

    def ensure_picture_duration(self, source: Path | str, required_duration: float,
                                work_dir: Path | str, fps: Fraction) -> Path:
        """If source audio outlasts its picture, repeat the picture tail first.

        This rare preparation step makes original STT timestamps address a real
        picture frame, rather than silently creating an audio-only extension.
        """
        source = Path(source)
        offset = self._video_start_offset(source)
        blank_frames = max(0, round(offset * float(fps)))
        if blank_frames:
            aligned = Path(work_dir) / "source_picture_aligned.mp4"
            logger.warning("Source picture starts %.3fs late; preserving the offset with %d blank frames.",
                           offset, blank_frames)
            source = normalize_video_start(source, aligned, fps, blank_frames, use_nvenc=self._gpu_encoder())
        # Metadata may be N/A or include a misleading stream start time. For
        # the normalized CFR grid, the actual frame count is a safer limit than
        # a container/stream duration estimate and is normally header-only.
        try:
            picture = _frame_count(source) / float(fps)
        except FFmpegError:
            picture = self._video_stream_duration(source)
            logger.warning("Frame count unavailable; picture duration uses stream metadata.")
        if required_duration <= picture + 0.5 / float(fps):
            return source
        plan = self._tail_plan(fps, picture, required_duration)
        output = Path(work_dir) / "source_picture_extended.mp4"
        logger.warning("Source picture ends %.3fs before its container; repeating its last pictures.",
                       required_duration - picture)
        render_video_plan(source, plan, output, workers=self.video_workers, use_nvenc=self._gpu_encoder())
        return output

    def mux(self, video_path: Path | str, audio_wav: Path | str, output_path: Path | str,
            *, required_video_duration: float | None = None, video_plan: VideoPlan | None = None) -> Path:
        video_path, audio_wav, output_path = Path(video_path), Path(audio_wav), Path(output_path)
        if required_video_duration is not None and (
            not math.isfinite(float(required_video_duration)) or float(required_video_duration) < 0
        ):
            raise ValueError("required_video_duration must be finite and non-negative")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if video_plan is not None and video_plan.changed:
            render_video_plan(video_path, video_plan, output_path, audio=audio_wav,
                              audio_bitrate=self.audio_bitrate, workers=self.video_workers,
                              use_nvenc=self._gpu_encoder())
            self._verify_picture(output_path, required_video_duration)
            return output_path

        # Without a supplied edit plan, retain the lossless-copy fast path when
        # possible; if extra picture is required, repeat a video tail instead of
        # freezing the final frame. The output is always checked afterwards.
        picture = None
        if required_video_duration is not None:
            try:
                picture = self._video_stream_duration(video_path)
            except FFmpegError:
                picture = self._duration_hint(video_path)
        if picture is not None and required_video_duration is not None and picture + 1e-4 < required_video_duration:
            return self._extend_and_mux(video_path, audio_wav, output_path, picture, required_video_duration)

        ffmpeg = ensure_tool("ffmpeg")
        common = ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", self.audio_bitrate,
                  "-ar", "44100", "-movflags", "+faststart"]
        attempts = [["-c:v", "copy"]]
        if self._gpu_encoder():
            attempts.append(["-c:v", "h264_nvenc", "-preset", "fast", "-cq", "23", "-pix_fmt", "yuv420p"])
        attempts.append(["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"])
        last_error: Exception | None = None
        for codec in attempts:
            try:
                cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
                       "-i", str(audio_wav), *codec, *common, str(output_path)]
                run_ffmpeg_with_progress(cmd, description="Mux video and dubbed audio",
                                         duration_seconds=self._duration_hint(video_path), stall_timeout=120.0)
                self._verify(output_path)
                if required_video_duration is not None:
                    actual = self._video_stream_duration(output_path)
                    if actual + 1e-4 < required_video_duration:
                        output_path.unlink(missing_ok=True)
                        return self._extend_and_mux(video_path, audio_wav, output_path, actual,
                                                    required_video_duration)
                return output_path
            except Exception as exc:
                last_error = exc
                logger.warning("Muxing method %s failed: %s", codec[1], str(exc)[:300])
                output_path.unlink(missing_ok=True)
        raise FFmpegError(f"All muxing methods failed: {last_error}")

    def _extend_and_mux(self, source: Path, audio: Path, output: Path,
                        source_duration: float, required: float) -> Path:
        fps = self.video_frame_rate(source)
        try:
            # A duration hint can be too short: use all actual source frames
            # so recovery never discards the end of the original picture.
            source_duration = _frame_count(source) / float(fps)
        except FFmpegError:
            logger.warning("Frame count unavailable; using picture duration hint for tail repetition.")
        plan = self._tail_plan(fps, source_duration, required)
        render_video_plan(source, plan, output, audio=audio, audio_bitrate=self.audio_bitrate,
                          workers=self.video_workers, use_nvenc=self._gpu_encoder())
        self._verify_picture(output, required)
        return output

    def _verify_picture(self, output: Path, required: float | None) -> None:
        self._verify(output)
        if required is not None and self._video_stream_duration(output) + 1e-4 < required:
            output.unlink(missing_ok=True)
            raise FFmpegError("Edited picture stream ends before dubbed speech")

    @staticmethod
    def _video_stream_duration(path: Path) -> float:
        ffprobe = ensure_tool("ffprobe")
        try:
            proc = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=start_time,duration", "-of", "json", str(path)],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", timeout=30, check=False,
            )
            stream = json.loads(proc.stdout or "{}")["streams"][0]
            duration, start = float(stream["duration"]), float(stream.get("start_time", 0))
            end = start + duration
            if proc.returncode or not all(map(math.isfinite, (start, duration, end))) or duration <= 0 or end <= 0:
                raise ValueError("invalid video stream timing")
            return end
        except (OSError, ValueError, KeyError, IndexError, TypeError, subprocess.TimeoutExpired) as exc:
            raise FFmpegError(f"Cannot verify picture-stream duration for {path.name}: {exc}") from exc

    @staticmethod
    def _duration_hint(video_path: Path):
        try:
            duration = get_media_duration(video_path)
        except Exception:
            return None
        return duration if duration and duration > 0 else None

    @staticmethod
    def _verify(output_path: Path) -> None:
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise FFmpegError(f"Muxing output was not created: {output_path}")
        if get_media_duration(output_path) <= 0:
            raise FFmpegError(f"Invalid muxing output duration: {output_path}")
