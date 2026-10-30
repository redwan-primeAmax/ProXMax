"""In-memory reuse of unchanged, already-prepared sync clips within one video."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .models import Segment


def _file_signature(path: Path | str | None) -> Optional[Tuple[Any, ...]]:
    if not path:
        return None
    try:
        file = Path(path).resolve()
        stat = file.stat()
        if not file.is_file():
            return None
        # ctime also catches content changes when callers restore mtime manually.
        return (str(file), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    except (OSError, ValueError):
        return None


class PreparedClipCache:
    """Only successful artifacts are reused; placement is always recomputed."""

    def __init__(self) -> None:
        self._directory: Optional[Path] = None
        self._entries: Dict[str, tuple] = {}

    def begin(self, directory: Path) -> None:
        directory = directory.resolve()
        if directory != self._directory:
            self._directory = directory
            self._entries.clear()

    @staticmethod
    def key(segment: Segment, window: Tuple[float, float], settings: tuple) -> Optional[tuple]:
        source = _file_signature(segment.tts_audio_path)
        return (source, tuple(window), settings) if source else None

    def restore(self, segment: Segment, out_path: Path, key: Optional[tuple]) -> bool:
        entry = self._entries.get(str(out_path.resolve()))
        if not key or not entry or entry[0] != key or entry[1] != _file_signature(out_path):
            return False
        metadata = entry[2]
        segment.tts_duration = metadata["tts_duration"]
        segment.stretched_audio_path = out_path
        segment.stretched_duration = metadata["stretched_duration"]
        segment.tempo_used = metadata["tempo_used"]
        segment.sync_overrun_before_seconds = metadata["sync_overrun_before_seconds"]
        segment.sync_trimmed_head_seconds = metadata["sync_trimmed_head_seconds"]
        segment.sync_trimmed_tail_seconds = metadata["sync_trimmed_tail_seconds"]
        return True

    def store(self, segment: Segment, out_path: Path, key: Optional[tuple], settings: tuple, window: tuple) -> None:
        artifact = _file_signature(out_path)
        if not key or not artifact or segment.stretched_audio_path != out_path:
            return
        if key != self.key(segment, window, settings):  # source changed during preparation
            return
        metadata = {
            name: getattr(segment, name) for name in (
                "tts_duration", "stretched_duration", "tempo_used",
                "sync_overrun_before_seconds", "sync_trimmed_head_seconds", "sync_trimmed_tail_seconds",
            )
        }
        self._entries[str(out_path.resolve())] = (key, artifact, metadata)
