"""Low-overhead, best-effort process/resource samples for report timelines."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

_MAX_SAMPLES = 100_000


def _proc_kib(path: str, key: str) -> float | None:
    try:
        for line in Path(path).read_text(encoding="ascii").splitlines():
            if line.startswith(key + ":"):
                return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return None


def _cgroup_mib(name: str) -> float | None:
    try:
        raw = (Path("/sys/fs/cgroup") / name).read_text(encoding="ascii").strip()
        return int(raw) / (1024.0 * 1024.0) if raw.isdigit() else None
    except (OSError, ValueError):
        return None


class ResourceSampler:
    """Sample parent process and Linux memory once a second, without a dependency.

    CPU and RSS exclude FFmpeg child processes. cgroup memory, when available,
    covers the whole group rather than just this video; GPU usage is not read.
    """

    def __init__(self, started: float) -> None:
        self.started = started
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._samples: list[dict[str, Any]] = []
        self._dropped = 0
        self._thread: threading.Thread | None = None
        self._errors = 0

    def _capture(self) -> None:
        try:
            sample = {"time": max(0.0, time.perf_counter() - self.started),
                      "cpu_seconds": time.process_time(),
                      "parent_rss_mib": _proc_kib("/proc/self/status", "VmRSS"),
                      "system_available_mib": _proc_kib("/proc/meminfo", "MemAvailable"),
                      "cgroup_current_mib": _cgroup_mib("memory.current"),
                      "cgroup_limit_mib": _cgroup_mib("memory.max"),
                      "python_threads": threading.active_count()}
            with self._lock:
                if len(self._samples) < _MAX_SAMPLES:
                    self._samples.append(sample)
                else:
                    self._dropped += 1
        except Exception:
            with self._lock:
                self._errors += 1

    def start(self) -> None:
        self._capture()
        def sample_loop() -> None:
            while not self._stop.wait(1.0):
                self._capture()
        try:
            self._thread = threading.Thread(target=sample_loop, name="proxmax-report-sampler", daemon=True)
            self._thread.start()
        except Exception:
            with self._lock:
                self._errors += 1

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"samples": [dict(item) for item in self._samples],
                    "dropped": self._dropped, "errors": self._errors}
