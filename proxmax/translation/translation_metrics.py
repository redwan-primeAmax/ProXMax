"""Thread-safe, secret-free, per-video Gemini performance diagnostics."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any

from .translation_errors import _status_code
from .reporting_runtime import bind_segments, record_event


def measured_session(method):
    """Bind one public translation operation to the current video generation."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        segments = args[0] if method.__name__ == "translate_segments" and args else []
        indexes = [int(s.index) for s in segments] if segments else []
        started = time.perf_counter()
        if indexes:
            record_event("translation_group_started", first_segment=min(indexes),
                         last_segment=max(indexes), count=len(indexes))
        error = None
        try:
            with self.diagnostics.session(), bind_segments(indexes):
                return method(self, *args, **kwargs)
        except BaseException as exc:
            error = exc
            raise
        finally:
            if indexes:
                record_event("translation_group_finished", first_segment=min(indexes),
                             last_segment=max(indexes), count=len(indexes),
                             elapsed_seconds=time.perf_counter() - started,
                             status="failed" if error else "success",
                             error_type=type(error).__name__ if error else None)
    return wrapped


class TranslationMetrics:
    """Isolate late requests after reset; concurrent durations are sums, not wall time."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._local = threading.local()
        self._current: dict[str, Any] = {}
        self.reset()

    def reset(self) -> None:
        """Start a new generation without waiting for uninterruptible HTTP calls.

        A thread already translating retains its old counters until it finishes;
        it cannot pollute the next video's results. Its cancellation token still
        prevents it from starting new attempts or delivering a late response.
        """
        with self._lock:
            prior_active = int(self._current.get("_active", 0))
            self._current = {
                "attempts": 0, "retries": 0, "failures": 0,
                "failure_types": {}, "attempt_seconds_sum": 0.0,
                "pacing_wait_seconds_sum": 0.0, "retry_wait_seconds_sum": 0.0,
                "quality_recheck_units": 0, "cache_hits": 0,
                "peak_in_flight": 0, "prior_in_flight_at_reset": prior_active,
                "_active": 0,
            }

    @contextmanager
    def session(self):
        old = getattr(self._local, "data", None)
        if old is None:
            with self._lock:
                self._local.data = self._current
        try:
            yield
        finally:
            if old is None:
                del self._local.data

    def _data(self) -> dict[str, Any]:
        return getattr(self._local, "data", None) or self._current

    def begin(self, retry: bool) -> dict[str, Any]:
        with self._lock:
            data = self._data()
            data["attempts"] += 1
            data["retries"] += int(retry)
            data["_active"] += 1
            data["peak_in_flight"] = max(data["peak_in_flight"], data["_active"])
            return data

    def end(self, token: dict[str, Any], seconds: float, error: BaseException | None) -> None:
        with self._lock:
            token["_active"] -= 1
            token["attempt_seconds_sum"] += max(0.0, seconds)
            if error is not None:
                token["failures"] += 1
                status = _status_code(error)
                # Never store response messages, transcript text, credentials or paths.
                kind = f"HTTP_{status}" if status is not None else type(error).__name__
                counts = token["failure_types"]
                counts[kind] = counts.get(kind, 0) + 1

    def add(self, key: str, value: int | float) -> None:
        with self._lock:
            self._data()[key] += value

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            data = {key: dict(value) if isinstance(value, dict) else value
                    for key, value in self._current.items() if key != "_active"}
            data["in_flight"] = self._current["_active"]
        for key in ("attempt_seconds_sum", "pacing_wait_seconds_sum", "retry_wait_seconds_sum"):
            data[key] = round(float(data[key]), 3)
        return data
