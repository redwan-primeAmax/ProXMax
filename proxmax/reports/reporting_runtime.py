"""Bounded, private-by-design runtime event recording for processing reports."""

from __future__ import annotations

import contextvars
import logging
import math
import re
import threading
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .reporting_common import _md
from .reporting_resources import ResourceSampler

_CURRENT = contextvars.ContextVar("proxmax_report_trace", default=None)
_SEGMENTS = contextvars.ContextVar("proxmax_report_segments", default=None)
_LIMIT = 100_000
_FIELDS = {"segment", "first_segment", "last_segment", "count", "attempt", "max_attempts",
           "retry", "delay_seconds", "elapsed_seconds", "clip_seconds", "status", "error_type",
           "http_status", "encoder", "strategy", "source_start_frame", "source_end_frame"}
_STAGES = {1: "extraction", 2: "STT", 3: "translation + TTS", 4: "sync / picture preparation",
           5: "mux / output copy", 6: "report generation"}


def _safe_fields(data: dict) -> dict:
    """Keep only bounded scalar diagnostics, never messages, paths or text."""
    clean = {}
    for key, value in data.items():
        if key not in _FIELDS or value is None:
            continue
        if isinstance(value, bool):
            clean[key] = value
        elif isinstance(value, int) and abs(value) < 10**12:
            clean[key] = value
        elif isinstance(value, float) and math.isfinite(value) and abs(value) < 10**12:
            clean[key] = round(value, 3)
        elif isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_\-]{0,79}", value):
            clean[key] = value
    return clean


def record_event(code: str, **details: Any) -> None:
    trace = _CURRENT.get()
    bounds = _SEGMENTS.get()
    if trace is not None:
        if bounds is not None:
            details.setdefault("first_segment", bounds[0])
            details.setdefault("last_segment", bounds[1])
        trace.record(code, **details)


@contextmanager
def bind_segments(indexes):
    token = _SEGMENTS.set((min(indexes), max(indexes)) if indexes else None)
    try:
        yield
    finally:
        _SEGMENTS.reset(token)


@contextmanager
def bind_trace(trace):
    token = _CURRENT.set(trace)
    try:
        yield
    finally:
        _CURRENT.reset(token)


class _Warnings(logging.Handler):
    def __init__(self, trace):
        super().__init__(level=logging.WARNING)
        self.trace = trace

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._capture(record)
        except Exception:
            # Observability must never interrupt media processing; report the loss.
            with self.trace._lock:
                self.trace._capture_errors += 1

    def _capture(self, record: logging.LogRecord) -> None:
        owner = _CURRENT.get()
        if owner is not None and owner is not self.trace:
            return  # A late request belongs to an earlier video's trace.
        template = str(record.msg).lower()
        name = record.name
        code = "warning" if record.levelno < logging.ERROR else "error"
        if "retry" in template or "আবার চেষ্টা" in template or "আবার তৈরি" in template:
            code = "retry_or_recovery"
        elif "fallback" in template or "failed" in template or "ব্যর্থ" in template:
            code = "failure_or_fallback"
        details: dict[str, Any] = {}
        args = record.args if isinstance(record.args, tuple) else ()
        bounds = _SEGMENTS.get()
        if bounds is not None:
            details.update(first_segment=bounds[0], last_segment=bounds[1])
        if template.startswith("segment %d") and args and isinstance(args[0], int):
            details["segment"] = args[0]
        if "gemini api error" in template and len(args) >= 4:
            code = "gemini_api_error"
            details.update(attempt=args[0], max_attempts=args[1], error_type=args[2], http_status=args[3])
        elif "retry %d/%d" in template and len(args) >= 3:
            code = "gemini_retry_scheduled"
            details.update(delay_seconds=args[0], attempt=args[1], max_attempts=args[2])
        elif name.endswith("gemini_protocol") and "status=%s" in template and len(args) >= 2:
            code = "gemini_model_fallback"
            details["http_status"] = args[1]
        elif name.endswith("video_render") and template.startswith("video %s encoding via") and len(args) >= 2:
            code = "video_encoder_failed"
            details.update(strategy=args[0], encoder=args[1])
        elif name.endswith("video_render") and template.startswith("video start normalization") and args:
            code = "video_start_encoder_failed"
            details["encoder"] = args[0]
        elif name.endswith("muxer") and template.startswith("virtual source tail failed") and args:
            code = "virtual_tail_fallback"
            details["error_type"] = args[0]
        elif name.endswith("muxer") and template.startswith("muxing method") and args:
            code = "mux_encoder_failed"
            details["encoder"] = args[0]
        elif name.endswith("sync_prepare") and "slowdown" in template:
            code = "sync_slowdown_fallback"
            if len(args) >= 2 and isinstance(args[1], BaseException):
                details["error_type"] = type(args[1]).__name__
        self.trace.record(code, level=record.levelname, location=f"{record.module}:{record.lineno}",
                          unscoped=owner is None, **details)


class OperationalTrace:
    """Capture known events; it is not a general process or visual monitor."""

    def __init__(self, started: float | None = None) -> None:
        self.started = time.perf_counter() if started is None else started
        self._lock = threading.RLock()
        self._events: list[dict] = []
        self._stages: list[dict] = []
        self._dropped = 0
        self._capture_errors = 0
        self._active_stage: int | None = None
        self._handler = _Warnings(self)
        self._sampler = ResourceSampler(self.started)
        self._token = None
        self.segments = []
        self.source_duration = 0.0

    def activate(self) -> None:
        self._token = _CURRENT.set(self)
        logging.getLogger("proxmax").addHandler(self._handler)
        self._sampler.start()

    def close(self) -> None:
        self._sampler.close()
        logging.getLogger("proxmax").removeHandler(self._handler)
        self._handler.close()
        if self._token is not None:
            _CURRENT.reset(self._token)
            self._token = None

    def record(self, code: str, *, level: str = "INFO", location: str = "", unscoped: bool = False,
               **details: Any) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code):
            return
        with self._lock:
            if len(self._events) >= _LIMIT:
                self._dropped += 1
                return
            self._events.append({"time": max(0.0, time.perf_counter() - self.started),
                                 "stage": self._active_stage, "level": level,
                                 "code": code, "location": location, "unscoped": unscoped,
                                 "details": _safe_fields(details)})

    def start_stage(self, number: int) -> None:
        with self._lock:
            self._active_stage = number
            self._stages.append({"number": number, "name": _STAGES.get(number, "other"),
                                 "start": max(0.0, time.perf_counter() - self.started),
                                 "end": None, "status": "running"})
        self.record("stage_started", count=number)

    def end_stage(self, number: int, error: BaseException | None = None) -> None:
        self.record("stage_failed" if error else "stage_finished", count=number,
                    error_type=type(error).__name__ if error else None)
        with self._lock:
            for stage in reversed(self._stages):
                if stage["number"] == number and stage["end"] is None:
                    stage["end"] = max(0.0, time.perf_counter() - self.started)
                    stage["status"] = "failed" if error else "completed"
                    break
            self._active_stage = None

    def snapshot(self) -> dict:
        resources = self._sampler.snapshot()
        with self._lock:
            return {"elapsed": max(0.0, time.perf_counter() - self.started),
                    "stages": [dict(item) for item in self._stages],
                    "events": [dict(item) for item in self._events], "dropped": self._dropped,
                    "capture_errors": self._capture_errors, "resources": resources}


def _resource_bins(samples: list[dict]) -> dict[int, dict]:
    bins = {}
    previous = None
    for sample in samples:
        current = dict(sample)
        current["cpu_percent"] = None
        if previous is not None:
            elapsed = current["time"] - previous["time"]
            used = current["cpu_seconds"] - previous["cpu_seconds"]
            if elapsed > 0 and used >= 0:
                current["cpu_percent"] = 100.0 * used / elapsed
        bins[int(current["time"])] = current
        previous = current
    return bins


def _resource_value(sample: dict | None, key: str, suffix: str = "") -> str:
    if sample is None or sample.get(key) is None:
        return "unavailable"
    value = sample[key]
    return f"{value:.1f}{suffix}" if isinstance(value, float) else f"{value}{suffix}"


def runtime_lines(snapshot: dict) -> list[str]:
    elapsed = float(snapshot.get("elapsed", 0.0))
    stages, events = snapshot.get("stages", []), snapshot.get("events", [])
    resources = snapshot.get("resources", {})
    samples = resources.get("samples", [])
    resource_bins = _resource_bins(samples)
    lines = ["## Processing elapsed-time diagnostics", "",
             "This is wall-clock time since this video's processing began, **not source or output media time**. "
             "Workers can overlap. Events are observations from instrumented code and warnings, not all possible failures.",
             "Provider messages, request/response content, credentials and absolute paths are not copied into these event records. "
             "A warning's module:line identifies the source template; unscoped worker warnings may not identify a segment. "
             "Segment fields are raw segment indexes; media tables show the corresponding U-number and segment index.",
             f"Elapsed at snapshot: {elapsed:.3f} s; captured events: {len(events)}; "
             f"events omitted after the {_LIMIT:,} event cap: {snapshot.get('dropped', 0)}; "
             f"warning-capture errors: {snapshot.get('capture_errors', 0)}. "
             f"Resource samples: {len(samples)}, omitted: {resources.get('dropped', 0)}, "
             f"sampling errors: {resources.get('errors', 0)}.",
             "Approximately once per second, CPU is the parent process average since the preceding sample "
             "(may exceed 100%); parent RSS excludes FFmpeg children. Cgroup memory, when available, "
             "covers the whole group; system available memory is the host's /proc value, not a container limit. "
             "GPU/VRAM and child-process CPU are not sampled. This snapshot excludes the final report append and cleanup.", "",
             "| Stage | Start elapsed (s) | End elapsed (s) | Status |",
             "|---|---:|---:|---|"]
    for stage in stages:
        end = stage.get("end")
        lines.append(f"| {stage['number']} {_md(stage['name'])} | {stage['start']:.3f} "
                     f"| {end:.3f} | {stage['status']} |" if end is not None else
                     f"| {stage['number']} {_md(stage['name'])} | {stage['start']:.3f} | running | {stage['status']} |")
    bins = defaultdict(list)
    for event in events:
        bins[int(event["time"])].append(event)
    lines += ["", "### Every processing second", "",
              "| Elapsed interval (s) | Active stage(s) | Recorded event counts | Parent CPU | Parent RSS (MiB) | Cgroup usage/limit (MiB) | Host available (MiB) | Python threads |",
              "|---|---|---|---:|---:|---|---:|---:|"]
    # Include an event occurring exactly at the current elapsed boundary.
    last = max(math.ceil(elapsed), max(bins, default=-1) + 1, max(resource_bins, default=-1) + 1)
    for second in range(last):
        active = [f"{s['number']} {s['name']}" for s in stages
                  if s['start'] < second + 1 and (s['end'] is None or s['end'] > second)]
        counts = Counter(item['code'] for item in bins[second])
        detail = ", ".join(f"{key}: {value}" for key, value in sorted(counts.items())) or "none observed"
        sample = resource_bins.get(second)
        cgroup = f"{_resource_value(sample, 'cgroup_current_mib')}/{_resource_value(sample, 'cgroup_limit_mib')}"
        lines.append(f"| {second}–{min(second + 1, elapsed):.3f} | {_md(', '.join(active) or 'between stages')} "
                     f"| {_md(detail)} | {_resource_value(sample, 'cpu_percent', '%')} "
                     f"| {_resource_value(sample, 'parent_rss_mib')} | {cgroup} "
                     f"| {_resource_value(sample, 'system_available_mib')} | {_resource_value(sample, 'python_threads')} |")
    lines += ["", "### Recorded event details", "",
              "| Elapsed (s) | Stage | Level | Event | Safe details | Source |",
              "|---:|---|---|---|---|---|"]
    for event in sorted(events, key=lambda item: item['time']):
        details = ", ".join(f"{key}={value}" for key, value in sorted(event.get('details', {}).items())) or "—"
        source = event.get("location") or "instrumented"
        if event.get("unscoped"):
            source += " (unscoped worker)"
        lines.append(f"| {event['time']:.3f} | {_md(event.get('stage') or 'between stages')} "
                     f"| {_md(event['level'])} | {_md(event['code'])} | {_md(details)} | {_md(source)} |")
    if not events:
        lines.append("| — | — | — | none captured | — | — |")
    return lines + [""]


def append_runtime(report: Path, snapshot: dict) -> None:
    with Path(report).open("a", encoding="utf-8") as handle:
        handle.write("\n" + "\n".join(runtime_lines(snapshot)) + "\n")
