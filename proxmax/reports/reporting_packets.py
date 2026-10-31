"""Bounded ffprobe packet-timestamp measurements for per-second picture reports."""

from __future__ import annotations

import csv
import io
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .utils import ensure_tool

# Limits RAM and ffprobe output for very long recordings; truncation is explicit.
_MAX_PACKETS = 500_000


def picture_packet_seconds(video: Path) -> dict[str, Any]:
    """Count video packets by presentation second; this does not decode frames.

    Packet PTS is measured relative to the container start. A video packet is
    not guaranteed to correspond one-to-one to a visible frame for all codecs.
    """
    started = time.perf_counter()
    if not video.is_file():
        return {"status": "unavailable", "reason": "video file not found", "bins": {}}
    try:
        tool = ensure_tool("ffprobe")
        origin_proc = subprocess.run(
            [tool, "-v", "error", "-show_entries", "format=start_time", "-of", "default=nw=1:nk=1", str(video)],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=30, check=True,
        )
        try:
            origin = float(origin_proc.stdout.strip())
            if not math.isfinite(origin):
                raise ValueError("nonfinite")
            origin_status = "measured"
        except ValueError:
            origin, origin_status = 0.0, "assumed_zero"
        proc = subprocess.run(
            [tool, "-v", "error", "-select_streams", "v:0", "-read_intervals", f"%+#{_MAX_PACKETS + 1}",
             "-show_entries", "packet=pts_time,duration_time,flags", "-of", "csv=p=0", str(video)],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=90, check=True,
        )
        rows = [row for row in csv.reader(io.StringIO(proc.stdout)) if row and any(row)]
        if not rows:
            raise ValueError("no video packets")
        partial = len(rows) > _MAX_PACKETS
        if partial:
            rows = rows[:_MAX_PACKETS]
        bins: dict[int, dict[str, Any]] = defaultdict(lambda: {"count": 0, "keyframes": 0,
                                                                "first_pts": float("inf"), "last_pts": -float("inf")})
        missing_pts = before_zero = missing_duration = 0
        last_end = 0.0
        for row in rows:
            try:
                pts = float(row[0]) - origin
                if not math.isfinite(pts):
                    raise ValueError("nonfinite")
            except (IndexError, TypeError, ValueError):
                missing_pts += 1
                continue
            if pts < -0.001:
                before_zero += 1
                continue
            pts = max(0.0, pts)
            try:
                duration = float(row[1])
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError("invalid duration")
            except (IndexError, TypeError, ValueError):
                duration = 0.0
                missing_duration += 1
            last_end = max(last_end, pts + duration)
            entry = bins[int(math.floor(pts + 1e-7))]
            entry["count"] += 1
            entry["keyframes"] += int(len(row) > 2 and "K" in row[2])
            entry["first_pts"] = min(entry["first_pts"], pts)
            entry["last_pts"] = max(entry["last_pts"], pts)
        if not bins:
            raise ValueError("no usable picture packet timestamps")
        return {"status": "partial" if partial else "ok", "bins": dict(bins),
                "scanned_packets": len(rows), "missing_pts": missing_pts,
                "before_origin": before_zero, "missing_duration": missing_duration,
                "end": last_end if not partial and not missing_pts and not missing_duration else None,
                "last_scanned_second": max(bins), "origin": origin_status,
                "scan_seconds": time.perf_counter() - started}
    except Exception as exc:
        # Never include stderr, media paths, URLs or container metadata in an error.
        return {"status": "unavailable", "reason": type(exc).__name__, "bins": {},
                "scan_seconds": time.perf_counter() - started}
