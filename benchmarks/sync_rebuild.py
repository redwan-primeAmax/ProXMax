"""Synthetic, offline comparison of fresh versus cached sync-retry preparation.

Run from the repository root: python benchmarks/sync_rebuild.py
Requires numpy and ffmpeg. Temporary synthetic WAV files are deleted afterward;
no Gemini, Edge, Whisper, network, or user's media is involved.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from proxmax import audio_fx  # noqa: E402
from proxmax.models import Segment  # noqa: E402
from proxmax.sync_engine import SyncEngine  # noqa: E402


def audio(seconds: float, frequency: int, rate: int = 24000) -> np.ndarray:
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (.15 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=int, default=171)
    parser.add_argument("--changed", type=int, default=53)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.changed <= args.clips <= 1000 or args.workers < 1:
        parser.error("require 1 <= changed <= clips <= 1000 and workers >= 1")
    logging.disable(logging.CRITICAL)

    with tempfile.TemporaryDirectory(prefix="proxmax-sync-bench-") as tmp:
        root = Path(tmp)
        segments = []
        for index in range(args.clips):
            path = root / f"source-{index:04d}.wav"
            audio_fx.write_wav(path, audio(.68 + (index % 6) * .07, 250 + index % 80), 24000)
            segments.append(Segment(index, index * 1.2, index * 1.2 + .90,
                                    f"Line {index}", "বাংলা", tts_audio_path=path))
        duration = args.clips * 1.2 + .2
        cached = SyncEngine(max_workers=args.workers, prefer_rubberband=False)
        started = time.perf_counter()
        cached.build_final_track(segments, duration, root / "cached", root / "initial.wav")
        initial = time.perf_counter() - started

        changed = np.linspace(0, args.clips - 1, args.changed, dtype=int)
        for index in changed:
            audio_fx.write_wav(segments[index].tts_audio_path,
                               audio(.58 + (index % 6) * .06, 250 + index % 80), 24000)
        started = time.perf_counter()
        cached.build_final_track(segments, duration, root / "cached", root / "cached.wav")
        cached_seconds = time.perf_counter() - started
        cached_stats = dict(cached.last_stats)
        reused = cached_stats["prepared_reused"]
        computed = cached_stats["prepared_computed"]
        assert reused == args.clips - args.changed and computed == args.changed

        fresh = SyncEngine(max_workers=args.workers, prefer_rubberband=False)
        started = time.perf_counter()
        fresh.build_final_track(segments, duration, root / "fresh", root / "fresh.wav")
        fresh_seconds = time.perf_counter() - started
        fresh_stats = dict(fresh.last_stats)
        a, rate_a = audio_fx.read_wav(root / "cached.wav")
        b, rate_b = audio_fx.read_wav(root / "fresh.wav")
        assert rate_a == rate_b and np.array_equal(a, b), "cached and fresh waveforms differ"
        for name in ("prepared_computed", "prepared_reused"):
            cached_stats.pop(name)
            fresh_stats.pop(name)
        assert cached_stats == fresh_stats, "cached and fresh sync diagnostics differ"
        print(json.dumps({
            "synthetic_clips": args.clips, "changed_clips": args.changed, "workers": args.workers,
            "initial_build_seconds": round(initial, 3),
            "fresh_rebuild_seconds": round(fresh_seconds, 3),
            "cached_rebuild_seconds": round(cached_seconds, 3),
            "prepared_reused": reused, "prepared_computed": computed,
            "speedup": round(fresh_seconds / cached_seconds, 2),
            "identical_waveform_and_sync_stats": True,
            "note": "Offline synthetic benchmark, not live end-to-end video performance.",
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
