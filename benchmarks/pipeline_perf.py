"""Offline stage 3/5 benchmarks using synthetic input, never user media or APIs.

Run from the repository root with FFmpeg/ffprobe installed:
    python benchmarks/pipeline_perf.py --repeats 3
The fake Gemini service sleeps to model network latency, without modelling a
real provider's throughput, retry/quota behavior or translation quality.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import tempfile
import threading
import time
from fractions import Fraction
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from proxmax.config import AppConfig  # noqa: E402
from proxmax.gemini_translator import GeminiTranslator  # noqa: E402
from proxmax.models import Segment  # noqa: E402
from proxmax.pipeline import DubbingPipeline  # noqa: E402
from proxmax.reporting_runtime import OperationalTrace  # noqa: E402
from proxmax.video_timeline import VideoPiece, VideoPlan  # noqa: E402
import proxmax.video_render as video_render  # noqa: E402


class FakeGemini:
    """Simulate latency while preserving paragraph IDs and tracking concurrency."""

    def __init__(self, delay: float):
        self.models = self
        self.delay = delay
        self.lock = threading.Lock()
        self.active = self.peak = self.calls = 0

    def generate_content(self, *, model, contents, config):
        with self.lock:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.delay)
            ids = [int(x) for x in re.findall(r"^Paragraph\s+(\d+)\s*:", contents, re.M)]
            result = {"translations": [
                {"id": i, "translation": "এটি একটি বাংলা অনুবাদের বাক্য।"} for i in ids
            ]}
            return type("Response", (), {"text": json.dumps(result, ensure_ascii=False),
                                         "candidates": []})()
        finally:
            with self.lock:
                self.active -= 1


class FakeTTS:
    def synthesize(self, segments, output_dir):
        time.sleep(.01)
        for segment in segments:
            segment.tts_audio_path = output_dir / f"{segment.index}.wav"
        return segments


def bench_translation(units: int, delay: float) -> dict:
    client = FakeGemini(delay)
    translator = GeminiTranslator(api_key="offline", client=client, batch_size=12,
                                  requests_per_minute=1_000_000)
    pipeline = DubbingPipeline.__new__(DubbingPipeline)
    pipeline.config = AppConfig(enable_overlap=True).validate()
    pipeline.translator, pipeline._tts = translator, FakeTTS()
    segments = [Segment(i, float(i), i + .8,
                        f"This is a sample sentence {chr(65 + i // 26)}{chr(65 + i % 26)}.")
                for i in range(units)]
    trace = OperationalTrace()
    trace.activate()
    try:
        trace.start_stage(3)
        with tempfile.TemporaryDirectory(prefix="proxmax-perf-tts-") as temp:
            started = time.perf_counter()
            translation, tts = pipeline._translate_and_synthesize(segments, Path(temp))
            elapsed = time.perf_counter() - started
        trace.end_stage(3)
        events = len(trace.snapshot()["events"])
    finally:
        trace.close()
    assert all(s.text_bn and s.tts_audio_path for s in segments)
    return {"wall_seconds": round(elapsed, 3), "translation_worker_seconds": round(translation, 3),
            "tts_call_seconds": round(tts, 3), "api_attempts": client.calls,
            "peak_simulated_requests": client.peak, "trace_events": events}


def make_plan(source_frames: int, units: int, repeats: int) -> VideoPlan:
    pieces = []
    repeat_at = {int(i * units / repeats) for i in range(repeats)}
    for i in range(units):
        start, end = i * source_frames // units, (i + 1) * source_frames // units
        count = end - start
        if i % 3 == 0:
            out = max(1, (count * 20 + 22) // 23)
            pieces.append(VideoPiece(start, end, out, "speed"))
        else:
            pieces.append(VideoPiece(start, end, count, "normal"))
        if i in repeat_at:
            length = max(2, min(18, count // 4))
            pieces.append(VideoPiece(end - length, end, length, "repeat"))
    return VideoPlan(Fraction(30), source_frames, tuple(pieces), sum(p.output_frames for p in pieces))


def bench_video(root: Path, source: Path, plan: VideoPlan, workers: int, run: int) -> dict:
    lock = threading.Lock()
    counts = {"ffmpeg_encodes": 0, "ffprobe_checks": 0}
    encode, probe = video_render.run_ffmpeg, video_render._frame_count

    def counted_encode(*args, **kwargs):
        with lock:
            counts["ffmpeg_encodes"] += 1
        return encode(*args, **kwargs)

    def counted_probe(*args, **kwargs):
        with lock:
            counts["ffprobe_checks"] += 1
        return probe(*args, **kwargs)

    output = root / f"render-{run}.mp4"
    trace = OperationalTrace()
    trace.activate()
    try:
        trace.start_stage(5)
        started = time.perf_counter()
        with mock.patch.object(video_render, "run_ffmpeg", side_effect=counted_encode), \
             mock.patch.object(video_render, "_frame_count", side_effect=counted_probe):
            video_render.render_video_plan(source, plan, output, workers=workers, use_nvenc=False)
        elapsed = time.perf_counter() - started
        trace.end_stage(5)
        events = len(trace.snapshot()["events"])
    finally:
        trace.close()
    assert probe(output) == plan.output_frames
    output.unlink()
    return {"wall_seconds": round(elapsed, 3), "trace_events": events, **counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--source-seconds", type=int, default=60)
    parser.add_argument("--latency", type=float, default=.12)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not (1 <= args.repeats <= 10 and 10 <= args.source_seconds <= 600 and
            .001 <= args.latency <= 10 and 1 <= args.workers <= 4):
        parser.error("unsupported benchmark parameters")
    logging.disable(logging.CRITICAL)
    units, repeats = 170, 97
    result = {"synthetic_units": units, "fake_request_latency_seconds": args.latency,
              "translation": [bench_translation(units, args.latency) for _ in range(args.repeats)]}
    with tempfile.TemporaryDirectory(prefix="proxmax-perf-video-") as temp:
        root, frames = Path(temp), args.source_seconds * 30
        source = root / "synthetic.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=640x360:rate=30", "-frames:v", str(frames),
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "25", str(source)], check=True)
        plan = make_plan(frames, units, repeats)
        result["video"] = {"source_seconds": args.source_seconds, "resolution": "640x360",
                           "pieces": len(plan.pieces), "output_frames": plan.output_frames,
                           "workers": args.workers,
                           "runs": [bench_video(root, source, plan, args.workers, i)
                                    for i in range(args.repeats)]}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
