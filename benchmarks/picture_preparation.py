"""Offline CPU benchmark: materialized vs verified virtual tiny source tails.

The source is synthetic. No media, credentials or network calls are used.
Example: python benchmarks/picture_preparation.py --source-seconds 60 --repeats 3
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.pipeline_perf import make_plan  # noqa: E402
from proxmax.muxer import VideoMuxer  # noqa: E402
from proxmax.video_render import _frame_count, render_video_plan  # noqa: E402


def run(root: Path, source: Path, fps: Fraction, duration: float, plan, name: str, n: int) -> dict:
    folder = root / f"{name}-{n}"
    folder.mkdir()
    muxer = VideoMuxer(use_hwaccel=False, video_workers=2)
    started = time.perf_counter()
    if name == "materialized":
        picture = muxer.ensure_picture_duration(source, duration, folder, fps)
        assert picture != source and _frame_count(picture) >= plan.source_frames
        source_map = None
    else:
        picture = muxer.prepare_picture_for_timeline(source, duration, folder, fps)
        source_map = muxer._virtual_tail
        assert picture == source and source_map is not None
        assert source_map.output_frames >= plan.source_frames
    prepared = time.perf_counter()
    output = folder / "result.mp4"
    render_video_plan(picture, plan, output, workers=2, use_nvenc=False, source_map=source_map)
    finished = time.perf_counter()
    assert _frame_count(output) == plan.output_frames
    return {"preparation_seconds": round(prepared - started, 3),
            "render_seconds": round(finished - prepared, 3),
            "total_seconds": round(finished - started, 3),
            "output_frames": plan.output_frames}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--source-seconds", type=int, default=60)
    args = parser.parse_args()
    if not (1 <= args.repeats <= 10 and 10 <= args.source_seconds <= 600):
        parser.error("unsupported benchmark parameters")
    logging.disable(logging.CRITICAL)
    fps = Fraction(30)
    with tempfile.TemporaryDirectory(prefix="proxmax-prep-bench-") as tmp:
        root = Path(tmp)
        source = root / "synthetic.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=640x360:rate=30", "-frames:v", str(args.source_seconds * 30),
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "25", str(source)], check=True)
        physical = _frame_count(source)
        duration = physical / float(fps) + .072
        plan = make_plan(round(duration * float(fps)), 170, 97)
        result = {"source_seconds": args.source_seconds, "resolution": "640x360", "fps": "30/1",
                  "source_audio_tail_extra_seconds": .072, "planned_pieces": len(plan.pieces),
                  "materialized": [], "virtual": []}
        for i in range(args.repeats):
            order = ("materialized", "virtual") if i % 2 == 0 else ("virtual", "materialized")
            for name in order:
                result[name].append(run(root, source, fps, duration, plan, name, i))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
