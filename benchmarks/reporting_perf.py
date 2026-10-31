"""Measure added per-second report cost on synthetic media, without APIs."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from proxmax import audio_fx  # noqa: E402
from proxmax.config import AppConfig  # noqa: E402
from proxmax.models import Segment, StageTimings, Word  # noqa: E402
import proxmax.reporting as reporting  # noqa: E402


def measure(root: Path, source: Path, output: Path, seconds: int, detail: bool) -> dict:
    segments = []
    for i in range(170):
        start = (seconds - 1) * i / 170
        unit = Segment(i, start, start + .8, "Synthetic sample sentence", "নমুনা বাংলা বাক্য",
                       words=[Word(start, start + .3, "Sample", .9)], placed_start=start,
                       placed_duration=.8, tts_audio_path=root / "clip.wav", stretched_audio_path=root / "clip.wav")
        segments.append(unit)
    kwargs = dict(video_path=source, output_video_path=output, output_dir=root, work_dir=root,
                  segments=segments, timings=StageTimings(audio_seconds=seconds), stats={},
                  config=AppConfig(), total_duration=seconds)
    started = time.perf_counter()
    if detail:
        result = reporting.write_detailed_report(**kwargs)
    else:
        with mock.patch.object(reporting, "postmux_seconds", return_value={"status": "unavailable", "seconds": []}), \
             mock.patch.object(reporting, "picture_packet_seconds", return_value={"status": "unavailable"}), \
             mock.patch.object(reporting, "media_timeline_lines", return_value=[]):
            result = reporting.write_detailed_report(**kwargs)
    elapsed = time.perf_counter() - started
    return {"seconds": round(elapsed, 3), "report_bytes": result.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-seconds", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not (10 <= args.source_seconds <= 600 and 1 <= args.repeats <= 10):
        parser.error("unsupported benchmark parameters")
    logging.disable(logging.CRITICAL)
    with tempfile.TemporaryDirectory(prefix="proxmax-report-bench-") as tmp:
        root, seconds = Path(tmp), args.source_seconds
        for filename, rate in (("audio_16k.wav", 16000), ("final_audio_bn.wav", 24000)):
            t = np.arange(seconds * rate, dtype=np.float32) / rate
            audio_fx.write_wav(root / filename, (.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), rate)
            del t
        output = root / "output.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=black:s=160x120:r=25",
                        "-i", str(root / "final_audio_bn.wav"), "-t", str(seconds), "-c:v", "libx264",
                        "-preset", "ultrafast", "-c:a", "aac", str(output)], check=True)
        result = {"source_seconds": seconds, "units": 170, "baseline_without_second_sections": [],
                  "with_second_sections_and_mux_decode": []}
        for i in range(args.repeats):
            for detail in ((False, True) if i % 2 == 0 else (True, False)):
                key = "with_second_sections_and_mux_decode" if detail else "baseline_without_second_sections"
                result[key].append(measure(root, output, output, seconds, detail))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
