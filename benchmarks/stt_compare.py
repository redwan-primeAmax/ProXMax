"""Compare Whisper checkpoints and optional filters on private, labeled audio.

Example (run from the repo):
  python benchmarks/stt_compare.py --manifest /path/to/manifest.json
Manifest: [{"audio": "/absolute/path/video.mp4", "reference": "spoken English"}]
Reference transcripts and media stay local; results contain metrics only. Model
weights may be downloaded by faster-whisper. These measurements are not CI and
must use representative, manually checked transcripts to be meaningful.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

# Allow direct execution without installing the repository as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from proxmax.audio_extraction import AudioExtractor
from proxmax.stt import SpeechToText
from proxmax.utils import get_media_duration


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.casefold())


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        raise ValueError("Reference must contain at least one English word")
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, 1):
        current = [i]
        for j, hyp_word in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref_word != hyp_word)))
        previous = current
    return previous[-1], len(ref)


def run_comparison(records, models, filters, device="auto", compute_type="float16"):
    results = []
    with tempfile.TemporaryDirectory(prefix="proxmax_stt_benchmark_") as tmp:
        root = Path(tmp)
        for model_name in models:
            stt = SpeechToText(model_size=model_name, device=device, compute_type=compute_type)
            started = time.perf_counter()
            stt._load()  # report one-time model load separately from inference
            load_seconds = time.perf_counter() - started
            try:
                for filter_name in filters:
                    extractor = AudioExtractor(audio_filter=filter_name)
                    metrics = {"errors": 0, "words": 0, "audio_seconds": 0.0,
                               "extraction_seconds": 0.0, "stt_seconds": 0.0}
                    for index, record in enumerate(records):
                        source = Path(record["audio"]).expanduser()
                        if not source.is_file():
                            raise FileNotFoundError(source)
                        wav = root / f"{index}_{filter_name}.wav"
                        started = time.perf_counter()
                        extractor.extract(source, wav)
                        metrics["extraction_seconds"] += time.perf_counter() - started
                        metrics["audio_seconds"] += get_media_duration(wav)
                        started = time.perf_counter()
                        segments = stt.transcribe(wav)
                        metrics["stt_seconds"] += time.perf_counter() - started
                        errors, count = word_errors(record["reference"], " ".join(s.text_en for s in segments))
                        metrics["errors"] += errors
                        metrics["words"] += count
                    audio_seconds = metrics["audio_seconds"]
                    results.append({
                        "model": model_name, "audio_filter": filter_name, "clips": len(records),
                        "word_error_rate": metrics["errors"] / metrics["words"],
                        "word_errors": metrics["errors"], "reference_words": metrics["words"],
                        "audio_seconds": round(audio_seconds, 3),
                        "model_load_seconds": round(load_seconds, 3),
                        "extraction_seconds": round(metrics["extraction_seconds"], 3),
                        "stt_seconds": round(metrics["stt_seconds"], 3),
                        "extract_plus_stt_rtf": (metrics["extraction_seconds"] + metrics["stt_seconds"]) / audio_seconds,
                    })
            finally:
                stt.release()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--models", nargs="+", default=["distil-large-v3", "large-v3"])
    parser.add_argument("--filters", nargs="+", choices=["none", "highpass60"], default=["none", "highpass60"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="float16")
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        parser.error("manifest must be a non-empty JSON array")
    print(json.dumps(run_comparison(records, args.models, args.filters, args.device, args.compute_type), indent=2))


if __name__ == "__main__":
    main()
