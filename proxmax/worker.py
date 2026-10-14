"""Notebook-এর আলাদা process-এ batch চালায়; kernel-এ কোনো ML model থাকে না।"""
import argparse
import json
import logging
from pathlib import Path

if __package__:
    from .pipeline import Config, atomic_json, run_batch
else:
    # Standalone notebook একই folder-এ এই দুই source file লেখে।
    from pipeline import Config, atomic_json, run_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    try:
        config = Config(**json.loads(Path(args.config).read_text(encoding="utf-8")))
        results = run_batch(config)
        failed = any(item["status"] == "failed" for item in results)
        outcome = {"status": "failed" if failed else "complete", "results": results}
    except Exception as error:
        logging.exception("কাজ সম্পন্ন হয়নি")
        outcome = {"status": "failed", "results": [],
                   "error": f"{type(error).__name__}: {error}"}
    atomic_json(args.results, outcome)
    return 1 if outcome["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
