"""নতুন process-এ batch; config/result pipe-এ, error সরাসরি terminal-এ।"""
import json
import logging
import sys

if __package__:
    from .pipeline import Config, run_batch
else:
    from pipeline import Config, run_batch

RESULT_PREFIX = "PROXMAX_RESULT\t"


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s", force=True)
    try:
        config = Config(**json.load(sys.stdin))
        results = run_batch(config)
        failed = any(item["status"] == "failed" for item in results)
        outcome = {"status": "failed" if failed else "complete", "results": results}
    except Exception as error:
        logging.exception("কাজ সম্পন্ন হয়নি")
        outcome = {"status": "failed", "results": [],
                   "error": f"{type(error).__name__}: {error}"}
    # Parent consumes this frame in memory. Human-readable tracebacks go to stderr.
    print("\n" + RESULT_PREFIX + json.dumps(outcome, ensure_ascii=False), flush=True)
    return 1 if outcome["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
