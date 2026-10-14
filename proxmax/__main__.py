"""Optional command-line entry point; the Colab notebook is fully standalone."""
import argparse
import json
import logging
from .pipeline import Config, run_batch


def main():
    parser = argparse.ArgumentParser(description="Translate English videos into Bengali dubbing")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--work-dir", required=True, help="Local SSD directory, not cloud-mounted storage")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--keep-local", action="store_true")
    parser.add_argument("--config", help="Optional JSON object of other Config overrides")
    args = parser.parse_args()
    overrides = {}
    if args.config:
        with open(args.config, encoding="utf-8") as stream:
            overrides = json.load(stream)
    overrides.update(input_dir=args.input_dir, output_dir=args.output_dir,
                     work_dir=args.work_dir, device=args.device, keep_local=args.keep_local)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    results = run_batch(Config(**overrides))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    raise SystemExit(1 if any(r["status"] == "failed" for r in results) else 0)


if __name__ == "__main__":
    main()
