#!/usr/bin/env python3
"""ProXMax — English → Bangla video dubbing tool (main entry point).

ব্যবহার (Google Colab-এ রিপো clone করার পর)::

    python main.py                      # config.yaml-এর সেটিংস দিয়ে পুরো ব্যাচ
    python main.py --model medium       # বেশি speed-এর জন্য ছোট মডেল
    python main.py --gemini-model gemini-3.5-flash-lite   # অনুবাদ মডেল
    python main.py --gender female      # নারী কণ্ঠে ডাবিং
    python main.py --force              # আউটপুট থাকলেও আবার প্রসেস করা

কণ্ঠ: বিল্ট-ইন বাংলা neural ভয়েস (`bn-BD-PradeepNeural`, পুরুষ ডিফল্ট;
`bn-BD-NabanitaNeural`, নারী) দিয়েই ডাবিং হয়।

অনুবাদে **Google Gemini API** ব্যবহৃত হয়। API key ধরা হয় এই ক্রমে:
``--gemini-api-key`` → ``config.yaml``-এর ``gemini.api_key`` →
``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` environment variable।
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# স্ক্রিপ্টটি যেখান থেকেই চালানো হোক, রিপোর রুট যেন import path-এ থাকে
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    print("▶ ProXMax চালু হচ্ছে — প্রয়োজনীয় মডিউল লোড হচ্ছে…", flush=True)

try:
    from proxmax.config import AppConfig                                   # noqa: E402
    from proxmax.pipeline import DubbingPipeline                           # noqa: E402
    from proxmax.utils import ensure_tool, get_logger, setup_logging       # noqa: E402
except ModuleNotFoundError as _import_error:                               # pragma: no cover
    print(
        f"❌ প্রয়োজনীয় প্যাকেজ পাওয়া যায়নি: '{_import_error.name}'\n"
        f"   নোটবুকের **ধাপ ৩** (ডিপেন্ডেন্সি ইনস্টল) সেলটি রান করুন, অথবা চালান:\n"
        f"   pip install -r requirements.txt",
        file=sys.stderr, flush=True,
    )
    raise SystemExit(2) from _import_error

logger = get_logger("proxmax.main")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "ProXMax: English ভিডিও → Bangla ডাবিং "
            "(faster-whisper + Gemini + বাংলা neural ভয়েস)"
        ),
    )
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "config.yaml"),
        help="config.yaml-এর পাথ (ডিফল্ট: রিপোর config.yaml)",
    )
    parser.add_argument("--input-dir", default=None, help="Input ভিডিওর ফোল্ডার (config override)")
    parser.add_argument("--output-dir", default=None, help="Output ফোল্ডার (config override)")
    parser.add_argument(
        "--model", default=None,
        help="Whisper মডেল: distil-large-v3 (ডিফল্ট, English-only) অথবা large-v3",
    )
    parser.add_argument("--device", default=None, help="STT device: auto | cuda | cpu")
    parser.add_argument(
        "--hotwords", default=None,
        help="অতিরিক্ত ভিডিওর নাম/শব্দ (কমা দিয়ে) — hotwords.txt-এর সাথে যুক্ত হবে",
    )
    parser.add_argument(
        "--hotwords-file", default=None,
        help="hotwords ফাইলের পাথ (ডিফল্ট: hotwords.txt)",
    )
    parser.add_argument(
        "--translator", default=None, choices=["gemini", "indictrans2"],
        help="অনুবাদ backend: gemini (ডিফল্ট) অথবা indictrans2",
    )
    parser.add_argument(
        "--gemini-model", default=None,
        help="Gemini অনুবাদ মডেল (ডিফল্ট: gemini-3.5-flash-lite)",
    )
    parser.add_argument(
        "--gemini-api-key", default=None,
        help="Gemini API key (না দিলে GEMINI_API_KEY environment variable পড়া হবে)",
    )
    parser.add_argument(
        "--gemini-max-input-tokens", type=int, default=None,
        help="প্রতি Gemini request-এর সর্বোচ্চ ইনপুট টোকেন (ডিফল্ট: 5000; ন্যূনতম: 1000)",
    )
    parser.add_argument(
        "--tts-engine", default=None, choices=["edge", "auto"],
        help="TTS ইঞ্জিন: edge (ডিফল্ট — বিল্ট-ইন বাংলা ভয়েস)",
    )
    parser.add_argument(
        "--gender", default=None, choices=["male", "female", "custom"],
        help="কণ্ঠের লিঙ্গ: male (ডিফল্ট, Pradeep) | female (Nabanita) | custom",
    )
    parser.add_argument(
        "--voice", default=None,
        help="edge-tts ভয়েস: bn-BD-PradeepNeural (পুরুষ, ডিফল্ট) / bn-BD-NabanitaNeural (নারী)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="output-এ ফাইল থাকলেও আবার প্রসেস করা (skip-existing বন্ধ)",
    )
    parser.add_argument(
        "--keep-temp", action="store_true",
        help="মধ্যবর্তী ফাইল (extracted wav, TTS ক্লিপ ইত্যাদি) রেখে দেওয়া (debug-এর জন্য)",
    )
    parser.add_argument("--log-level", default=None, help="INFO (ডিফল্ট) | DEBUG | WARNING")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = AppConfig.from_yaml(args.config)
    config.apply_overrides(
        {
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
            "whisper_model": args.model,
            "whisper_device": args.device,
            "stt_hotwords": args.hotwords,
            "stt_hotwords_file": args.hotwords_file,
            "translator_backend": args.translator,
            "gemini_model": args.gemini_model,
            "gemini_api_key": args.gemini_api_key,
            "gemini_max_input_tokens": args.gemini_max_input_tokens,
            "tts_engine": args.tts_engine,
            "tts_voice": args.voice,
            "tts_gender": ("custom" if args.voice else args.gender),
            "log_level": args.log_level,
            "skip_existing": False if args.force else None,
            "keep_temp_files": True if args.keep_temp else None,
        }
    )

    setup_logging(config.log_level)

    logger.info("=" * 70)
    logger.info("🎙️  ProXMax — English → Bangla ভিডিও ডাবিং")
    logger.info("=" * 70)
    logger.info("Input   : %s", config.input_dir)
    logger.info("Output  : %s", config.output_dir)
    logger.info("Work dir: %s (সাময়িক ফাইল)", config.resolved_work_dir)
    translator_label = (
        f"gemini:{config.gemini_model}"
        if config.translator_backend == "gemini"
        else f"indictrans2:{config.indictrans2_model}"
    )
    voice_label = f"বিল্ট-ইন {config.resolved_tts_voice} ({config.tts_gender})"
    logger.info(
        "STT=%s | অনুবাদ=%s | কণ্ঠ=%s | tempo %.2f–%.2f",
        config.whisper_model, translator_label, voice_label,
        config.min_tempo, config.max_tempo,
    )
    if config.stt_hotwords:
        logger.info("অতিরিক্ত Whisper hotwords: %s", config.stt_hotwords)
    logger.info(
        "কণ্ঠ (বিল্ট-ইন পথ): %s | লিঙ্গ: %s",
        config.resolved_tts_voice, config.tts_gender,
    )
    if config.tts_gender == "custom" and not config.resolved_tts_voice.startswith("bn-"):
        logger.warning("⚠️ tts.voice বাংলা ভয়েস নয়: %s", config.resolved_tts_voice)

    if config.translator_backend == "gemini" and not config.resolved_gemini_api_key():
        logger.error(
            "❌ Gemini API key পাওয়া যায়নি — কিছুই শুরু করা হলো না।\n"
            "   নোটবুকের **ধাপ ৪**-এ `GEMINI_API_KEY = \"...\"` এর ভেতরে key পেস্ট করে "
            "সেই সেলটি রান করুন (অথবা Colab Secrets-এ GEMINI_API_KEY রাখুন), "
            "তারপর আবার চালান।\n"
            "   key নিন: https://aistudio.google.com/app/apikey"
        )
        return 2

    try:
        ensure_tool("ffmpeg")
    except Exception as exc:
        logger.error(str(exc))
        return 2
    if not shutil.which("ffprobe"):
        logger.warning("ffprobe পাওয়া যায়নি — duration জানতে ffmpeg fallback ব্যবহৃত হবে।")

    try:
        config.validate()
        pipeline = DubbingPipeline(config)
    except Exception as exc:
        logger.error("সেটআপে সমস্যা: %s", exc)
        return 2

    try:
        summary = pipeline.run_batch()
    except KeyboardInterrupt:
        logger.warning("ব্যবহারকারী কাজ থামিয়ে দিয়েছেন (Ctrl+C)।")
        return 130
    finally:
        if config.release_gpu_when_done:
            pipeline.close()

    if not summary.results:
        return 0
    return 1 if summary.failed else 0


if __name__ == "__main__":
    sys.exit(main())
