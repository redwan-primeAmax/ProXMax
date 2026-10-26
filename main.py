#!/usr/bin/env python3
"""ProXMax — English → Bangla video dubbing tool (main entry point).

ব্যবহার (Google Colab-এ রিপো clone করার পর)::

    python main.py --setup-runtime      # সেটআপ: মডেল আগেই নামানো + CUDA যাচাই
    python main.py --check-gemini       # Gemini API কাজ করছে কি না যাচাই
    python main.py                      # config.yaml-এর সেটিংস দিয়ে পুরো ব্যাচ
    python main.py --model medium       # বেশি speed-এর জন্য ছোট মডেল
    python main.py --gemini-model gemini-3.5-flash-lite   # অনুবাদ মডেল
    python main.py --force              # আউটপুট থাকলেও আবার প্রসেস করা

কণ্ঠ: **কিছুই দিতে হয় না** — বিল্ট-ইন বাংলা neural ভয়েস
(`bn-BD-PradeepNeural`, পুরুষ) দিয়েই ডাবিং হয়। ভয়েস ক্লোনিং নেই (কোনো
নমুনা, Hugging Face token বা অতিরিক্ত ইনস্টল লাগে না)।

নাম-শব্দের নির্ভুলতা: রিপোর ``hotwords.txt`` ফাইল নিজে থেকেই পড়া হয়
(ফুটবল/FC Mobile-এর শত শত নাম) — ব্যবহারকারীকে কিছু টাইপ করতে হয় না।

অনুবাদে **Google Gemini API** ব্যবহৃত হয়। API key ধরা হয় এই ক্রমে:
``--gemini-api-key`` → ``config.yaml``-এর ``gemini.api_key`` →
``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` environment variable।

Input ভিডিও নেওয়া হয় ``/content/drive/MyDrive/Video/input/`` থেকে
এবং ডাবিং করা ভিডিও রাখা হয় ``/content/drive/MyDrive/Video/output/``
ফোল্ডারে (পাথ বদলাতে ``config.yaml`` অথবা ``--input-dir/--output-dir``)।
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
    # প্রথম লগটাই সাথে সাথে দেখা যাবে — নোটবুকে "কিছুই হচ্ছে না" ভাব দূর হয়
    print("▶ ProXMax চালু হচ্ছে — প্রয়োজনীয় মডিউল লোড হচ্ছে…", flush=True)

try:
    from proxmax.config import AppConfig                                   # noqa: E402
    from proxmax.pipeline import DubbingPipeline               # noqa: E402
    from proxmax.utils import ensure_tool, get_logger, setup_logging  # noqa: E402
except ModuleNotFoundError as _import_error:                   # pragma: no cover
    print(
        f"❌ প্রয়োজনীয় প্যাকেজ পাওয়া যায়নি: '{_import_error.name}'\n"
        f"   নোটবুকের **ধাপ ৪** (Dependency ইনস্টল) সেলটি রান করুন, অথবা চালান:\n"
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
        help="Whisper মডেল: large-v3 (ডিফল্ট, best accuracy) অথবা medium (বেশি speed)",
    )
    parser.add_argument("--device", default=None, help="STT device: auto | cuda | cpu")
    parser.add_argument(
        "--hotwords", default=None,
        help="নাম/শব্দ (কমা দিয়ে) — Whisper এগুলো সহজে চিনবে। না দিলে "
             "রিপোর hotwords.txt ফাইল নিজে থেকেই পড়া হবে। "
             "যেমন: \"FC Mobile, Ballon d'Or, OVR, star pass\"",
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
        help="Gemini API key (না দিলে GEMINI_API_KEY environment variable পড়া হবে; "
             "CLI-তে key দেওয়া নিরাপদ নয়)",
    )
    parser.add_argument(
        "--voice", default=None,
        help="বিল্ট-ইন বাংলা ভয়েস: bn-BD-PradeepNeural (পুরুষ, ডিফল্ট) / "
             "bn-BD-NabanitaNeural (নারী)",
    )
    parser.add_argument(
        "--check-gemini", action="store_true",
        help="শুধু Gemini API কাজ করছে কি না যাচাই করে বেরিয়ে যাও (একটি ছোট অনুবাদ)",
    )
    parser.add_argument(
        "--prefetch-model", action="store_true",
        help="Whisper মডেল আগেই নামিয়ে রাখো (আগে থেকে থাকলে কিছুই করে না) — "
             "তাহলে প্রসেসিংয়ের সময় আর ডাউনলোডের অপেক্ষা করতে হয় না",
    )
    parser.add_argument(
        "--setup-runtime", action="store_true",
        help="Whisper মডেল আগেই নামাও + GPU/CUDA ঠিক আছে কি না দেখো "
             "(দরকার হলে ভারী nvidia প্যাকেজ ইনস্টল করে)",
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


def _setup_runtime(config: AppConfig, check_cuda: bool = True) -> int:
    """সেটআপ ধাপ: Whisper মডেল আগেই নামায়, দরকার হলে CUDA ঠিক করে।

    এতে ভিডিও প্রসেসিং শুরু করার সময় মডেল ডাউনলোডের জন্য অপেক্ষা করতে হয় না
    (ব্যবহারকারীর অভিযোগ ছিল ঠিক এটাই)।

    Args:
        config: অ্যাপ্লিকেশন কনফিগ।
        check_cuda: GPU/CUDA যাচাই করে দরকার হলে ``nvidia-*`` ইনস্টল করবে কি না।

    Returns:
        ``0`` সফল, ``1`` মডেল নামানো যায়নি (প্রথম রানে নিজেই নামবে)।
    """
    import subprocess
    import sys

    from proxmax.model_cache import cuda_runtime_ok, model_is_cached, prefetch_whisper

    ok, elapsed = prefetch_whisper(config.whisper_model)
    if not ok:
        logger.warning(
            "⚠️ মডেল আগেই নামানো যায়নি — প্রথম ভিডিওর সময় নিজেই নামবে "
            "(একবারই, তারপর ক্যাশে থাকবে)।"
        )

    if check_cuda:
        # libs লোড করার চেষ্টা (torch/lib সহ) — সেটআপেই একবার
        from proxmax.stt import _ensure_cuda_libraries

        _ensure_cuda_libraries()
        if cuda_runtime_ok():
            logger.info("✅ GPU ঠিক আছে — দ্রুত ডিকোডিং হবে (CUDA libs প্রস্তুত)।")
        elif shutil.which("nvidia-smi") is None:
            # GPU-ই নেই (CPU রানটাইম) — ভারী CUDA হুইল নামানোর কোনো মানে নেই
            logger.warning(
                "ℹ️ GPU পাওয়া যায়নি — CPU-তে চলবে (ধীর কিন্তু সম্পূর্ণ)। "
                "দ্রুত করতে Colab-এ `Runtime → Change runtime type → T4 GPU` দিন।"
            )
        else:
            logger.warning(
                "ℹ️ GPU আছে, কিন্তু CUDA লাইব্রেরি মিলছে না — ইনস্টল করা হচ্ছে…"
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--no-input",
                 "nvidia-cublas-cu12==12.6.4.1", "nvidia-cudnn-cu12==9.5.1.17"],
                check=False,
            )
            if cuda_runtime_ok():
                logger.info("✅ GPU চালু হয়েছে (CUDA libs ইনস্টল সম্পন্ন)।")
            else:
                logger.warning(
                    "⚠️ GPU এখনো পাওয়া যাচ্ছে না — CPU-তে চলবে (ধীর কিন্তু সম্পূর্ণ)। "
                    "Colab-এ `Runtime → Change runtime type → T4 GPU` দেখা করে নিন।"
                )

    if model_is_cached(config.whisper_model):
        logger.info(
            "✅ সেটআপ শেষ — %s মডেল প্রস্তুত (%.0fs)। এখন ধাপ ৭-এ ডাবিং শুরু করতে পারেন।",
            config.whisper_model, elapsed,
        )
    return 0 if ok else 1


def _check_gemini(config: AppConfig) -> int:
    """Gemini API সত্যিই কাজ করছে কি না যাচাই করে (নোটবুকের key-সেলের পরে)।

    একটি ছোট বাক্য অনুবাদ করে দেখা হয় — key, মডেলের নাম আর নেটওয়ার্ক তিনটিই
    একসাথে পরীক্ষা হয়। ব্যর্থ হলে পরিষ্কার বার্তা ও exit code 1, সফল হলে 0।

    Args:
        config: অ্যাপ্লিকেশন কনফিগ (key/model ইতিমধ্যেই লোড করা)।

    Returns:
        ``0`` সফল, ``1`` ব্যর্থ, ``2`` key নেই।
    """
    from proxmax.translator import TranslationError, create_translator

    key = config.resolved_gemini_api_key()
    if not key:
        logger.error("❌ Gemini API key নেই — যাচাই করা সম্ভব নয়।")
        return 2

    logger.info("🔎 Gemini API যাচাই করা হচ্ছে (model=%s)…", config.gemini_model)
    translator = create_translator(
        "gemini",
        gemini_api_key=key,
        gemini_model=config.gemini_model,
        gemini_batch_size=config.gemini_batch_size,
        gemini_max_chars_per_request=config.gemini_max_chars_per_request,
        gemini_max_retries=config.gemini_max_retries,
        gemini_retry_base_delay=config.gemini_retry_base_delay,
        gemini_retry_max_delay=config.gemini_retry_max_delay,
        gemini_retry_cap_delay=config.gemini_retry_cap_delay,
        gemini_requests_per_minute=config.gemini_requests_per_minute,
        gemini_temperature=config.gemini_temperature,
        gemini_timeout=config.gemini_timeout,
        gemini_max_output_tokens=config.gemini_max_output_tokens,
        char_rate=config.translation_char_rate,
        indictrans2_model=config.indictrans2_model,
    )
    try:
        result = translator.verify()
    except TranslationError as exc:
        logger.error(
            "❌ Gemini API কাজ করছে না: %s\n"
            "   • key ঠিক আছে কি না দেখুন (https://aistudio.google.com/app/apikey)\n"
            "   • মডেলের নাম বৈধ কি না দেখুন: %s\n"
            "   • key-সেলে নতুন করে পেস্ট করে সেলটি আবার রান করুন।",
            exc, config.gemini_model,
        )
        return 1
    except Exception as exc:  # noqa: BLE001 - সব ত্রুটি একই বার্তায়
        logger.error("❌ Gemini API যাচাই ব্যর্থ: %s", exc)
        return 1
    finally:
        try:
            translator.close()
        except Exception:  # pragma: no cover - best effort
            pass

    logger.info(
        "✅ Gemini API ঠিক আছে | model=%s | সময় %.2fs | নমুনা: %r",
        result.get("model"), result.get("elapsed", 0.0), result.get("sample", ""),
    )
    logger.info("   এখন ধাপ ৭-এ ডাবিং শুরু করতে পারেন — key নিয়ে আর ভাবতে হবে না।")
    return 0


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
            "translator_backend": args.translator,
            "gemini_model": args.gemini_model,
            "gemini_api_key": args.gemini_api_key,
            # --voice দিলে সেটিই চূড়ান্ত (gender=custom), নাহলে gender-ই প্রধান
            "tts_voice": args.voice,
            "tts_gender": "custom" if args.voice else None,
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
    voice_label = (
        f"বিল্ট-ইন {config.resolved_tts_voice} ({config.tts_gender})"
    )
    logger.info(
        "STT=%s | অনুবাদ=%s | কণ্ঠ=%s | tempo %.2f–%.2f",
        config.whisper_model, translator_label, voice_label,
        config.min_tempo, config.max_tempo,
    )
    hotword_entries = config.hotword_entries()
    if hotword_entries:
        source = "stt.hotwords" if (config.stt_hotwords or "").strip() else (
            str(config.hotwords_path) if config.hotwords_path else "?"
        )
        logger.info(
            "Whisper hotwords: %dটি নাম (%s) — যেমন: %s",
            len(hotword_entries), source, ", ".join(hotword_entries[:6]),
        )
    logger.info("কণ্ঠ (বিল্ট-ইন পথ): %s | লিঙ্গ: %s",
                config.resolved_tts_voice, config.tts_gender)
    if config.tts_gender == "custom" and not config.resolved_tts_voice.startswith("bn-"):
        logger.warning("⚠️ tts.voice বাংলা ভয়েস নয়: %s", config.resolved_tts_voice)

    # সেটআপ ধাপ: মডেল আগেই নামিয়ে রাখা (+ দরকার হলে CUDA ঠিক করা)
    if args.prefetch_model or args.setup_runtime:
        return _setup_runtime(config, check_cuda=args.setup_runtime)

    # Gemini key ছাড়া অনুবাদই সম্ভব নয় — শুরুতেই স্পষ্ট error, অপেক্ষা নয়
    if config.translator_backend == "gemini" and not config.resolved_gemini_api_key():
        logger.error(
            "❌ Gemini API key পাওয়া যায়নি — কিছুই শুরু করা হলো না।\n"
            "   নোটবুকের **ধাপ ৬**-এ `GEMINI_API_KEY = \"...\"` এর ভেতরে key পেস্ট করে "
            "সেই সেলটি রান করুন (অথবা Colab Secrets-এ GEMINI_API_KEY রাখুন), "
            "তারপর আবার চালান।\n"
            "   key নিন: https://aistudio.google.com/app/apikey"
        )
        return 2

    # নোটবুকের key-সেলের পরে এটাই চলে: API সত্যিই কাজ করছে কি না দেখা হয়
    if args.check_gemini:
        return _check_gemini(config)

    # ffmpeg ছাড়া এক ধাপও এগোনো যাবে না — শুরুতেই পরিষ্কার error দেওয়া ভালো
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
        # যেকোনো অবস্থাতেই GPU/VRAM ও API client ছেড়ে দেওয়া হয়
        if config.release_gpu_when_done:
            pipeline.close()

    if not summary.results:
        # Input খালি / ফোল্ডার নেই — উপরে বিস্তারিত বার্তা দেওয়া হয়েছে
        return 0
    return 1 if summary.failed else 0


if __name__ == "__main__":
    sys.exit(main())
