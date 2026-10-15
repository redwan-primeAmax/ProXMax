"""পরীক্ষিত source থেকে সংক্ষিপ্ত, standalone বাংলা Colab notebook তৈরি করে।"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "ProXMax_English_to_Bangla.ipynb"


def make_notebook():
    cells = []

    def cell(kind, text, hidden=False):
        text = text.strip() + "\n"
        metadata = {"cellView": "form", "jupyter": {"source_hidden": True}} if hidden else {}
        entry = {"cell_type": kind, "id": hashlib.sha256(text.encode()).hexdigest()[:12],
                 "metadata": metadata, "source": text.splitlines(keepends=True)}
        if kind == "code":
            entry.update(execution_count=None, outputs=[])
        cells.append(entry)

    sources = {name: (ROOT / "proxmax" / name).read_text(encoding="utf-8")
               for name in ("colab_runtime.py", "colab_ui.py", "gemini.py", "language.py", "pipeline.py", "worker.py")}
    requirements = (ROOT / "requirements-colab.txt").read_text(encoding="utf-8")
    cell("code", f'''
# @title ▶ ProXMax · Setup ও App
# @markdown **Domain: FC Mobile** · **Model: gemini-3.5-flash-lite**
# @markdown **T4 GPU** বেছে ▶ চাপুন। API-key box-এ key paste করে **Enter** দিন; code খুলতে হবে না।
# @markdown ইনপুট: **My Drive / Video / input** → ফলাফল: **Video / output**।
RUN_RESULT = None
print("ProXMax চালু হচ্ছে…", flush=True)
import sys
import shutil
import subprocess
import runpy
from pathlib import Path

if sys.version_info < (3, 10):
    raise RuntimeError("Python 3.10 বা নতুন সংস্করণ প্রয়োজন।")
try:
    import google.colab
except ImportError as error:
    raise RuntimeError("এই notebook Google Colab-এ খুলুন।") from error

APP_DIR = Path("/content/proxmax-app")
APP_DIR.mkdir(parents=True, exist_ok=True)
SOURCES = {sources!r}
for name, source in SOURCES.items():
    (APP_DIR / name).write_text(source, encoding="utf-8")
# প্রতি run-এ নতুন helper; kernel-এ torch/transformers import হয় না।
RUNTIME = runpy.run_path(str(APP_DIR / "colab_runtime.py"))
UI = runpy.run_path(str(APP_DIR / "colab_ui.py"))

def _run_app():
    api_key = None
    try:
        CONFIG = {{
            "work_dir": "/content/proxmax-work",
            "device": "auto",                 # ASR/TTS-এর জন্য; Gemini-তে GPU লাগে না
            "domain": "football",            # এই notebook-এ সবসময় FC Mobile
            "translation_backend": "gemini",
            "translation_model": "gemini-3.5-flash-lite",
            "translation_batch_size": 4,
            "max_source_tokens": 8192,
            "max_target_tokens": 8192,
            "min_speed": 0.90,
            "max_speed": 1.25,
            "overflow_policy": "error",       # সীমা ছাড়িয়ে speech প্রকাশ করা হবে না
            "comfortable_speed": 1.25,
            "keep_local": True,
            "continue_on_error": True,
        }}
        REMOUNT_DRIVE = False  # শুধু Drive সংযোগ নষ্ট হলে True করুন
        print("Domain: FC Mobile (football)", flush=True)
        print("Model:", CONFIG["translation_model"], flush=True)
        if CONFIG["translation_backend"] == "gemini":
            api_key = UI["read_api_key"]()
        print("Environment setup শুরু হচ্ছে…", flush=True)
        WORKER_PYTHON = RUNTIME["setup_environment"](APP_DIR, {requirements!r})
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            subprocess.run(["apt-get", "-qq", "update"], check=True)
            subprocess.run(["apt-get", "-qq", "install", "-y", "ffmpeg"], check=True)
        print("Setup প্রস্তুত।")
        print("Drive সংযোগ যাচাই হচ্ছে…", flush=True)
        DRIVE_ROOT = RUNTIME["ensure_drive_mounted"](force_remount=REMOUNT_DRIVE)
        INPUT_DIR = DRIVE_ROOT / "Video" / "input"
        OUTPUT_DIR = DRIVE_ROOT / "Video" / "output"
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print("Drive প্রস্তুত। ফলাফল পাবেন: My Drive / Video / output")
        CONFIG.update(input_dir=str(INPUT_DIR), output_dir=str(OUTPUT_DIR))
        print("ভিডিও অনুবাদ শুরু হচ্ছে…")
        return RUNTIME["run_isolated"](
            WORKER_PYTHON, APP_DIR / "worker.py", CONFIG, api_key=api_key
        )
    finally:
        api_key = None  # error/interrupt হলেও credential reference রাখা হবে না

try:
    RUN_RESULT = _run_app()
except KeyboardInterrupt:
    print("কাজ বন্ধ হয়েছে; আবার এই cell চালাতে পারেন।")
finally:
    del _run_app

if RUN_RESULT is not None:
    for result in RUN_RESULT["results"]:
        if result["status"] == "complete":
            print("সম্পন্ন:", result["output"])
            if result.get("review_required_segments"):
                print("এই অংশগুলোর ASR/অনুবাদ যাচাই করুন:", result["review_required_segments"])
        else:
            print("অসম্পূর্ণ:", Path(result["source"]).name, "—", result.get("error", ""))
    if RUN_RESULT["status"] != "complete":
        print(RUN_RESULT.get("error", "কিছু ভিডিও সম্পন্ন হয়নি। উপরের error দেখুন।"))
        print("Error উপরের cell output-এ আছে; কোনো error/log ফাইল সংরক্ষণ করা হয়নি।")
    else:
        print("সব ভিডিও Drive-এ সংরক্ষিত। প্রকাশের আগে বাংলা অডিও ও subtitle যাচাই করুন।")
''', hidden=True)
    return {"cells": cells, "metadata": {
        "colab": {"name": "ProXMax_English_to_Bangla.ipynb", "provenance": []},
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }, "nbformat": 4, "nbformat_minor": 5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Notebook পুরোনো হলে ব্যর্থ হবে")
    args = parser.parse_args()
    content = json.dumps(make_notebook(), ensure_ascii=False, indent=1) + "\n"
    if args.check:
        if not TARGET.exists() or TARGET.read_text(encoding="utf-8") != content:
            raise SystemExit("Notebook আপডেট করুন: python scripts/build_notebook.py")
        print("Notebook হালনাগাদ আছে")
    else:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_text(content, encoding="utf-8")
        print(TARGET)


if __name__ == "__main__":
    main()
