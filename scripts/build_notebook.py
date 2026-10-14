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
               for name in ("colab_runtime.py", "pipeline.py", "worker.py")}
    requirements = (ROOT / "requirements-colab.txt").read_text(encoding="utf-8")
    cell("code", f'''
# @title ▶ Setup ও App চালু করুন · ইংরেজি ভিডিও → বাংলা
# @markdown **T4 GPU** বেছে ▶ চাপুন। ইনপুট: **My Drive / Video / input** → ফলাফল: **Video / output**।
# @markdown প্রথমবার Drive-এর অনুমতি ও model download লাগবে। আবার চালাতেও এই একটি cell যথেষ্ট।
RUN_RESULT = None
REMOUNT_DRIVE = False # @param {{type:"boolean"}}
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
# run_path প্রতি run-এ সর্বশেষ helper পড়ে; kernel-এ torch/transformers import হয় না।
RUNTIME = runpy.run_path(str(APP_DIR / "colab_runtime.py"))
WORKER_PYTHON = RUNTIME["setup_environment"](APP_DIR, {requirements!r})
if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
    subprocess.run(["apt-get", "-qq", "update"], check=True)
    subprocess.run(["apt-get", "-qq", "install", "-y", "ffmpeg"], check=True)
print("Setup প্রস্তুত।")
# শুধু Drive সংযোগ নষ্ট হলে REMOUNT_DRIVE=True করুন।
DRIVE_ROOT = RUNTIME["ensure_drive_mounted"](force_remount=REMOUNT_DRIVE)
INPUT_DIR = DRIVE_ROOT / "Video" / "input"
OUTPUT_DIR = DRIVE_ROOT / "Video" / "output"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Drive প্রস্তুত। ফলাফল পাবেন: My Drive / Video / output")
CONFIG = {{
    "input_dir": str(INPUT_DIR),
    "output_dir": str(OUTPUT_DIR),
    "work_dir": "/content/proxmax-work",
    "device": "auto",                 # প্রয়োজনে "cpu"
    "translation_batch_size": 4,
    "overflow_policy": "stretch",     # বেশি দ্রুত কণ্ঠ হলে ফলাফলে চিহ্ন থাকবে
    "comfortable_speed": 1.50,
    "keep_local": False,               # সফল কাজের checkpoint রাখতে True
    "continue_on_error": True,
}}
print("ভিডিও অনুবাদ শুরু হচ্ছে…")
try:
    RUN_RESULT = RUNTIME["run_isolated"](
        WORKER_PYTHON, APP_DIR / "worker.py", CONFIG, APP_DIR / "runs"
    )
except KeyboardInterrupt:
    print("কাজ বন্ধ করা হয়েছে। Worker-ও বন্ধ হয়েছে; আবার এই cell চালাতে পারেন।")

if RUN_RESULT is not None:
    for result in RUN_RESULT["results"]:
        if result["status"] == "complete":
            print("সম্পন্ন:", result["output"])
            if result.get("review_required_segments"):
                print("দ্রুত উচ্চারণের অংশগুলো শুনে দেখুন:", result["review_required_segments"])
        else:
            print("অসম্পূর্ণ:", Path(result["source"]).name, "—", result.get("error", ""))
    if RUN_RESULT["status"] != "complete":
        print(RUN_RESULT.get("error", "কিছু ভিডিও সম্পন্ন হয়নি। উপরের error দেখুন।"))
        print("বিস্তারিত log:", RUN_RESULT["log"])
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
