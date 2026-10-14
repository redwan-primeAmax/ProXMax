"""Generate a standalone, output-free Colab notebook from the tested Python source."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "ProXMax_English_to_Bangla.ipynb"


def make_notebook():
    cells = []

    def cell(kind, text):
        text = text.strip() + "\n"
        entry = {"cell_type": kind, "id": hashlib.sha256(text.encode()).hexdigest()[:12],
                 "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            entry.update(execution_count=None, outputs=[])
        cells.append(entry)

    cell("markdown", """
# ProXMax · English → Bangla Video Translator

**Standalone Google Colab notebook — no repository clone or API key required.**

Select **Runtime → Change runtime type → T4 GPU** (recommended), then run cells top to bottom.
CPU is supported but substantially slower. First use downloads several GB of public model weights.
The notebook runs recognition, translation and speech generation locally on the Colab runtime;
no hosted transcription/TTS service receives your video. Colab and mounted Drive are still cloud services.

### Pipeline
`Drive input → local copy → FFmpeg audio → faster-whisper → NLLB → MMS Bengali TTS
→ per-phrase duration fitting → disk-backed audio timeline → video mux → verified Drive output`

- Input: `/content/drive/MyDrive/Video/input` (your `drive/Video/input`).
- Output: `/content/drive/MyDrive/Video/output/<unique-job>/translated_bn.mp4`.
- All expensive processing and checkpoints use `/content/proxmax-work`, **not Drive**.
- English speech only; one generic Bengali voice. This is timed dubbing, **not** lip-sync,
  speaker preservation, voice cloning or a certified translation.
- The original soundtrack is replaced: music, ambience and English dialogue are not retained.
- Review Bengali subtitles and `review_required_segments` before sharing. Accuracy, naturalness
  and speed depend on the recording; no fixed real-time performance is promised.
- **Model licensing:** NLLB and MMS checkpoints have non-commercial restrictions. Review their
  model cards before use: [NLLB](https://huggingface.co/facebook/nllb-200-distilled-600M),
  [MMS Bengali](https://huggingface.co/facebook/mms-tts-ben),
  [Whisper conversion](https://huggingface.co/Systran/faster-whisper-large-v3-turbo).

Only process media you have the right to translate. Outputs/checkpoints contain spoken content.
""")
    cell("markdown", """
## 1 · Install dependencies
Run this in a **fresh Colab runtime** before importing ML libraries. Colab's torch/CUDA stack
is retained. FFmpeg/ffprobe are installed only if absent. If you previously imported a different
version of these libraries, restart the runtime after installation and rerun from this cell.
Model weights are downloaded later from Hugging Face; internet is required for the first run.
""")
    requirements = (ROOT / "requirements-colab.txt").read_text()
    cell("code", f'''
import os
import sys
import subprocess
import shutil
from pathlib import Path

if sys.version_info < (3, 10):
    raise RuntimeError("Python 3.10+ is required")
if "google.colab" not in sys.modules:
    try:
        import google.colab
    except ImportError as exc:
        raise RuntimeError("This notebook's setup targets Google Colab; see README for local usage") from exc

requirements = {requirements!r}
requirements_path = Path("/content/proxmax-requirements.txt")
requirements_path.write_text(requirements)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-r", str(requirements_path)], check=True)
if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
    subprocess.run(["apt-get", "-qq", "update"], check=True)
    subprocess.run(["apt-get", "-qq", "install", "-y", "ffmpeg"], check=True)

# Keep caches local: repeated model reads must not hit the Drive mount.
os.environ["HF_HOME"] = "/content/proxmax-model-cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import torch
from packaging.version import Version
if Version(torch.__version__.split("+")[0]) < Version("2.6"):
    raise RuntimeError("Use a current Colab runtime with torch >= 2.6 for safe checkpoint loading")
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none — CPU mode")
if torch.cuda.is_available() and (torch.version.cuda or "").split(".")[0] != "12":
    print("WARNING: this CTranslate2 pin targets CUDA 12 + cuDNN 9. See architecture troubleshooting.")
print(subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0])
''')
    cell("markdown", """
##  · Mount Drive and prepare folders
Authorize the mount through Colab's own Google sign-in prompt (do not paste credentials into code).
Place videos directly inside `MyDrive/Video/input`; discovery is non-recursive.
The notebook never deletes input videos. Shared Drives require changing paths in the configuration cell.
""".replace("##  ·", "## 2 ·"))
    cell("code", """
from google.colab import drive
from pathlib import Path

drive.mount("/content/drive")
DRIVE_ROOT = Path("/content/drive/MyDrive")
if not DRIVE_ROOT.is_dir():
    raise RuntimeError("Google Drive was not mounted successfully")
INPUT_DIR = DRIVE_ROOT / "Video" / "input"
OUTPUT_DIR = DRIVE_ROOT / "Video" / "output"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Upload English videos to:", INPUT_DIR)
print("Completed bundles will be saved to:", OUTPUT_DIR)
""")
    cell("markdown", """
## 3 · Load the modular pipeline
The following definition cells are embedded from `proxmax/pipeline.py`. They do not download
models or process media until the run cell. Each ML stage releases its model before the next
stage starts. Rerunning definitions is safe; do not run two batches concurrently on one work directory.
""")
    source = (ROOT / "proxmax" / "pipeline.py").read_text()
    sections = re.split(r"^# %% (.+)\n", source, flags=re.MULTILINE)
    for i in range(1, len(sections), 2):
        cell("markdown", f"### 3.{(i + 1) // 2} · {sections[i]}")
        cell("code", sections[i + 1])
    cell("markdown", """
## 4 · Configure the run
Defaults balance quality and free-tier resource use. Set `device="cpu"` for CPU-only use;
consider `asr_model="Systran/faster-whisper-small.en"` on CPU (lower accuracy).
For faster GPU runs, try `asr_beam_size=1` and `translation_beams=1`.
For CUDA OOM, reduce `translation_batch_size` to 1 or use a smaller Whisper model.

`overflow_policy="stretch"` keeps complete speech in each time slot and flags rates above
`comfortable_speed`. Set it to `"error"` to stop rather than render potentially rushed speech.
Shorter clips are silence-padded, and each clip is placed at an absolute source timestamp, so
rounding errors never accumulate through concatenation. Up to 0.30 seconds of unused following
silence may be borrowed, but clips never overlap.

Set `keep_local=True` to retain successful checkpoints for review/rerendering (uses local disk).
Failed jobs are always retained. Identical input bytes/configuration/dependency versions reuse
local ASR/translation checkpoints and per-phrase TTS caches. Colab runtime resets **erase** them.
Changing parameters creates a separate job. For model reproducibility, replace revision `main`
with model Hub commit IDs; actual resolved commits are recorded in the output manifest.
""")
    cell("code", """
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)

CONFIG = Config(
    input_dir=str(INPUT_DIR),
    output_dir=str(OUTPUT_DIR),
    work_dir="/content/proxmax-work",
    device="auto",
    translation_batch_size=4,
    overflow_policy="stretch",
    comfortable_speed=1.50,
    keep_local=False,
    continue_on_error=True,
)
CONFIG.validate()
print(json.dumps(asdict(CONFIG), indent=2))
""")
    cell("markdown", """
## 5 · Translate every input video and publish to Drive
The batch runs **sequentially**, preventing multiple large models/videos competing for GPU RAM.
No-speech recordings produce a silent audio track with a warning; recordings with no audio stream
fail explicitly. Mixed/non-English speech and overlapping speakers are outside this version's scope.

Each completed output folder contains an MP4, English/Bangla SRT files, detailed segment JSON,
ASR diagnostics and a manifest. `manifest.json` is copied **last**, after checksum verification;
a folder without a complete manifest is not a completed result. `batch_results.json` records
both successes and errors. Other videos continue after an error by default.

Re-running after an interruption resumes local completed stages. A changed input has a new
content hash and cannot reuse the old job. Source files must not be edited during processing.
""")
    cell("code", """
RESULTS = run_batch(CONFIG)
for result in RESULTS:
    print(json.dumps(result, ensure_ascii=False, indent=2))
failed = [r for r in RESULTS if r["status"] == "failed"]
if failed:
    print(f"WARNING: {len(failed)} video(s) failed. See errors above and local failure.json files.")
else:
    print("All videos processed and copied to Google Drive.")
""")
    cell("markdown", """
## 6 · Inspect completion and quality flags
Open the output folder in Google Drive to play/download the video. This avoids embedding a
potentially large video as base64 in notebook outputs. **Listen to a short sample first** and
review names, numbers, idioms, low-confidence ASR and rapid speech. Inspect the entire translation
before publishing; a low speed factor does not guarantee semantic accuracy.
""")
    cell("code", """
for result in RESULTS:
    if result["status"] != "complete":
        continue
    folder = Path(result["output"])
    manifest = read_json(folder / "manifest.json")
    print("\nVIDEO:", folder / "translated_bn.mp4")
    print("Phrases:", manifest["segment_count"])
    print("Rapid-speech review IDs:", manifest["review_required_segments"])
    print("No speech:", manifest["no_speech_detected"])
    print("Runtime (seconds):", result["seconds"])
    records = read_json(folder / "segments.json")
    for record in records[:3]:
        print(f"  {record['start']:.2f}–{record['slot_end']:.2f}s | {record['speed_factor']:.2f}x")
        print("  EN:", record["english"])
        print("  BN:", record["bangla"])
""".replace('print("\nVIDEO:', 'print("\\nVIDEO:'))
    cell("markdown", """
## Troubleshooting and review workflow
- **No files:** verify `MyDrive/Video/input`, supported extensions and that Drive upload finished.
- **OOM:** restart the runtime to release other notebooks' allocations. Use batch size 1,
  shorter input files or `Systran/faster-whisper-small.en`; CPU fallback is explicit, not silent.
- **CUDA/cuDNN load error:** this stack targets CUDA 12/cuDNN 9. Use a compatible Colab runtime
  or `device="cpu"`; don't randomly replace Colab's torch/CUDA packages. Future Colab image
  changes may require updating the CTranslate2 pin.
- **Hub/network error:** internet must reach Hugging Face for public checkpoint downloads.
  Rerun after connectivity recovers; no paid API/token is normally needed.
- **Rushed Bengali:** with local checkpoints retained, open the job's `translation.json`, edit
  only the `bangla` string for the relevant phrase, and rerun with identical settings. Changed
  text invalidates its TTS clip cache. Do not change IDs or timestamps. Edits are not automatic
  linguistic shortening; they need a fluent reviewer. Published `segments.json` alone is not
  used as an input checkpoint.
- **Runtime disconnected:** local cache/checkpoints survive only as long as that runtime does.
  Completed Drive bundles remain. Publication is manifest-last but mounted Drive cannot provide
  a cross-file transaction or a guarantee of immediate remote durability.
- **Unexpected timestamps:** ordinary media timestamps are retained and extraction fills audio
  gaps; severely damaged/discontinuous media may need repair before translation.

### Deployment notes
The GitHub version includes architecture details, lightweight tests, a media smoke test and a
notebook generator. Regenerate after editing the source. Never commit Drive content, model weights,
credentials or executed notebook outputs. Model licensing is independent from repository code.
""")
    return {"cells": cells, "metadata": {
        "colab": {"name": "ProXMax_English_to_Bangla.ipynb", "provenance": []},
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }, "nbformat": 4, "nbformat_minor": 5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail if the committed notebook is stale")
    args = parser.parse_args()
    content = json.dumps(make_notebook(), ensure_ascii=False, indent=1) + "\n"
    if args.check:
        if not TARGET.exists() or TARGET.read_text() != content:
            raise SystemExit("Notebook is stale: run python scripts/build_notebook.py")
        print("Notebook is up to date")
    else:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_text(content, encoding="utf-8")
        print(TARGET)


if __name__ == "__main__":
    main()
