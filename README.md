# ProXMax — English → Bangla Video Translator

A **standalone Google Colab notebook** for local-first video dubbing, with timestamp alignment, sequential model loading, resumable stages and automatic Google Drive output.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/redwan-primeAmax/ProXMax/blob/main/notebooks/ProXMax_English_to_Bangla.ipynb)

> The badge works once this notebook is pushed/merged into `main`. Before then, download [`ProXMax_English_to_Bangla.ipynb`](notebooks/ProXMax_English_to_Bangla.ipynb) and use **Colab → File → Upload notebook**. No clone or API key is required by the notebook.

## Quick start

1. Open the notebook in Google Colab and select **Runtime → Change runtime type → T4 GPU**.
2. Run the setup and Drive-mount cells. Authorize through Colab's Google sign-in prompt.
3. Upload English videos into **My Drive → Video → input**.
4. Run the remaining cells. The input scan is non-recursive; videos run one at a time.
5. Find each completed bundle in **My Drive → Video → output → `<unique-job>`**.

| Purpose | Colab path |
|---|---|
| Requested `drive/Video/input` | `/content/drive/MyDrive/Video/input` |
| Local media and checkpoints | `/content/proxmax-work` |
| Local model cache | `/content/proxmax-model-cache` |
| Requested `drive/Video/output` | `/content/drive/MyDrive/Video/output` |

Input videos are **copied to local storage before any probing, decoding or inference**. Originals are never modified. Only finished artifacts and the batch summary are written to Drive. Shared Drive users can change the input/output paths in the configuration cell.

## Architecture

```text
Google Drive / Video / input
           │ sequential copy + content hash
           ▼
Local Colab SSD → FFmpeg → 16 kHz mono English audio
           │
           ▼
faster-whisper large-v3-turbo · VAD · English word timestamps
           │ unload ASR model
           ▼
NLLB-200 distilled 600M · eng_Latn → ben_Beng · bounded batches
           │ unload translation model
           ▼
MMS Bengali VITS · reusable uroman · local, single-voice synthesis
           │ per-phrase cache; unload TTS model
           ▼
Pitch-preserving duration fit → absolute-timestamp PCM timeline
           │
           ▼
FFmpeg · copy H.264 video where possible, otherwise transcode
           │ validate durations; copy files; verify checksums
           ▼
Google Drive / Video / output / <unique-job> / manifest.json LAST
```

**[Full architecture blueprint](docs/ARCHITECTURE.md)** · **[Colab acceptance checklist](docs/COLAB_SMOKE_TEST.md)**

### What is included

- Faster-whisper with CUDA INT8/FP16 or CPU INT8 decoding, VAD and word timestamps.
- Batched local English-to-Bengali translation; no silently truncated input/output text.
- Local Bengali speech synthesis, with no paid service or voice API.
- No cumulative concatenation drift: clips are placed at absolute source times.
- Phrase speed warnings; optional strict mode rejects over-compressed speech instead of clipping words.
- Disk-backed audio timeline; only one inference model is resident at a time.
- Local ASR/translation checkpoints, per-phrase TTS caching, atomic file writes and per-video errors.
- Verified Drive copies and a manifest-last completion protocol.
- Separate English/Bangla subtitles and machine-readable diagnostics for review.

### Output bundle

```text
Video/output/
├── batch_results.json                  # successes and errors from the latest batch
└── <safe-name>-<content-hash>-<recipe-hash>/
    ├── translated_bn.mp4               # original picture + Bengali audio
    ├── english.srt                     # source phrase subtitles
    ├── bangla.srt                      # Bengali subtitles (UTF-8, separate file)
    ├── segments.json                   # texts, times, speed factors, review flags
    ├── asr.json                        # words/confidence, source segments, ASR settings
    └── manifest.json                   # completion marker, hashes, versions and model commits
```

A folder without a complete `manifest.json` is **not a completed bundle**. Copy verification confirms bytes read back from the mounted filesystem, not instantaneous durability of Google's remote servers.

## Requirements and tradeoffs

- Current Colab, Python 3.10+, torch 2.6+; GPU recommended. The pinned CTranslate2 stack targets **CUDA 12/cuDNN 9**. The notebook keeps Colab's torch installation rather than replacing it.
- Allow several GB for first-run model downloads, **at least 10 GB of free local disk for a small trial**, and additional room for your video. The pipeline reserves 6 GiB plus media estimates. Model downloads require Hugging Face access; later inference is local.
- English → Bangla only. One generic voice; no diarization, voice cloning or lip-sync. Mixed-language and overlapping speech are not supported reliably.
- The original audio is **replaced**, including music and ambience; no source-separation model is included. Only the first real video stream and first audio stream are processed. Subtitles are not burned into the picture.
- Silent recordings with an audio stream produce a silent output track and warning; video without an audio stream fails explicitly.
- Translation quality and speed depend on content/hardware. Names, numbers, idioms, ASR errors and rushed phrases need fluent review. No benchmark, accuracy percentage or faster-than-real-time guarantee is claimed.
- Whisper loads the source audio into host memory. The final timeline is disk-backed, but this is not an unbounded/constant-memory streaming system; split very long videos.

### Tuning

Change fields in the notebook's `Config(...)` cell:

| Goal | Setting | Tradeoff |
|---|---|---|
| Faster inference | `asr_beam_size=1`, `translation_beams=1` | May reduce recognition/translation quality |
| Lower translation VRAM | `translation_batch_size=1` | Lower throughput |
| CPU / smaller ASR | `device="cpu"`, `asr_model="Systran/faster-whisper-small.en"` | Slower than a GPU; less accurate ASR |
| Reject rushed dubbing | `overflow_policy="error"` | Stops if speech needs more than the comfort limit |
| Preserve review checkpoints | `keep_local=True` | Uses more local disk |
| Reproducible model weights | Set all three `*_revision` fields to Hub commit IDs | Defaults follow `main` |

Defaults stretch complete speech to fit its slot and flag rates above `1.50×`. Speech is never cropped to fit. Shorter speech is padded with silence. These flags indicate timing risk, not semantic correctness.

Failed jobs retain local checkpoints. Rerun with the same input/settings to resume. With `keep_local=True`, edit only `bangla` strings in the local job's `translation.json` to correct wording; changed phrases are resynthesized. A Colab runtime reset loses local checkpoints; completed Drive bundles remain. Do not run concurrent batches against the same work/output directories.

## Repository layout

```text
notebooks/ProXMax_English_to_Bangla.ipynb  # complete standalone deliverable
proxmax/pipeline.py                      # canonical modular implementation
proxmax/__main__.py                      # optional command-line runner
docs/ARCHITECTURE.md                     # design, reliability, limitations
docs/COLAB_SMOKE_TEST.md                 # real-model acceptance procedure
scripts/build_notebook.py                # deterministic source → notebook generator
tests/                                  # unit + real FFmpeg integration tests
requirements-colab.txt                   # pinned core inference stack
requirements-dev.txt                    # lightweight test dependencies
.github/workflows/ci.yml                 # notebook and pipeline checks
```

### Development and validation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
# Install ffmpeg AND ffprobe through your OS package manager for media tests.
python scripts/build_notebook.py
python scripts/build_notebook.py --check
python -m pytest -q
```

The notebook embeds the tested source; regenerate it after changing `proxmax/pipeline.py` or dependencies. CI checks notebook freshness, schema, code-cell syntax, clean outputs, timing math, caching, failed publication, and real synthetic-media extraction/fitting/muxing. Media tests skip when FFmpeg/ffprobe are absent.

**Validation boundary:** lightweight tests use synthetic media and stub inference for the orchestration test. They do not establish ASR/translation quality, model-download availability, current Colab image compatibility or actual GPU throughput. Run the [Colab smoke test](docs/COLAB_SMOKE_TEST.md) before a release.

### Optional local command line

Install a compatible torch 2.6+ build, FFmpeg/ffprobe, and `requirements-colab.txt` in an isolated Python environment, then run from the repository root:

```bash
python -m proxmax \
  --input-dir /path/to/input \
  --output-dir /path/to/output \
  --work-dir /path/to/local-ssd/work \
  --device auto
```

`--config settings.json` accepts other `Config` fields; explicit CLI path/device/keep-local arguments take precedence. Exit status is nonzero if any video fails. The CLI does not mount Google Drive.

## Privacy, licensing and publishing

Use media you are authorized to process. Input, transcripts, local checkpoints and output all contain user content. Processing happens on the Colab VM; it is not on-device processing on your own computer. Model downloads contact Hugging Face, but inference does not send speech/text to an external inference API.

Review the model cards and licenses before distributing or commercially using the system:

- [faster-whisper large-v3-turbo](https://huggingface.co/Systran/faster-whisper-large-v3-turbo) / [Whisper](https://github.com/openai/whisper)
- [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) — CC-BY-NC-4.0 model restrictions.
- [MMS Bengali TTS](https://huggingface.co/facebook/mms-tts-ben) — CC-BY-NC-4.0 model restrictions.

**This model combination is not an unrestricted commercial-use stack.** No repository code license is chosen on the owner's behalf. Add the appropriate code license before open-source distribution; it does not override the checkpoints' licenses.

Notebooks are committed without outputs. `.gitignore` excludes media, model weights, Drive content, runtime caches and common credential files. Check `git diff` before publishing; keep checkpoints and transcripts out of Git.
