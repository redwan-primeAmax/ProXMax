"""Local-first English-to-Bangla dubbing. Heavy dependencies are imported by stage.

This file is also the source for the standalone Colab notebook. After editing it,
run `python scripts/build_notebook.py` and commit both files.
"""

# %% Configuration and standard-library utilities
from __future__ import annotations

import ctypes
import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import wave
from dataclasses import asdict, dataclass
from typing import Callable

PIPELINE_VERSION = "1.0.0"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}
LOG = logging.getLogger("proxmax")


@dataclass(frozen=True)
class Config:
    input_dir: str = "/content/drive/MyDrive/Video/input"
    output_dir: str = "/content/drive/MyDrive/Video/output"
    work_dir: str = "/content/proxmax-work"
    # These are public, downloadable checkpoints, not hosted inference APIs.
    asr_model: str = "Systran/faster-whisper-large-v3-turbo"
    translation_model: str = "facebook/nllb-200-distilled-600M"
    tts_model: str = "facebook/mms-tts-ben"
    # Replace 'main' with Hub commit IDs for reproducible model snapshots.
    asr_revision: str = "main"
    translation_revision: str = "main"
    tts_revision: str = "main"
    device: str = "auto"  # auto, cuda, or cpu
    asr_beam_size: int = 5
    translation_beams: int = 3
    translation_batch_size: int = 4
    cpu_threads: int = 2
    max_utterance_seconds: float = 12.0
    max_utterance_words: int = 40
    split_gap_seconds: float = 0.65
    borrow_silence_seconds: float = 0.30
    comfortable_speed: float = 1.50
    overflow_policy: str = "stretch"  # stretch and flag, or error (never cut words)
    max_source_tokens: int = 480
    max_target_tokens: int = 512
    max_tts_chars: int = 220
    sample_rate: int = 16000
    seed: int = 42
    # Successful jobs are removed locally only AFTER verified publication.
    keep_local: bool = False
    continue_on_error: bool = True
    min_free_gb: float = 6.0
    subprocess_timeout: int = 7200

    def validate(self):
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.overflow_policy not in {"stretch", "error"}:
            raise ValueError("overflow_policy must be stretch or error")
        for name in ("asr_beam_size", "translation_beams", "translation_batch_size",
                     "cpu_threads", "max_utterance_seconds", "max_utterance_words",
                     "comfortable_speed", "max_source_tokens", "max_target_tokens",
                     "max_tts_chars", "sample_rate", "subprocess_timeout"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.comfortable_speed < 1:
            raise ValueError("comfortable_speed must be at least 1")
        for name in ("borrow_silence_seconds", "split_gap_seconds", "min_free_gb"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        source, destination, work = map(
            lambda p: Path(p).expanduser().resolve(),
            (self.input_dir, self.output_dir, self.work_dir),
        )
        if source == destination or source in destination.parents or destination in source.parents:
            raise ValueError("Input and output directories must be separate, non-nested folders")
        for drive_path in (source, destination):
            if work == drive_path or work in drive_path.parents or drive_path in work.parents:
                raise ValueError("Work directory must be separate from input/output")
        if str(work).startswith("/content/drive/"):
            raise ValueError("Work directory must be local, not on mounted Google Drive")


def run_command(args, timeout=7200):
    """No shell interpolation: filenames may contain spaces, Unicode, or quotes."""
    result = subprocess.run([str(x) for x in args], capture_output=True, text=True,
                            timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed ({result.returncode}):\n{result.stderr[-6000:]}")
    return result.stdout


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    try:
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_copy(source, destination):
    """Publish to a temporary sibling, verify bytes, then rename on Drive's mount."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = sha256_file(source)
    if destination.is_file() and sha256_file(destination) == expected:
        return expected
    temp = destination.with_name(destination.name + ".partial-" + uuid.uuid4().hex)
    try:
        shutil.copyfile(source, temp)
        if sha256_file(temp) != expected:
            raise IOError(f"Checksum mismatch while publishing {destination}")
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)
    return expected


def safe_stem(name):
    # Limit bytes as well as characters so long Unicode filenames remain portable.
    stem = re.sub(r"[^\w.-]+", "_", Path(name).stem, flags=re.UNICODE).strip("._") or "video"
    return stem.encode("utf-8")[:100].decode("utf-8", errors="ignore") or "video"


def package_versions():
    versions = {}
    for name in ("torch", "transformers", "faster-whisper", "ctranslate2", "uroman",
                 "numpy", "soundfile", "sentencepiece"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def recipe_id(config):
    recipe = asdict(config)
    for key in ("input_dir", "output_dir", "work_dir", "keep_local", "continue_on_error",
                "min_free_gb", "subprocess_timeout"):
        recipe.pop(key)
    encoded = json.dumps({"pipeline": PIPELINE_VERSION, "config": recipe,
                          "packages": package_versions()}, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


# %% Local staging, probing and media extraction

def probe(path, timeout=120):
    return json.loads(run_command([
        "ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path
    ], timeout))


def media_info(path):
    data = probe(path)
    videos = [s for s in data["streams"] if s["codec_type"] == "video"
              and not s.get("disposition", {}).get("attached_pic", 0)]
    audios = [s for s in data["streams"] if s["codec_type"] == "audio"]
    if not videos:
        raise ValueError(f"No video stream in {Path(path).name}")
    duration = float(data.get("format", {}).get("duration", 0))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Input must have a finite, positive duration")
    return {"duration": duration, "video_index": videos[0]["index"],
            "audio_index": audios[0]["index"] if audios else None,
            "video_codec": videos[0]["codec_name"], "probe": data}


def stage_input(source, config):
    """Read Drive once for staging; every subsequent media operation is local."""
    root = Path(config.work_dir)
    root.mkdir(parents=True, exist_ok=True)
    size = source.stat().st_size
    # Reserve for source, rendered video, PCM/intermediates and model downloads.
    required = size * 3 + config.min_free_gb * 1024**3
    if shutil.disk_usage(root).free < required:
        raise OSError("Insufficient local disk: free space or increase Colab storage")
    fd, name = tempfile.mkstemp(prefix="staging-", suffix=source.suffix.lower(), dir=root)
    digest = hashlib.sha256()
    before = source.stat()
    try:
        with os.fdopen(fd, "wb") as out, source.open("rb") as inp:
            for block in iter(lambda: inp.read(8 * 1024 * 1024), b""):
                out.write(block)
                digest.update(block)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise IOError("Input changed while copying; retry when its upload has finished")
        job_id = f"{safe_stem(source.name)}-{digest.hexdigest()[:12]}-{recipe_id(config)}"
        job = root / job_id
        job.mkdir(exist_ok=True)
        local = job / ("input" + source.suffix.lower())
        os.replace(name, local)
        return job, local, digest.hexdigest()
    finally:
        Path(name).unlink(missing_ok=True)


def extract_audio(local_video, destination, info, config):
    if info["audio_index"] is None:
        raise ValueError("Input has no audio stream to translate; add/select a video with English speech")
    temp = Path(destination).with_suffix(".partial.wav")
    run_command([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", local_video, "-map", f"0:{info['audio_index']}", "-vn",
        "-af", "aresample=16000:async=1:first_pts=0", "-ac", "1", "-ar", "16000",
        "-t", str(info["duration"]), "-c:a", "pcm_s16le", temp,
    ], config.subprocess_timeout)
    os.replace(temp, destination)


# %% Timestamp grouping and subtitle generation (no ML dependencies)

def group_words(words, duration, config):
    """Keep absolute ASR times; split at sentences, pauses and bounded phrase lengths."""
    output, group = [], []
    previous_end = 0.0

    def flush():
        if group:
            output.append({"id": len(output), "start": group[0]["start"],
                           "end": group[-1]["end"],
                           "english": " ".join(w["text"] for w in group).strip()})
            group.clear()

    for word in words:
        text = str(word.get("text", "")).strip()
        start, end = float(word["start"]), float(word["end"])
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("Nonfinite ASR timestamp")
        start, end = max(0.0, previous_end, start), min(duration, end)
        if not text or end <= start:
            continue
        if group and (
            start - group[-1]["end"] >= config.split_gap_seconds
            or end - group[0]["start"] > config.max_utterance_seconds
            or len(group) >= config.max_utterance_words
            or re.search(r"[.!?][\"')\]]*$", group[-1]["text"])
        ):
            flush()
        group.append({"start": start, "end": end, "text": text})
        previous_end = end
    flush()
    return output


def validate_segments(segments, duration, translated=False):
    previous_end = 0.0
    for index, segment in enumerate(segments):
        start, end = float(segment["start"]), float(segment["end"])
        if (segment["id"] != index or not math.isfinite(start) or not math.isfinite(end)
                or start < previous_end - 1e-6 or end <= start or end > duration + 1e-6):
            raise ValueError(f"Invalid/nonmonotonic segment {index}")
        if not str(segment["english"]).strip():
            raise ValueError(f"Empty English text in segment {index}")
        if translated and not str(segment.get("bangla", "")).strip():
            raise ValueError(f"Empty Bangla translation in segment {index}")
        previous_end = end


def srt_time(seconds):
    ms = max(0, round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def write_srt(segments, key, path):
    lines = []
    for i, segment in enumerate(segments, 1):
        # Bengali subtitles cover the actual allowed dubbing slot, including borrowed silence.
        end = segment.get("slot_end", segment["end"]) if key == "bangla" else segment["end"]
        lines.append(f"{i}\n{srt_time(segment['start'])} --> {srt_time(end)}\n{segment[key]}\n")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def atempo_filter(rate):
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Tempo must be finite and positive")
    factors = []
    while rate > 2.0:
        factors.append(2.0)
        rate /= 2.0
    while rate < 0.5:
        factors.append(0.5)
        rate /= 0.5
    factors.append(rate)
    return ",".join(f"atempo={f:.10f}" for f in factors)


def split_tts_text(text, limit):
    """Bound VITS memory use without dropping words or splitting Unicode codepoints."""
    chunks, current = [], ""
    for word in text.split():
        if len(word) > limit:
            raise ValueError("A translation contains an overlong token; edit it before synthesis")
        if current and len(current) + len(word) + 1 > limit:
            chunks.append(current)
            current = ""
        current = (current + " " + word).strip()
        if current.endswith(("।", ".", "?", "!")):
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    if not chunks:
        raise ValueError("Cannot synthesize empty text")
    return chunks


# %% Device setup and English speech recognition

def torch_device(config):
    import torch
    torch.set_num_threads(config.cpu_threads)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select a Colab GPU or device='cpu'")
    return "cuda" if config.device != "cpu" and torch.cuda.is_available() else "cpu"


def release_memory():
    gc.collect()
    if "torch" in sys.modules:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def preload_cuda_libraries():
    """Help CTranslate2 locate Colab/PyTorch's CUDA 12/cuDNN 9 shared libraries."""
    import site
    roots = [Path(p) / "nvidia" for p in site.getsitepackages()]
    for package, library in (("cublas", "libcublasLt.so.12"),
                             ("cublas", "libcublas.so.12"), ("cudnn", "libcudnn.so.9")):
        for root in roots:
            candidate = root / package / "lib" / library
            if candidate.is_file():
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                break


def transcribe(audio_path, duration, config):
    import ctranslate2
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download
    device = torch_device(config)
    if device == "cuda":
        preload_cuda_libraries()
    supported = ctranslate2.get_supported_compute_types(device)
    compute_type = "int8_float16" if device == "cuda" else "int8"
    if compute_type not in supported:
        compute_type = "float16" if device == "cuda" else "float32"
    LOG.info("ASR: %s (%s / %s)", config.asr_model, device, compute_type)
    model = None
    try:
        model_path = snapshot_download(config.asr_model, revision=config.asr_revision,
                                       allow_patterns=["*.json", "*.bin", "*.txt"])
        model = WhisperModel(model_path, device=device, compute_type=compute_type,
                             cpu_threads=config.cpu_threads, num_workers=1)
        # Non-batched decoding is deliberately conservative on free-tier RAM/VRAM.
        stream, info = model.transcribe(
            str(audio_path), language="en", task="transcribe",
            beam_size=config.asr_beam_size, word_timestamps=True,
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
        )
        words, raw_segments = [], []
        for segment in stream:  # Consume the lazy generator while the model is alive.
            raw_segments.append({"start": segment.start, "end": segment.end,
                                 "text": segment.text, "avg_logprob": segment.avg_logprob,
                                 "no_speech_prob": segment.no_speech_prob})
            if segment.words:
                words.extend({"start": w.start, "end": w.end, "text": w.word,
                              "probability": w.probability} for w in segment.words)
            elif segment.text.strip():
                words.append({"start": segment.start, "end": segment.end,
                              "text": segment.text})
        segments = group_words(words, duration, config)
        validate_segments(segments, duration)
        return {"segments": segments, "words": words, "raw_segments": raw_segments,
                "language": "en", "device": device, "compute_type": compute_type,
                "model_commit": Path(model_path).name,
                "duration_after_vad": info.duration_after_vad}
    finally:
        del model
        release_memory()


# %% Batched English-to-Bangla translation

def translate(segments, config):
    if not segments:
        return {"segments": [], "model_commit": None}
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    device = torch_device(config)
    tokenizer = model = inputs = generated = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.translation_model, revision=config.translation_revision,
            src_lang="eng_Latn", use_fast=True, trust_remote_code=False,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config.translation_model, revision=config.translation_revision,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=False,
        ).to(device).eval()
        bangla_id = tokenizer.convert_tokens_to_ids("ben_Beng")
        if bangla_id is None or bangla_id == tokenizer.unk_token_id:
            raise ValueError("Translation tokenizer does not support ben_Beng")
        output = []
        for offset in range(0, len(segments), config.translation_batch_size):
            batch = segments[offset:offset + config.translation_batch_size]
            texts = [s["english"] for s in batch]
            lengths = [len(tokenizer.encode(t)) for t in texts]
            if max(lengths) > config.max_source_tokens:
                raise ValueError("ASR phrase exceeds translation token limit; reduce max_utterance_words")
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=False).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, forced_bos_token_id=bangla_id,
                    num_beams=config.translation_beams, max_new_tokens=config.max_target_tokens,
                )
            for segment, ids in zip(batch, generated):
                # NLLB's decoder start token can itself be EOS: ignore the first token.
                nonpadding = [t for t in ids.tolist() if t != tokenizer.pad_token_id]
                if (tokenizer.eos_token_id not in ids[1:].tolist()
                        or len(nonpadding) >= config.max_target_tokens + 1):
                    # A forced EOS at max_length is still potentially truncated text.
                    raise ValueError(f"Translation hit the generation limit at segment {segment['id']}")
                text = tokenizer.decode(ids, skip_special_tokens=True).strip()
                if not text or not re.search(r"[\u0980-\u09FF]", text):
                    raise ValueError(f"Empty/non-Bengali output at segment {segment['id']}; review source text")
                output.append({**segment, "bangla": text})
            LOG.info("Translated %s/%s phrases", len(output), len(segments))
        return {"segments": output, "model_commit": getattr(model.config, "_commit_hash", None)}
    finally:
        del inputs, generated, model, tokenizer
        release_memory()


# %% Local Bengali speech synthesis and duration fitting

def fit_audio(raw_path, output_path, slot_seconds, config):
    """Speed up only when needed; never truncate spoken samples to force a fit."""
    import numpy as np
    import soundfile as sf
    native = sf.info(raw_path)
    raw_seconds = native.frames / native.samplerate
    target_samples = round(slot_seconds * config.sample_rate)
    if target_samples <= 0 or raw_seconds <= 0:
        raise ValueError("Audio and timestamp slot must have positive duration")
    rate = max(1.0, raw_seconds / slot_seconds)
    flagged = rate > config.comfortable_speed
    if flagged and config.overflow_policy == "error":
        raise ValueError(f"Dub needs {rate:.2f}x speed; edit/shorten translation or use overflow_policy='stretch'")
    if flagged:
        LOG.warning("Dub needs %.2fx speed (comfort limit %.2fx); flagged for review",
                    rate, config.comfortable_speed)
    temp = Path(output_path).with_suffix(".tempo.wav")
    try:
        for attempt in range(4):
            # Resample before tempo so target counts refer to the timeline sample rate.
            run_command([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", raw_path, "-af", f"aresample={config.sample_rate}," + atempo_filter(rate),
                "-ar", str(config.sample_rate), "-ac", "1", "-c:a", "pcm_s16le", temp,
            ], config.subprocess_timeout)
            samples, sr = sf.read(temp, dtype="float32")
            if len(samples) <= target_samples:
                break
            # atempo is not sample-exact. Retry faster instead of cropping phonemes.
            rate *= (len(samples) / target_samples) * 1.015
            if config.overflow_policy == "error" and rate > config.comfortable_speed:
                raise ValueError("Sample-exact fit would exceed comfortable_speed; edit the translation")
        else:
            raise RuntimeError("Could not fit speech without truncation; edit this translation")
        if not len(samples) or not np.isfinite(samples).all() or np.max(np.abs(samples)) < 1e-5:
            raise ValueError("Synthesis produced empty, silent or nonfinite audio")
        # Pad silence, never speech-crop. This also gives exact per-slot length.
        samples = np.pad(samples, (0, target_samples - len(samples)))
        partial = Path(output_path).with_suffix(".partial.wav")
        sf.write(partial, samples, sr, subtype="PCM_16")
        os.replace(partial, output_path)
        return {"raw_seconds": raw_seconds, "speed_factor": rate,
                "review_required": rate > config.comfortable_speed,
                "slot_seconds": target_samples / config.sample_rate}
    finally:
        temp.unlink(missing_ok=True)


def synthesize(segments, duration, job, config):
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoTokenizer, VitsModel, set_seed
    from uroman import Uroman
    device = torch_device(config)
    clips_dir = Path(job) / "clips"
    clips_dir.mkdir(exist_ok=True)
    tokenizer = model = romanizer = tokens = result = None
    records = []
    recipe = recipe_id(config)
    tts_commit = None
    try:
        for i, segment in enumerate(segments):
            next_start = segments[i + 1]["start"] if i + 1 < len(segments) else duration
            slot_end = min(duration, next_start, segment["end"] + config.borrow_silence_seconds)
            start_sample = round(segment["start"] * config.sample_rate)
            end_sample = round(slot_end * config.sample_rate)
            slot_seconds = (end_sample - start_sample) / config.sample_rate
            if slot_seconds <= 0:
                raise ValueError(f"Segment {i} has no available audio slot")
            signature = hashlib.sha256(json.dumps(
                {"segment": segment, "slot_end": slot_end, "recipe": recipe},
                sort_keys=True, ensure_ascii=False,
            ).encode()).hexdigest()[:16]
            clip = clips_dir / f"{i:05d}-{signature}.wav"
            metadata = clip.with_suffix(".json")
            if clip.exists() and metadata.exists():
                record = read_json(metadata)
                if record.get("sha256") == sha256_file(clip):
                    records.append(record)
                    tts_commit = record.get("tts_model_commit", tts_commit)
                    continue
            if model is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    config.tts_model, revision=config.tts_revision, trust_remote_code=False)
                model = VitsModel.from_pretrained(
                    config.tts_model, revision=config.tts_revision,
                    trust_remote_code=False,
                ).to(device).eval()  # FP32 is deliberate: VITS can be unstable in FP16.
                tts_commit = getattr(model.config, "_commit_hash", None)
                if getattr(tokenizer, "is_uroman", False):
                    romanizer = Uroman()
                    # Reuse one Romanizer, rather than construct one for every tokenization.
                    tokenizer.is_uroman = False
            set_seed(config.seed + i)
            parts = []
            native_rate = model.config.sampling_rate
            for text in split_tts_text(segment["bangla"], config.max_tts_chars):
                if romanizer is not None:
                    text = romanizer.romanize_string(text, lcode="ben")
                tokens = tokenizer(text, return_tensors="pt").to(device)
                if tokens.input_ids.shape[-1] <= 1:
                    raise ValueError(f"TTS tokenizer has no pronounceable text in segment {i}")
                with torch.inference_mode():
                    result = model(**tokens).waveform
                part = result[0].float().cpu().numpy()
                if not len(part) or not np.isfinite(part).all():
                    raise ValueError(f"Invalid TTS waveform in segment {i}")
                if parts:
                    parts.append(np.zeros(round(0.08 * native_rate), dtype=np.float32))
                parts.append(part)
            audio = np.concatenate(parts)
            peak = float(np.max(np.abs(audio)))
            if peak > 0:
                audio *= min(1.0, 0.95 / peak)  # Prevent clipping without boosting noise.
            raw = clips_dir / f"{i:05d}-raw.wav"
            sf.write(raw, audio, native_rate, subtype="PCM_16")
            timing = fit_audio(raw, clip, slot_seconds, config)
            record = {**segment, **timing, "slot_end": end_sample / config.sample_rate,
                      "start_sample": start_sample, "clip": clip.name,
                      "sha256": sha256_file(clip), "tts_model_commit": tts_commit}
            atomic_json(metadata, record)
            raw.unlink(missing_ok=True)
            records.append(record)
            LOG.info("Synthesized %s/%s phrases", i + 1, len(segments))
        return records, tts_commit
    finally:
        del tokens, result, model, tokenizer, romanizer
        release_memory()


# %% Disk-backed audio timeline and final video muxing

def assemble_timeline(records, duration, job, config):
    import numpy as np
    import soundfile as sf
    job = Path(job)
    total_samples = math.ceil(duration * config.sample_rate)
    pcm = job / "timeline.pcm"
    # A sparse file + mmap avoids an in-RAM waveform proportional to video length.
    with pcm.open("wb") as stream:
        stream.truncate(total_samples * 2)
    timeline = np.memmap(pcm, dtype="<i2", mode="r+", shape=(total_samples,))
    previous_end = 0
    try:
        for record in records:
            samples, rate = sf.read(job / "clips" / record["clip"], dtype="int16")
            if rate != config.sample_rate or samples.ndim != 1:
                raise ValueError("Clip does not match mono timeline format")
            start = record["start_sample"]
            end = start + len(samples)
            if start < previous_end or end > total_samples:
                raise ValueError("Dubbing clip would overlap or exceed the video")
            timeline[start:end] = samples
            previous_end = end
        timeline.flush()
        output = job / "dub.wav"
        partial = job / "dub.partial.wav"
        with wave.open(str(partial), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(config.sample_rate)
            for offset in range(0, total_samples, config.sample_rate * 30):
                stream.writeframesraw(timeline[offset:offset + config.sample_rate * 30].tobytes())
        os.replace(partial, output)
        return output
    finally:
        del timeline
        pcm.unlink(missing_ok=True)


def mux_video(local_video, audio, destination, info, config):
    destination = Path(destination)
    temp = destination.with_suffix(".partial.mp4")
    base = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", local_video, "-i", audio, "-map", f"0:{info['video_index']}",
            "-map", "1:a:0", "-map_metadata", "-1", "-map_chapters", "-1"]
    suffix = ["-c:a", "aac", "-b:a", "96k", "-metadata:s:a:0", "language=ben",
              "-t", str(info["duration"]), "-movflags", "+faststart", temp]
    # H.264 is broadly browser-compatible. Other codecs are explicitly normalized.
    copied = info["video_codec"] == "h264"
    if copied:
        try:
            run_command(base + ["-c:v", "copy"] + suffix, config.subprocess_timeout)
        except RuntimeError:
            LOG.warning("Video stream-copy failed; retrying with H.264 encoding")
            copied = False
    if not copied:
        run_command(base + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
                            "-threads", str(config.cpu_threads)] + suffix,
                    config.subprocess_timeout)
    output = media_info(temp)
    audio_streams = [s for s in output["probe"]["streams"] if s["codec_type"] == "audio"]
    if not audio_streams or abs(output["duration"] - info["duration"]) > 0.25:
        raise ValueError("Final video duration/audio validation failed")
    audio_duration = float(audio_streams[0].get("duration", output["duration"]))
    if abs(audio_duration - info["duration"]) > 0.25:
        raise ValueError("Final audio does not cover the expected video duration")
    os.replace(temp, destination)
    return {"video_stream_copied": copied, "duration": output["duration"]}


# %% Checkpointed orchestration and verified publication

def cached_stage(path, compute: Callable):
    path = Path(path)
    if path.exists():
        LOG.info("Resuming checkpoint %s", path.name)
        return read_json(path)
    result = compute()
    atomic_json(path, result)
    return result


def publish_bundle(job, output_dir, artifacts, manifest):
    """Manifest-last publication: its presence marks the verified bundle complete."""
    destination = Path(output_dir) / Path(job).name
    destination.mkdir(parents=True, exist_ok=True)
    # Invalidate an earlier completion marker before replacing any bundle files.
    (destination / "manifest.json").unlink(missing_ok=True)
    checksums = {}
    for name in artifacts:
        checksums[name] = atomic_copy(Path(job) / name, destination / name)
    completed = {**manifest, "status": "complete", "artifacts_sha256": checksums}
    atomic_json(Path(job) / "manifest.json", completed)
    atomic_copy(Path(job) / "manifest.json", destination / "manifest.json")
    return destination


def process_video(source, config):
    started = time.perf_counter()
    job, local_video, source_hash = stage_input(Path(source), config)
    stage_times = {}
    current_stage = "probe"

    def timed(name, compute):
        nonlocal current_stage
        current_stage = name
        tick = time.perf_counter()
        try:
            return compute()
        finally:
            stage_times[name] = round(time.perf_counter() - tick, 3)

    try:
        info = media_info(local_video)
        # Estimate PCM/audio disk space before decoding a very long/compressed source.
        if shutil.disk_usage(job).free < info["duration"] * config.sample_rate * 12 + config.min_free_gb * 1024**3:
            raise OSError("Insufficient disk for decoded audio/timeline intermediates")
        if info["duration"] * config.sample_rate * 2 >= 4_000_000_000:
            raise ValueError("Video exceeds WAV size limit; split it into shorter videos")
        wav = job / "english.wav"
        if not (job / "asr.json").exists():
            timed("extract", lambda: extract_audio(local_video, wav, info, config))
        asr = timed("asr", lambda: cached_stage(
            job / "asr.json", lambda: transcribe(wav, info["duration"], config)))
        validate_segments(asr["segments"], info["duration"])
        translations = timed("translation", lambda: cached_stage(
            job / "translation.json", lambda: translate(asr["segments"], config)))
        validate_segments(translations["segments"], info["duration"], translated=True)
        if not translations["segments"]:
            LOG.warning("No speech detected: output will contain a silent Bengali audio track")
        records, tts_commit = timed("tts", lambda: synthesize(
            translations["segments"], info["duration"], job, config))
        audio = timed("timeline", lambda: assemble_timeline(records, info["duration"], job, config))
        mux = timed("mux", lambda: mux_video(local_video, audio, job / "translated_bn.mp4", info, config))
        atomic_json(job / "segments.json", records)
        write_srt(records, "english", job / "english.srt")
        write_srt(records, "bangla", job / "bangla.srt")
        manifest = {"pipeline_version": PIPELINE_VERSION, "source_name": Path(source).name,
                    "source_sha256": source_hash, "config": asdict(config),
                    "packages": package_versions(), "source_duration": info["duration"],
                    "language_pair": "eng_Latn -> ben_Beng", "segment_count": len(records),
                    "no_speech_detected": not bool(records),
                    "review_required_segments": [r["id"] for r in records if r["review_required"]],
                    "model_commits": {"asr": asr["model_commit"],
                                      "translation": translations["model_commit"], "tts": tts_commit},
                    "stages_seconds_this_run": stage_times,
                    "processing_seconds_before_publish": round(time.perf_counter() - started, 3),
                    "mux": mux}
        destination = timed("publish", lambda: publish_bundle(
            job, config.output_dir,
            ["translated_bn.mp4", "english.srt", "bangla.srt", "segments.json", "asr.json"], manifest))
        result = {"source": str(source), "status": "complete", "output": str(destination),
                  "review_required_segments": manifest["review_required_segments"],
                  "seconds": round(time.perf_counter() - started, 3)}
        if not config.keep_local:
            shutil.rmtree(job)  # job is always a generated child of the local work directory.
        return result
    except Exception as error:
        atomic_json(job / "failure.json", {"status": "failed", "stage": current_stage,
                    "error": f"{type(error).__name__}: {error}", "source": str(source),
                    "stages_seconds_this_run": stage_times})
        LOG.error("Failed in %s. Local checkpoints retained at %s", current_stage, job)
        raise
    finally:
        release_memory()


def run_batch(config):
    config.validate()
    for executable in ("ffmpeg", "ffprobe"):
        if not shutil.which(executable):
            raise RuntimeError(f"{executable} is missing; run the notebook installation cell")
    input_dir = Path(config.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Create {input_dir} and upload English videos before running")
    sources = sorted(p for p in input_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if not sources:
        raise FileNotFoundError(f"No supported videos in {input_dir} (scan is non-recursive)")
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    results = []
    for source in sources:
        LOG.info("Processing %s", source.name)
        try:
            results.append(process_video(source, config))
        except Exception as error:
            LOG.exception("Could not translate %s", source.name)
            results.append({"source": str(source), "status": "failed",
                            "error": f"{type(error).__name__}: {error}"})
            if not config.continue_on_error:
                atomic_json(Path(config.output_dir) / "batch_results.json", results)
                raise
    atomic_json(Path(config.output_dir) / "batch_results.json", results)
    return results
