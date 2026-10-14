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
from dataclasses import asdict, dataclass, field
from typing import Callable

if __package__:
    from .gemini import GeminiClient
    from .language import (group_timed_words, prepare_segments, relevant_glossary,
                           quality_issues, parse_translations, translation_messages, spoken_bengali)
else:
    from gemini import GeminiClient
    from language import (group_timed_words, prepare_segments, relevant_glossary,
                          quality_issues, parse_translations, translation_messages, spoken_bengali)

PIPELINE_VERSION = "2.1.0"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}
LOG = logging.getLogger("proxmax")


@dataclass(frozen=True)
class Config:
    input_dir: str = "/content/drive/MyDrive/Video/input"
    output_dir: str = "/content/drive/MyDrive/Video/output"
    work_dir: str = "/content/proxmax-work"
    # ASR/TTS are local public checkpoints; translation defaults to the Gemini API.
    asr_model: str = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    translation_model: str = "gemini-3.5-flash-lite"
    translation_backend: str = "gemini"  # nllb is an explicit offline alternative
    gemini_timeout_seconds: float = 120.0
    gemini_http_retries: int = 2
    domain: str = "general"  # football is an opt-in notebook control
    glossary: dict = field(default_factory=dict)
    translation_retries: int = 1
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
    min_utterance_seconds: float = 2.0
    segment_hard_seconds: float = 24.0
    segment_hard_words: int = 80
    merge_gap_seconds: float = 0.35
    split_gap_seconds: float = 0.65
    borrow_silence_seconds: float = 0.30
    min_speed: float = 0.90
    max_speed: float = 1.25  # hard cap, including every fit retry
    comfortable_speed: float = 1.25
    overflow_policy: str = "error"  # legacy 'stretch' is also bounded by max_speed
    max_source_tokens: int = 8192
    max_target_tokens: int = 8192
    max_tts_chars: int = 220
    sample_rate: int = 16000
    seed: int = 42
    # Successful jobs are removed locally only AFTER verified publication.
    keep_local: bool = False
    continue_on_error: bool = True
    min_free_gb: float = 6.0
    subprocess_timeout: int = 7200

    def validate(self):
        if self.translation_backend not in {"gemini", "nllb"}:
            raise ValueError("translation_backend must be gemini or nllb")
        if type(self.gemini_http_retries) is not int or not 0 <= self.gemini_http_retries <= 3:
            raise ValueError("gemini_http_retries must be an integer from 0 to 3")
        if type(self.translation_batch_size) is not int or not 1 <= self.translation_batch_size <= 16:
            raise ValueError("translation_batch_size must be an integer from 1 to 16")
        if self.domain not in {"general", "football"}:
            raise ValueError("domain must be general or football")
        if not isinstance(self.glossary, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip()
            for k, v in self.glossary.items()
        ):
            raise ValueError("glossary must map nonempty English strings to Bengali strings")
        if type(self.translation_retries) is not int or not 0 <= self.translation_retries <= 3:
            raise ValueError("translation_retries must be an integer from 0 to 3")
        if not (0.90 <= self.min_speed <= 1.0 <= self.max_speed <= 1.25):
            raise ValueError("Speech speed must stay within 0.90–1.25x")
        if not (1 <= self.comfortable_speed <= self.max_speed):
            raise ValueError("comfortable_speed must be between 1 and max_speed")
        if self.segment_hard_seconds < self.max_utterance_seconds or self.segment_hard_words < self.max_utterance_words:
            raise ValueError("Hard segmentation limits must not be smaller than soft targets")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.overflow_policy not in {"stretch", "error"}:
            raise ValueError("overflow_policy must be stretch or error")
        for name in ("asr_beam_size", "translation_beams", "translation_batch_size",
                     "cpu_threads", "max_utterance_seconds", "max_utterance_words",
                     "comfortable_speed", "min_utterance_seconds", "segment_hard_seconds", "segment_hard_words",
                     "max_source_tokens", "max_target_tokens",
                     "max_tts_chars", "sample_rate", "subprocess_timeout", "gemini_timeout_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.comfortable_speed < 1:
            raise ValueError("comfortable_speed must be at least 1")
        for name in ("borrow_silence_seconds", "split_gap_seconds", "merge_gap_seconds", "min_free_gb"):
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
    for name in ("torch", "transformers", "faster-whisper", "av", "ctranslate2", "uroman",
                 "numpy", "soundfile", "sentencepiece"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def recipe_id(config):
    recipe = asdict(config)
    for key in ("input_dir", "output_dir", "work_dir", "keep_local", "continue_on_error",
                "min_free_gb", "subprocess_timeout", "gemini_timeout_seconds", "gemini_http_retries"):
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
    return group_timed_words(words, duration, config)


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


def download_asr_model(config):
    """Use the public CT2 Turbo repository listed by faster-whisper v1.2.1."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    repo_id = config.asr_model
    if repo_id in {"Systran/faster-whisper-large-v3-turbo", "large-v3-turbo", "turbo"}:
        repo_id = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
        LOG.warning("ASR model ID সংশোধন করা হয়েছে: %s", repo_id)
    try:
        # token=False also ignores an invalid token inherited from a previous session.
        return snapshot_download(
            repo_id, revision=config.asr_revision, token=False,
            allow_patterns=["config.json", "preprocessor_config.json", "model.bin",
                            "tokenizer.json", "vocabulary.*"],
        )
    except HfHubHTTPError as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        if status in {401, 403, 404}:
            raise RuntimeError(
                f"ASR model পাওয়া যায়নি বা প্রবেশাধিকার নেই: {repo_id} "
                f"(revision={config.asr_revision}, HTTP {status})। "
                "Model ID/revision ও Hugging Face সংযোগ যাচাই করুন। "
                "Default public model-এর জন্য login/token প্রয়োজন নেই।"
            ) from error
        raise


def load_asr_audio(audio_path):
    """Read the FFmpeg-normalized WAV without faster-whisper's PyAV decoder."""
    import numpy as np
    import soundfile as sf

    with sf.SoundFile(str(audio_path)) as source:
        if source.samplerate != 16000 or source.channels != 1:
            raise ValueError(
                "ASR-এর জন্য 16 kHz mono WAV প্রয়োজন; "
                f"পাওয়া গেছে {source.samplerate} Hz / {source.channels} channel। "
                "FFmpeg audio extraction ধাপটি আবার চালান।"
            )
        audio = source.read(dtype="float32", always_2d=False)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("ASR audio খালি বা তাতে অবৈধ sample আছে; input ভিডিও যাচাই করুন।")
    return np.ascontiguousarray(audio, dtype=np.float32)


def transcribe(audio_path, duration, config):
    import ctranslate2
    from faster_whisper import WhisperModel
    device = torch_device(config)
    if device == "cuda":
        preload_cuda_libraries()
    supported = ctranslate2.get_supported_compute_types(device)
    compute_type = "int8_float16" if device == "cuda" else "int8"
    if compute_type not in supported:
        compute_type = "float16" if device == "cuda" else "float32"
    LOG.info("ইংরেজি শনাক্তকরণ: %s (%s / %s)", config.asr_model, device, compute_type)
    model = audio = stream = None
    try:
        audio = load_asr_audio(audio_path)
        model_path = download_asr_model(config)
        model = WhisperModel(model_path, device=device, compute_type=compute_type,
                             cpu_threads=config.cpu_threads, num_workers=1)
        # Non-batched decoding is deliberately conservative on free-tier RAM/VRAM.
        # ndarray input skips decode_audio()/av.open(); FFmpeg already decoded it.
        stream, info = model.transcribe(
            audio, language="en", task="transcribe",
            beam_size=config.asr_beam_size, word_timestamps=True,
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
            initial_prompt=("FC Mobile football commentary. Ballon d'Or, OVR, CAM, center back, "
                            "Star Pass, Gerd Müller, Lev Yashin, Kaká, Beckenbauer, Cannavaro, "
                            "Gullit, Matthäus, Messi, Ronaldo, Dembélé."
                            if config.domain == "football" else None),
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
        del stream, audio, model
        release_memory()


# %% Batched English-to-Bangla translation

def translate_nllb(segments, config):
    if not segments:
        return {"segments": [], "model_commit": None}
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    device = torch_device(config)
    tokenizer = model = inputs = generated = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.translation_model, revision=config.translation_revision, token=False,
            src_lang="eng_Latn", use_fast=True, trust_remote_code=False,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config.translation_model, revision=config.translation_revision, token=False,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=False,
        ).to(device).eval()
        bangla_id = tokenizer.convert_tokens_to_ids("ben_Beng")
        if bangla_id is None or bangla_id == tokenizer.unk_token_id:
            raise ValueError("Translation tokenizer does not support ben_Beng")
        output = []
        target_limit = min(config.max_target_tokens, 512)
        for offset in range(0, len(segments), config.translation_batch_size):
            batch = segments[offset:offset + config.translation_batch_size]
            texts = [s["english_normalized"] for s in batch]
            lengths = [len(tokenizer.encode(t)) for t in texts]
            if max(lengths) > min(config.max_source_tokens, 480):
                raise ValueError("ASR phrase exceeds translation token limit; reduce max_utterance_words")
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=False).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, forced_bos_token_id=bangla_id,
                    num_beams=config.translation_beams, max_new_tokens=target_limit,
                )
            for segment, ids in zip(batch, generated):
                # NLLB's decoder start token can itself be EOS: ignore the first token.
                nonpadding = [t for t in ids.tolist() if t != tokenizer.pad_token_id]
                if (tokenizer.eos_token_id not in ids[1:].tolist()
                        or len(nonpadding) >= target_limit + 1):
                    # A forced EOS at max_length is still potentially truncated text.
                    raise ValueError(f"Translation hit the generation limit at segment {segment['id']}")
                text = tokenizer.decode(ids, skip_special_tokens=True).strip()
                if not text or not re.search(r"[\u0980-\u09FF]", text):
                    raise ValueError(f"Empty/non-Bengali output at segment {segment['id']}; review source text")
                output.append({**segment, "bangla": text, "bangla_concise": text})
            LOG.info("অনুবাদ হয়েছে: %s/%s অংশ", len(output), len(segments))
        return {"segments": output, "model_commit": getattr(model.config, "_commit_hash", None)}
    finally:
        del inputs, generated, model, tokenizer
        release_memory()


def candidate_issues(segment, text, config):
    source = segment.get("english_normalized", segment["english"])
    glossary = relevant_glossary(source, config.domain, config.glossary)
    return quality_issues(source, text, glossary, config.domain)


def translate_gemini(segments, config, job=None):
    client = GeminiClient(config)
    progress = Path(job) / "translation_progress.json" if job is not None else None
    signature = hashlib.sha256(json.dumps(
        {"segments": segments, "recipe": recipe_id(config)},
        sort_keys=True, ensure_ascii=False,
    ).encode()).hexdigest()
    output, receipts = [], []
    if progress is not None and progress.exists():
        saved = read_json(progress)
        if saved.get("signature") != signature:
            raise ValueError("Gemini progress checkpoint settings/source বদলেছে; checkpoint যাচাই করুন")
        cached = saved["segments"]
        if not isinstance(cached, list) or len(cached) > len(segments):
            raise ValueError("Invalid Gemini progress checkpoint")
        parsed = parse_translations(json.dumps([
            {"id": item["id"], "bangla": item["bangla"], "concise": item["bangla_concise"]}
            for item in cached], ensure_ascii=False), segments[:len(cached)])
        output = [{**source, "bangla": item["bangla"], "bangla_concise": item["concise"]}
                  for source, item in zip(segments, parsed)]
        receipts = saved["receipts"]
        LOG.info("Gemini checkpoint থেকে %s অংশ পুনর্ব্যবহার", len(output))
    for offset in range(len(output), len(segments), config.translation_batch_size):
        batch = segments[offset:offset + config.translation_batch_size]
        before = segments[offset - 1]["english_normalized"] if offset else ""
        after_index = offset + len(batch)
        after = segments[after_index]["english_normalized"] if after_index < len(segments) else ""
        glossary = relevant_glossary(" ".join(s["english_normalized"] for s in batch), config.domain, config.glossary)
        feedback, best, best_issues = "", None, float("inf")
        for attempt in range(config.translation_retries + 1):
            messages = translation_messages(batch, before, after, glossary, feedback, config.domain)
            response, receipt = client.generate(messages, config)
            receipts.append({"cue_ids": [s["id"] for s in batch], "attempt": attempt + 1, **receipt})
            try:
                parsed = parse_translations(response, batch)
            except (ValueError, TypeError):
                feedback = "Invalid JSON/schema. Return a complete array, with exactly the requested IDs and both Bengali versions."
                continue
            errors = []
            for source, item in zip(batch, parsed):
                full = candidate_issues(source, item["bangla"], config)
                concise = candidate_issues(source, item["concise"], config)
                if full or concise:
                    errors.append({"id": source["id"], "full": full, "concise": concise})
            issue_count = sum(len(e["full"]) + len(e["concise"]) for e in errors)
            if issue_count < best_issues:
                best, best_issues = parsed, issue_count
            if not errors:
                break
            feedback = json.dumps(errors, ensure_ascii=False)
        if best is None:
            raise ValueError(f"Gemini JSON/IDs সঠিক নয়, IDs={[s['id'] for s in batch]}; অসম্পূর্ণ অনুবাদ গ্রহণ করা হয়নি")
        for source, item in zip(batch, best):
            output.append({**source, "bangla": item["bangla"], "bangla_concise": item["concise"]})
        if progress is not None:
            atomic_json(progress, {"signature": signature, "segments": output, "receipts": receipts})
        LOG.info("Gemini অনুবাদ হয়েছে: %s/%s অংশ", len(output), len(segments))
    return {"segments": output, "model_commit": None,
            "api": {"provider": "Google Gemini", "requested_model": config.translation_model,
                    "model_versions": sorted({r["model_version"] for r in receipts if r.get("model_version")}),
                    "requests": receipts}}


def translate(segments, config, job=None):
    if not segments:
        return {"segments": [], "model_commit": None}
    prepared = prepare_segments(segments, config)
    result = (translate_gemini(prepared, config, job) if config.translation_backend == "gemini"
              else translate_nllb(prepared, config))
    for segment in result["segments"]:
        segment["translation_checks"] = {
            "full": candidate_issues(segment, segment["bangla"], config),
            "concise": candidate_issues(segment, segment["bangla_concise"], config),
        }
    return result


# %% Local Bengali speech synthesis and duration fitting

class TimingOverflow(ValueError):
    """A complete utterance cannot fit without exceeding the hard speed limit."""


def trim_edge_silence(audio, sample_rate):
    """Only trim low-level outer silence, retaining 50 ms around audible content."""
    import numpy as np
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Invalid TTS waveform")
    peak = float(np.max(np.abs(audio)))
    active = np.flatnonzero(np.abs(audio) > max(1e-5, peak * 0.001))
    if not len(active):
        raise ValueError("TTS produced silent audio")
    margin = round(0.05 * sample_rate)
    return audio[max(0, active[0] - margin):min(len(audio), active[-1] + margin + 1)]


def fit_audio(raw_path, output_path, slot_seconds, config):
    """Hard 0.90–1.25x bounds; never crop speech or exceed the cap on retries."""
    import numpy as np
    import soundfile as sf
    if not (0.90 <= config.min_speed <= 1 <= config.max_speed <= 1.25):
        raise ValueError("Speech speed must stay within 0.90–1.25x")
    cap = min(config.max_speed, config.comfortable_speed) if config.overflow_policy == "error" else config.max_speed
    if not config.min_speed <= cap <= config.max_speed:
        raise ValueError("The effective speed cap is outside the configured bounds")
    native = sf.info(raw_path)
    raw_seconds = native.frames / native.samplerate
    target_samples = round(slot_seconds * config.sample_rate)
    if target_samples <= 0 or raw_seconds <= 0:
        raise ValueError("Audio and timestamp slot must have positive duration")
    # Small atempo window/rounding differences are measured, not solved by truncation.
    if raw_seconds / cap > slot_seconds + 0.08:
        raise TimingOverflow(f"{raw_seconds:.3f}s speech / {slot_seconds:.3f}s slot needs "
                             f"{raw_seconds / slot_seconds:.2f}x; hard limit {cap:.2f}x")
    rate = min(cap, max(config.min_speed, raw_seconds / slot_seconds))
    temp = Path(output_path).with_suffix(".tempo.wav")
    partial = Path(output_path).with_suffix(".partial.wav")
    try:
        for attempt in range(4):
            run_command([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", raw_path, "-af", f"aresample={config.sample_rate}," + atempo_filter(rate),
                "-ar", str(config.sample_rate), "-ac", "1", "-c:a", "pcm_s16le", temp,
            ], config.subprocess_timeout)
            samples, sr = sf.read(temp, dtype="float32")
            if len(samples) <= target_samples:
                break
            proposed = min(cap, rate * len(samples) / target_samples * 1.015)
            if proposed <= rate + 1e-9:
                raise TimingOverflow(f"অডিও {cap:.2f}x গতিতেও slot-এ ধরছে না; অনুবাদ সংক্ষিপ্ত করুন")
            rate = proposed
        else:
            raise TimingOverflow("সীমার মধ্যে sample-exact fit হয়নি; speech কাটা হয়নি")
        if not len(samples) or not np.isfinite(samples).all() or np.max(np.abs(samples)) < 1e-5:
            raise ValueError("Synthesis produced empty, silent or nonfinite audio")
        speech_seconds = len(samples) / sr
        samples = np.pad(samples, (0, target_samples - len(samples)))
        sf.write(partial, samples, sr, subtype="PCM_16")
        os.replace(partial, output_path)
        return {"raw_seconds": raw_seconds, "speed_factor": rate,
                "speech_seconds": speech_seconds,
                "review_required": rate > config.comfortable_speed,
                "slot_seconds": target_samples / config.sample_rate}
    finally:
        temp.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)


def usable_translations(segment, config):
    choices = []
    for variant, text in (("full", segment["bangla"]),
                          ("concise", segment.get("bangla_concise", ""))):
        if not text or any(text == old[1] for old in choices):
            continue
        issues = candidate_issues(segment, text, config)
        if issues:
            LOG.warning("অনুবাদ যাচাই, id=%s %s: %s", segment["id"], variant, "; ".join(issues))
        else:
            choices.append((variant, text))
    return choices


def synthesize(segments, duration, job, config):
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoTokenizer, VitsModel, set_seed
    from uroman import Uroman
    choices = [usable_translations(segment, config) for segment in segments]
    invalid = [s["id"] for s, variants in zip(segments, choices) if not variants]
    if invalid:
        raise ValueError(f"অনুবাদ যাচাই ব্যর্থ, IDs={invalid}। {Path(job) / 'translation.json'}-এ "
                         "bangla লেখা ঠিক করুন; শুধু speed কমালে অর্থের ভুল ঠিক হবে না।")
    device = torch_device(config)
    clips_dir = Path(job) / "clips"
    clips_dir.mkdir(exist_ok=True)
    tokenizer = model = romanizer = tokens = result = None
    records, overflows = [], []
    recipe = recipe_id(config)
    tts_commit = None
    try:
        for i, (segment, variants) in enumerate(zip(segments, choices)):
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
                if (record.get("sha256") == sha256_file(clip)
                        and config.min_speed <= record.get("speed_factor", 0) <= config.max_speed
                        and record.get("bangla") in [text for _, text in variants]):
                    records.append(record)
                    tts_commit = record.get("tts_model_commit", tts_commit)
                    continue
            if model is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    config.tts_model, revision=config.tts_revision, token=False, trust_remote_code=False)
                model = VitsModel.from_pretrained(
                    config.tts_model, revision=config.tts_revision, token=False,
                    trust_remote_code=False,
                ).to(device).eval()
                tts_commit = getattr(model.config, "_commit_hash", None)
                if getattr(tokenizer, "is_uroman", False):
                    romanizer = Uroman()
                    tokenizer.is_uroman = False
            native_rate = model.config.sampling_rate
            raw = clips_dir / f"{i:05d}-raw.wav"
            fitted = False
            for variant, spoken_text in variants:
                set_seed(config.seed + i)
                parts = []
                tts_text = spoken_bengali(spoken_text)
                for text in split_tts_text(tts_text, config.max_tts_chars):
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
                        parts.append(np.zeros(round(0.06 * native_rate), dtype=np.float32))
                    parts.append(part)
                audio = np.concatenate(parts)
                tts_seconds = len(audio) / native_rate
                audio = trim_edge_silence(audio, native_rate)
                audio *= min(1.0, 0.95 / float(np.max(np.abs(audio))))
                sf.write(raw, audio, native_rate, subtype="PCM_16")
                try:
                    timing = fit_audio(raw, clip, slot_seconds, config)
                except TimingOverflow as error:
                    LOG.warning("সময় মিলছে না, id=%s (%s): %s", i, variant, error)
                    continue
                record = {**segment, **timing, "bangla": spoken_text,
                          "bangla_full": segment["bangla"], "spoken_variant": variant,
                          "tts_text": tts_text,
                          "tts_seconds": tts_seconds,
                          "edge_silence_removed": tts_seconds - len(audio) / native_rate,
                          "slot_end": end_sample / config.sample_rate,
                          "start_sample": start_sample, "clip": clip.name,
                          "sha256": sha256_file(clip), "tts_model_commit": tts_commit}
                record["translation_checks"] = {
                    "full": candidate_issues(segment, segment["bangla"], config),
                    "concise": candidate_issues(segment, segment.get("bangla_concise", ""), config),
                }
                record["review_required"] = bool(record["review_required"] or segment.get("boundary_review")
                                                   or segment.get("source_review") or segment.get("source_corrections")
                                                   or variant == "concise")
                atomic_json(metadata, record)
                records.append(record)
                fitted = True
                LOG.info("বাংলা অডিও: %s/%s, %s, %.3fx", i + 1, len(segments), variant, timing["speed_factor"])
                break
            raw.unlink(missing_ok=True)
            if not fitted:
                overflows.append(i)
        if overflows:
            raise TimingOverflow(f"IDs={overflows}: বৈধ অনুবাদ বিকল্পগুলোর speech নির্ধারিত speed সীমায় ধরেনি। "
                                 f"{Path(job) / 'translation.json'}-এ অর্থ রেখে bangla সংক্ষিপ্ত করুন। "
                                 "সফল clip রাখা হয়েছে; speech কাটা/overlap বা বেশি speed-এ প্রকাশ করা হয়নি।")
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
            if not config.min_speed <= record["speed_factor"] <= config.max_speed:
                raise ValueError("Cached speech exceeds the hard speed bounds")
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
            LOG.warning("ভিডিও stream-copy হয়নি; H.264-এ encode করা হচ্ছে")
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
        LOG.info("আগের checkpoint ব্যবহার হচ্ছে: %s", path.name)
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
            job / "translation.json", lambda: translate(asr["segments"], config, job)))
        validate_segments(translations["segments"], info["duration"], translated=True)
        if not translations["segments"]:
            LOG.warning("কথা শনাক্ত হয়নি; ফলাফলে নীরব audio track থাকবে")
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
                    "quality_status": "human_review_required",
                    "speed_limits": {"min": config.min_speed, "max": config.max_speed},
                    "translation_backend": config.translation_backend,
                    "translation_api": translations.get("api"),
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
    except Exception:
        # Errors go to the terminal only; media/checkpoints remain resumable.
        LOG.error("%s ধাপে সমস্যা। Checkpoint রাখা হয়েছে: %s", current_stage, job)
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
    if config.translation_backend == "gemini":
        # Fail before expensive ASR if the exact requested model/key is unavailable.
        GeminiClient(config).check_model(config)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    results = []
    for source in sources:
        LOG.info("কাজ চলছে: %s", source.name)
        try:
            results.append(process_video(source, config))
        except Exception as error:
            LOG.exception("অনুবাদ হয়নি: %s", source.name)
            results.append({"source": str(source), "status": "failed",
                            "error": f"{type(error).__name__}: {error}"})
            if not config.continue_on_error:
                raise
    return results
