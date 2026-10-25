"""Audio I/O and lightweight DSP helpers (numpy only, no extra dependency).

এই মডিউলটাই পুরো অডিও স্তরের ভিত্তি: ffmpeg দিয়ে যেকোনো ফাইলকে
float32 mono-তে decode করা, WAV পড়া/লেখা, peak/RMS normalize, fade,
নীরবতা কাটা — সব এখানে। কোনো ভারী লাইব্রেরি (scipy/librosa) ছাড়াই
কাজ হয়, তাই Colab-এ GPU/সময় বাঁচে।

সব ফাংশন 1-D ``numpy.float32`` অ্যারে নিয়ে কাজ করে, মানের পরিসর
``[-1.0, 1.0]``।
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .utils import capture_bytes, ensure_tool, get_logger

logger = get_logger(__name__)

#: শূন্যের খুব কাছের মান — log হিসাবের সময় 0 division ঠেকায়
_EPS = 1e-9


# =============================================================================
# Decode / encode
# =============================================================================
def decode_audio(
    path: Path | str,
    sample_rate: int = 16000,
    filters: Optional[str] = None,
) -> np.ndarray:
    """Decode any media file into a mono float32 waveform via ffmpeg.

    একবারই পুরো অডিও মেমরিতে আসে — বারবার ফাইল পড়ার খরচ (Drive/FUSE-এ
    খুব ধীর) এড়ানো যায়। ১০ মিনিটের অডিও 16 kHz float32 ≈ ৩৮ MB,
    তাই T4-এ মেমরি সংকট নেই।

    Args:
        path: অডিও/ভিডিও ফাইলের পাথ।
        sample_rate: আউটপুট sample rate (STT-এর জন্য 16000, TTS-এর জন্য 24000)।
        filters: ঐচ্ছিক ffmpeg audio filter (যেমন ``highpass=f=70,loudnorm=I=-16``)।

    Returns:
        ``numpy.float32`` অ্যারে, amplitude ``[-1, 1]``।

    Raises:
        RuntimeError: ফাইল decode করা না গেলে (corrupt বা অডিও stream নেই)।
    """
    ffmpeg = ensure_tool("ffmpeg")
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(int(sample_rate)),
    ]
    if filters:
        cmd += ["-af", filters]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "-"]
    # stdin বন্ধ + stall detection: নোটবুকে ffmpeg যেন কখনো ঝুলে না যায়
    raw, stderr_text, returncode = capture_bytes(
        cmd, description=f"অডিও decode ({Path(path).name})"
    )
    if returncode != 0:
        tail = (stderr_text or "").strip()[-500:]
        raise RuntimeError(f"অডিও decode করা যায়নি ({Path(path).name}): {tail or 'unknown ffmpeg error'}")
    audio = np.frombuffer(raw, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError(f"অডিও ফাইল খালি (কোনো sample নেই): {Path(path).name}")
    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)


def read_wav(path: Path | str) -> Tuple[np.ndarray, int]:
    """Read a PCM WAV file into ``(float32 mono array, sample_rate)``."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:  # pragma: no cover - 24-bit WAV খুব কম দেখা যায়
        raise ValueError(f"অসমর্থিত WAV sample width: {width} bytes ({path})")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), rate


def write_wav(path: Path | str, samples: np.ndarray, sample_rate: int) -> Path:
    """Write a mono float32 array as a 16-bit PCM WAV file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2").tobytes()
    with wave.open(str(out), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm)
    return out


# =============================================================================
# Level helpers
# =============================================================================
def amp_to_db(value: float) -> float:
    """Amplitude → dBFS (``0.0`` → ``-inf``)।"""
    return 20.0 * float(np.log10(max(abs(float(value)), _EPS)))


def db_to_amp(db: float) -> float:
    """dBFS → amplitude।"""
    return float(10.0 ** (float(db) / 20.0))


def rms_dbfs(audio: np.ndarray) -> float:
    """Root-mean-square level of a clip in dBFS (silence → ``-100``)."""
    if audio.size == 0:
        return -100.0
    return amp_to_db(float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))))


def peak_normalize(audio: np.ndarray, peak_dbfs: float = -1.0) -> np.ndarray:
    """Scale a clip so its loudest sample sits at ``peak_dbfs``."""
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak < _EPS:
        return audio
    return (audio * (db_to_amp(peak_dbfs) / peak)).astype(np.float32)


def rms_normalize(audio: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    """Scale a clip to a target RMS level, then guard against clipping."""
    if audio.size == 0:
        return audio
    current = rms_dbfs(audio)
    if current <= -90.0:
        return audio
    scaled = audio * db_to_amp(target_dbfs - current)
    return np.clip(scaled, -0.98, 0.98).astype(np.float32)


def fade(audio: np.ndarray, sample_rate: int, in_ms: int = 12, out_ms: int = 40) -> np.ndarray:
    """Apply short linear fade-in/fade-out (ক্লিক/pop আটকায়)।"""
    if audio.size == 0:
        return audio
    out = audio.copy()
    n_in = min(int(sample_rate * max(0, in_ms) / 1000), out.size)
    if n_in > 0:
        out[:n_in] *= np.linspace(0.0, 1.0, n_in, dtype=np.float32)
    n_out = min(int(sample_rate * max(0, out_ms) / 1000), out.size)
    if n_out > 0:
        out[-n_out:] *= np.linspace(1.0, 0.0, n_out, dtype=np.float32)
    return out


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_dbfs: float = -45.0,
    pad_ms: int = 40,
) -> np.ndarray:
    """Trim leading/trailing near-silence, keeping ``pad_ms`` of padding."""
    if audio.size == 0:
        return audio
    frame = max(1, int(sample_rate * 0.01))  # 10 ms
    n_frames = audio.size // frame
    if n_frames == 0:
        return audio
    frames = audio[: n_frames * frame].reshape(n_frames, frame)
    loud = np.max(np.abs(frames), axis=1) > db_to_amp(threshold_dbfs)
    if not bool(loud.any()):
        return audio
    first = int(np.argmax(loud))
    last = int(len(loud) - 1 - np.argmax(loud[::-1]))
    pad = int(sample_rate * pad_ms / 1000)
    start = max(0, first * frame - pad)
    end = min(audio.size, (last + 1) * frame + pad)
    return audio[start:end]


def silence(seconds: float, sample_rate: int) -> np.ndarray:
    """Return a zero-filled array of the requested duration."""
    return np.zeros(max(0, int(round(seconds * sample_rate))), dtype=np.float32)
