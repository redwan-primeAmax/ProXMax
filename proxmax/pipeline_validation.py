"""Optional multilingual ASR selection for post-mux Bengali coverage."""

from __future__ import annotations

from .stt import SpeechToText
from .utils import get_logger
from .whisper_model import is_english_only_model

logger = get_logger(__name__)


def bengali_validator(source_stt, config):
    """Return (STT, owned), loading a separate multilingual model only if needed.

    This helper is reached only when post-mux validation is explicitly enabled.
    It can download a large model and adds a complete extra transcription pass.
    """
    if not is_english_only_model(getattr(source_stt, "model_size", "")):
        return source_stt, False
    logger.info("Bengali QA requires multilingual Whisper; loading separate large-v3 (optional extra pass).")
    if getattr(source_stt, "_model", None) is not None:
        source_stt.release()  # avoid holding both large models in VRAM at once
    validator = SpeechToText(
        model_size="large-v3",
        device=config.whisper_device,
        compute_type=config.whisper_compute_type,
        beam_size=config.whisper_beam_size,
        vad_min_silence_ms=config.vad_min_silence_ms,
        chunk_min_seconds=config.chunk_min_seconds,
        chunk_max_seconds=config.chunk_max_seconds,
        chunk_pad_ms=config.chunk_pad_ms,
    )
    return validator, True
