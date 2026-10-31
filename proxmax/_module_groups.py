"""Domain folders with one shared module identity for old and new import paths.

Legacy code loads ``proxmax.stt`` (etc.) directly; adding the physical domain
folders to the package search path preserves that name and its relative imports.
The domain packages then alias the already-loaded modules, so patching either
``proxmax.stt`` or ``proxmax.speech.stt`` affects the same object.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

GROUPS = {
    "audio": ("audio_extraction", "audio_fx", "audio_quality", "ffmpeg_utils"),
    "speech": ("chunking", "hotword_repair", "prosody", "segmenter", "speech_rate",
               "stt", "stt_inference", "stt_seams", "whisper_model"),
    "translation": ("gemini_operations", "gemini_protocol", "gemini_repair", "gemini_token_budget",
                    "gemini_translator", "indictrans2_translator", "translation_errors",
                    "translation_metrics", "translation_parsing", "translation_quality", "translator",
                    "translator_base", "translator_config"),
    "sync": ("sync_assembly", "sync_cache", "sync_engine", "sync_math", "sync_prepare", "sync_silence"),
    "video": ("muxer", "video_render", "video_timeline", "virtual_picture"),
    "reports": ("reporting", "reporting_activity", "reporting_common", "reporting_failure",
                "reporting_media", "reporting_packets", "reporting_resources", "reporting_runtime",
                "reporting_seconds", "reporting_timeline"),
    "workflow": ("discovery", "pipeline", "pipeline_batch", "pipeline_helpers", "pipeline_integrity",
                 "pipeline_parallel", "pipeline_process", "pipeline_translation", "pipeline_validation"),
}


def extend_package_path(search_path, root: Path) -> None:
    """Allow historic ``proxmax.<module>`` imports to find moved files."""
    for group in GROUPS:
        directory = str(root / group)
        if directory not in search_path:
            search_path.append(directory)


def alias_group(package_name: str) -> None:
    """Expose ``proxmax.<group>.<module>`` as the exact legacy module object."""
    parent, group = package_name.rsplit(".", 1)
    package = sys.modules[package_name]
    for name in GROUPS[group]:
        module = importlib.import_module(f"{parent}.{name}")
        sys.modules[f"{package_name}.{name}"] = module
        setattr(package, name, module)
