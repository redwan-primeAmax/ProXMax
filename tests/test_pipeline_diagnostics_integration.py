"""Offline synthetic pipeline: real extraction, sync, mux and per-second report."""
from __future__ import annotations

import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from benchmarks.pipeline_perf import FakeGemini
from proxmax import audio_fx
from proxmax.audio_extraction import AudioExtractor
from proxmax.config import AppConfig
from proxmax.gemini_translator import GeminiTranslator
from proxmax.models import Segment, Word
from proxmax.muxer import VideoMuxer
from proxmax.pipeline import DubbingPipeline
from proxmax.sync_engine import SyncEngine
from proxmax.video_render import _frame_count
from tests._support import FFMPEG


@unittest.skipUnless(FFMPEG, "FFmpeg required")
class TestSyntheticPipelineDiagnostics(unittest.TestCase):
    def test_real_media_pipeline_records_three_clocks_and_keeps_all_clips(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inp, out, work = root / "in", root / "out", root / "work"
            inp.mkdir(); out.mkdir(); work.mkdir()
            audio = root / "source.wav"
            rate = 16000
            t = np.arange(round(rate * 5.072)) / rate
            audio_fx.write_wav(audio, (.15 * np.sin(2 * math.pi * 220 * t)).astype(np.float32), rate)
            source = inp / "synthetic.mp4"
            subprocess.run([FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i",
                            "testsrc2=size=160x120:rate=25:duration=5", "-i", str(audio),
                            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(source)],
                           check=True, capture_output=True)
            segments = [
                Segment(0, .2, .8, "First line", words=[Word(.2, .7, "First", .9)]),
                Segment(1, 1.2, 2.2, "Second line", words=[Word(1.2, 1.8, "Second", .3)]),
                Segment(2, 3.4, 4.6, "Third line", words=[Word(3.4, 4.0, "Third", .8)]),
            ]
            class STT:
                last_stats = {"synthetic": True}
                def transcribe(self, _audio): return segments
                def release(self): pass
            class TTS:
                name = "offline-fake"
                def describe(self): return "offline-fake"
                def set_budget(self, *_args): pass
                def synthesize(self, units, folder):
                    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
                    for unit in units:
                        duration = (.4, 1.7, .65)[unit.index]
                        x = np.arange(round(24000 * duration)) / 24000
                        path = folder / f"clip_{unit.index}.wav"
                        audio_fx.write_wav(path, (.2 * np.sin(2 * math.pi * (330 + unit.index * 70) * x)).astype(np.float32), 24000)
                        unit.tts_audio_path, unit.tts_duration = path, duration
            cfg = AppConfig(input_dir=str(inp), output_dir=str(out), work_dir=str(work),
                            skip_existing=False, save_report=True, enable_overlap=True,
                            gemini_batch_size=1, gemini_concurrency=2).validate()
            pipeline = DubbingPipeline.__new__(DubbingPipeline)
            pipeline.config = cfg
            pipeline.extractor = AudioExtractor(use_hwaccel=False)
            pipeline.stt, pipeline._tts = STT(), TTS()
            pipeline.translator = GeminiTranslator(api_key="FAKE_SECRET_DO_NOT_REPORT", client=FakeGemini(.005),
                                                    batch_size=1, requests_per_minute=1_000_000)
            pipeline.sync_engine = SyncEngine(sample_rate=24000, max_workers=2, prefer_rubberband=False)
            pipeline.muxer = VideoMuxer(use_hwaccel=False, video_workers=2)
            with mock.patch("proxmax.pipeline_process.build_units", return_value=[]):
                result = pipeline.process_one(source, out)
            self.assertEqual(result.status, "success", result.error)
            self.assertEqual(result.segment_count, 3)
            self.assertEqual(result.stats["sync"]["unplaced"], 0)
            self.assertEqual(result.stats["sync"]["overlaps"], 0)
            self.assertTrue(all(segment.placed_duration is not None for segment in segments))
            self.assertTrue(all(.9 <= segment.tempo_used <= 1 for segment in segments))
            self.assertEqual(_frame_count(result.output_path), pipeline.sync_engine.video_plan.output_frames)
            self.assertTrue(result.stats["picture_preparation"]["details"].get("virtual_tail"))
            report = (out / "synthetic_report.markdown").read_text(encoding="utf-8")
            for expected in ("Source clock — every second", "Edited output clock — every second",
                             "Every processing second", "gemini_attempt_started", "sync_clip_finished",
                             "video_group_finished", "Post-mux waveform status: **ok**", "low-confidence word"):
                self.assertIn(expected, report)
            self.assertNotIn("FAKE_SECRET_DO_NOT_REPORT", report)
            self.assertNotIn(str(root), report)
            self.assertFalse((work / "synthetic").exists())


if __name__ == "__main__":
    unittest.main()
