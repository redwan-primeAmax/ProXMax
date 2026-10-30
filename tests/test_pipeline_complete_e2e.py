"""Complete fallback through real audio extraction and held-frame video muxing."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from proxmax import audio_fx
from proxmax.audio_extraction import AudioExtractor
from proxmax.audio_quality import assess_bengali_coverage
from proxmax.config import AppConfig
from proxmax.models import Segment, Word
from proxmax.muxer import VideoMuxer
from proxmax.pipeline import DubbingPipeline
from proxmax.prosody import subtitles_for_unit
from proxmax.reporting_timeline import _unit_timing_rows
from proxmax.sync_engine import SyncEngine
from tests._support import FFMPEG


@unittest.skipUnless(FFMPEG and shutil.which("ffprobe"), "ffmpeg and ffprobe required")
class TestCompletePipeline(unittest.TestCase):
    def test_conflicting_windows_preserve_all_speech_and_extend_video(self):
        class STT:
            last_stats = {}
            def transcribe(self, _path):
                return [Segment(0, 0, 0, "First sentence"),
                        Segment(1, 0, .1, "Second sentence")]
        class Translator:
            calls = 0
            def translate_segments(self, segments):
                for seg in segments:
                    seg.text_bn = "প্রথম বাক্য।" if seg.index == 0 else "দ্বিতীয় বাক্য।"
            def shorten_to_durations(self, segments, targets):
                self.calls += 1  # no change: original clips must survive
        class TTS:
            def describe(self):
                return "fake"
            def set_budget(self, *_args):
                pass
            def synthesize(self, segments, output_dir):
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                for seg in segments:
                    rate = 24000
                    samples = np.arange(round(.6 * rate)) / rate
                    clip = (.2 * np.sin(2 * np.pi * (300 + seg.index * 120) * samples)).astype(np.float32)
                    path = output_dir / f"seg_{seg.index:04d}.wav"
                    audio_fx.write_wav(path, clip, rate)
                    seg.tts_audio_path, seg.tts_duration = path, .6

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source.mp4", root / "out.mp4"
            proc = subprocess.run([
                FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=red:s=128x96:r=25:d=0.52",
                "-f", "lavfi", "-i", "sine=frequency=220:duration=0.52",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source),
            ], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            pipeline = DubbingPipeline.__new__(DubbingPipeline)
            pipeline.config = AppConfig(enable_overlap=False).validate()
            pipeline.extractor, pipeline.stt, pipeline.translator = AudioExtractor(), STT(), Translator()
            pipeline._tts, pipeline.muxer = TTS(), VideoMuxer(use_hwaccel=False)
            pipeline.sync_engine = SyncEngine(sample_rate=24000, max_workers=1, prefer_rubberband=False)
            with mock.patch("proxmax.pipeline_process.build_units", return_value=[]):
                segments, _timings, stats = pipeline._run_steps(source, root / "work", output)

            self.assertTrue(output.is_file())
            self.assertEqual(pipeline.translator.calls, 1)
            self.assertEqual(stats["sync"]["unplaced"], 0)
            self.assertEqual(stats["sync"]["overlaps"], 0)
            self.assertGreater(stats["sync"]["video_extension_seconds"], .5)
            self.assertEqual([s.start for s in segments], [0, 0])
            self.assertAlmostEqual(segments[1].placed_start, .68, places=3)
            self.assertAlmostEqual(segments[0].placed_duration, .6, places=3)
            self.assertAlmostEqual(segments[1].placed_duration, .6, places=3)
            speech_end = segments[-1].actual_start + segments[-1].placed_duration
            self.assertGreaterEqual(pipeline.muxer._video_stream_duration(output), speech_end)

            rows = _unit_timing_rows(segments, .52, pipeline.config)
            self.assertEqual(rows[1]["actual_start"], segments[1].placed_start)
            self.assertEqual(rows[1]["unit"].start, 0)
            cue = subtitles_for_unit(segments[1])[0]
            self.assertAlmostEqual(cue.start, segments[1].placed_start, places=3)
            self.assertAlmostEqual(cue.end, speech_end, places=3)
            words = [Word(.72, .82, "দ্বিতীয়"), Word(.85, .95, "বাক্য।")]
            coverage = assess_bengali_coverage([segments[1]], words)
            self.assertEqual(coverage["units"][0]["status"], "likely_complete")
            self.assertEqual(coverage["units"][0]["source_start"], 0.0)
            self.assertAlmostEqual(coverage["units"][0]["start"], .68, places=3)


if __name__ == "__main__":
    unittest.main()
