"""Real source stream offsets remain aligned through extraction and picture edits."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from proxmax import audio_fx
from proxmax.audio_extraction import AudioExtractor
from proxmax.models import Segment
from proxmax.muxer import VideoMuxer
from proxmax.sync_engine import SyncEngine
from tests._support import FFMPEG


@unittest.skipUnless(FFMPEG and shutil.which("ffprobe"), "ffmpeg and ffprobe required")
class TestSourceStreamOffsets(unittest.TestCase):
    def source(self, output, delay_video):
        filt = "[0:v]setpts=PTS+0.4/TB[v]" if delay_video else "[1:a]asetpts=PTS+0.4/TB[a]"
        maps = ["-map", "[v]", "-map", "1:a"] if delay_video else ["-map", "0:v", "-map", "[a]"]
        proc = subprocess.run([
            FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=96x64:r=25:d=2",
            "-f", "lavfi", "-i", "sine=frequency=220:duration=2", "-filter_complex", filt,
            *maps, "-vsync", "0", "-c:v", "libx264", "-c:a", "aac", str(output),
        ], capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode(errors="replace"))

    def frames(self, path):
        out = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-vsync", "0", "-f", "rawvideo",
                              "-pix_fmt", "gray", "-"], check=True, capture_output=True).stdout
        return np.frombuffer(out, np.uint8).reshape(-1, 64, 96)

    def test_late_audio_is_padded_before_stt_instead_of_moved_to_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "late audio.mp4"
            self.source(source, delay_video=False)
            for audio_filter in ("none", "highpass60"):
                with self.subTest(audio_filter=audio_filter):
                    wav = AudioExtractor(audio_filter=audio_filter, sample_rate=8000).extract(
                        source, root / f"{audio_filter}.wav"
                    )
                    audio, sr = audio_fx.read_wav(wav)
                    first_sound = np.flatnonzero(np.abs(audio) > .01)[0] / sr
                    self.assertAlmostEqual(first_sound, .4, delta=.025)
                    self.assertGreater(len(audio) / sr, 2.39)

    def test_pipeline_uses_measured_streams_when_container_hint_is_wrong(self):
        from unittest import mock
        from proxmax.config import AppConfig
        from proxmax.pipeline import DubbingPipeline

        class STT:
            last_stats = {}
            first_sound = None
            def transcribe(self, path):
                samples, rate = audio_fx.read_wav(path)
                self.first_sound = np.flatnonzero(np.abs(samples) > .01)[0] / rate
                return [Segment(0, .4, .8, "source speech")]
        class Translator:
            def translate_segments(self, segments):
                for segment in segments:
                    segment.text_bn = "বাংলা বক্তব্য।"
        class TTS:
            def describe(self): return "fake"
            def set_budget(self, *_args): pass
            def synthesize(self, segments, directory):
                Path(directory).mkdir(parents=True, exist_ok=True)
                for segment in segments:
                    path = Path(directory) / f"{segment.index}.wav"
                    t = np.arange(7200) / 24000
                    audio_fx.write_wav(path, (.2 * np.sin(2 * np.pi * 350 * t)).astype(np.float32), 24000)
                    segment.tts_audio_path, segment.tts_duration = path, .3

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "late audio.mp4"
            self.source(source, delay_video=False)
            for hint in (.2, 20.0):
                with self.subTest(hint=hint):
                    p = DubbingPipeline.__new__(DubbingPipeline)
                    p.config = AppConfig(enable_overlap=False).validate()
                    p.extractor, p.stt, p.translator = AudioExtractor(), STT(), Translator()
                    p._tts, p.muxer = TTS(), VideoMuxer(use_hwaccel=False)
                    p.sync_engine = SyncEngine(sample_rate=24000, max_workers=1, prefer_rubberband=False)
                    with mock.patch("proxmax.pipeline_process.get_media_duration", return_value=hint), \
                         mock.patch("proxmax.pipeline_process.build_units", return_value=[]):
                        segments, timing, stats = p._run_steps(source, root / f"work-{hint}", root / f"out-{hint}.mp4")
                    self.assertAlmostEqual(p.stt.first_sound, .4, delta=.025)
                    self.assertAlmostEqual(timing.audio_seconds, 2.42, delta=.05)
                    self.assertAlmostEqual(segments[0].actual_start, .4, places=2)
                    self.assertEqual(stats["sync"]["unplaced"], 0)
                    self.assertLess(p.sync_engine.video_plan.duration, 4)
                    self.assertGreater(p.sync_engine.video_plan.duration, 2)

    def test_late_video_is_not_shifted_earlier_when_picture_is_edited(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "late video.mp4"
            self.source(source, delay_video=True)
            muxer = VideoMuxer(use_hwaccel=False)
            fps = muxer.video_frame_rate(source)
            self.assertAlmostEqual(muxer._video_start_offset(source), .4, places=2)
            aligned = muxer.ensure_picture_duration(source, 2.4, root, fps)
            original, frames = self.frames(source), self.frames(aligned)
            self.assertEqual((len(original), len(frames)), (50, 60))
            self.assertLess(float(frames[:10].mean()), 3.0)
            self.assertLess(np.mean(np.abs(frames[10:].astype(float) - original.astype(float))), 4)
            self.assertAlmostEqual(muxer.picture_timeline_duration(source, fps), 2.4)

            t = np.arange(8000) / 8000
            audio_fx.write_wav(root / "line.wav", (.2 * np.sin(2 * np.pi * 330 * t)).astype(np.float32), 8000)
            seg = Segment(0, .8, 1.1, "line", "বাক্য", tts_audio_path=root / "line.wav")
            sync = SyncEngine(sample_rate=8000, slot_gap_ms=0, borrow_silence_ms=0)
            dub = sync.build_final_track([seg], 2.4, root, root / "dub.wav", video_fps=fps)
            output = root / "out.mp4"
            muxer.mux(aligned, dub, output, video_plan=sync.video_plan,
                      required_video_duration=sync.video_plan.duration)
            actual = self.frames(output)
            self.assertLess(np.mean(np.abs(actual[:28].astype(float) - frames[:28].astype(float))), 4)
            self.assertAlmostEqual(seg.actual_start, .8, places=3)
            self.assertEqual(sync.last_stats["unplaced"], 0)


if __name__ == "__main__":
    unittest.main()
