"""No untranslated or missing voice-over unit is silently accepted."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from proxmax import audio_fx
from proxmax.config import AppConfig
from proxmax.models import Segment, Word
from proxmax.pipeline import DubbingPipeline
from proxmax.pipeline_integrity import recover_invalid_waveforms, recover_missing_speech
from proxmax.pipeline_helpers import _report_is_usable
from proxmax.prosody import build_units
from proxmax.sync_engine import SyncEngine
from tests._support import FFMPEG


def make_audio(path, seconds, rate=24000):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(round(rate * seconds)) / rate
    audio_fx.write_wav(path, (.2 * np.sin(2 * np.pi * 330 * t)).astype(np.float32), rate)


class TestSpeechAccounting(unittest.TestCase):
    def test_prior_report_with_missing_speech_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old_report.markdown"
            template = ("# report\n" + "diagnostic " * 40 + "\n"
                        "**Transcript units:** 3; **source STT pieces:** 3; "
                        "**translated units:** {}; **units with TTS clips:** {}; **placed units:** {}.\n"
                        "**unplaced clips:** {};")
            for translated, tts, placed, unplaced, usable in ((3, 3, 3, 0, True),
                    (3, 3, 2, 1, False), (2, 2, 2, 0, False)):
                with self.subTest(unplaced=unplaced, translated=translated):
                    path.write_text(template.format(translated, tts, placed, unplaced), encoding="utf-8")
                    self.assertEqual(_report_is_usable(path), usable)

    def test_zero_and_short_source_segments_survive_unit_building(self):
        segs = [Segment(0, 0, 0, "First"), Segment(1, .01, .02, "tiny"),
                Segment(2, 1, 2, "Complete sentence.")]
        units = build_units(segs)
        self.assertEqual([child.index for unit in units for child in unit.children], [0, 1, 2])
        self.assertIn("First", " ".join(unit.text_en for unit in units))
        self.assertIn("tiny", " ".join(unit.text_en for unit in units))

    def test_targeted_translation_and_tts_recovery_does_not_redo_good_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "good.wav"
            make_audio(first, .2)
            segments = [Segment(0, 0, 1, "good", "ভালো", tts_audio_path=first),
                        Segment(1, 1, 2, "missing translation"),
                        Segment(2, 2, 3, "missing audio", "অডিও")]
            pipeline = mock.Mock()
            def translate(group):
                self.assertEqual([s.index for s in group], [1])
                group[0].text_bn = "অনুবাদ"
            def synthesize(group, destination):
                self.assertEqual([s.index for s in group], [1, 2])
                for seg in group:
                    path = destination / f"{seg.index}.wav"
                    make_audio(path, .2)
                    seg.tts_audio_path = path
            pipeline.translator.translate_segments.side_effect = translate
            pipeline._synthesize.side_effect = synthesize
            recover_missing_speech(pipeline, segments, root)
            self.assertEqual(segments[0].tts_audio_path, first)
            self.assertTrue(all(s.tts_audio_path.is_file() for s in segments))
            self.assertEqual(pipeline.translator.translate_segments.call_count, 1)
            self.assertEqual(pipeline._synthesize.call_count, 1)

    def test_failed_recovery_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = mock.Mock()
            untranslated = Segment(1, 0, 1, "English")
            with self.assertRaisesRegex(RuntimeError, "Missing Bengali translation.*1"):
                recover_missing_speech(pipeline, [untranslated], root)
            missing_audio = Segment(2, 1, 2, "English", "বাংলা")
            with self.assertRaisesRegex(RuntimeError, "Missing TTS audio.*2"):
                recover_missing_speech(pipeline, [missing_audio], root)
            missing_audio.tts_audio_path = root / "nonexistent.wav"
            pipeline._synthesize.side_effect = OSError("network failed")
            with self.assertRaisesRegex(RuntimeError, "TTS recovery failed.*2"):
                recover_missing_speech(pipeline, [missing_audio], root)

    def test_silent_waveform_is_targeted_for_resynthesis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            silent = root / "silent.wav"
            audio_fx.write_wav(silent, np.zeros(2400, np.float32), 24000)
            seg = Segment(0, 0, 1, "English", "বাংলা", tts_audio_path=silent,
                          sync_issue="stretched audio measurement failed (ValueError)")
            pipeline = mock.Mock()
            def synthesize(group, destination):
                path = destination / "recovered.wav"
                make_audio(path, .2)
                group[0].tts_audio_path = path
            pipeline._synthesize.side_effect = synthesize
            seconds, count = recover_invalid_waveforms(pipeline, [seg], root)
            self.assertEqual(count, 1)
            self.assertGreaterEqual(seconds, 0)
            self.assertEqual(seg.tts_audio_path.name, "recovered.wav")
            pipeline._synthesize.assert_called_once()

    def test_words_without_source_transcript_fail_before_translation(self):
        class Extractor:
            def extract(self, source, target, **_kwargs):
                make_audio(target, 1)
                return target
        class STT:
            last_stats = {}
            def transcribe(self, _path):
                return [Segment(0, 0, .5, "valid"), Segment(1, .5, 1, words=[Word(.6, .8, "name")])]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = DubbingPipeline.__new__(DubbingPipeline)
            pipeline.config, pipeline.extractor, pipeline.stt = AppConfig(), Extractor(), STT()
            with mock.patch("proxmax.pipeline_process.get_media_duration", return_value=1), \
                 self.assertRaisesRegex(RuntimeError, "Recognized source words have no transcript"):
                pipeline._run_steps(root / "video.mp4", root, root / "out.mp4")


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class TestNoRewriteOrchestration(unittest.TestCase):
    def test_silent_tts_is_retried_before_final_sync(self):
        class Extractor:
            def extract(self, source, target, **_kwargs):
                make_audio(target, 2)
                return target
        class STT:
            last_stats = {}
            def transcribe(self, _path):
                return [Segment(0, 0, .8, "First"), Segment(1, 1, 1.8, "Second")]
        class Translator:
            def translate_segments(self, segments):
                for seg in segments:
                    seg.text_bn = "বাংলা বাক্য।"
        class TTS:
            calls = 0
            def describe(self):
                return "fake"
            def set_budget(self, *_args):
                pass
            def synthesize(self, segments, output_dir):
                self.calls += 1
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                for seg in segments:
                    path = Path(output_dir) / f"{seg.index}.wav"
                    if self.calls == 1 and seg.index == 0:
                        audio_fx.write_wav(path, np.zeros(7200, np.float32), 24000)
                    else:
                        make_audio(path, .3)
                    seg.tts_audio_path, seg.tts_duration = path, .3
        class Muxer:
            def video_frame_rate(self, source): return 25
            def ensure_picture_duration(self, source, required, work, fps): return source
            def mux(self, source, audio, output, **_kwargs):
                Path(output).write_bytes(b"stub")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = DubbingPipeline.__new__(DubbingPipeline)
            pipeline.config = AppConfig(enable_overlap=False).validate()
            pipeline.extractor, pipeline.stt, pipeline.translator = Extractor(), STT(), Translator()
            pipeline._tts, pipeline.muxer = TTS(), Muxer()
            pipeline.sync_engine = SyncEngine(sample_rate=24000, max_workers=1, prefer_rubberband=False)
            with mock.patch("proxmax.pipeline_process.get_media_duration", return_value=2), \
                 mock.patch("proxmax.pipeline_process.build_units", return_value=[]):
                segments, _timings, stats = pipeline._run_steps(root / "video.mp4", root, root / "out.mp4")
            self.assertEqual(pipeline._tts.calls, 2)
            self.assertEqual(stats["tts"]["recovered_invalid_waveforms"], 1)
            self.assertEqual(stats["sync"]["unplaced"], 0)
            self.assertTrue(all(s.placed_duration for s in segments))

    def _run(self, root, mode):
        class Extractor:
            def extract(self, source, target, **_kwargs):
                make_audio(target, 2)
                return target
        class STT:
            last_stats = {}
            def transcribe(self, _path):
                return [Segment(0, 0, .5, "We have 126 OVR"), Segment(1, .8, 1.0, "Next card")]
        class Translator:
            calls = 0
            def translate_segments(self, segments):
                for seg in segments:
                    seg.text_bn = "আমাদের কাছে ১২৬ ওভিআর আছে।" if seg.index == 0 else "পরের কার্ড।"
            def shorten_to_durations(self, segments, targets):
                self.calls += 1
                if mode == "error":
                    raise RuntimeError("Gemini failed")
                segments[0].text_bn = "আমাদের কাছে ২৬ ওভিআর আছে।"  # lose a digit
        class TTS:
            def describe(self):
                return "fake"
            def set_budget(self, *_args):
                pass
            def synthesize(self, segments, output_dir):
                for seg in segments:
                    path = Path(output_dir) / f"seg_{seg.index}.wav"
                    duration = 2.0 if seg.index == 0 else .25
                    make_audio(path, duration)
                    seg.tts_audio_path, seg.tts_duration = path, duration
        class Muxer:
            def video_frame_rate(self, source): return 25
            def ensure_picture_duration(self, source, required, work, fps): return source
            def mux(self, source, audio, output, **_kwargs):
                Path(output).write_bytes(b"stub")
        pipeline = DubbingPipeline.__new__(DubbingPipeline)
        pipeline.config = AppConfig(enable_overlap=False).validate()
        pipeline.extractor, pipeline.stt, pipeline.translator = Extractor(), STT(), Translator()
        pipeline._tts, pipeline.muxer = TTS(), Muxer()
        pipeline.sync_engine = SyncEngine(sample_rate=24000, max_workers=1, prefer_rubberband=False)
        with mock.patch("proxmax.pipeline_process.get_media_duration", return_value=2), \
             mock.patch("proxmax.pipeline_process.build_units", return_value=[]):
            result = pipeline._run_steps(root / "video.mp4", root, root / "out.mp4")
        return pipeline, result

    def test_overlong_audio_never_calls_rewrite_and_uses_picture_repetition(self):
        for mode in ("rejected", "error"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                pipeline, (segments, _timings, stats) = self._run(Path(tmp), mode)
                self.assertEqual(pipeline.translator.calls, 0)
                self.assertNotIn("tts_fit_retry", stats)
                self.assertEqual(stats["sync"]["unplaced"], 0)
                self.assertEqual(stats["sync"]["overlaps"], 0)
                self.assertEqual(segments[0].text_bn, "আমাদের কাছে ১২৬ ওভিআর আছে।")
                self.assertTrue(all(s.placed_duration for s in segments))
                self.assertEqual(segments[0].tempo_used, 1.0)
                self.assertGreater(segments[0].video_added_seconds, 0)


if __name__ == "__main__":
    unittest.main()
