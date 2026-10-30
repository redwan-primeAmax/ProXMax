"""Conservative quiet-speech, Bengali spoken-form and QA regressions."""

import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from proxmax.config import AppConfig
from proxmax.models import Segment, Word
from proxmax.pipeline import DubbingPipeline
from proxmax.prosody import retime_children, subtitles_for_unit
from proxmax.stt import SpeechToText
from proxmax.sync_silence import trim_silence_for_overrun
from proxmax.text_utils import prepare_for_speech
from proxmax.tts import EdgeTTSEngine, build_tts_engine
from proxmax.tts.base import BaseTTSEngine
from proxmax.whisper_model import _resolve_cached_model_path, is_english_only_model


class _Engine(BaseTTSEngine):
    def synthesize(self, segments, output_dir):
        return segments


class TestVoiceSafety(unittest.TestCase):
    def test_only_exact_dib_is_spelled_for_speech_not_names_or_subtitles(self):
        original = "আমি দিব। দিবস, দিবা, দিবেন, দিবো ও হব। ইয়াসিন।"
        spoken = prepare_for_speech(original)
        self.assertEqual(spoken, "আমি দিবো। দিবস, দিবা, দিবেন, দিবো ও হব। ইয়াসিন।")
        self.assertEqual(original, "আমি দিব। দিবস, দিবা, দিবেন, দিবো ও হব। ইয়াসিন।")
        self.assertEqual(prepare_for_speech("(দিব), দিব?"), "(দিবো), দিবো?")
        self.assertEqual(prepare_for_speech("দিবসের দিন"), "দিবসের দিন")

    def test_quiet_ending_and_interior_activity_are_not_silence_trimmed(self):
        rate = 1000
        quiet = np.full(200, 10 ** (-65 / 20), np.float32)
        clip = np.r_[np.full(100, .2), np.zeros(300), quiet, np.zeros(300), np.full(100, .1), quiet, np.zeros(200)]
        trimmed = trim_silence_for_overrun(clip, rate, 10)
        self.assertEqual(np.count_nonzero(trimmed.audio == quiet[0]), len(quiet) * 2)
        self.assertEqual(trimmed.internal_cuts, 2)

    def test_tts_fade_only_affects_final_eight_milliseconds(self):
        engine = _Engine(trim_silence=False)
        rate = engine.sample_rate
        audio = np.full(rate, .2, np.float32)
        output = engine._postprocess(audio)
        self.assertAlmostEqual(float(output[-int(rate * .04)]), float(output[rate // 2]), places=5)
        self.assertAlmostEqual(float(output[-1]), 0.0, places=6)
        self.assertEqual(len(output), len(audio))

    def test_opt_in_tts_trim_also_preserves_minus_65_dbfs_tail(self):
        engine = _Engine(trim_silence=True, trim_pad_ms=25)
        rate = engine.sample_rate
        quiet = np.full(int(.1 * rate), 10 ** (-65 / 20), np.float32)
        clip = np.r_[np.full(rate // 4, .1, np.float32), np.zeros(rate // 5), quiet, np.zeros(rate // 4)]
        result = engine._finalize(clip)
        # Postprocessing may scale and fade, but the protected quiet tail remains.
        self.assertGreater(result.size, int(.5 * rate))

    def test_shifted_short_multisentence_cues_never_exceed_actual_speech(self):
        text = "প্রথম বাক্য। দ্বিতীয় বাক্য। তৃতীয় বাক্য।"
        unit = Segment(0, 0, .3, "source", text, placed_start=2.0, placed_duration=.15)
        cues = subtitles_for_unit(unit, min_seconds=.1)
        self.assertEqual(" ".join(c.text_bn for c in cues), text)
        self.assertAlmostEqual(cues[0].start, 2.0, places=3)
        self.assertAlmostEqual(cues[-1].end, 2.15, places=3)
        self.assertTrue(all(2.0 <= c.start < c.end <= 2.15 for c in cues))
        self.assertTrue(all(a.end <= b.start for a, b in zip(cues, cues[1:])))
        unit.children = [Segment(0, 0, .1), Segment(1, .1, .3)]
        retime_children(unit)
        self.assertAlmostEqual(unit.children[-1].end, 2.15, places=3)

    def test_keep_temp_retains_raw_mp3_and_default_deletes_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for keep in (False, True):
                raw = root / f"{keep}.mp3"
                raw.write_bytes(b"raw voice")
                engine = EdgeTTSEngine(keep_raw=keep)
                with mock.patch("proxmax.tts.edge_engine.ensure_tool", return_value="ffmpeg"), \
                     mock.patch("proxmax.tts.edge_engine.run_ffmpeg"):
                    engine._to_wav(raw, root / f"{keep}.wav")
                self.assertEqual(raw.exists(), keep)
            with mock.patch.object(EdgeTTSEngine, "prepare"):
                self.assertTrue(build_tts_engine(AppConfig(keep_temp_files=True)).keep_raw)
                self.assertFalse(build_tts_engine(AppConfig()).keep_raw)


class TestOptionalBengaliValidation(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("faster_whisper"), "faster-whisper not installed")
    def test_faster_whisper_exposes_both_selected_checkpoints(self):
        from faster_whisper.utils import available_models
        self.assertTrue({"distil-large-v3", "large-v3"}.issubset(set(available_models())))

    def test_distil_cache_is_never_mistaken_for_large_v3(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "faster-whisper-distil-large-v3"
            model.mkdir()
            (model / "model.bin").write_bytes(b"mock weights")
            fake_utils = types.ModuleType("faster_whisper.utils")
            fake_utils.download_model = lambda *_args, **_kwargs: None
            with mock.patch.dict("sys.modules", {"faster_whisper": types.ModuleType("faster_whisper"),
                                                  "faster_whisper.utils": fake_utils}), \
                 mock.patch.dict("os.environ", {"PROXMAX_WHISPER_MODEL_PATH": str(model),
                                                 "PROXMAX_WHISPER_MODEL_NAME": "distil-large-v3"}):
                self.assertEqual(Path(_resolve_cached_model_path("distil-large-v3")), model)
                self.assertIsNone(_resolve_cached_model_path("large-v3"))
            with mock.patch.dict("sys.modules", {"faster_whisper": types.ModuleType("faster_whisper"),
                                                  "faster_whisper.utils": fake_utils}), \
                 mock.patch.dict("os.environ", {"PROXMAX_WHISPER_MODEL_PATH": str(model),
                                                 "PROXMAX_WHISPER_MODEL_NAME": ""}):
                self.assertIsNone(_resolve_cached_model_path("large-v3"))

    def test_english_only_checkpoints_reject_bengali_decode(self):
        for model in ("distil-large-v3", "distil-large-v2", "small.en"):
            with self.subTest(model=model):
                self.assertTrue(is_english_only_model(model))
                stt = SpeechToText(model_size=model)
                with mock.patch("proxmax.stt_inference.decode_audio") as decode, \
                     self.assertRaisesRegex(ValueError, "English-only"):
                    stt.transcribe_words_in_language("unused.wav", language="bn")
                decode.assert_not_called()
        self.assertFalse(is_english_only_model("large-v3"))

    def test_enabled_bengali_qa_uses_separate_multilingual_model_and_releases_it(self):
        pipeline = DubbingPipeline.__new__(DubbingPipeline)
        pipeline.config = AppConfig(audio_validation_min_coverage=.5)
        pipeline.extractor = mock.Mock()
        pipeline.extractor.extract.return_value = Path("post_mux.wav")
        pipeline.stt = mock.Mock(model_size="distil-large-v3", _model=object())
        validator = mock.Mock()
        validator.transcribe_words_in_language.return_value = [
            Word(2.1, 2.2, "দ্বিতীয়"), Word(2.3, 2.5, "বাক্য।")
        ]
        segment = Segment(0, 0, 1, "", "দ্বিতীয় বাক্য।", placed_start=2, placed_duration=1)
        with mock.patch("proxmax.pipeline_validation.SpeechToText", return_value=validator) as factory:
            result = pipeline._validate_dub_audio(Path("out.mp4"), Path("work"), [segment], 3.0)
        self.assertEqual(factory.call_args.kwargs["model_size"], "large-v3")
        pipeline.stt.release.assert_called_once()
        validator.release.assert_called_once()
        validator.transcribe_words_in_language.assert_called_once_with(Path("post_mux.wav"), language="bn")
        self.assertEqual(result["units"][0]["status"], "likely_complete")

    def test_multilingual_model_is_reused_and_unavailable_is_honest(self):
        pipeline = DubbingPipeline.__new__(DubbingPipeline)
        pipeline.config = AppConfig()
        pipeline.extractor = mock.Mock()
        pipeline.stt = mock.Mock(model_size="large-v3")
        pipeline.stt.transcribe_words_in_language.side_effect = RuntimeError("unavailable")
        with mock.patch("proxmax.pipeline_validation.SpeechToText") as factory:
            result = pipeline._validate_dub_audio(Path("out.mp4"), Path("work"), [], 1.0)
        factory.assert_not_called()
        self.assertEqual(result["status"], "unavailable")
        pipeline.stt.release.assert_not_called()


if __name__ == "__main__":
    unittest.main()
