"""Focused regression tests for ProXMax."""





from tests._support import *





@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class TestGapFilling(unittest.TestCase):
    """বড় ফাঁক ভরানোর ধাপ: ছোট লাইন ভরাট করে শুধু সেগুলোরই আবার TTS।"""

    class _Translator(_FakeTranslator):
        def __init__(self, expand: bool = True):
            super().__init__()
            self.expand = expand
            self.expand_calls = 0

        def translate_segments(self, segments):
            for seg in segments:
                seg.text_bn = "ছোট।"
            return segments

        def expand_to_fit(self, segments, targets):
            self.expand_calls += 1
            changed = 0
            for seg, target in zip(segments, targets):
                if target - float(seg.tts_duration or 0.0) >= 0.9:
                    seg.text_bn = "এটি এখন অনেক বেশি ভরাট করে লেখা একটি লাইন যাতে সময়টা পুরো ভরে যায়।"
                    changed += 1
            return changed

    class _PaddedTTS(_FakeTTS):
        """edge-tts-এর মতো দুই প্রান্তে নীরবতা জোড়া ক্লিপ (trim হবে)।"""

        def synthesize(self, segments, output_dir):
            from proxmax.tts.base import BaseTTSEngine

            class _Trimmer(BaseTTSEngine):
                name = "trimmer"

                def synthesize(self, *_args, **_kwargs):  # pragma: no cover
                    return []

            engine = _Trimmer(trim_silence=True, trim_pad_ms=25)
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for seg in segments:
                speech = max(0.5, len(seg.text_bn) / 13.0)
                body = (0.3 * np.sin(2 * np.pi * 220 * np.arange(int(speech * 24000)) / 24000)).astype(np.float32)
                padded = np.concatenate([
                    np.zeros(int(0.15 * 24000), dtype=np.float32),
                    body,
                    np.zeros(int(0.6 * 24000), dtype=np.float32),
                ])
                final = engine._finalize(padded)
                path = output_dir / f"seg_{seg.index:04d}.wav"
                audio_fx.write_wav(path, final, 24000)
                seg.tts_audio_path = path
                seg.tts_duration = final.size / 24000
            return segments

    def _pipeline(self, root: Path, translator, tts):
        from proxmax.audio_extraction import AudioExtractor
        from proxmax.muxer import VideoMuxer
        from proxmax.pipeline import DubbingPipeline

        class _Stt:
            last_stats = {"chunks": 1}

            def __init__(self):
                self.released = False

            def transcribe(self, _path):
                return [
                    Segment(0, 0.3, 12.9, "a long english sentence about the squad", ""),
                    Segment(1, 12.9, 18.0, "and another short one here", ""),
                    Segment(2, 18.0, 23.0, "the final line of this clip", ""),
                ]

            def release(self):
                self.released = True

        input_dir, output_dir = root / "in", root / "out"
        input_dir.mkdir()
        video = input_dir / "demo.mp4"
        subprocess.run(
            [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=c=navy:s=320x240:r=25:d=24",
             "-f", "lavfi", "-i", "sine=frequency=220:duration=24",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(video)],
            capture_output=True, text=True,
        )
        cfg = AppConfig(
            input_dir=str(input_dir), output_dir=str(output_dir),
            work_dir=str(root / "work"), save_report=True,
        ).validate()
        pipeline = DubbingPipeline.__new__(DubbingPipeline)
        pipeline.config = cfg
        pipeline.extractor = AudioExtractor(use_hwaccel=False)
        pipeline.stt = _Stt()
        pipeline.translator = translator
        pipeline.tts = tts
        pipeline._tts = tts
        pipeline._profile = None
        pipeline._profile_checked = True
        pipeline.sync_engine = SyncEngine(sample_rate=24000, max_workers=2,
                                          fill_extension=cfg.fill_extension)
        pipeline.muxer = VideoMuxer(use_hwaccel=False)
        return pipeline

    def test_short_lines_are_filled_and_resynthesized(self):
        with tempfile.TemporaryDirectory() as tmp:
            translator = self._Translator()
            pipeline = self._pipeline(Path(tmp), translator, self._PaddedTTS())
            summary = pipeline.run_batch()
            self.assertEqual(len(summary.succeeded), 1, summary.failed)
            self.assertEqual(translator.expand_calls, 1)
            self.assertGreater(summary.succeeded[0].stats.get("filled_lines", 0), 0)
            self.assertEqual(summary.succeeded[0].stats["sync"]["overlaps"], 0)

    def test_expansion_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            translator = self._Translator()
            pipeline = self._pipeline(Path(tmp), translator, self._PaddedTTS())
            pipeline.config.translation_expand_pass = False
            pipeline.config = pipeline.config.validate()
            summary = pipeline.run_batch()
            self.assertEqual(len(summary.succeeded), 1, summary.failed)
            self.assertEqual(translator.expand_calls, 0)
            self.assertNotIn("filled_lines", summary.succeeded[0].stats)
