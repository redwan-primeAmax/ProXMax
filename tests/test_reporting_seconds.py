"""Per-second source/output and elapsed-time diagnostics, including failures."""
from __future__ import annotations

import asyncio
import contextvars
import logging
import math
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np

from proxmax import audio_fx
from proxmax.config import AppConfig
from proxmax.models import Segment, StageTimings, Word
from proxmax.pipeline import DubbingPipeline
from proxmax.pipeline_helpers import _report_is_usable
from proxmax.reporting import write_detailed_report
from proxmax.reporting_activity import activity_seconds, postmux_seconds
from proxmax.reporting_packets import picture_packet_seconds
from proxmax.reporting_runtime import OperationalTrace, record_event, runtime_lines
from proxmax.video_timeline import VideoPiece, VideoPlan
from tests._support import FFMPEG


class TestReportTrace(unittest.TestCase):
    def test_late_context_does_not_leak_warnings_into_next_video(self):
        logger = logging.getLogger("proxmax.gemini_protocol")
        old = OperationalTrace()
        old.activate()
        old_context = contextvars.copy_context()
        old.close()
        current = OperationalTrace()
        current.activate()
        try:
            record_event("tts_clip_started", segment=7)
            old_context.run(logger.warning, "late previous secret %s", "DO_NOT_PUBLISH")
            logger.error("Gemini API error (attempt %d/%d) | %s | status=%s | %s",
                         2, 101, "TimeoutError", 429, "DO_NOT_PUBLISH")
            snapshot = current.snapshot()
        finally:
            current.close()
        codes = [event["code"] for event in snapshot["events"]]
        self.assertEqual(codes, ["tts_clip_started", "gemini_api_error"])
        text = "\n".join(runtime_lines(snapshot))
        self.assertIn("segment=7", text)
        self.assertIn("http_status=429", text)
        self.assertIn("Every processing second", text)
        self.assertNotIn("DO_NOT_PUBLISH", text)
        self.assertNotIn("late previous", text)

    def test_resource_samples_are_labeled_and_missing_is_not_zero(self):
        snapshot = {"elapsed": 2.1, "events": [], "stages": [], "resources": {"samples": [
            {"time": .0, "cpu_seconds": 10., "parent_rss_mib": 100.0,
             "system_available_mib": None, "cgroup_current_mib": 200.0,
             "cgroup_limit_mib": None, "python_threads": 3},
            {"time": 1., "cpu_seconds": 10.5, "parent_rss_mib": 110.0,
             "system_available_mib": 400.0, "cgroup_current_mib": 210.0,
             "cgroup_limit_mib": 1024.0, "python_threads": 4},
        ]}}
        text = "\n".join(runtime_lines(snapshot))
        self.assertIn("50.0%", text)
        self.assertIn("210.0/1024.0", text)
        self.assertIn("200.0/unavailable", text)
        self.assertIn("GPU/VRAM and child-process CPU are not sampled", text)
        self.assertIn("unavailable", text)

    def test_event_limit_is_explicit_not_silent(self):
        with mock.patch("proxmax.reporting_runtime._LIMIT", 2):
            trace = OperationalTrace()
            trace.record("one")
            trace.record("two")
            trace.record("three")
            result = trace.snapshot()
            self.assertEqual(result["dropped"], 1)
            self.assertIn("event cap: 1", "\n".join(runtime_lines(result)))

    def test_tts_retry_and_completion_have_safe_timestamps_and_segment_id(self):
        from proxmax.tts.edge_engine import EdgeTTSEngine
        calls = []
        class FakeCommunication:
            def __init__(self, *_args, **_kwargs):
                pass
            async def save(self, path):
                calls.append(path)
                if len(calls) == 1:
                    raise TimeoutError("PRIVATE_PROVIDER_RESPONSE")
                Path(path).write_bytes(b"fake")
        trace = OperationalTrace()
        trace.activate()
        try:
            with tempfile.TemporaryDirectory() as temp, \
                 mock.patch("edge_tts.Communicate", FakeCommunication), \
                 mock.patch("asyncio.sleep", new_callable=mock.AsyncMock) as sleep:
                engine = EdgeTTSEngine(max_retries=3)
                segment = Segment(7, 0, 1, "Hello", "বাংলা")
                with mock.patch.object(engine, "_convert_and_finalize", return_value=.8):
                    asyncio.run(engine._synthesize_one(segment, Path(temp)))
                self.assertEqual(sleep.await_args_list, [mock.call(1.0)])
                self.assertEqual(segment.tts_duration, .8)
            snapshot = trace.snapshot()
        finally:
            trace.close()
        events = snapshot["events"]
        self.assertEqual([item["code"] for item in events],
                         ["tts_clip_started", "tts_attempt_failed", "tts_clip_finished"])
        self.assertEqual(events[1]["details"]["segment"], 7)
        self.assertEqual(events[1]["details"]["delay_seconds"], 1.0)
        self.assertEqual(events[1]["details"]["error_type"], "TimeoutError")
        self.assertNotIn("PRIVATE_PROVIDER_RESPONSE", str(events))

    def test_sync_error_record_omits_raw_exception_details(self):
        from proxmax.sync_engine import SyncEngine
        from proxmax.sync_prepare import prepare_clip
        with tempfile.TemporaryDirectory() as temp:
            clip = Path(temp) / "voice.wav"
            clip.write_bytes(b"fake")
            segment = Segment(4, 0, 1, "Hello", "বাংলা", tts_audio_path=clip)
            with mock.patch("proxmax.sync_prepare.audio_fx.read_wav",
                            side_effect=RuntimeError("PRIVATE_PROVIDER_RESPONSE /private/path")):
                self.assertIsNone(prepare_clip(SyncEngine(), segment, (0, 1), Path(temp) / "out.wav"))
            self.assertEqual(segment.sync_issue, "TTS preparation failed (RuntimeError)")
            self.assertNotIn("PRIVATE_PROVIDER_RESPONSE", segment.sync_issue)
            self.assertNotIn("/private/path", segment.sync_issue)

    def test_failure_report_is_written_before_cleanup_and_cannot_trigger_skip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "video.mp4"
            source.write_bytes(b"synthetic")
            cfg = AppConfig(input_dir=str(root / "in"), output_dir=str(root), work_dir=str(root / "work"),
                            skip_existing=False, save_report=True).validate()
            pipeline = DubbingPipeline.__new__(DubbingPipeline)
            pipeline.config = cfg
            def steps(_source, work, _output):
                samples = np.full(12000, .1, np.float32)
                audio_fx.write_wav(work / "audio_16k.wav", samples, 8000)
                pipeline._trace.source_duration = 1.5
                pipeline._trace.segments = [Segment(0, .2, 1.1, "A source", words=[Word(.3, .7, "word", .2)])]
                with pipeline._step(3, "translation"):
                    logging.getLogger("proxmax.gemini_protocol").error(
                        "Gemini API error (attempt %d/%d) | %s | status=%s | %s",
                        3, 101, "TimeoutError", 429, "PRIVATE_PROVIDER_RESPONSE")
                    raise RuntimeError(f"SECRET_KEY at {root}/private")
            pipeline._run_steps = steps
            result = pipeline.process_one(source, root)
            self.assertEqual(result.status, "failed")
            report = root / "video_report.markdown"
            self.assertTrue(report.is_file())
            self.assertFalse(_report_is_usable(report))
            text = report.read_text(encoding="utf-8")
            self.assertIn("PROCESSING FAILED", text)
            self.assertIn("stage_failed", text)
            self.assertIn("http_status=429", text)
            self.assertIn("Source clock — every second", text)
            self.assertIn("word [0.30–0.70, p=0.20]", text)
            self.assertNotIn("SECRET_KEY", text)
            self.assertNotIn("PRIVATE_PROVIDER_RESPONSE", text)
            self.assertNotIn(str(root), text)
            self.assertFalse((root / "work" / "video").exists())


@unittest.skipUnless(FFMPEG, "FFmpeg required")
class TestPerSecondMediaReport(unittest.TestCase):
    def test_output_audio_decode_and_different_clock_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            work, out = root / "work", root / "out"
            work.mkdir(); out.mkdir()
            rate, duration = 8000, 3.2
            t = np.arange(round(rate * duration)) / rate
            tone = (.2 * np.sin(2 * math.pi * 440 * t)).astype(np.float32)
            source_audio = tone.copy()
            source_audio[rate:2 * rate] = 0
            post_audio = tone.copy()
            post_audio[rate:2 * rate] = 0
            audio_fx.write_wav(work / "audio_16k.wav", source_audio, rate)
            audio_fx.write_wav(work / "final_audio_bn.wav", tone, rate)
            audio_fx.write_wav(work / "post.wav", post_audio, rate)
            output = out / "output.mp4"
            subprocess.run([FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i",
                            "color=black:s=160x120:r=25:d=3.2", "-i", str(work / "post.wav"),
                            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(output)],
                           check=True, capture_output=True)
            source = root / "source.mp4"
            source.write_bytes(output.read_bytes())
            # The published output, not a stale/invalid local precursor, is measured.
            (work / output.name).write_bytes(b"invalid local precursor")
            post = postmux_seconds(output, work, duration)
            packets = picture_packet_seconds(output)
            self.assertEqual(packets["status"], "ok")
            self.assertEqual(packets["bins"][0]["count"], 25)
            self.assertEqual(packets["bins"][1]["count"], 25)
            with mock.patch("proxmax.reporting_packets._MAX_PACKETS", 30):
                partial = picture_packet_seconds(output)
            self.assertEqual(partial["status"], "partial")
            self.assertEqual(partial["scanned_packets"], 30)
            self.assertEqual(post["status"], "ok")
            self.assertEqual(len(post["seconds"]), 4)
            self.assertGreater(post["seconds"][0]["rms_dbfs"], -30)
            self.assertLess(post["seconds"][1]["rms_dbfs"], -55)
            self.assertFalse(activity_seconds({"error": "unavailable"}))
            segment = Segment(0, .8, 1.3, "A word", "একটি শব্দ", words=[Word(.9, 1.2, "word", .25)],
                              tts_audio_path=work / "tts.wav", stretched_audio_path=work / "stretched.wav",
                              placed_start=.8, placed_duration=1.8, tempo_used=.95,
                              translation_issues=["review"], video_speed=1.1)
            pieces = (VideoPiece(0, 25, 25), VideoPiece(25, 50, 22, "speed"),
                      VideoPiece(40, 50, 10, "repeat"), VideoPiece(50, 80, 30))
            plan = VideoPlan(Fraction(25), 80, pieces, sum(p.output_frames for p in pieces))
            report = write_detailed_report(video_path=source, output_video_path=output, output_dir=out,
                                           work_dir=work, segments=[segment], timings=StageTimings(),
                                           stats={}, config=AppConfig(), total_duration=3.2, video_plan=plan)
            text = report.read_text(encoding="utf-8")
            self.assertIn("Source and edited output are **different clocks**", text)
            self.assertIn("Source clock — every second", text)
            self.assertIn("Edited output clock — every second", text)
            self.assertIn("word [0.90–1.20, p=0.25]", text)
            self.assertIn("low-confidence word", text)
            self.assertIn("P2 speed", text)
            self.assertIn("P3 repeat", text)
            self.assertIn("Source packet scan: **ok**", text)
            self.assertIn("Output packet scan: **ok**", text)
            self.assertIn("25 packets; PTS", text)
            self.assertIn("output picture packet coverage ends before this interval ends", text)
            self.assertIn("post-mux near-silent while pre-mux active", text)
            self.assertIn("post-mux waveform ends early within this interval", text)
            self.assertIn("U001:C1", text)
            self.assertIn("not visual recognition", text)
            self.assertNotIn(str(root), text)

    def test_delayed_final_audio_is_measured_on_the_output_clock(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rate = 8000
            t = np.arange(2 * rate) / rate
            wav = root / "tone.wav"
            audio_fx.write_wav(wav, (.2 * np.sin(2 * math.pi * 330 * t)).astype(np.float32), rate)
            output = root / "delayed.mp4"
            subprocess.run([FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i",
                            "color=black:s=160x120:r=25:d=4", "-itsoffset", "1", "-i", str(wav),
                            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(output)],
                           check=True, capture_output=True)
            result = postmux_seconds(output, root, 4.0)
            self.assertEqual(result["status"], "ok")
            self.assertLess(result["seconds"][0]["rms_dbfs"], -55)
            self.assertGreater(result["seconds"][1]["rms_dbfs"], -30)

    def test_missing_output_is_unavailable_not_zero_or_silence(self):
        with tempfile.TemporaryDirectory() as temp:
            result = postmux_seconds(Path(temp) / "missing.mp4", Path(temp))
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["seconds"], [])


if __name__ == "__main__":
    unittest.main()
