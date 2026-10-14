"""Real FFmpeg media tests; inference is stubbed, no weights/network/GPU required."""
from dataclasses import replace
from pathlib import Path
import shutil
import wave

import numpy as np
import pytest
import soundfile as sf

from proxmax import pipeline as p

pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="ffmpeg and ffprobe are required for media integration tests",
)


@pytest.fixture
def config(tmp_path):
    return p.Config(input_dir=str(tmp_path / "input"), output_dir=str(tmp_path / "output"),
                    work_dir=str(tmp_path / "work"), min_free_gb=0,
                    cpu_threads=1, device="cpu", keep_local=True)


def make_video(path, duration=3, audio=True, codec="libx264", delay=0):
    path.parent.mkdir(exist_ok=True, parents=True)
    args = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
            f"color=c=navy:s=160x90:r=25:d={duration}"]
    if audio:
        args += ["-itsoffset", str(delay), "-f", "lavfi", "-i",
                 f"sine=frequency=440:sample_rate=16000:duration={duration-delay}"]
    args += ["-c:v", codec, "-threads", "1", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac"]
    args += ["-t", str(duration), path]
    p.run_command(args)
    return path


def tone(path, seconds, sr=16000):
    t = np.arange(round(seconds * sr)) / sr
    sf.write(path, (0.15 * np.sin(2 * np.pi * 350 * t)).astype(np.float32), sr)


def test_fit_audio_sample_exact_and_strict_overflow(tmp_path, config):
    raw, clip = tmp_path / "raw.wav", tmp_path / "clip.wav"
    tone(raw, 1.4)
    timing = p.fit_audio(raw, clip, .8, config)
    assert sf.info(clip).frames == 12800
    assert timing["speed_factor"] >= 1.75 - 1e-9
    assert timing["review_required"]
    with pytest.raises(ValueError, match="Dub needs"):
        p.fit_audio(raw, clip, .8, replace(config, overflow_policy="error"))
    tone(raw, .2)
    p.fit_audio(raw, clip, .8, config)
    samples, sr = sf.read(clip)
    assert len(samples) == 12800
    assert np.max(np.abs(samples[-4000:])) == 0


def test_absolute_timeline_has_silence_and_no_cumulative_drift(tmp_path, config):
    (tmp_path / "clips").mkdir()
    tone(tmp_path / "clips" / "one.wav", .2)
    tone(tmp_path / "clips" / "two.wav", .3)
    records = [{"clip": "one.wav", "start_sample": 8000},
               {"clip": "two.wav", "start_sample": 32000}]
    wav = p.assemble_timeline(records, 3, tmp_path, config)
    samples, sr = sf.read(wav)
    assert sr == 16000 and len(samples) == 48000
    assert np.max(np.abs(samples[:8000])) == 0
    assert np.max(np.abs(samples[11200:32000])) == 0
    assert np.max(np.abs(samples[32000:36800])) > .1
    assert not (tmp_path / "timeline.pcm").exists()
    with pytest.raises(ValueError, match="overlap"):
        p.assemble_timeline([records[0], {**records[1], "start_sample": 9000}], 3, tmp_path, config)


def test_no_speech_timeline(tmp_path, config):
    wav = p.assemble_timeline([], 1.25, tmp_path, config)
    with wave.open(str(wav)) as audio:
        assert audio.getnframes() == 20000
        assert set(audio.readframes(20000)) == {0}


def test_extract_and_mux_h264_preserve_duration(tmp_path, config):
    source = make_video(tmp_path / "source with ' spaces.mp4")
    info = p.media_info(source)
    english = tmp_path / "english.wav"
    p.extract_audio(source, english, info, config)
    assert sf.info(english).samplerate == 16000
    assert sf.info(english).channels == 1
    dub = p.assemble_timeline([], info["duration"], tmp_path, config)
    output = tmp_path / "translated.mp4"
    result = p.mux_video(source, dub, output, info, config)
    assert result["video_stream_copied"]
    assert abs(p.media_info(output)["duration"] - info["duration"]) < .15
    streams = p.probe(output)["streams"]
    assert [s["codec_name"] for s in streams] == ["h264", "aac"]
    assert streams[1]["tags"]["language"] == "ben"


def test_delayed_audio_extraction_fills_leading_silence(tmp_path, config):
    source = make_video(tmp_path / "delayed.mp4", delay=.5)
    info = p.media_info(source)
    wav = tmp_path / "extracted.wav"
    p.extract_audio(source, wav, info, config)
    samples, sr = sf.read(wav)
    # AAC preroll can precede the tone; inspect comfortably before its start.
    assert np.max(np.abs(samples[:round(.35 * sr)])) < .002
    assert np.max(np.abs(samples[round(.6 * sr):round(.8 * sr)])) > .05


def test_non_h264_transcodes_and_missing_audio_fails(tmp_path, config):
    source = make_video(tmp_path / "source.mkv", codec="mpeg4")
    info = p.media_info(source)
    dub = p.assemble_timeline([], info["duration"], tmp_path, config)
    result = p.mux_video(source, dub, tmp_path / "translated.mp4", info, config)
    assert not result["video_stream_copied"]
    silent_video = make_video(tmp_path / "no-audio.mp4", audio=False)
    with pytest.raises(ValueError, match="no audio"):
        p.extract_audio(silent_video, tmp_path / "unused.wav", p.media_info(silent_video), config)


def test_whole_pipeline_publication_resume_and_cleanup(config, monkeypatch):
    source = make_video(Path(config.input_dir) / "example.mp4")
    calls = []
    segments = [{"id": 0, "start": .4, "end": 1.4, "english": "Hello world."}]

    def asr(*_):
        calls.append("asr")
        return {"segments": segments, "model_commit": "stub-asr"}

    def translation(items, _):
        calls.append("translation")
        return {"segments": [{**item, "bangla": "হ্যালো বিশ্ব।"} for item in items],
                "model_commit": "stub-translation"}

    def tts(items, duration, job, cfg):
        (job / "clips").mkdir(exist_ok=True)
        raw, clip = job / "raw.wav", job / "clips" / "clip.wav"
        tone(raw, .8)
        timing = p.fit_audio(raw, clip, 1, cfg)
        return [{**items[0], **timing, "slot_end": 1.4, "start_sample": 6400,
                 "clip": "clip.wav", "sha256": p.sha256_file(clip)}], "stub-tts"

    monkeypatch.setattr(p, "transcribe", asr)
    monkeypatch.setattr(p, "translate", translation)
    monkeypatch.setattr(p, "synthesize", tts)
    result = p.run_batch(config)[0]
    assert result["status"] == "complete"
    dest = Path(result["output"])
    manifest = p.read_json(dest / "manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["segment_count"] == 1
    for artifact, checksum in manifest["artifacts_sha256"].items():
        assert p.sha256_file(dest / artifact) == checksum
    assert source.exists()
    # Retry a retained job: source is staged again, but ML stage checkpoints are reused.
    assert p.run_batch(config)[0]["status"] == "complete"
    assert calls == ["asr", "translation"]
    # Operational keep_local is not part of the inference recipe and reuses the same job.
    assert p.run_batch(replace(config, keep_local=False))[0]["status"] == "complete"
    assert not (Path(config.work_dir) / dest.name).exists()
    assert source.exists() and (dest / "translated_bn.mp4").exists()
