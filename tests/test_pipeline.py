from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys

import pytest

from proxmax import pipeline as p


@pytest.fixture
def config(tmp_path):
    return p.Config(input_dir=str(tmp_path / "input"), output_dir=str(tmp_path / "output"),
                    work_dir=str(tmp_path / "work"), min_free_gb=0)


def test_defaults_and_validation(config):
    config.validate()
    for overrides in ({"device": "gpu"}, {"translation_batch_size": 0},
                      {"overflow_policy": "clip"}, {"comfortable_speed": 0.9},
                      {"min_free_gb": -1}, {"sample_rate": math.nan},
                      {"input_dir": config.output_dir},
                      {"work_dir": config.input_dir + "/local"},
                      {"output_dir": config.input_dir + "/output"},
                      {"work_dir": "/content/drive/MyDrive/work"}):
        with pytest.raises(ValueError):
            replace(config, **overrides).validate()


def test_import_does_not_load_ml_libraries():
    # Imports must stay lightweight so tests and notebook generation need no GPU.
    import subprocess
    output = subprocess.check_output([
        sys.executable, "-c", "import proxmax,sys; print('torch' in sys.modules, 'transformers' in sys.modules)"
    ], text=True)
    assert output.strip() == "False False"


def test_group_words_preserves_absolute_timing(config):
    words = [{"start": 1, "end": 1.4, "text": " Hello"},
             {"start": 1.5, "end": 2, "text": " world."},
             {"start": 3.2, "end": 3.6, "text": " Second"},
             {"start": 3.7, "end": 4.4, "text": " sentence."}]
    segments = p.group_words(words, 10, config)
    assert segments == [
        {"id": 0, "start": 1, "end": 2, "english": "Hello world."},
        {"id": 1, "start": 3.2, "end": 4.4, "english": "Second sentence."},
    ]
    p.validate_segments(segments, 10)


def test_grouping_word_and_time_bounds_and_overlap(config):
    words = [{"start": i * .2, "end": (i + 1) * .2, "text": f"word{i}"} for i in range(15)]
    output = p.group_words(words, 4, replace(config, max_utterance_words=3))
    assert len(output) == 5
    assert all(len(s["english"].split()) <= 3 for s in output)
    output = p.group_words(words, 4, replace(config, max_utterance_seconds=1))
    assert all(s["end"] - s["start"] <= 1.000001 for s in output)
    output = p.group_words([{"start": -.3, "end": .4, "text": "a"},
                            {"start": .3, "end": .7, "text": "b"},
                            {"start": 2, "end": 2.5, "text": "outside"}], 1, config)
    assert output[0]["start"] == 0
    assert output[0]["english"] == "a b"
    assert p.group_words([], 10, config) == []
    with pytest.raises(ValueError, match="Nonfinite"):
        p.group_words([{"start": float("nan"), "end": 1, "text": "a"}], 5, config)


def test_validate_segments_rejects_overlap_and_empty_translation():
    one = {"id": 0, "start": 1, "end": 2, "english": "Hello"}
    p.validate_segments([one], 3)
    for segments in ([{**one, "start": 2}], [{**one, "id": 4}],
                     [one, {**one, "id": 1, "start": 1.5, "end": 3}]):
        with pytest.raises(ValueError):
            p.validate_segments(segments, 5)
    with pytest.raises(ValueError, match="Bangla"):
        p.validate_segments([one], 5, translated=True)


@pytest.mark.parametrize("rate", [.03, .5, 1, 1.9, 2, 2.1, 4.5, 16])
def test_atempo_factorization(rate):
    factors = [float(x.split("=")[1]) for x in p.atempo_filter(rate).split(",")]
    assert all(.5 <= f <= 2 for f in factors)
    assert math.prod(factors) == pytest.approx(rate, abs=1e-8)


@pytest.mark.parametrize("rate", [0, -1, math.inf, math.nan])
def test_atempo_rejects_invalid_rates(rate):
    with pytest.raises(ValueError):
        p.atempo_filter(rate)


def test_bengali_srt_is_utf8_and_uses_slot_end(tmp_path):
    path = tmp_path / "bn.srt"
    p.write_srt([{"start": 1.25, "end": 2, "slot_end": 2.3, "bangla": "হ্যালো বিশ্ব।"}], "bangla", path)
    assert path.read_text() == "1\n00:00:01,250 --> 00:00:02,300\nহ্যালো বিশ্ব।\n"
    assert p.srt_time(3599.9996) == "01:00:00,000"
    assert p.srt_time(0) == "00:00:00,000"


def test_tts_split_keeps_text():
    text = "আমি বাংলায় কথা বলি। এটি একটি উদাহরণ।"
    chunks = p.split_tts_text(text, 20)
    assert all(len(chunk) <= 20 for chunk in chunks)
    assert " ".join(chunks) == text
    with pytest.raises(ValueError, match="overlong"):
        p.split_tts_text("x" * 30, 10)
    with pytest.raises(ValueError, match="empty"):
        p.split_tts_text("  ", 10)


def test_local_staging_content_addressed(config):
    source = Path(config.input_dir) / "বাংলা video ';.MP4"
    source.parent.mkdir()
    source.write_bytes(b"sample-not-real-video")
    job, local, digest = p.stage_input(source, config)
    assert local.parent == job and job.parent == Path(config.work_dir)
    assert local.suffix == ".mp4"
    assert local.read_bytes() == source.read_bytes()
    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
    assert p.stage_input(source, config)[0] == job
    source.write_bytes(b"changed")
    assert p.stage_input(source, config)[0] != job
    assert not list(Path(config.work_dir).glob("staging-*"))


def test_recipe_ignores_paths_but_not_inference_settings(config):
    original = p.recipe_id(config)
    assert original == p.recipe_id(replace(config, output_dir="elsewhere", keep_local=True))
    assert original != p.recipe_id(replace(config, translation_beams=1))
    assert original != p.recipe_id(replace(config, tts_revision="some-commit"))


def test_unicode_stem_length():
    stem = p.safe_stem("বাংলা" * 100 + ".mp4")
    assert len(stem.encode()) <= 100
    assert "/" not in stem
    assert p.safe_stem("...mp4") == "video"


def test_checkpoint_computes_once_and_does_not_cache_failure(tmp_path):
    path = tmp_path / "checkpoint.json"
    called = []
    assert p.cached_stage(path, lambda: called.append(1) or {"bangla": "বাংলা"}) == {"bangla": "বাংলা"}
    assert p.cached_stage(path, lambda: 1 / 0) == {"bangla": "বাংলা"}
    assert called == [1]
    failed = tmp_path / "failed.json"
    with pytest.raises(ZeroDivisionError):
        p.cached_stage(failed, lambda: 1 / 0)
    assert not failed.exists()


def test_publish_verifies_bundle_and_manifest_is_last(tmp_path, monkeypatch):
    job, output = tmp_path / "job", tmp_path / "output"
    job.mkdir()
    (job / "video.txt").write_text("example")
    (job / "bangla.srt").write_text("বাংলা")
    calls = []
    original = p.atomic_copy

    def copied(source, destination):
        calls.append(Path(destination).name)
        return original(source, destination)

    monkeypatch.setattr(p, "atomic_copy", copied)
    dest = p.publish_bundle(job, output, ["video.txt", "bangla.srt"], {"status": "working"})
    manifest = p.read_json(dest / "manifest.json")
    assert calls[-1] == "manifest.json"
    assert manifest["status"] == "complete"
    assert manifest["artifacts_sha256"]["video.txt"] == p.sha256_file(job / "video.txt")
    # No stale completion marker may survive a partial rerender/publication.
    with pytest.raises(FileNotFoundError):
        p.publish_bundle(job, output, ["missing"], {})
    assert not (dest / "manifest.json").exists()


def test_copy_failure_never_replaces_existing_destination(tmp_path, monkeypatch):
    source, dest = tmp_path / "source", tmp_path / "dest"
    source.write_text("new data")
    dest.write_text("old data")
    monkeypatch.setattr(p.shutil, "copyfile", lambda a, b: Path(b).write_text("corrupted"))
    with pytest.raises(IOError, match="Checksum"):
        p.atomic_copy(source, dest)
    assert dest.read_text() == "old data"
    assert not list(tmp_path.glob("*.partial-*"))


def test_batch_continues_and_reports_failures(config, monkeypatch):
    input_dir = Path(config.input_dir)
    input_dir.mkdir()
    (input_dir / "a.mp4").touch()
    (input_dir / "b.MKV").touch()
    (input_dir / "ignore.txt").touch()
    monkeypatch.setattr(p.shutil, "which", lambda _: "/fake")

    def process(source, _):
        if source.stem == "a":
            raise RuntimeError("test failure")
        return {"source": str(source), "status": "complete"}

    monkeypatch.setattr(p, "process_video", process)
    result = p.run_batch(config)
    assert [r["status"] for r in result] == ["failed", "complete"]
    assert p.read_json(Path(config.output_dir) / "batch_results.json") == result
    with pytest.raises(RuntimeError, match="test failure"):
        p.run_batch(replace(config, continue_on_error=False))


def test_empty_input_is_actionable(config, monkeypatch):
    Path(config.input_dir).mkdir()
    monkeypatch.setattr(p.shutil, "which", lambda _: "/fake")
    with pytest.raises(FileNotFoundError, match="No supported videos"):
        p.run_batch(config)
