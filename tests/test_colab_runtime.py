"""Rerun checks without Google credentials, a GPU, or model downloads."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from proxmax import colab_runtime as runtime


def fake_installer(monkeypatch, root):
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if "venv" in args:
            python = root / "venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(runtime.subprocess, "run", run)
    return calls


def test_setup_rerun_does_not_reinstall_or_touch_kernel_packages(tmp_path, monkeypatch):
    calls = fake_installer(monkeypatch, tmp_path)
    requirements = "transformers==4.57.6\n"
    first = runtime.setup_environment(tmp_path, requirements)
    second = runtime.setup_environment(tmp_path, requirements)
    assert first == second == tmp_path / "venv" / "bin" / "python"
    assert sum("venv" in call for call in calls) == 1
    pip_calls = [call for call in calls if "pip" in call]
    assert len(pip_calls) == 1
    assert pip_calls[0][0] == str(first)  # not the notebook's sys.executable
    assert json.loads((tmp_path / "environment.json").read_text())["fingerprint"]


def test_changed_requirements_reinstall_once(tmp_path, monkeypatch):
    calls = fake_installer(monkeypatch, tmp_path)
    runtime.setup_environment(tmp_path, "a==1\n")
    runtime.setup_environment(tmp_path, "a==2\n")
    runtime.setup_environment(tmp_path, "a==2\n")
    assert sum("pip" in call for call in calls) == 2


def test_failed_install_is_retryable_and_has_no_success_marker(tmp_path, monkeypatch):
    calls = fake_installer(monkeypatch, tmp_path)
    successful = runtime.subprocess.run

    def broken(args, **kwargs):
        if "pip" in args:
            raise subprocess.CalledProcessError(1, args)
        return successful(args, **kwargs)

    monkeypatch.setattr(runtime.subprocess, "run", broken)
    with pytest.raises(subprocess.CalledProcessError):
        runtime.setup_environment(tmp_path, "a==1\n")
    assert not (tmp_path / "environment.json").exists()
    monkeypatch.setattr(runtime.subprocess, "run", successful)
    runtime.setup_environment(tmp_path, "a==1\n")
    assert (tmp_path / "environment.json").exists()
    assert sum("venv" in call for call in calls) == 1


def test_stale_or_invalid_environment_marker_is_rechecked(tmp_path, monkeypatch):
    calls = fake_installer(monkeypatch, tmp_path)
    runtime.setup_environment(tmp_path, "a==1\n")
    normal = runtime.subprocess.run
    unhealthy = True

    def check_failure(args, **kwargs):
        nonlocal unhealthy
        if "-c" in args and unhealthy:
            unhealthy = False
            return subprocess.CompletedProcess(args, 1)
        return normal(args, **kwargs)

    monkeypatch.setattr(runtime.subprocess, "run", check_failure)
    runtime.setup_environment(tmp_path, "a==1\n")
    assert sum("pip" in call for call in calls) == 2
    (tmp_path / "environment.json").write_text("{bad-json")
    runtime.setup_environment(tmp_path, "a==1\n")
    assert sum("pip" in call for call in calls) == 3


@pytest.fixture
def drive_stub(tmp_path, monkeypatch):
    mount_point = tmp_path / "drive"
    root = mount_point / "MyDrive"
    state = SimpleNamespace(mounted=False, calls=[])
    monkeypatch.setattr(runtime.os.path, "ismount", lambda _: state.mounted)

    def mount(path, force_remount=False):
        state.calls.append((path, force_remount))
        state.mounted = True
        root.mkdir(parents=True, exist_ok=True)

    google = ModuleType("google")
    google.__path__ = []
    colab = ModuleType("google.colab")
    colab.drive = SimpleNamespace(mount=mount)
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    return mount_point, root, state


def test_mount_rerun_is_quiet_and_mounts_only_once(drive_stub):
    point, root, state = drive_stub
    assert runtime.ensure_drive_mounted(point) == root
    assert runtime.ensure_drive_mounted(point) == root
    assert state.calls == [(str(point), False)]


def test_existing_local_folder_is_not_mistaken_for_drive(drive_stub):
    point, root, state = drive_stub
    root.mkdir(parents=True)
    runtime.ensure_drive_mounted(point)
    assert state.calls == [(str(point), False)]


def test_remount_is_only_for_explicit_request(drive_stub):
    point, root, state = drive_stub
    runtime.ensure_drive_mounted(point)
    root.rmdir()
    with pytest.raises(RuntimeError, match="REMOUNT_DRIVE=True"):
        runtime.ensure_drive_mounted(point)
    assert len(state.calls) == 1
    runtime.ensure_drive_mounted(point, force_remount=True)
    assert state.calls[-1] == (str(point), True)


def test_stale_fuse_error_is_actionable(drive_stub, monkeypatch):
    point, root, state = drive_stub
    runtime.ensure_drive_mounted(point)
    original = Path.iterdir

    def broken(path):
        if path == root:
            raise OSError("Transport endpoint is not connected")
        return original(path)

    monkeypatch.setattr(Path, "iterdir", broken)
    with pytest.raises(RuntimeError, match="REMOUNT_DRIVE=True"):
        runtime.ensure_drive_mounted(point)
    assert len(state.calls) == 1
    runtime.ensure_drive_mounted(point, force_remount=True)
    assert state.calls[-1][1] is True


@pytest.fixture
def fake_worker(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text('''
import argparse, json, os, sys
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--results')
args = parser.parse_args()
config = json.loads(Path(args.config).read_text())
print('নতুন worker', flush=True)
if config.get('crash'):
    raise SystemExit(7)
Path(args.results).write_text(json.dumps({
    'status': 'complete', 'results': [], 'pid': os.getpid(),
    'cached_fake_module': 'fake_parent_module' in sys.modules,
    'pythonpath': os.environ.get('PYTHONPATH'),
    'hf_home': os.environ['HF_HOME']
}))
''', encoding="utf-8")
    return worker


def test_new_process_each_run_and_failure_does_not_poison_next(tmp_path, fake_worker, monkeypatch):
    monkeypatch.setitem(sys.modules, "fake_parent_module", ModuleType("fake_parent_module"))
    monkeypatch.setenv("PYTHONPATH", "/fake-parent-cache")
    state = tmp_path / "runs"
    one = runtime.run_isolated(sys.executable, fake_worker, {}, state, echo=False)
    failed = runtime.run_isolated(sys.executable, fake_worker, {"crash": True}, state, echo=False)
    two = runtime.run_isolated(sys.executable, fake_worker, {}, state, echo=False)
    assert one["status"] == two["status"] == "complete"
    assert failed["status"] == "failed" and failed["returncode"] == 7
    assert failed["results"] == []  # no previous successful result reused
    assert one["pid"] != two["pid"] != os.getpid()
    assert not two["cached_fake_module"]
    assert two["pythonpath"] is None
    assert two["hf_home"] == "/content/proxmax-model-cache"
    assert len({one["log"], failed["log"], two["log"]}) == 3
    assert "নতুন worker" in Path(two["log"]).read_text(encoding="utf-8")
    assert len(list(state.glob("*.config.json"))) == 3


def test_real_worker_reports_preflight_failure_and_exits(tmp_path):
    worker = Path(__file__).resolve().parents[1] / "proxmax" / "worker.py"
    config = {"input_dir": str(tmp_path / "input"), "output_dir": str(tmp_path / "output"),
              "work_dir": str(tmp_path / "work"), "device": "invalid"}
    result = runtime.run_isolated(sys.executable, worker, config, tmp_path / "runs", echo=False)
    assert result["status"] == "failed" and result["returncode"] == 1
    assert "ValueError" in result["error"]
    assert result["results"] == []


def process_is_running(pid):
    try:
        # A killed orphan can briefly be a zombie until PID 1 reaps it.
        status = Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0]
        return status not in {"Z", "X"}
    except FileNotFoundError:
        return False


@pytest.mark.skipif(sys.platform != "linux", reason="Colab process-group cleanup is Linux-specific")
def test_interrupt_stops_worker_and_grandchild(tmp_path, monkeypatch):
    worker = tmp_path / "long_worker.py"
    pids = tmp_path / "pids.json"
    worker.write_text(f'''
import json, os, signal, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
def stop(signum, frame):
    child.wait(timeout=4)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
Path({str(pids)!r}).write_text(json.dumps([os.getpid(), child.pid]))
print('READY', flush=True)
time.sleep(120)
''')

    def interrupt(*args, **kwargs):
        if "READY" in args[0]:
            raise KeyboardInterrupt

    monkeypatch.setattr(runtime, "print", interrupt, raising=False)
    with pytest.raises(KeyboardInterrupt):
        runtime.run_isolated(sys.executable, worker, {}, tmp_path / "runs")
    assert all(not process_is_running(pid) for pid in json.loads(pids.read_text()))


def test_timeout_escalates_to_sigkill(monkeypatch):
    sent, waits = [], []

    def wait(timeout=None):
        waits.append(timeout)
        if timeout is not None:
            raise subprocess.TimeoutExpired("worker", timeout)
        return -9

    monkeypatch.setattr(runtime.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    runtime.stop_worker(SimpleNamespace(pid=42, wait=wait), timeout=.01)
    assert sent[0] == (42, runtime.signal.SIGTERM)
    assert (42, runtime.signal.SIGKILL) in sent
    assert waits == [.01, None]
