"""Colab setup ও আলাদা worker চালানোর সহায়ক কোড; এখানে ML import করা হয় না।"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def setup_environment(root, requirements):
    """Colab-এর প্যাকেজ না বদলে নিজস্ব venv বানায়; সফল setup পুনরায় ব্যবহার করে।"""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    environment = root / "venv"
    python = environment / "bin" / "python"
    marker = root / "environment.json"
    # Colab-এর pip ও torch দেখা যাবে; নতুন প্যাকেজ কেবল এই venv-এ লেখা হবে।
    # --without-pip ব্যবহার করায় Debian/Colab-এর ensurepip দরকার হয় না।
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages",
                        "--without-pip", str(environment)], check=True, env=env)
    requirement_file = root / "requirements.txt"
    requirement_file.write_text(requirements, encoding="utf-8")
    fingerprint = hashlib.sha256(
        (requirements + sys.version + sys.executable).encode()).hexdigest()
    # Parent kernel-এর import cache নয়, নতুন interpreter-এর প্যাকেজ যাচাই করা হয়।
    check_code = """
import importlib.metadata as metadata
import sys
from packaging.requirements import Requirement
from packaging.version import Version
if Version(metadata.version('torch').split('+')[0]) < Version('2.6'):
    raise SystemExit('torch >= 2.6 প্রয়োজন; বর্তমান Colab runtime ব্যবহার করুন।')
for line in open(sys.argv[1], encoding='utf-8'):
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    requirement = Requirement(line)
    if requirement.marker and not requirement.marker.evaluate():
        continue
    if not requirement.specifier.contains(metadata.version(requirement.name)):
        raise SystemExit(1)
"""
    ready = False
    if marker.exists():
        try:
            ready = json.loads(marker.read_text())["fingerprint"] == fingerprint
        except (ValueError, KeyError):
            pass
    if ready:
        ready = subprocess.run([str(python), "-c", check_code, str(requirement_file)],
                               capture_output=True, text=True, env=env).returncode == 0
    if not ready:
        marker.unlink(missing_ok=True)
        print("প্রথমবারের প্যাকেজ setup চলছে…", flush=True)
        subprocess.run([str(python), "-m", "pip", "--isolated", "install", "--no-user", "--quiet",
                        "--disable-pip-version-check", "-r", str(requirement_file)], check=True, env=env)
        subprocess.run([str(python), "-c", check_code, str(requirement_file)], check=True, env=env)
        temp = marker.with_suffix(".partial.json")
        temp.write_text(json.dumps({"fingerprint": fingerprint}), encoding="utf-8")
        os.replace(temp, marker)
    else:
        print("আগের setup ব্যবহার করা হচ্ছে; আবার install লাগবে না।")
    return python


def ensure_drive_mounted(mount_point="/content/drive", force_remount=False):
    """সক্রিয় mount পুনর্ব্যবহার করে; শুধু directory থাকাকে mount ধরে না।"""
    mount_point = Path(mount_point)
    root = mount_point / "MyDrive"
    try:
        mounted = os.path.ismount(mount_point)
        accessible = root.is_dir() if mounted else False
        if accessible:
            # Stale FUSE mount-এ stat সফল হলেও directory পড়া ব্যর্থ হতে পারে।
            next(root.iterdir(), None)
    except OSError as error:
        if not force_remount:
            raise RuntimeError("Drive সংযোগে সমস্যা। REMOUNT_DRIVE=True করে এই cell আবার চালান।") from error
        mounted = accessible = False
    if not force_remount and mounted:
        if not accessible:
            raise RuntimeError("MyDrive পাওয়া যাচ্ছে না। REMOUNT_DRIVE=True করে এই cell আবার চালান।")
        return root
    from google.colab import drive
    drive.mount(str(mount_point), force_remount=force_remount)
    if not os.path.ismount(mount_point) or not root.is_dir():
        raise RuntimeError("Drive mount হয়নি। Google-এর অনুমতি দিয়ে আবার চেষ্টা করুন।")
    return root


def stop_worker(process, timeout=5):
    """Interrupt হলে worker এবং তার FFmpeg subprocess-গুলোও বন্ধ করে।"""
    def send(sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    send(signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        send(signal.SIGKILL)
        process.wait()
    finally:
        # Parent আগে থামলেও group-এর কোনো FFmpeg যেন চলতে না থাকে।
        send(signal.SIGKILL)


RESULT_PREFIX = "PROXMAX_RESULT\t"


def run_isolated(python, worker, config, echo=True):
    """Fresh worker; config/results travel through pipes, never diagnostic files."""
    payload = json.dumps(config, ensure_ascii=False)
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
               HF_HOME="/content/proxmax-model-cache", HF_HUB_DISABLE_TELEMETRY="1",
               HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    process = subprocess.Popen(
        [str(python), str(Path(worker).resolve())],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=env, start_new_session=True,
    )
    outcome = None
    try:
        try:
            process.stdin.write(payload)
            process.stdin.flush()
        except BrokenPipeError:
            pass  # An early startup error is still printed by the stdout loop.
        finally:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        for line in process.stdout:
            if line.startswith(RESULT_PREFIX):
                try:
                    message = json.loads(line[len(RESULT_PREFIX):])
                    if (not isinstance(message, dict)
                            or message.get("status") not in {"complete", "failed"}
                            or not isinstance(message.get("results"), list)):
                        raise ValueError("Invalid worker result")
                    outcome = message
                except (ValueError, TypeError):
                    if echo:
                        print("Worker-এর ফলাফল পড়া যায়নি।", flush=True)
                continue  # Do not dump the machine-readable JSON into the notebook.
            if echo:
                print(line, end="", flush=True)
        returncode = process.wait()
    except BaseException:
        stop_worker(process)
        raise
    finally:
        process.stdout.close()
        if not process.stdin.closed:
            process.stdin.close()
    if outcome is None:
        outcome = {"status": "failed", "results": [],
                   "error": f"Worker ফলাফল দেয়নি (code {returncode})। উপরের terminal output দেখুন।"}
        if echo:
            print(outcome["error"], flush=True)
    if returncode:
        outcome["status"] = "failed"
    return {**outcome, "returncode": returncode}
