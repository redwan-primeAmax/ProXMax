"""Colab setup ও আলাদা worker চালানোর সহায়ক কোড; এখানে ML import করা হয় না।"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid


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


def run_isolated(python, worker, config, state_dir, echo=True):
    """প্রতিবার নতুন Python/CUDA process; আগের run-এর GPU/import state ধরে রাখে না।"""
    state_dir = Path(state_dir).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    config_path = state_dir / f"{run_id}.config.json"
    results_path = state_dir / f"{run_id}.results.json"
    log_path = state_dir / f"{run_id}.log"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
               HF_HOME="/content/proxmax-model-cache", HF_HUB_DISABLE_TELEMETRY="1")
    # আগের notebook থেকে পাওয়া PYTHONPATH/PYTHONHOME যেন venv isolation নষ্ট না করে।
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    process = subprocess.Popen(
        [str(python), str(Path(worker).resolve()), "--config", str(config_path),
         "--results", str(results_path)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=env, start_new_session=True,
    )
    try:
        with log_path.open("w", encoding="utf-8") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if echo:
                    print(line, end="", flush=True)
        returncode = process.wait()
    except BaseException:
        stop_worker(process)
        raise
    finally:
        process.stdout.close()
    try:
        outcome = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        outcome = {"status": "failed", "results": [],
                   "error": f"Worker বন্ধ হয়েছে (code {returncode})। নিচের log দেখুন, তারপর আবার run করুন।"}
    if returncode:
        outcome["status"] = "failed"
    return {**outcome, "returncode": returncode, "log": str(log_path)}
