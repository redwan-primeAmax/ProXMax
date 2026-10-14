import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "ProXMax_English_to_Bangla.ipynb"


def test_notebook_schema_and_code_cells():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"notebook-cell-{i}")
            assert cell.outputs == []
            assert cell.execution_count is None
    assert notebook.metadata.accelerator == "GPU"


def test_notebook_contains_complete_pipeline_without_repo_clone():
    notebook = json.loads(NOTEBOOK.read_text())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "git clone" not in code
    for name in ("transcribe", "translate", "synthesize", "mux_video", "run_batch"):
        assert f"def {name}(" in code
    assert "drive.mount" in code
    assert "/content/proxmax-work" in code
    assert "MyDrive" in code


def test_generated_notebook_matches_source():
    subprocess.run([sys.executable, "scripts/build_notebook.py", "--check"], cwd=ROOT, check=True)


def test_embedded_sources_exactly_match_and_compile():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    setup = next(cell for cell in notebook.cells
                 if cell.cell_type == "code" and "SOURCES =" in cell.source)
    tree = ast.parse(setup.source)
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "SOURCES" for t in node.targets))
    sources = ast.literal_eval(assignment.value)
    assert set(sources) == {"pipeline.py", "colab_runtime.py", "worker.py"}
    for name, code in sources.items():
        assert code == (ROOT / "proxmax" / name).read_text(encoding="utf-8")
        compile(code, name, "exec")
    assert setup.metadata.cellView == "form"


def test_notebook_is_short_bengali_and_never_imports_ml_into_kernel():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    assert len(notebook.cells) == 1
    cell = notebook.cells[0]
    assert cell.cell_type == "code"
    assert cell.metadata.cellView == "form"
    assert cell.metadata.jupyter.source_hidden is True
    assert "# @title ▶ Setup ও App চালু করুন · ইংরেজি ভিডিও → বাংলা" in cell.source
    descriptions = [line for line in cell.source.splitlines() if line.startswith("# @markdown")]
    assert sum(map(len, descriptions)) < 700
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        # Inspect code, not quoted source strings written for the separate worker.
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(a.name.split(".")[0] not in {"torch", "transformers", "proxmax"}
                           for a in node.names)
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in {"torch", "transformers", "proxmax"}
    runner = notebook.cells[-1].source
    assert 'RUNTIME["run_isolated"]' in runner
    assert "RUN_RESULT = None" in runner
    assert "KeyboardInterrupt" in runner


@pytest.mark.parametrize("mode", ["success", "interrupt", "setup_failure"])
def test_single_cell_runs_setup_mount_and_worker_in_order(tmp_path, monkeypatch, mode):
    import runpy
    import shutil
    from types import ModuleType

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    tree = ast.parse(notebook.cells[0].source)
    # Keep all writes in the test's temporary folder, not in /content.
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "APP_DIR" for target in node.targets
        ):
            node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()),
                                  args=[ast.Constant(str(tmp_path / "app"))], keywords=[])
    ast.fix_missing_locations(tree)
    calls = []
    google = ModuleType("google")
    google.__path__ = []
    colab = ModuleType("google.colab")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(shutil, "which", lambda name: f"/fake/{name}")

    def setup(root, requirements):
        calls.append("setup")
        assert (root / "pipeline.py").is_file()
        assert "faster-whisper==" in requirements
        if mode == "setup_failure":
            raise RuntimeError("setup failed")
        return Path(sys.executable)

    def mount(force_remount=False):
        calls.append("mount")
        assert force_remount is False
        return tmp_path / "MyDrive"

    def run(python, worker, config, state_dir):
        calls.append("run")
        assert python == Path(sys.executable)
        assert worker.is_file()
        assert Path(config["input_dir"]).is_dir()
        assert Path(config["output_dir"]).is_dir()
        if mode == "interrupt":
            raise KeyboardInterrupt
        return {"status": "complete", "results": []}

    monkeypatch.setattr(runpy, "run_path", lambda path: {
        "setup_environment": setup, "ensure_drive_mounted": mount, "run_isolated": run,
    })
    namespace = {"RUN_RESULT": {"status": "stale-result"}}
    code = compile(tree, "single-colab-cell", "exec")
    if mode == "setup_failure":
        with pytest.raises(RuntimeError, match="setup failed"):
            exec(code, namespace)
        assert calls == ["setup"]
        assert namespace["RUN_RESULT"] is None
    else:
        exec(code, namespace)
        exec(code, namespace)  # the same cell works again without preceding cells
        assert calls == ["setup", "mount", "run"] * 2
        assert namespace["RUN_RESULT"] == (
            {"status": "complete", "results": []} if mode == "success" else None
        )
