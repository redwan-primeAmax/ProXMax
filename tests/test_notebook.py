import ast
import json
from pathlib import Path
import subprocess
import sys

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
