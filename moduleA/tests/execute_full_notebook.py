"""Execute the original workload of one Module A notebook, preserving real outputs.

Run on an environment with moduleA/requirements.txt installed.
This is a FULL training run, unlike run_short_training.py.
"""
import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: execute_full_notebook.py <moduleA-relative-notebook>")
    relative_path = sys.argv[1]
    notebook_path = (REPO / "moduleA" / relative_path).resolve()
    if not notebook_path.is_relative_to(REPO / "moduleA"):
        raise ValueError("Path escapes Module A")
    notebook = nbformat.read(notebook_path, as_version=4)
    setup_cells = {}
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == "code" and ("!pip install swig" in cell.source or
            '!pip install "gymnasium[box2d]"' in cell.source):
            setup_cells[i] = cell.source
            cell.source = "# Dependencies already installed by the GitHub Actions environment."
    client = NotebookClient(notebook, timeout=7200, kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
        allow_errors=False)
    print(f"FULL EXECUTION START: {relative_path}", flush=True)
    try:
        client.execute()
    finally:
        for i, old_source in setup_cells.items():
            notebook.cells[i].source = old_source
    nbformat.write(notebook, notebook_path)
    executed = sum(c.cell_type == "code" and c.execution_count is not None
                   for c in notebook.cells)
    print(f"FULL EXECUTION SUCCESS: {relative_path}, executed code cells={executed}",
          flush=True)


if __name__ == "__main__":
    main()
