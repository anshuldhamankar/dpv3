
import nbformat
from nbformat.v4 import new_notebook, new_code_cell
import os

def append_to_notebook(notebook_path, code_block):
    if os.path.exists(notebook_path):
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
    else:
        nb = new_notebook()

    nb.cells.append(new_code_cell(code_block))

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
