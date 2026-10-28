"""
Tests that the Docker image's workspace setup script runs on the bare ROS base image.

The image build runs the script with the system Python before any Python package is
installed, so an import of anything outside the standard library stops the build.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

WORKSPACE_SETUP_SCRIPT = (
    Path(__file__).parents[2] / ".github" / "docker" / "setup_workspace.py"
)
"""
The script the Docker image build runs to set up the ROS workspace.
"""

# %% import inspection


def imported_top_level_modules(script: Path) -> set[str]:
    """
    :param script: The Python file to inspect.
    :return: The top-level name of every module the file imports by absolute import.
    """
    syntax_tree = ast.parse(script.read_text())
    modules = set()
    for node in ast.walk(syntax_tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            modules.add(node.module.partition(".")[0])
    return modules


# %% tests


def test_workspace_setup_script_imports_only_the_standard_library():
    """
    Every module the workspace setup script imports belongs to the standard library.
    """
    assert (
        imported_top_level_modules(WORKSPACE_SETUP_SCRIPT) - sys.stdlib_module_names
        == set()
    )
