"""
Tests for the script that sets up the ROS workspace of the Docker image.
"""

from __future__ import annotations

import ast
import sys

from cognitive_robot_abstract_machine.orm_interfaces import REPOSITORY_ROOT

# %% runs on the base image's Python


def test_the_workspace_setup_imports_only_the_standard_library() -> None:
    """
    The script runs on the base image's Python before any package is installed, so an
    import from outside the standard library fails the image build.
    """
    script = REPOSITORY_ROOT / ".github" / "docker" / "setup_workspace.py"
    imported_modules = set()
    for node in ast.walk(ast.parse(script.read_text())):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module)
    top_level_modules = {module.split(".")[0] for module in imported_modules}
    assert top_level_modules - sys.stdlib_module_names == set()
