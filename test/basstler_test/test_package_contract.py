"""
The contract :mod:`basstler` has to keep: everything importable from the repository root
with no install, every module importable on its own, every entry point reachable through
``python3 -m``, every workflow that runs a module installing its dependencies first, and
nothing left behind under ``.claude/``.

:mod:`basstler.package_layout` discovers the modules and the entry points from the package
itself, so nothing here is a second list of them, and the workflows are read from the
directory holding them rather than named.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from .constants import ToolingDirectory
from .script_runner import ScriptRunner
from basstler.package_layout import (
    PACKAGE_DIRECTORY,
    REPOSITORY_ROOT,
    PackageModule,
    command_line_entry_points,
    package_modules,
)

CLAUDE_DIRECTORY = REPOSITORY_ROOT / ".claude"
"""
The directory the migration emptied of Python.

Its SKILL.md files, settings.json and bash entry points stay; not one ``.py`` file does.
"""

WORKFLOWS_DIRECTORY = REPOSITORY_ROOT / ".github" / "workflows"
"""
Where this repository's Actions workflows live.

Read as a directory rather than named one by one, so a workflow added later is checked
without anyone recording that it exists.
"""


@dataclass(frozen=True, kw_only=True)
class InterpreterRunner(ScriptRunner):
    """
    Runs this interpreter itself, for the checks that are about the import rather than
    about any one entry point.
    """

    @property
    def command(self) -> tuple[str, ...]:
        """:return: This interpreter, with no module named yet."""
        return (sys.executable,)


def run_from_repository_root(*arguments: str) -> subprocess.CompletedProcess[str]:
    """
    Run this interpreter from the repository root with an environment that cannot help
    it find the package.

    ``PYTHONPATH`` is removed so a pass proves the zero-install import really comes from
    the repository root being the working directory, rather than from whatever the
    caller's shell happened to export.

    :param arguments: Arguments to the interpreter, e.g. ``("-c", "import basstler")``.
    :return: The completed process, with output captured as text.
    """
    return InterpreterRunner(
        project_root=REPOSITORY_ROOT,
        removed_variable_prefixes=("PYTHONPATH",),
    ).run(*arguments)


SHELL_CONFIGURATION = ToolingDirectory.HOOKS.path / "resolve-personal-notes-config.sh"
"""
The file the bash callers source, which is where a module gets a shell name.
"""

MODULE_VARIABLE = re.compile(r'^([A-Z_]+)="(basstler\.[a-z_]+)"', re.MULTILINE)
"""
One ``NAME="basstler.module"`` assignment in the shell configuration.
"""

DEPENDENCY_INSTALL = re.compile(
    r"(pip install|uv pip install|uv sync)[^\n]*" r"(BASSTLER_PACKAGE_DIRECTORY|basstler)"
)
"""
A step that installs this package, and with it the dependencies its metadata declares,
however the caller spells the directory.
"""


def names_of(module: PackageModule) -> frozenset[str]:
    """
    Every spelling a caller can invoke a module by.

    Bash callers go through a shell variable rather than the dotted path, so the variable
    names are read from the configuration that assigns them instead of written out here.

    :param module: The module to name.
    :return: Its import path, plus any shell variable holding that path.
    """
    assignments = MODULE_VARIABLE.findall(SHELL_CONFIGURATION.read_text())
    return frozenset({module.import_path}) | {
        variable for variable, path in assignments if path == module.import_path
    }


def installs_the_dependencies(caller_source: str) -> bool:
    """
    :param caller_source: A caller's own file.
    :return: Whether it installs this package's dependencies before running anything.
    """
    return DEPENDENCY_INSTALL.search(caller_source) is not None


# %% the package exists and is reachable with no install


def test_the_package_imports_from_the_repository_root_with_no_install():
    """
    The zero-install contract: a fresh clone with no pip step can import the package,
    and what it imports is this repository's copy rather than an installed one.
    """
    result = run_from_repository_root("-c", "import basstler; print(basstler.__file__)")

    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == PACKAGE_DIRECTORY / "__init__.py"


# %% each module stands on its own


@pytest.mark.parametrize("module", package_modules(), ids=lambda module: module.name)
def test_every_module_imports_on_its_own(module: PackageModule):
    """
    Each module imports in an interpreter that has imported nothing else.

    One subprocess per module rather than one for all of them: an import cycle only bites
    whichever module a caller reaches first, so a suite that imports them together can
    stay green while a single-module entry point is broken.
    """
    result = run_from_repository_root("-c", f"import {module.import_path}")

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "module", command_line_entry_points(), ids=lambda module: module.name
)
def test_every_entry_point_answers_help_through_the_module_runner(
    module: PackageModule,
):
    """
    Each entry point is reachable as ``python -m basstler.<name>``.

    That form rather than a path to the file: a module run by path puts its own directory
    on ``sys.path`` instead of the repository root, so its absolute imports of its
    siblings would not resolve. Every bash caller invokes them this way.
    """
    result = run_from_repository_root("-m", module.import_path, "--help")

    assert result.returncode == 0, result.stderr


# %% what a workflow runs, and what it installs first


def workflows_that_run_a_module() -> tuple[tuple[Path, PackageModule], ...]:
    """
    Every Actions workflow that runs a module of this package, paired with that module.

    Discovered from the workflow directory and the package itself rather than declared:
    an Actions runner is the one caller no session hook reaches, so a workflow added
    later has to be found rather than remembered.

    :return: Each workflow's path and the module it runs, in path then name order.
    """
    return tuple(
        (workflow_path, module)
        for workflow_path in sorted(WORKFLOWS_DIRECTORY.glob("*.yml"))
        for module in package_modules()
        if any(name in workflow_path.read_text() for name in names_of(module))
    )


def test_every_workflow_that_runs_a_module_installs_the_dependencies_first():
    """
    A runner starts with nothing installed and no hook runs on it, so a workflow that
    invokes a module installs this package's dependencies itself.

    The emptiness assertion is the scan's own coverage: a discovery that finds no
    workflow checks nothing, and a parametrization over it would pass while doing so.
    """
    running = workflows_that_run_a_module()

    assert running != ()
    assert [
        workflow_path.name
        for workflow_path, _ in running
        if not installs_the_dependencies(workflow_path.read_text())
    ] == []


# %% nothing is left behind


def test_no_python_module_remains_under_the_claude_directory():
    """
    The migration's own completeness assertion from the other side.

    ``.claude/`` keeps its SKILL.md files, settings.json and bash entry points - Claude
    Code discovers those by path - and not one Python file.
    """
    remaining_module_paths = sorted(
        str(path.relative_to(REPOSITORY_ROOT))
        for path in CLAUDE_DIRECTORY.rglob("*.py")
    )

    assert remaining_module_paths == []
