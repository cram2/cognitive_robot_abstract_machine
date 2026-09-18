"""
Running one of the things this repository ships, the way a caller runs it.

Every suite here has the same shape to express - invoke a module or a bash script from
some project root, capture both streams, and read the exit status - and had been writing
it out again per suite, each with its own environment handling. The shape is here once;
what differs between callers is the two things below it: what to run, and which variables
the run must not inherit.
"""

from __future__ import annotations

import os
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class ScriptRunner(ABC):
    """
    Runs something from a project root and hands back the finished process.

    The process is returned rather than asserted on, because an exit status and a stderr
    are usually what a test is about.
    """

    project_root: Path
    """
    The directory to run in, standing for the checkout a real caller runs from.
    """

    removed_variable_prefixes: tuple[str, ...] = ()
    """
    Variables the run must not inherit, by the prefix of their name.

    A whole name is its own prefix, so this covers both a family
    (``CLAUDE_PERSONAL_NOTES_``) and a single variable (``PYTHONPATH``). Removing rather
    than overriding is the point: a test asserting what happens when a credential is
    absent cannot express that by setting it to something.
    """

    @property
    @abstractmethod
    def command(self) -> tuple[str, ...]:
        """:return: What to run, before the arguments of any particular call."""

    def environment(self, overrides: Mapping[str, str]) -> dict[str, str]:
        """
        :param overrides: Variables this call sets.
        :return: This process's own environment, minus what
            :attr:`removed_variable_prefixes` names, plus the overrides.
        """
        inherited = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith(self.removed_variable_prefixes)
        }
        return {**inherited, **overrides}

    def run(
        self, *arguments: str, **environment_overrides: str
    ) -> subprocess.CompletedProcess[str]:
        """
        :param arguments: The arguments this call passes.
        :param environment_overrides: Variables to set for this run, for the tests that
            exercise resolution from the environment.
        :return: The finished subprocess, with both streams captured as text.
        """
        return subprocess.run(
            [*self.command, *arguments],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            env=self.environment(environment_overrides),
        )


@dataclass(frozen=True, kw_only=True)
class PythonModuleRunner(ScriptRunner):
    """
    Runs a module of this repository's package as ``python3 -m``.

    By module rather than by path, which is how every shell entry point invokes them: an
    interpreter given a file path puts that file's own directory on ``sys.path`` in
    place of the project root, so the module's imports of its siblings stop resolving.
    """

    module_name: str
    """
    The dotted import path, e.g. ``"basstler.stack"`` - read off the imported module
    rather than spelled out, so a rename cannot leave this behind.
    """

    @property
    def command(self) -> tuple[str, ...]:
        """:return: This interpreter, running that module."""
        return (sys.executable, "-m", self.module_name)


@dataclass(frozen=True, kw_only=True)
class BashScriptRunner(ScriptRunner):
    """
    Runs one of the bash entry points that stay under ``.claude/``.
    """

    script_path: Path
    """
    The script to run, usually the scratch layout's own copy rather than the
    repository's.
    """

    @property
    def command(self) -> tuple[str, ...]:
        """:return: bash, running that script."""
        return ("bash", str(self.script_path))
