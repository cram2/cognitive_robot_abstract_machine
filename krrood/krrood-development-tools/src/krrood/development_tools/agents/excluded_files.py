"""
The files a repository keeps out of ``krrood check``.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from krrood.development_tools.agents.core_boundary import (
    DeclarationFile,
    PyprojectKey,
)

# %% the exclusions a repository declares


@dataclass(kw_only=True, frozen=True)
class ExcludedFiles:
    """
    The files left unchecked, as ``[tool.krrood.check]`` of the ``pyproject.toml`` at a
    repository's root declares them.
    """

    repository_root: Path
    """
    The directory holding the repository's ``pyproject.toml``.
    """

    patterns: tuple[str, ...]
    """
    Regular expressions, each searched for in a file's path relative to
    :attr:`repository_root`, written with forward slashes.
    """

    @classmethod
    def from_repository(cls, *, repository_root: Path) -> ExcludedFiles:
        """
        :param repository_root: The repository's root directory.
        :return: The files it excludes; none when it declares no exclusions.
        """
        pyproject_path = repository_root / DeclarationFile.PYPROJECT
        if not pyproject_path.is_file():
            return cls(repository_root=repository_root, patterns=())
        pyproject = tomllib.loads(pyproject_path.read_text())
        check = (
            pyproject.get(PyprojectKey.TOOL, {})
            .get(PyprojectKey.KRROOD, {})
            .get(PyprojectKey.CHECK, {})
        )
        return cls(
            repository_root=repository_root,
            patterns=tuple(check.get(PyprojectKey.EXCLUDE, ())),
        )

    def excludes(self, *, path: Path) -> bool:
        """
        :param path: A file.
        :return: Whether one of :attr:`patterns` matches it; never for a file outside
            the repository.
        """
        absolute_path = path.resolve()
        repository_root = self.repository_root.resolve()
        if not absolute_path.is_relative_to(repository_root):
            return False
        relative_path = absolute_path.relative_to(repository_root)
        return any(
            re.search(pattern, relative_path.as_posix()) for pattern in self.patterns
        )
