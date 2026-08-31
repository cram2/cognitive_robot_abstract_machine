"""
Every workspace member declares its dependencies statically in ``pyproject.toml``.

Tools that read the workspace without building it (uv, PyCharm's project model) only
see ``[project] dependencies``; a member that resolves them dynamically from a
requirements file, or that imports a sibling package it never declared, silently
drifts out of the workspace graph.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# %% workspace description

REPOSITORY_ROOT = Path(__file__).parents[2]
"""Root of the monorepo, which holds the workspace ``pyproject.toml``."""


@dataclass(frozen=True)
class WorkspaceMember:
    """
    One package of the uv workspace, read from its ``pyproject.toml`` and ``src`` tree.
    """

    directory: Path
    """Directory holding the member's ``pyproject.toml`` and ``src`` tree."""

    @property
    def name(self) -> str:
        """Directory name, which is also the member's import and distribution name."""
        return self.directory.name

    @property
    def project_table(self) -> dict[str, Any]:
        """The ``[project]`` table of the member's ``pyproject.toml``."""
        return tomllib.loads((self.directory / "pyproject.toml").read_text())["project"]

    @property
    def dynamic_fields(self) -> list[str]:
        """Metadata fields the member leaves to its build backend."""
        return self.project_table.get("dynamic", [])

    @property
    def declared_dependencies(self) -> set[str]:
        """Canonical distribution names listed under ``[project] dependencies``."""
        return {
            canonicalize_name(Requirement(specifier).name)
            for specifier in self.project_table.get("dependencies", [])
        }

    def imported_members(self, members: list[WorkspaceMember]) -> set[str]:
        """Canonical names of the workspace members the ``src`` tree imports."""
        import_pattern = re.compile(
            r"^\s*(?:from|import)\s+("
            + "|".join(re.escape(member.name) for member in members)
            + r")(?=[.\s]|$)",
            re.MULTILINE,
        )
        imported = set()
        for source_file in (self.directory / "src").rglob("*.py"):
            imported.update(
                import_pattern.findall(source_file.read_text(errors="ignore"))
            )
        imported.discard(self.name)
        return {canonicalize_name(name) for name in imported}


def workspace_members() -> list[WorkspaceMember]:
    """Read the members from ``[tool.uv.workspace]`` of the root ``pyproject.toml``."""
    root_project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
    return [
        WorkspaceMember(REPOSITORY_ROOT / member)
        for member in root_project["tool"]["uv"]["workspace"]["members"]
    ]


MEMBERS = workspace_members()
"""All workspace members, resolved once for parametrization."""


# %% declaration checks


@pytest.mark.parametrize("member", MEMBERS, ids=[member.name for member in MEMBERS])
def test_dependencies_are_declared_statically(member: WorkspaceMember) -> None:
    assert "dependencies" not in member.dynamic_fields


@pytest.mark.parametrize("member", MEMBERS, ids=[member.name for member in MEMBERS])
def test_imported_workspace_members_are_declared(member: WorkspaceMember) -> None:
    undeclared_members = member.imported_members(MEMBERS) - member.declared_dependencies
    assert undeclared_members == set()
