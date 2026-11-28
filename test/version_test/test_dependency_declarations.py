"""
Every workspace member declares its dependencies statically in ``pyproject.toml``, so
tools that read the workspace without building it, such as uv, see every member it
imports.
"""

from __future__ import annotations

import ast
import itertools
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing_extensions import Any

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

# %% reading imports


@dataclass(kw_only=True)
class ImportedModuleFinder(ast.NodeVisitor):
    """
    Collects the absolute module names a module imports, leaving out the blocks only a
    type checker reads.
    """

    modules: list[str]
    """
    The modules found so far.
    """

    @classmethod
    def modules_imported_at_run_time(cls, *, tree: ast.Module) -> list[str]:
        """
        :param tree: A parsed module.
        :return: The absolute names of the modules it imports when it runs.
        """
        finder = cls(modules=[])
        finder.visit(tree)
        return finder.modules

    def visit_If(self, node: ast.If) -> None:
        """
        Skip a block only a type checker reads.
        """
        if "TYPE_CHECKING" in ast.unparse(node.test):
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        """
        Record each module an ``import`` statement names.
        """
        self.modules.extend(alias.name for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """
        Record the module an absolute ``from`` import reads from.
        """
        if node.level == 0 and node.module is not None:
            self.modules.append(node.module)


# %% workspace description

REPOSITORY_ROOT = Path(__file__).parents[2]
"""
Root of the monorepo, which holds the workspace ``pyproject.toml``.
"""


@dataclass(frozen=True, kw_only=True)
class WorkspaceMember:
    """
    One package of the uv workspace, read from its ``pyproject.toml`` and ``src`` tree.
    """

    directory: Path
    """
    Directory holding the member's ``pyproject.toml`` and ``src`` tree.
    """

    @property
    def project_table(self) -> dict[str, Any]:
        """
        The ``[project]`` table of the member's ``pyproject.toml``.
        """
        return tomllib.loads((self.directory / "pyproject.toml").read_text())["project"]

    @property
    def name(self) -> str:
        """
        The member's distribution name, as an installer compares names.
        """
        return canonicalize_name(self.project_table["name"])

    @property
    def dynamic_fields(self) -> list[str]:
        """
        Metadata fields the member leaves to its build backend.
        """
        return self.project_table.get("dynamic", [])

    @property
    def declared_dependencies(self) -> set[str]:
        """
        Canonical distribution names listed under ``[project] dependencies``.
        """
        return {
            canonicalize_name(Requirement(specifier).name)
            for specifier in self.project_table.get("dependencies", [])
        }

    @property
    def optional_dependencies(self) -> set[str]:
        """
        Canonical distribution names an extra of the member installs.
        """
        return {
            canonicalize_name(Requirement(specifier).name)
            for extra in self.project_table.get("optional-dependencies", {}).values()
            for specifier in extra
        }

    @property
    def oldest_python(self) -> Version:
        """
        The oldest Python 3 minor release the member's ``requires-python`` admits.
        """
        supported = SpecifierSet(self.project_table.get("requires-python", ""))
        return next(
            version
            for version in (Version(f"3.{minor}") for minor in itertools.count())
            if supported.contains(version)
        )

    @property
    def provided_packages(self) -> set[str]:
        """
        The dotted names the member's ``src`` tree is imported by: each top-level
        package, or, for a namespace package several members share, each package and
        module of the member below it.
        """
        source = self.directory / "src"
        if not source.is_dir():
            return set()
        provided = set()
        for top_level in source.iterdir():
            if not top_level.is_dir() or top_level.suffix:
                continue
            if (top_level / "__init__.py").exists():
                provided.add(top_level.name)
                continue
            provided.update(
                f"{top_level.name}.{portion.stem}"
                for portion in top_level.iterdir()
                if portion.is_dir() or portion.suffix == ".py"
            )
        return provided

    def imported_members(self, *, members: list[WorkspaceMember]) -> set[str]:
        """
        Canonical names of the workspace members the ``src`` tree imports when it runs;
        an import only a type checker reads, under ``TYPE_CHECKING``, needs none.
        """
        member_by_package = {
            package: member.name
            for member in members
            for package in member.provided_packages
        }
        imported = set()
        for source_file in (self.directory / "src").rglob("*.py"):
            tree = ast.parse(source_file.read_text(errors="ignore"))
            for module in ImportedModuleFinder.modules_imported_at_run_time(tree=tree):
                imported.update(
                    member
                    for package, member in member_by_package.items()
                    if module == package or module.startswith(f"{package}.")
                )
        imported.discard(self.name)
        return imported

    def declares_dependency_on(
        self, *, member_name: str, members: list[WorkspaceMember]
    ) -> bool:
        """
        Whether the member lists the named member among its dependencies, or lists the
        member whose directory holds it and is released together with it.
        """
        if member_name in self.declared_dependencies | self.optional_dependencies:
            return True
        members_by_name = {member.name: member for member in members}
        return any(
            members_by_name[member_name].directory.is_relative_to(
                members_by_name[declared].directory
            )
            for declared in self.declared_dependencies & members_by_name.keys()
            if declared != member_name
        )

    def members_depended_on(self, *, members: list[WorkspaceMember]) -> set[str]:
        """
        Canonical names of the workspace members the member depends on, directly or
        through other members' declared dependencies.
        """
        members_by_name = {member.name: member for member in members}
        reached = set()
        pending = list(self.declared_dependencies & members_by_name.keys())
        while pending:
            name = pending.pop()
            if name in reached:
                continue
            reached.add(name)
            pending.extend(
                members_by_name[name].declared_dependencies & members_by_name.keys()
            )
        return reached


def workspace_members() -> list[WorkspaceMember]:
    """
    Read the members from ``[tool.uv.workspace]`` of the root ``pyproject.toml``.
    """
    root_project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
    return [
        WorkspaceMember(directory=REPOSITORY_ROOT / member)
        for member in root_project["tool"]["uv"]["workspace"]["members"]
    ]


MEMBERS = workspace_members()
"""
All workspace members, resolved once for parametrization.
"""


# %% declaration checks


@pytest.mark.parametrize("member", MEMBERS, ids=[member.name for member in MEMBERS])
def test_dependencies_are_declared_statically(member: WorkspaceMember) -> None:
    """
    No member leaves its dependencies to its build backend.
    """
    assert "dependencies" not in member.dynamic_fields


@pytest.mark.parametrize("member", MEMBERS, ids=[member.name for member in MEMBERS])
def test_imported_workspace_members_are_declared(member: WorkspaceMember) -> None:
    """
    A member another member imports when it runs is a dependency or an extra of it.
    """
    undeclared_members = {
        imported
        for imported in member.imported_members(members=MEMBERS)
        if not member.declares_dependency_on(member_name=imported, members=MEMBERS)
    }
    assert undeclared_members == set()


@pytest.mark.parametrize("member", MEMBERS, ids=[member.name for member in MEMBERS])
def test_a_member_admits_no_python_older_than_its_dependencies_do(
    member: WorkspaceMember,
) -> None:
    """
    On a Python a dependency rejects, an installer would otherwise fall back to an older
    release of the member instead of failing.
    """
    members_by_name = {other.name: other for other in MEMBERS}
    dependencies_needing_a_newer_python = {
        name
        for name in member.members_depended_on(members=MEMBERS)
        if members_by_name[name].oldest_python > member.oldest_python
    }
    assert dependencies_needing_a_newer_python == set()


# %% layering


def test_semantic_digital_twin_depends_on_no_package_that_depends_on_it() -> None:
    """
    No workspace member that :mod:`semantic_digital_twin` depends on, directly or
    through other members, depends on :mod:`semantic_digital_twin` in turn.
    """
    semantic_digital_twin = next(
        member for member in MEMBERS if member.name == "semantic-digital-twin"
    )
    assert semantic_digital_twin.name not in semantic_digital_twin.members_depended_on(
        members=MEMBERS
    )
