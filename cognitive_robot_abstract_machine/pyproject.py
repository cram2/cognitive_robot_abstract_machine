"""
The parts of a ``pyproject.toml`` the repository's tools read.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from packaging.utils import canonicalize_name

from cognitive_robot_abstract_machine.repository_files import RepositoryFile

# %% keys of a pyproject.toml


class PyprojectKey(StrEnum):
    """The keys of a ``pyproject.toml`` the repository's tools read."""

    PROJECT = "project"
    NAME = "name"
    DEPENDENCIES = "dependencies"
    TOOL = "tool"
    UV = "uv"
    WORKSPACE = "workspace"
    MEMBERS = "members"
    SETUPTOOLS = "setuptools"
    DYNAMIC = "dynamic"
    VERSION = "version"
    ATTRIBUTE = "attr"
    PACKAGE_DIRECTORY = "package-dir"
    PACKAGES = "packages"
    FIND = "find"
    WHERE = "where"
    ROOT_PACKAGE = ""
    """The ``package-dir`` entry every package not mapped by name falls
    under."""


# %% the pyproject.toml model
@dataclass(frozen=True, kw_only=True)
class Pyproject:
    """The parts of a ``pyproject.toml`` the repository's tools read.

    Only :meth:`from_text` knows where in the file they are.
    """

    distribution_name: str
    """The ``[project]`` name as an installer compares names."""

    declared_version: str | None
    """The version the ``[project]`` table states; ``None`` when it states none."""

    dependencies: list[str]
    """The ``[project]`` dependencies as written."""

    version_attribute: str
    """The dotted path of the attribute setuptools reads the version from;
    empty when it reads it from elsewhere."""

    named_package_directories: dict[str, str]
    """The directory setuptools finds each package in, where ``package-dir``
    names the package."""

    default_package_directory: str | None
    """The directory setuptools finds every package not named in
    :attr:`named_package_directories` in; ``None`` without one."""

    find_search_directories: list[str] | None
    """The directories setuptools searches for packages; ``None`` when it does
    not search."""

    workspace_members: list[str]
    """The ``[tool.uv.workspace]`` members; empty without a workspace."""

    @classmethod
    def from_text(cls, *, text: str) -> Pyproject:
        """
        :param text: The content of a ``pyproject.toml``.
        :return: What the tools read of it.
        """
        document = tomllib.loads(text)
        project = document[PyprojectKey.PROJECT]
        tool = document.get(PyprojectKey.TOOL, {})
        setuptools = tool.get(PyprojectKey.SETUPTOOLS, {})
        package_directories = dict(setuptools.get(PyprojectKey.PACKAGE_DIRECTORY, {}))
        default_package_directory = package_directories.pop(
            PyprojectKey.ROOT_PACKAGE, None
        )
        packages = setuptools.get(PyprojectKey.PACKAGES, [])
        search = (
            packages.get(PyprojectKey.FIND, {}).get(PyprojectKey.WHERE, ["."])
            if isinstance(packages, dict)
            else None
        )
        return cls(
            distribution_name=canonicalize_name(project[PyprojectKey.NAME]),
            declared_version=project.get(PyprojectKey.VERSION),
            dependencies=project.get(PyprojectKey.DEPENDENCIES, []),
            version_attribute=setuptools.get(PyprojectKey.DYNAMIC, {})
            .get(PyprojectKey.VERSION, {})
            .get(PyprojectKey.ATTRIBUTE, ""),
            named_package_directories=package_directories,
            default_package_directory=default_package_directory,
            find_search_directories=search,
            workspace_members=tool.get(PyprojectKey.UV, {})
            .get(PyprojectKey.WORKSPACE, {})
            .get(PyprojectKey.MEMBERS, []),
        )

    @classmethod
    def from_directory(cls, *, directory: Path) -> Pyproject:
        """
        :param directory: A directory holding a ``pyproject.toml``.
        :return: What the tools read of that file.
        """
        return cls.from_text(text=(directory / RepositoryFile.PYPROJECT).read_text())
