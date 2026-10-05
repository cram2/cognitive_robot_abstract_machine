"""
Synchronize the root VERSION file into every package's version module.

A package is found from its own ``pyproject.toml``: one whose setuptools configuration
reads its version from a module of its own is written to, wherever that configuration
puts the module.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class RepositoryFile(StrEnum):
    """
    The files this script reads, by name.
    """

    VERSION = "VERSION"
    """
    The repository's one version, at its root.
    """

    PYPROJECT = "pyproject.toml"
    """
    A package's metadata, at the root of the package's directory.
    """


class SetuptoolsKey(StrEnum):
    """
    The keys of a ``pyproject.toml`` that say where a package's version module is.
    """

    TOOL = "tool"
    SETUPTOOLS = "setuptools"
    DYNAMIC = "dynamic"
    VERSION = "version"
    ATTRIBUTE = "attr"
    PACKAGE_DIRECTORY = "package-dir"
    PACKAGES = "packages"
    FIND = "find"
    WHERE = "where"
    ROOT_PACKAGE = ""
    """
    The ``package-dir`` entry every package not mapped by name falls under.
    """


@dataclass(frozen=True)
class VersionedPackage:
    """
    A package whose version setuptools reads from a module of the package's own.
    """

    directory: Path
    """
    The directory the package's ``pyproject.toml`` is in.
    """

    setuptools: dict[str, Any]
    """
    That file's ``[tool.setuptools]`` table.
    """

    @classmethod
    def from_directory(cls, directory: Path) -> VersionedPackage | None:
        """
        :param directory: A directory holding a ``pyproject.toml``.
        :return: Its package, or ``None`` when it reads no version from a module of its
            own - a ``__version__`` set in the package's ``__init__`` is not this script's
            to write.
        """
        pyproject = tomllib.loads((directory / RepositoryFile.PYPROJECT).read_text())
        setuptools = pyproject.get(SetuptoolsKey.TOOL, {}).get(
            SetuptoolsKey.SETUPTOOLS, {}
        )
        package = cls(directory, setuptools)
        if len(package.version_module_path) < 2:
            return None
        return package

    @property
    def version_module_path(self) -> list[str]:
        """
        :return: The dotted path of the module the version is read from, split, e.g.
            ``["krrood", "_version"]``; empty when the version is not read from one.
        """
        attribute = (
            self.setuptools.get(SetuptoolsKey.DYNAMIC, {})
            .get(SetuptoolsKey.VERSION, {})
            .get(SetuptoolsKey.ATTRIBUTE, "")
        )
        return attribute.split(".")[:-1]

    @property
    def package_directory(self) -> Path:
        """
        :return: Where setuptools finds the package: the directory ``package-dir`` maps it
            to, or the one it maps every package to, or the first ``packages.find``
            directory, or else the package's own name beside its ``pyproject.toml``.
        """
        name = self.version_module_path[0]
        mapping = self.setuptools.get(SetuptoolsKey.PACKAGE_DIRECTORY, {})
        if name in mapping:
            return self.directory / mapping[name]
        if SetuptoolsKey.ROOT_PACKAGE in mapping:
            return self.directory / mapping[SetuptoolsKey.ROOT_PACKAGE] / name
        packages = self.setuptools.get(SetuptoolsKey.PACKAGES, [])
        if isinstance(packages, dict):
            search = packages.get(SetuptoolsKey.FIND, {}).get(
                SetuptoolsKey.WHERE, ["."]
            )
            return self.directory / search[0] / name
        return self.directory / name

    @property
    def version_file(self) -> Path:
        """
        :return: The module this package reads its version from.
        """
        return self.package_directory.joinpath(
            *self.version_module_path[1:]
        ).with_suffix(".py")


@dataclass(frozen=True)
class Repository:
    """
    This repository, whose root holds the version every package carries.
    """

    root: Path
    """
    The repository's root directory.
    """

    @classmethod
    def of_this_script(cls) -> Repository:
        """
        :return: The repository this script is checked out in.
        """
        return cls(Path(__file__).resolve().parents[1])

    @property
    def version(self) -> str:
        """
        :return: The version every package is to carry.
        """
        return (self.root / RepositoryFile.VERSION).read_text().strip()

    def versioned_packages(self) -> list[VersionedPackage]:
        """
        :return: The root's package and every package one directory below it, where
            each reads its version from a module of its own.
        """
        directories = [self.root] + sorted(
            pyproject.parent
            for pyproject in self.root.glob(f"*/{RepositoryFile.PYPROJECT}")
        )
        packages = (
            VersionedPackage.from_directory(directory) for directory in directories
        )
        return [package for package in packages if package is not None]


def main() -> None:
    repository = Repository.of_this_script()
    for package in repository.versioned_packages():
        package.version_file.write_text(f'__version__ = "{repository.version}"\n')
        print(f"Updated {package.version_file}")


if __name__ == "__main__":
    main()
