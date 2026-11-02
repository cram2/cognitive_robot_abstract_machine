"""Synchronize the root VERSION file into every package of the workspace.

For each package that reads its version from a module of its own, the
script writes that module and raises every dependency on another such
package to ``>=`` the repository's version, so no package is installable
next to an older release of another. The packages are the root's and the
members of its ``[tool.uv.workspace]``.

``--check`` writes nothing and exits non-zero when a file would change.
"""

from __future__ import annotations

import argparse
import re
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


# %% names the script reads
class RepositoryFile(StrEnum):
    """The files this script reads, by name."""

    VERSION = "VERSION"
    """The repository's one version, at its root."""

    PYPROJECT = "pyproject.toml"
    """A package's metadata, at the root of the package's directory."""


class PyprojectKey(StrEnum):
    """The keys of a ``pyproject.toml`` the script reads."""

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


class CommandFlag(StrEnum):
    """The command-line flags of this script."""

    CHECK = "--check"


# %% the pyproject.toml model
@dataclass(frozen=True)
class Pyproject:
    """The parts of a ``pyproject.toml`` this script reads.

    Only :meth:`from_text` knows where in the file they are.
    """

    distribution_name: str
    """The ``[project]`` name as an installer compares names."""

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
    def from_text(cls, text: str) -> Pyproject:
        """
        :param text: The content of a ``pyproject.toml``.
        :return: What this script reads of it.
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
    def from_directory(cls, directory: Path) -> Pyproject:
        """
        :param directory: A directory holding a ``pyproject.toml``.
        :return: What this script reads of that file.
        """
        return cls.from_text((directory / RepositoryFile.PYPROJECT).read_text())


# %% a package
@dataclass(frozen=True)
class VersionedPackage:
    """A package whose version setuptools reads from a module of the package's
    own."""

    directory: Path
    """The directory the package's ``pyproject.toml`` is in."""

    pyproject: Pyproject
    """What this script reads of that file."""

    @classmethod
    def from_directory(cls, directory: Path) -> VersionedPackage | None:
        """
        :param directory: A directory holding a ``pyproject.toml``.
        :return: Its package, or ``None`` when it reads no version from a module of its
            own - a ``__version__`` set in the package's ``__init__`` is not this script's
            to write.
        """
        package = cls(directory, Pyproject.from_directory(directory))
        if len(package.version_module_path) < 2:
            return None
        return package

    @property
    def distribution_name(self) -> str:
        """
        :return: The package's name as an installer compares names.
        """
        return self.pyproject.distribution_name

    @property
    def version_module_path(self) -> list[str]:
        """
        :return: The dotted path of the module the version is read from, split, e.g.
            ``["krrood", "_version"]``; empty when the version is not read from one.
        """
        return self.pyproject.version_attribute.split(".")[:-1]

    @property
    def package_directory(self) -> Path:
        """
        :return: Where setuptools finds the package: the directory ``package-dir`` maps it
            to, or the one it maps every package to, or the first ``packages.find``
            directory, or else the package's own name beside its ``pyproject.toml``.
        """
        name = self.version_module_path[0]
        if name in self.pyproject.named_package_directories:
            return self.directory / self.pyproject.named_package_directories[name]
        if self.pyproject.default_package_directory is not None:
            return self.directory / self.pyproject.default_package_directory / name
        if self.pyproject.find_search_directories is not None:
            return self.directory / self.pyproject.find_search_directories[0] / name
        return self.directory / name

    @property
    def version_file(self) -> Path:
        """
        :return: The module this package reads its version from.
        """
        return self.package_directory.joinpath(
            *self.version_module_path[1:]
        ).with_suffix(".py")

    @property
    def pyproject_file(self) -> Path:
        """
        :return: The package's ``pyproject.toml``.
        """
        return self.directory / RepositoryFile.PYPROJECT

    def version_module_text(self, version: str) -> str:
        """
        :param version: The version the package is to carry.
        :return: The content its version module is to have.
        """
        return f'__version__ = "{version}"\n'

    def pyproject_text(self, siblings: set[str], version: str) -> str:
        """
        :param siblings: The names, as an installer compares them, of the packages whose
            release carries ``version``.
        :param version: The version every sibling is released at.
        :return: The content of the package's ``pyproject.toml`` with every ``[project]``
            dependency on a sibling bounded to ``>=version``. A bound already there is
            replaced; extras and markers stay, and so does everything else in the file.
        """
        text = self.pyproject_file.read_text()
        for specifier in self.pyproject.dependencies:
            requirement = Requirement(specifier)
            name = canonicalize_name(requirement.name)
            if name not in siblings or name == self.distribution_name:
                continue
            text = self.replace_dependency(
                text, specifier, self.bounded(specifier, version)
            )
        return text

    @staticmethod
    def bounded(specifier: str, version: str) -> str:
        """
        :param specifier: A dependency as the ``dependencies`` array lists it.
        :param version: The lowest version it may be satisfied by.
        :return: The dependency with its version specifier replaced by ``>=version``;
            its extras and its environment marker, written as they were, stay.
        """
        requirement = Requirement(specifier)
        extras = (
            f"[{','.join(sorted(requirement.extras))}]" if requirement.extras else ""
        )
        marker = specifier[specifier.index(";") :] if requirement.marker else ""
        return f"{requirement.name}{extras}>={version}{marker}"

    @staticmethod
    def replace_dependency(text: str, old: str, new: str) -> str:
        """
        :param text: The content of a ``pyproject.toml``.
        :param old: A dependency as the ``[project]`` ``dependencies`` array lists it.
        :param new: What to list instead.
        :return: ``text`` with that array's ``old`` entry replaced by ``new``; nothing
            else, such as a dependency group naming the same package, is touched.
        """
        array = re.search(
            rf"^{PyprojectKey.DEPENDENCIES}\s*=\s*\[.*?^\]",
            text,
            re.DOTALL | re.MULTILINE,
        )
        entry = re.compile(rf"([\"']){re.escape(old)}\1")
        replaced = entry.sub(lambda match: f"{match[1]}{new}{match[1]}", array[0])
        return text[: array.start()] + replaced + text[array.end() :]


# %% the repository
@dataclass(frozen=True)
class ManagedFile:
    """A file this script keeps in sync, with the content it is to have."""

    path: Path
    """Where the file is."""

    text: str
    """The content the file is to have."""

    @property
    def is_outdated(self) -> bool:
        """
        :return: Whether the file is missing or its content differs from :attr:`text`.
        """
        return not self.path.exists() or self.path.read_text() != self.text


@dataclass(frozen=True)
class Repository:
    """This repository, whose root holds the version every package carries."""

    root: Path
    """The repository's root directory."""

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

    @property
    def package_directories(self) -> list[Path]:
        """
        :return: The root and every member of the root's ``[tool.uv.workspace]``.
        """
        members = Pyproject.from_directory(self.root).workspace_members
        return [self.root] + sorted(self.root / member for member in members)

    def versioned_packages(self) -> list[VersionedPackage]:
        """
        :return: The packages among the root and the workspace members that read their
            version from a module of their own.
        """
        packages = (
            VersionedPackage.from_directory(directory)
            for directory in self.package_directories
        )
        return [package for package in packages if package is not None]

    def expected_files(self) -> list[ManagedFile]:
        """
        :return: Every file this script manages, with the content it is to have.
        """
        packages = self.versioned_packages()
        siblings = {package.distribution_name for package in packages}
        files = []
        for package in packages:
            files.append(
                ManagedFile(
                    package.version_file, package.version_module_text(self.version)
                )
            )
            files.append(
                ManagedFile(
                    package.pyproject_file,
                    package.pyproject_text(siblings, self.version),
                )
            )
        return files

    def outdated_files(self) -> list[ManagedFile]:
        """
        :return: The managed files whose content differs from what they are to have.
        """
        return [file for file in self.expected_files() if file.is_outdated]

    def synchronize(self) -> list[ManagedFile]:
        """Write every managed file whose content is outdated.

        :return: The files written.
        """
        outdated = self.outdated_files()
        for file in outdated:
            file.path.write_text(file.text)
        return outdated


# %% command line
def main() -> int:
    """
    :return: The exit status: non-zero when ``--check`` finds an outdated file.
    """
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument(
        CommandFlag.CHECK,
        action="store_true",
        help="Write nothing; fail when a file is outdated.",
    )
    repository = Repository.of_this_script()
    if parser.parse_args().check:
        outdated = repository.outdated_files()
        for file in outdated:
            print(f"Outdated {file.path}; run scripts/sync_version.py")
        return int(bool(outdated))
    for file in repository.synchronize():
        print(f"Updated {file.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
