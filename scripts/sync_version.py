"""Synchronize the root VERSION file into every package of the workspace: the root and
the members of its ``[tool.uv.workspace]``. Dependencies between them require at least
that version, and exactly it between packages released together.

``--check`` writes nothing and exits non-zero when a file would change.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from cognitive_robot_abstract_machine.pyproject import Pyproject, PyprojectKey


# %% names the script reads
class RepositoryFile(StrEnum):
    """The files this script reads, by name."""

    VERSION = "VERSION"
    """The repository's one version, at its root."""

    PYPROJECT = "pyproject.toml"
    """A package's metadata, at the root of the package's directory."""


class CommandFlag(StrEnum):
    """The command-line flags of this script."""

    CHECK = "--check"


class VersionOperator(StrEnum):
    """How a dependency on another package of the workspace is bounded."""

    AT_LEAST = ">="
    """A package released on its own accepts any later release of the other."""

    EXACTLY = "=="
    """Packages released together install only with each other's release."""


# %% a package
@dataclass(frozen=True, kw_only=True)
class VersionedPackage:
    """A package whose version setuptools reads from a module of the package's
    own, or one that declares the version of the packages its directory holds."""

    directory: Path
    """The directory the package's ``pyproject.toml`` is in."""

    pyproject: Pyproject
    """What this script reads of that file."""

    @classmethod
    def from_directory(
        cls, *, directory: Path, holds_other_packages: bool
    ) -> VersionedPackage | None:
        """
        :param directory: A directory holding a ``pyproject.toml``.
        :param holds_other_packages: Whether another package of the workspace is in the
            directory.
        :return: Its package, or ``None`` when it reads no version from a module of its
            own and declares none for packages it holds - a ``__version__`` set in the
            package's ``__init__`` is not this script's to write.
        """
        package = cls(
            directory=directory, pyproject=Pyproject.from_directory(directory=directory)
        )
        if package.has_version_module:
            return package
        if holds_other_packages and package.pyproject.declared_version is not None:
            return package
        return None

    @property
    def has_version_module(self) -> bool:
        """
        :return: Whether setuptools reads the version from a module of the package.
        """
        return len(self.version_module_path) >= 2

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

    def version_module_text(self, *, version: str) -> str:
        """
        :param version: The version the package is to carry.
        :return: The content its version module is to have.
        """
        return f'__version__ = "{version}"\n'

    def pyproject_text(
        self, *, siblings: set[str], released_together: set[str], version: str
    ) -> str:
        """
        :param siblings: The names, as an installer compares them, of the packages whose
            release carries ``version``.
        :param released_together: The names of the siblings released together with this
            package.
        :param version: The version every sibling is released at.
        :return: The content of the package's ``pyproject.toml`` with every ``[project]``
            dependency on a sibling bounded to ``==version`` if it is released together
            with this package and to ``>=version`` otherwise, and the declared version, if
            any, set to ``version``. A bound already there is replaced; extras and markers
            stay, and so does everything else in the file.
        """
        text = self.pyproject_file.read_text()
        if self.pyproject.declared_version is not None:
            text = self.replace_declared_version(text=text, version=version)
        for specifier in self.pyproject.dependencies:
            requirement = Requirement(specifier)
            name = canonicalize_name(requirement.name)
            if name not in siblings or name == self.distribution_name:
                continue
            operator = (
                VersionOperator.EXACTLY
                if name in released_together
                else VersionOperator.AT_LEAST
            )
            text = self.replace_dependency(
                text=text,
                old=specifier,
                new=self.bounded(
                    specifier=specifier, operator=operator, version=version
                ),
            )
        return text

    @staticmethod
    def replace_declared_version(*, text: str, version: str) -> str:
        """
        :param text: The content of a ``pyproject.toml`` whose ``[project]`` table states
            a version.
        :param version: The version to state instead.
        :return: ``text`` with that version replaced.
        """
        return re.sub(
            rf'^({PyprojectKey.VERSION}\s*=\s*)(["\'])[^"\']*\2',
            lambda match: f"{match[1]}{match[2]}{version}{match[2]}",
            text,
            count=1,
            flags=re.MULTILINE,
        )

    @staticmethod
    def bounded(*, specifier: str, operator: VersionOperator, version: str) -> str:
        """
        :param specifier: A dependency as the ``dependencies`` array lists it.
        :param operator: How the dependency is bounded by ``version``.
        :param version: The version that bounds it.
        :return: The dependency with its version specifier replaced by ``operator`` and
            ``version``; its extras and its environment marker, written as they were, stay.
        """
        requirement = Requirement(specifier)
        extras = (
            f"[{','.join(sorted(requirement.extras))}]" if requirement.extras else ""
        )
        marker = specifier[specifier.index(";") :] if requirement.marker else ""
        return f"{requirement.name}{extras}{operator}{version}{marker}"

    @staticmethod
    def replace_dependency(*, text: str, old: str, new: str) -> str:
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


@dataclass(frozen=True, kw_only=True)
class Repository:
    """This repository, whose root holds the version every package carries."""

    root: Path
    """The repository's root directory."""

    @classmethod
    def from_this_checkout(cls) -> Repository:
        """
        :return: The repository this script is checked out in.
        """
        return cls(root=Path(__file__).resolve().parents[1])

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
        members = Pyproject.from_directory(directory=self.root).workspace_members
        return [self.root] + sorted(self.root / member for member in members)

    def versioned_packages(self) -> list[VersionedPackage]:
        """
        :return: The packages among the root and the workspace members that read their
            version from a module of their own or declare it for packages they hold.
        """
        packages = (
            VersionedPackage.from_directory(
                directory=directory,
                holds_other_packages=bool(self.packages_held_by(directory=directory)),
            )
            for directory in self.package_directories
        )
        return [package for package in packages if package is not None]

    def packages_held_by(self, *, directory: Path) -> list[Path]:
        """
        :param directory: A package directory.
        :return: The other package directories inside it; none for the root, whose
            members are released on their own.
        """
        if directory == self.root:
            return []
        return [
            other
            for other in self.package_directories
            if other != directory and other.is_relative_to(directory)
        ]

    def release_directory(self, *, package: VersionedPackage) -> Path:
        """
        :param package: A package of the workspace.
        :return: The outermost package directory, the root aside, that holds the package
            or is its own; packages with the same one are released together.
        """
        return min(
            (
                directory
                for directory in self.package_directories
                if directory != self.root
                and package.directory.is_relative_to(directory)
            ),
            key=lambda directory: len(directory.parts),
            default=self.root,
        )

    def expected_files(self) -> list[ManagedFile]:
        """
        :return: Every file this script manages, with the content it is to have.
        """
        packages = self.versioned_packages()
        siblings = {package.distribution_name for package in packages}
        files = []
        for package in packages:
            if package.has_version_module:
                files.append(
                    ManagedFile(
                        package.version_file,
                        package.version_module_text(version=self.version),
                    )
                )
            released_together = {
                other.distribution_name
                for other in packages
                if self.release_directory(package=other)
                == self.release_directory(package=package)
            }
            files.append(
                ManagedFile(
                    package.pyproject_file,
                    package.pyproject_text(
                        siblings=siblings,
                        released_together=released_together,
                        version=self.version,
                    ),
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
    repository = Repository.from_this_checkout()
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
