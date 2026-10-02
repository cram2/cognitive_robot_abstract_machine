#!/usr/bin/env python3
"""
What this package needs installed, and which of it this environment is missing.

The declaration is ``pyproject.toml``'s ``[project] dependencies``, which is where every
package in this repository states them, so there is one list rather than a metadata table
and a requirements file that can disagree.

Kept as a module rather than as a snippet inside the shell that calls it: the bash entry
points ask this question before anything is installed, and a question with parsing in it
is real, testable code wherever it is written.

Usage:
    python3 -m basstler.dependencies [--declaration <pyproject.toml>]

Prints one requirement specifier per missing dependency, and nothing at all when the
environment already has them - which is what a caller passes straight to ``pip install``.

..note:: Imports nothing outside the standard library. It runs before any install, so a
    dependency of its own would be the one thing it could never report.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from importlib.metadata import distributions
from pathlib import Path

from basstler.locations import PackageLocation
from basstler.standard_streams import StandardStreamHandler

logger = StandardStreamHandler.logger_for(__name__)
"""
This module's logger, which is also what its command prints through.
"""


class PyprojectKey(StrEnum):
    """
    The keys of ``pyproject.toml`` this module reads.
    """

    PROJECT = "project"
    """
    The table a package's own metadata lives in.
    """

    DEPENDENCIES = "dependencies"
    """
    The field of that table listing the requirement specifiers.
    """


@dataclass
class UnreadableDependencyDeclarationError(Exception):
    """
    Raised when the declaration cannot be read, so nothing can be said about what is
    missing.

    Distinct from *nothing is missing*, which is what a caller would otherwise conclude
    from an empty answer and act on by installing nothing.
    """

    declaration: Path
    """
    The file that was to be read.
    """

    def __str__(self) -> str:
        """:return: What to tell a reader who asked what is missing and cannot be told."""
        return (
            f"{self.declaration} does not exist, so this package's dependencies cannot "
            f"be read"
        )


@dataclass(frozen=True)
class Dependency:
    """
    One requirement this package declares.
    """

    specifier: str
    """
    The requirement as ``pyproject.toml`` writes it, version bounds and all.
    """

    constraint_start: re.Pattern[str] = field(
        default=re.compile(r"[<>=!~;\[ ]"), repr=False, compare=False
    )
    """
    The first character that ends a distribution's name and begins a version bound, an extra
    or an environment marker, in a PEP 508 specifier.
    """

    @property
    def distribution_name(self) -> str:
        """:return: The distribution this requirement names, without its constraints."""
        return self.constraint_start.split(self.specifier, maxsplit=1)[0].strip()

    @property
    def is_missing(self) -> bool:
        """
        Presence rather than version: an installed distribution is left alone, which is
        what lets a session start run this on every start and install nothing. Names are
        compared as PEP 503 does, so a declaration need not match a distribution's own
        spelling.

        :return: Whether this environment has no distribution of that name.
        """
        return next(iter(distributions(name=self.distribution_name)), None) is None


@dataclass(frozen=True)
class DependencyDeclaration:
    """
    A ``pyproject.toml`` and the dependencies it declares.
    """

    path: Path
    """
    The ``pyproject.toml`` to read.
    """

    @classmethod
    def of_this_package(cls) -> DependencyDeclaration:
        """
        :return: This package's own metadata, found beside the modules it declares the
            dependencies of.
        """
        return cls(PackageLocation.DEPENDENCY_DECLARATION.value)

    def dependencies(self) -> tuple[Dependency, ...]:
        """
        :return: Every dependency it declares, in the order it declares them.
        :raises UnreadableDependencyDeclarationError: If the file is absent.
        """
        if not self.path.is_file():
            raise UnreadableDependencyDeclarationError(self.path)
        project = tomllib.loads(self.path.read_text(encoding="utf-8"))
        return tuple(
            Dependency(specifier)
            for specifier in project[PyprojectKey.PROJECT].get(
                PyprojectKey.DEPENDENCIES, []
            )
        )

    def missing(self) -> tuple[Dependency, ...]:
        """
        :return: The declared dependencies this environment does not have.
        :raises UnreadableDependencyDeclarationError: If the file is absent.
        """
        return tuple(
            dependency for dependency in self.dependencies() if dependency.is_missing
        )


class ExitCode(IntEnum):
    """
    How the command ended, as its caller reads it.
    """

    SUCCESS = 0
    """
    The missing dependencies, if any, were printed.
    """

    UNREADABLE_DECLARATION = 1
    """
    The declaration could not be read, so nothing was printed.
    """


def main() -> ExitCode:
    """
    Print one specifier per missing dependency, for a caller to hand to an installer.

    :return: How the command ended.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--declaration",
        type=Path,
        default=DependencyDeclaration.of_this_package().path,
        help="The pyproject.toml to read the dependencies from",
    )
    declaration = DependencyDeclaration(parser.parse_args().declaration)
    if not declaration.path.is_file():
        logger.error(str(UnreadableDependencyDeclarationError(declaration.path)))
        return ExitCode.UNREADABLE_DECLARATION
    for dependency in declaration.missing():
        logger.info(dependency.specifier)
    return ExitCode.SUCCESS


if __name__ == "__main__":
    sys.exit(main())
