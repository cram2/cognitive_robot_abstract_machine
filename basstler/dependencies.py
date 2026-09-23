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
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import distributions
from pathlib import Path

DECLARATION_PATH = Path(__file__).parent / "pyproject.toml"
"""
This package's own metadata, found beside the modules it declares the dependencies of.
"""

PROJECT_TABLE = "project"
"""
The ``pyproject.toml`` table a package's own metadata lives in.
"""

DEPENDENCIES_FIELD = "dependencies"
"""
The field of that table listing the requirement specifiers.
"""

CONSTRAINT_START = re.compile(r"[<>=!~;\[ ]")
"""
The first character that ends a distribution's name and begins a version bound, an extra
or an environment marker.
"""

NAME_SEPARATORS = re.compile(r"[-_.]+")
"""
The characters a distribution name may be spelled with interchangeably, per PEP 503.
"""


class UnreadableDependencyDeclarationError(Exception):
    """
    Raised when the declaration cannot be read, so nothing can be said about what is
    missing.

    Distinct from *nothing is missing*, which is what a caller would otherwise conclude
    from an empty answer and act on by installing nothing.
    """


def unreadable_declaration_message(declaration: Path) -> str:
    """
    :param declaration: The file that was to be read.
    :return: What to tell a reader who asked what is missing and cannot be told.
    """
    return (
        f"{declaration} does not exist, so this package's dependencies cannot be read"
    )


def canonical_name(distribution_name: str) -> str:
    """
    :param distribution_name: A distribution's name as anyone spells it.
    :return: The one spelling ``PyYAML``, ``pyyaml`` and ``py-yaml`` share.
    """
    return NAME_SEPARATORS.sub("-", distribution_name).lower()


@lru_cache(maxsize=1)
def installed_distribution_names() -> frozenset[str]:
    """
    :return: The canonical name of every distribution installed in this environment.
    """
    return frozenset(
        canonical_name(installed.metadata["Name"])
        for installed in distributions()
        if installed.metadata["Name"]
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

    @property
    def distribution_name(self) -> str:
        """:return: The distribution this requirement names, without its constraints."""
        return CONSTRAINT_START.split(self.specifier, maxsplit=1)[0].strip()

    @property
    def is_missing(self) -> bool:
        """
        Presence rather than version: an installed distribution is left alone, which is
        what lets a session start run this on every start and install nothing.

        :return: Whether this environment has no distribution of that name.
        """
        return canonical_name(self.distribution_name) not in (
            installed_distribution_names()
        )


def declared_dependencies(
    declaration: Path = DECLARATION_PATH,
) -> tuple[Dependency, ...]:
    """
    :param declaration: The ``pyproject.toml`` to read.
    :return: Every dependency it declares, in the order it declares them.
    :raises UnreadableDependencyDeclarationError: If the file is absent or unparseable.
    """
    if not declaration.is_file():
        raise UnreadableDependencyDeclarationError(
            unreadable_declaration_message(declaration)
        )
    project = tomllib.loads(declaration.read_text(encoding="utf-8"))
    return tuple(
        Dependency(specifier)
        for specifier in project[PROJECT_TABLE].get(DEPENDENCIES_FIELD, [])
    )


def missing_dependencies(
    declaration: Path = DECLARATION_PATH,
) -> tuple[Dependency, ...]:
    """
    :param declaration: The ``pyproject.toml`` to read.
    :return: The declared dependencies this environment does not have.
    """
    return tuple(
        dependency
        for dependency in declared_dependencies(declaration)
        if dependency.is_missing
    )


def main() -> None:
    """
    Print one specifier per missing dependency, for a caller to hand to an installer.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--declaration",
        type=Path,
        default=DECLARATION_PATH,
        help="The pyproject.toml to read the dependencies from",
    )
    arguments = parser.parse_args()
    if not arguments.declaration.is_file():
        sys.exit(unreadable_declaration_message(arguments.declaration))
    for dependency in missing_dependencies(arguments.declaration):
        print(dependency.specifier)


if __name__ == "__main__":
    main()
