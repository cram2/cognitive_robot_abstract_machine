"""
What this package contains, discovered rather than listed.

Which modules exist and which of them answer as a command line are both read from the
package itself - the directory, and each module's own source. Nothing about the package
is written down here.

No module is held to the standard library: the dependencies are installed before anything
runs one. A session gets them from ``session-start.sh``, which installs whatever is
missing on every start, and an Actions workflow installs them in a step of its own.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PACKAGE_DIRECTORY = Path(__file__).parent
"""
This package's own directory, which is also the directory it *is* rather than lives under.
"""

REPOSITORY_ROOT = PACKAGE_DIRECTORY.parent
"""
The repository root, which is the directory ``basstler`` imports from with no install.
"""


@dataclass(frozen=True)
class PackageModule:
    """
    One module of this package.
    """

    name: str
    """
    The module's name within the package, e.g. ``"stack"``.
    """

    is_command_line_entry_point: bool
    """
    Whether the module guards a block on being run as ``__main__``, which is what makes
    ``python3 -m basstler.<name> --help`` answer rather than do nothing.
    """

    @property
    def import_path(self) -> str:
        """:return: The dotted path this module is imported by."""
        return f"basstler.{self.name}"


def _has_a_main_block(source: Path) -> bool:
    """
    :param source: A module's file.
    :return: Whether it guards a top-level block on being run as ``__main__``.
    """
    return any(
        isinstance(node, ast.If) and "__main__" in ast.dump(node.test)
        for node in ast.parse(source.read_text()).body
    )


@lru_cache(maxsize=1)
def package_modules() -> tuple[PackageModule, ...]:
    """
    :return: Every module in this package, in name order.
    """
    return tuple(
        PackageModule(path.stem, _has_a_main_block(path))
        for path in sorted(PACKAGE_DIRECTORY.glob("*.py"))
        if path.stem != "__init__"
    )


def command_line_entry_points() -> tuple[PackageModule, ...]:
    """
    :return: The modules a script or a session invokes directly.
    """
    return tuple(
        module for module in package_modules() if module.is_command_line_entry_point
    )
