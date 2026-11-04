"""
Collects the semantic digital twin tests in a process that cannot import ROS, and exits
with the exit code of that collection.

Run as a subprocess by :mod:`test_collection_without_ros`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder
from pathlib import Path
from types import ModuleType

import pytest
from typing_extensions import Self, Sequence

# %% hiding ROS


class RosEnvironmentVariable(StrEnum):
    """
    Environment variables a sourced ROS installation sets.
    """

    INSTALL_PREFIXES = "AMENT_PREFIX_PATH"
    """
    The directories ROS and the sourced workspaces are installed in.
    """


@dataclass
class RosImportBlocker(MetaPathFinder):
    """
    Makes every module installed under a ROS install prefix unimportable, as if ROS were
    not installed.
    """

    install_prefixes: list[Path] = field(default_factory=list)
    """
    The directories whose modules cannot be imported.
    """

    @classmethod
    def from_environment(cls) -> Self:
        """
        :return: A blocker for the install prefixes of the sourced ROS installation, which
            blocks nothing where no ROS installation is sourced.
        """
        prefixes = os.environ.get(RosEnvironmentVariable.INSTALL_PREFIXES, "")
        return cls([Path(prefix) for prefix in prefixes.split(os.pathsep) if prefix])

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        """
        :return: Nothing, leaving the module to the finders after this one.
        :raises ModuleNotFoundError: If the module is installed under a ROS install
            prefix.
        """
        specification = PathFinder.find_spec(fullname, path)
        if specification is None:
            return None
        locations = [
            Path(location)
            for location in [
                specification.origin,
                *(specification.submodule_search_locations or []),
            ]
            if location is not None
        ]
        if any(
            location.is_relative_to(prefix)
            for location in locations
            for prefix in self.install_prefixes
        ):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


# %% collection


def main() -> None:
    """
    Collects the tests beside this script with ROS hidden.
    """
    sys.meta_path.insert(0, RosImportBlocker.from_environment())
    sys.exit(pytest.main(["--collect-only", "--quiet", str(Path(__file__).parent)]))


if __name__ == "__main__":
    main()
