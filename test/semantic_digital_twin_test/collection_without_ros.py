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
from pathlib import Path

import pytest
from typing_extensions import Self

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
class RosInstallation:
    """
    A sourced ROS installation and the workspaces sourced on top of it.
    """

    install_prefixes: list[Path] = field(default_factory=list)
    """
    The directories the installation and its workspaces are installed in.
    """

    @classmethod
    def from_environment(cls) -> Self:
        """
        :return: The installation the environment has sourced, which has no install
            prefixes where no ROS installation is sourced.
        """
        prefixes = os.environ.get(RosEnvironmentVariable.INSTALL_PREFIXES, "")
        return cls([Path(prefix) for prefix in prefixes.split(os.pathsep) if prefix])

    def hide_from(self, module_search_path: list[str]) -> None:
        """
        Removes every directory of this installation from the module search path, so
        neither its modules nor the pytest plugins it registers can be found.
        """
        module_search_path[:] = [
            entry
            for entry in module_search_path
            if not any(
                Path(entry).is_relative_to(prefix) for prefix in self.install_prefixes
            )
        ]


# %% collection


def main() -> None:
    """
    Collects the tests beside this script with ROS hidden.
    """
    RosInstallation.from_environment().hide_from(sys.path)
    sys.exit(pytest.main(["--collect-only", "--quiet", str(Path(__file__).parent)]))


if __name__ == "__main__":
    main()
