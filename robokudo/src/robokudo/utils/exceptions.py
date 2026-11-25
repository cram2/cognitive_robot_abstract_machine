"""
Errors raised while resolving RoboKudo resources.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass
from pathlib import Path

from robokudo.exceptions import RoboKudoError


# %% ROS package assets
@dataclass(kw_only=True)
class RosPackageMissing(RoboKudoError, OSError):
    """
    A requested ROS package is absent from the active package index.
    """

    ros_package_name: str
    """
    Package whose shared resources cannot be located.
    """

    def error_message(self) -> str:
        return f"ROS2 package {self.ros_package_name!r} does not exist."

    def suggest_correction(self) -> str:
        return (
            "Install the ROS package and source its workspace before resolving assets."
        )


@dataclass(kw_only=True)
class PackageFileMissing(RoboKudoError, OSError):
    """
    A requested path does not exist inside a resolved ROS package.
    """

    file_path: Path
    """
    Package asset that could not be found.
    """

    def error_message(self) -> str:
        return f"The desired path {self.file_path} does not exist."

    def suggest_correction(self) -> str:
        return "Check the package-relative path and install the requested asset."
