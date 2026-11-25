"""
Resolve installed ROS package directories and their shared assets.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass
from pathlib import Path

from ament_index_python.packages import (
    get_package_share_directory,
    get_packages_with_prefixes,
)

from robokudo.utils.exceptions import PackageFileMissing, RosPackageMissing


# %% ROS package resolution
@dataclass(kw_only=True)
class FileLoader:
    """
    Resolve shared package assets and report unavailable packages or paths.
    """

    @staticmethod
    def get_ros_package_path(*, ros_package_name: str) -> Path:
        """
        Resolve the shared directory of an installed ROS package.

        :raises RosPackageMissing: If the active package index contains no such package.
        """
        if ros_package_name not in get_packages_with_prefixes():
            raise RosPackageMissing(ros_package_name=ros_package_name)
        return Path(get_package_share_directory(ros_package_name))

    @staticmethod
    def get_path_to_file_in_ros_package(
        *, ros_package_name: str, relative_path: str
    ) -> Path:
        """
        Resolve an existing asset relative to a ROS package's shared directory.

        :raises RosPackageMissing: If the package is not installed.
        :raises PackageFileMissing: If the resolved asset does not exist.
        """
        ros_package_path = FileLoader.get_ros_package_path(
            ros_package_name=ros_package_name
        )
        full_path = ros_package_path / relative_path
        if not full_path.exists():
            raise PackageFileMissing(file_path=full_path)
        return full_path
