"""
Verify ROS package paths and package-relative files resolve or report missing data.
"""

from __future__ import annotations

import pytest

from robokudo.utils.file_loader import FileLoader


class TestUtilsFileLoader:
    """
    Check installed ROS package assets and missing-path errors.
    """

    def test_file_loader_find_robokudo_ros_package(self):
        assert FileLoader.get_ros_package_path(ros_package_name="robokudo_ros")

    def test_file_loader_non_existing_ros_package(self):
        assert pytest.raises(
            OSError,
            FileLoader.get_ros_package_path,
            ros_package_name="skdaopsi2098392183robokud0",
        )

    def test_file_loader_find_file_in_robokudo_ros_package(self):
        assert FileLoader.get_path_to_file_in_ros_package(
            ros_package_name="robokudo_ros", relative_path="package.xml"
        )

    def test_file_loader_existing_file_in_non_existing_ros_package(self):
        assert pytest.raises(
            OSError,
            FileLoader.get_path_to_file_in_ros_package,
            ros_package_name="skdaopsi2098392183robokud0",
            relative_path="package.xml",
        )

    def test_file_loader_non_existing_file_in_existing_ros_package(self):
        assert pytest.raises(
            OSError,
            FileLoader.get_path_to_file_in_ros_package,
            ros_package_name="robokudo",
            relative_path="skdaopsi2098392183robokud0.xml",
        )
