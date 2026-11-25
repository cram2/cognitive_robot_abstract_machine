import pytest

from robokudo.utils.file_loader import FileLoader


class TestUtilsFileLoader:
    def test_file_loader_find_robokudo_ros_package(self):
        assert FileLoader.get_ros_pkg_path(ros_pkg_name="robokudo_ros")

    def test_file_loader_non_existing_ros_package(self):
        assert pytest.raises(
            OSError,
            FileLoader.get_ros_pkg_path,
            ros_pkg_name="skdaopsi2098392183robokud0",
        )

    def test_file_loader_find_file_in_robokudo_ros_package(self):
        assert FileLoader.get_path_to_file_in_ros_package(
            ros_pkg_name="robokudo_ros", relative_path="package.xml"
        )

    def test_file_loader_existing_file_in_non_existing_ros_package(self):
        assert pytest.raises(
            OSError,
            FileLoader.get_path_to_file_in_ros_package,
            ros_pkg_name="skdaopsi2098392183robokud0",
            relative_path="package.xml",
        )

    def test_file_loader_non_existing_file_in_existing_ros_package(self):
        assert pytest.raises(
            OSError,
            FileLoader.get_path_to_file_in_ros_package,
            ros_pkg_name="robokudo",
            relative_path="skdaopsi2098392183robokud0.xml",
        )
