import os

import pytest

from semantic_digital_twin.adapters.package_resolver import (
    PackageUriResolver,
    PrefixPathPackageLocator,
    ROSPackageLocator,
)
from semantic_digital_twin.exceptions import PackageResolutionError


def _make_package(base_directory, *segments):
    """
    :return: The path of a package share directory created under ``base_directory``.
    """
    package_directory = os.path.join(base_directory, *segments)
    os.makedirs(package_directory, exist_ok=True)
    return package_directory


# %% resolving install prefix layouts


class TestPrefixPathPackageLocator:
    """
    Resolution of a package name by searching install prefixes directly on disk.
    """

    def test_resolves_package_directory_from_ament_prefix_path(
        self, tmp_path, monkeypatch
    ):
        expected = _make_package(tmp_path, "some_package", "share", "some_package")
        monkeypatch.setenv("AMENT_PREFIX_PATH", str(tmp_path))
        monkeypatch.delenv("CMAKE_PREFIX_PATH", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "empty_home"))

        locator = PrefixPathPackageLocator()

        assert locator.resolve("some_package") == expected

    def test_resolves_package_directory_from_cmake_prefix_path(
        self, tmp_path, monkeypatch
    ):
        expected = _make_package(tmp_path, "some_package", "share", "some_package")
        monkeypatch.delenv("AMENT_PREFIX_PATH", raising=False)
        monkeypatch.setenv("CMAKE_PREFIX_PATH", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path / "empty_home"))

        locator = PrefixPathPackageLocator()

        assert locator.resolve("some_package") == expected

    def test_resolves_merged_install_share_layout(self, tmp_path, monkeypatch):
        expected = _make_package(tmp_path, "share", "some_package")
        monkeypatch.setenv("AMENT_PREFIX_PATH", str(tmp_path))
        monkeypatch.delenv("CMAKE_PREFIX_PATH", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "empty_home"))

        locator = PrefixPathPackageLocator()

        assert locator.resolve("some_package") == expected

    def test_resolves_bare_package_directory(self, tmp_path, monkeypatch):
        expected = _make_package(tmp_path, "some_package")
        monkeypatch.setenv("AMENT_PREFIX_PATH", str(tmp_path))
        monkeypatch.delenv("CMAKE_PREFIX_PATH", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "empty_home"))

        locator = PrefixPathPackageLocator()

        assert locator.resolve("some_package") == expected

    def test_resolves_against_workspace_glob_under_home_directory(
        self, tmp_path, monkeypatch
    ):
        expected = _make_package(
            tmp_path, "robot_ws", "install", "some_package", "share", "some_package"
        )
        monkeypatch.delenv("AMENT_PREFIX_PATH", raising=False)
        monkeypatch.delenv("CMAKE_PREFIX_PATH", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))

        locator = PrefixPathPackageLocator()

        assert locator.resolve("some_package") == expected

    def test_additional_prefixes_take_precedence_over_environment(
        self, tmp_path, monkeypatch
    ):
        environment_prefix = tmp_path / "from_environment"
        additional_prefix = tmp_path / "from_additional_prefixes"
        environment_match = _make_package(
            str(environment_prefix), "some_package", "share", "some_package"
        )
        additional_match = _make_package(
            str(additional_prefix), "some_package", "share", "some_package"
        )
        monkeypatch.setenv("AMENT_PREFIX_PATH", str(environment_prefix))
        monkeypatch.delenv("CMAKE_PREFIX_PATH", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "empty_home"))

        locator = PrefixPathPackageLocator(additional_prefixes=[str(additional_prefix)])

        assert locator.resolve("some_package") == additional_match
        assert additional_match != environment_match

    def test_raises_package_resolution_error_when_not_found_anywhere(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("AMENT_PREFIX_PATH", raising=False)
        monkeypatch.delenv("CMAKE_PREFIX_PATH", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "empty_home"))

        locator = PrefixPathPackageLocator()

        with pytest.raises(PackageResolutionError) as error:
            locator.resolve("missing_package")
        assert error.value.package_name == "missing_package"


# %% integration with the default resolver chain


class TestROSPackageLocatorDefaultChain:
    """
    The default locator chain closes the gap for every consumer that builds a resolver
    with no explicit configuration.
    """

    def test_default_chain_resolves_via_prefix_path_locator(
        self, tmp_path, monkeypatch
    ):
        expected = _make_package(tmp_path, "some_package", "share", "some_package")
        monkeypatch.setenv("AMENT_PREFIX_PATH", str(tmp_path))
        monkeypatch.delenv("CMAKE_PREFIX_PATH", raising=False)
        monkeypatch.delenv("ROS_PACKAGE_PATH", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "empty_home"))

        resolved = ROSPackageLocator().resolve("some_package")

        assert resolved == expected

    def test_package_uri_resolver_resolves_a_file_via_prefix_path(
        self, tmp_path, monkeypatch
    ):
        package_directory = _make_package(
            tmp_path, "some_package", "share", "some_package"
        )
        expected_file = os.path.join(package_directory, "model.urdf")
        with open(expected_file, "w") as file:
            file.write("<robot/>")
        monkeypatch.setenv("AMENT_PREFIX_PATH", str(tmp_path))
        monkeypatch.delenv("CMAKE_PREFIX_PATH", raising=False)
        monkeypatch.delenv("ROS_PACKAGE_PATH", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "empty_home"))

        resolved = PackageUriResolver().resolve("package://some_package/model.urdf")

        assert resolved == expected_file
