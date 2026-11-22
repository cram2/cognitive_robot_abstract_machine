"""
Verify dynamic loading of RoboKudo modules and invalid module errors.
"""

from __future__ import annotations

import pytest
import rclpy

from robokudo.utils.module_loader import ModuleLoader


class TestUtilsModuleLoader(object):
    """
    Exercise module discovery, reloading, and invalid import paths.
    """

    @pytest.fixture(autouse=True)
    def module_loader(self):
        return ModuleLoader()

    invalid_modules = [
        ("robokudo", "non_existent_module.py"),
        ("non_existent_package", "non_existent_module.py"),
    ]

    def test_load_ae(self, module_loader: ModuleLoader):
        assert module_loader.load_ae(ros_package_name="robokudo", module_name="demo.py")

    @pytest.mark.parametrize(["pkg_name", "module_name"], invalid_modules)
    def test_load_ae_invalid_input(
        self, module_loader: ModuleLoader, pkg_name: str, module_name: str
    ):
        assert pytest.raises(
            ImportError,
            module_loader.load_ae,
            ros_package_name=pkg_name,
            module_name=module_name,
        )

    def test_load_annotator(self, module_loader: ModuleLoader):
        assert module_loader.load_annotator(
            ros_package_name="robokudo", module_name="query.py"
        )

    @pytest.mark.parametrize(["pkg_name", "module_name"], invalid_modules)
    def test_load_annotator_invalid_input(
        self, module_loader: ModuleLoader, pkg_name: str, module_name: str
    ):
        assert pytest.raises(
            ImportError,
            module_loader.load_annotator,
            ros_package_name=pkg_name,
            module_name=module_name,
        )

    def test_load_camera_config(self, module_loader: ModuleLoader):
        assert module_loader.load_camera_config(
            ros_package_name="robokudo", module_name="config_tiago.py"
        )

    @pytest.mark.parametrize(["pkg_name", "module_name"], invalid_modules)
    def test_load_camera_config_invalid_input(
        self, module_loader: ModuleLoader, pkg_name: str, module_name: str
    ):
        assert pytest.raises(
            ImportError,
            module_loader.load_camera_config,
            ros_package_name=pkg_name,
            module_name=module_name,
        )

    def test_load_io(self, module_loader: ModuleLoader):
        assert module_loader.load_io(
            ros_package_name="robokudo", module_name="camera_interface.py"
        )

    @pytest.mark.parametrize(["pkg_name", "module_name"], invalid_modules)
    def test_load_io_invalid_input(
        self, module_loader: ModuleLoader, pkg_name: str, module_name: str
    ):
        assert pytest.raises(
            ImportError,
            module_loader.load_io,
            ros_package_name=pkg_name,
            module_name=module_name,
        )

    def test_load_world_descriptor(self, module_loader: ModuleLoader):
        assert module_loader.load_world_descriptor("robokudo", "world_iai_kitchen20.py")

    @pytest.mark.parametrize(["pkg_name", "module_name"], invalid_modules)
    def test_load_world_descriptor_invalid_input(
        self, module_loader: ModuleLoader, pkg_name: str, module_name: str
    ):
        assert pytest.raises(
            ImportError, module_loader.load_world_descriptor, pkg_name, module_name
        )

    def test_load_tree_components(self, module_loader: ModuleLoader):
        assert module_loader.load_tree_components(
            ros_package_name="robokudo", module_name="query_based_task_scheduler.py"
        )

    @pytest.mark.parametrize(["pkg_name", "module_name"], invalid_modules)
    def test_load_tree_components_invalid_input(
        self, module_loader: ModuleLoader, pkg_name: str, module_name: str
    ):
        assert pytest.raises(
            ImportError,
            module_loader.load_tree_components,
            ros_package_name=pkg_name,
            module_name=module_name,
        )

    def test_load_types(self, module_loader: ModuleLoader):
        assert module_loader.load_types(
            ros_package_name="robokudo", module_name="annotation.py"
        )

    @pytest.mark.parametrize(["pkg_name", "module_name"], invalid_modules)
    def test_load_types_invalid_input(
        self, module_loader: ModuleLoader, pkg_name: str, module_name: str
    ):
        assert pytest.raises(
            ImportError,
            module_loader.load_types,
            ros_package_name=pkg_name,
            module_name=module_name,
        )

    def test_load_utils(self, module_loader: ModuleLoader):
        assert module_loader.load_utils(
            ros_package_name="robokudo", module_name="module_loader.py"
        )

    @pytest.mark.parametrize(["pkg_name", "module_name"], invalid_modules)
    def test_load_utils_invalid_input(
        self, module_loader: ModuleLoader, pkg_name: str, module_name: str
    ):
        assert pytest.raises(
            ImportError,
            module_loader.load_utils,
            ros_package_name=pkg_name,
            module_name=module_name,
        )
