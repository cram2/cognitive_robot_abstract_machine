"""Dynamic module loading utilities for Robokudo.

This module provides functionality for dynamically loading Robokudo modules and components.
It supports loading:

* Analysis engines
* Annotators
* Camera configurations
* Action servers
* IO modules
* World descriptors
* Tree components
* Types and utilities

The module handles:

* ROS package path resolution
* Module type management
* Dynamic module importing
* File path resolution
"""

from __future__ import annotations

import enum
import importlib
import importlib.resources
import logging
import sys
import warnings
from importlib.resources import as_file
from pathlib import Path

from typing_extensions import TYPE_CHECKING, List, Any, Optional

if TYPE_CHECKING:
    from types import ModuleType
    from robokudo.analysis_engine import AnalysisEngineInterface
    from robokudo.world_descriptor import BaseWorldDescriptor
    from robokudo.descriptors.camera_configs.base_camera_config import (
        BaseCameraConfig,
    )


class RobokudoModuleType(enum.Enum):
    """Enumeration of Robokudo module types.

    Defines the standard module types and their paths within a ROS package.
    """

    Annotator = ["annotators"]
    """Annotator modules"""

    AnalysisEngine = ["descriptors", "analysis_engines"]
    """Analysis engine descriptors"""

    CameraConfig = ["descriptors", "camera_configs"]
    """Camera configuration descriptors"""

    IO = ["io"]
    """Input/output modules"""

    WorldDescriptor = ["descriptors", "worlds"]
    """World descriptor modules"""

    TreeComponents = ["tree_components"]
    """Behavior tree components"""

    Types = ["types"]
    """Type modules"""

    Utils = ["utils"]
    """Utility modules"""

    Data = ["data"]  # If you have Python modules in `robokudo.data`
    """Data modules"""


class ModuleLoader:
    """Dynamic module loader for Robokudo components.

    Handles loading of various Robokudo module types from ROS packages.
    Provides path resolution and module importing functionality.
    """

    def __init__(self) -> None:
        """Initialize module loader with logger."""

        self.logger = logging.getLogger(self.__class__.__name__)
        """The logger for the module loader instance."""

    def _load_module(
        self, ros_pkg_name: str, module_type: RobokudoModuleType, module_name: str
    ) -> ModuleType:
        """
        Dynamically import a submodule of the 'robokudo' package (or another package).
        E.g., 'robokudo.descriptors.analysis_engines.demo'

        :param ros_pkg_name: Name of ROS package
        :param module_type: Type of module to load
        :param module_name: Name of module to load
        :return: Loaded module object
        """
        # Remove any trailing '.py' from the user's input, if present
        module_name = module_name.replace(".py", "")

        # Combine [ros_pkg_name, *subfolders, module_name] into a single dotted path
        subfolders = module_type.value
        import_path = ".".join([ros_pkg_name, *subfolders, module_name])

        self.logger.debug(f"Loading module: {import_path}")

        if import_path in sys.modules:
            self.logger.debug(f"Reloading module: {import_path}")
            loaded_module = importlib.reload(sys.modules[import_path])
        else:
            loaded_module = importlib.import_module(import_path)
        return loaded_module

    def load_ae(self, ros_pkg_name: str, module_name: str) -> AnalysisEngineInterface:
        """Load an Analysis Engine (AE). Expects a class `AnalysisEngine` in the loaded module.

        The ROS package must be in the same workspace with path structure:
        ``$package_path/src/robokudo_example_package/descriptors/analysis_engines/``

        :param ros_pkg_name: Name of ROS package containing AE
        :param module_name: Name of analysis engine module
        :return: Root of loaded analysis engine
        """
        module_type = RobokudoModuleType.AnalysisEngine
        loaded_module = self._load_module(ros_pkg_name, module_type, module_name)

        # e.g. the module should define: class AnalysisEngine: ...
        loaded_ae = loaded_module.AnalysisEngine()
        return loaded_ae

    def load_annotator(self, ros_pkg_name: str, module_name: str) -> ModuleType:
        """Load an annotator module. You can adjust the returned object as needed.

        :param ros_pkg_name: Name of ROS package containing annotator
        :param module_name: Name of annotator module
        :return: The requested module. This is not pointing to an Annotator in this module. It's just the module.
        :rtype: module
        """
        module_type = RobokudoModuleType.Annotator
        loaded_module = self._load_module(ros_pkg_name, module_type, module_name)
        return loaded_module

    def load_camera_config(self, ros_pkg_name: str, module_name: str) -> Any:
        """Load a camera config module. Expects class `CameraConfig`.

        :param ros_pkg_name: Name of ROS package containing config
        :param module_name: Name of camera config module
        :return: Loaded camera configuration
        """
        from robokudo.descriptors.camera_configs.base_camera_config import (
            BaseCameraConfig,
        )

        module_type = RobokudoModuleType.CameraConfig
        loaded_module = self._load_module(ros_pkg_name, module_type, module_name)
        for name, obj in loaded_module.__dict__.items():
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseCameraConfig)
                and obj != BaseCameraConfig
            ):
                return obj()
        return None
        # return loaded_module.CameraConfig()

    def load_io(self, ros_pkg_name: str, module_name: str) -> ModuleType:
        """Load an I/O module. Customize this if there's a specific class to instantiate.

        :param ros_pkg_name: Name of ROS package containing IO module
        :param module_name: Name of IO module
        :return: Loaded IO module
        """
        module_type = RobokudoModuleType.IO
        loaded_module = self._load_module(ros_pkg_name, module_type, module_name)
        # return loaded_module.YourIoClass()
        return loaded_module

    def load_world_descriptor(
        self, ros_pkg_name: str, module_name: str
    ) -> BaseWorldDescriptor:
        """Load a WorldDescriptor given the module name and the ros package name.

        The path within the package is meant to be:
        $package_path/src/PACKAGE_NAME/descriptors/worlds/.

        :param ros_pkg_name: Name of ROS package containing world descriptor
        :param module_name: Name of world descriptor module
        :return: Loaded world descriptor
        """
        module_type = RobokudoModuleType.WorldDescriptor

        loaded_module = self._load_module(
            ros_pkg_name=ros_pkg_name, module_type=module_type, module_name=module_name
        )
        loaded_world_descriptor = loaded_module.WorldDescriptor()
        return loaded_world_descriptor

    def load_tree_components(self, ros_pkg_name: str, module_name: str) -> ModuleType:
        """Load tree components. If there's a main class, instantiate it here.

        :param ros_pkg_name: Name of ROS package containing components
        :param module_name: Name of tree components module
        :return: Loaded tree components module
        """
        module_type = RobokudoModuleType.TreeComponents
        loaded_module = self._load_module(ros_pkg_name, module_type, module_name)
        return loaded_module

    def load_types(self, ros_pkg_name: str, module_name: str) -> ModuleType:
        """Load a 'types' module, or a class if needed.

        :param ros_pkg_name: Name of ROS package containing types
        :param module_name: Name of types module
        :return: Loaded types module
        """
        module_type = RobokudoModuleType.Types
        loaded_module = self._load_module(ros_pkg_name, module_type, module_name)
        return loaded_module

    def load_utils(self, ros_pkg_name: str, module_name: str) -> ModuleType:
        """Load a 'utils' module, or a class if needed.

        :param ros_pkg_name: Name of ROS package containing utilities
        :param module_name: Name of utility module
        :return: Loaded utility module
        """
        module_type = RobokudoModuleType.Utils
        loaded_module = self._load_module(ros_pkg_name, module_type, module_name)
        return loaded_module

    def get_file_paths(
        self,
        ros_pkg_name: str,
        module_type: RobokudoModuleType,
        dir_name: str,
        file_extension: Optional[str] = None,
    ) -> List[str]:  # pragma: no cover
        """Get paths to files in module directory.

        If you previously used this to read data files from the 'source' folder, you can
        either remove it or refactor to read from:
          1) get_package_share_directory('robokudo'), or
          2) standard Python package resources.

        If you truly need to load data from the installed package, see:
          - `ament_index_python` (to find share dir)
          - or `importlib.resources`

        For now, you might remove this method or keep it if you adapt the logic.

        :param ros_pkg_name: Name of ROS package
        :param module_type: Type of module to search
        :param dir_name: Name of directory to search
        :param file_extension: Optional file extension filter
        :return: List of paths to matching files
        """
        warnings.warn(
            "`get_file_paths()` no longer scans source directories. "
            "Please store data files in share/robokudo and retrieve them via "
            "`ament_index_python.packages.get_package_share_directory` or "
            "`importlib.resources`."
        )
        return []

    @staticmethod
    def get_module_path(module_name: str) -> Path:
        files = importlib.resources.files(module_name)
        with as_file(files) as path:
            return Path(path)
