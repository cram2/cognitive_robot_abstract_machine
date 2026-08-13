"""File path resolution utilities for RoboKudo.

This module provides utilities for resolving file paths in ROS packages.

:module: file_loader
:synopsis: File path resolution utilities for ROS packages
:moduleauthor: RoboKudo Team

:Dependencies:
    * pathlib
    * rospkg
    * logging
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory


class FileLoader:
    """File path resolution helper for ROS packages.

    This class provides methods for resolving file paths in ROS packages.

    :Example:

    .. code-block:: python

        pkg_path = FileLoader.get_ros_pkg_path('my_package')
        file_path = FileLoader.get_path_to_file_in_ros_package('my_package', 'config/params.yaml')
    """

    @staticmethod
    def get_ros_pkg_path(ros_pkg_name: str) -> Path:
        """
        Get a Path object to a ROS2 package given the name

        Throws OSError if ROS2 package can't be found.

        :param ros_pkg_name: name of a ros2 package
        :return: path object with the ros2 package path
        :raises OSError: If ROS package cannot be found

        :Example:

        .. code-block:: python

            pkg_path = loader.get_ros_pkg_path('my_package')
            assert pkg_path.exists()
        """
        try:
            package_path = Path(get_package_share_directory(ros_pkg_name))
        except:
            raise OSError(f"ROS2 package with name '{ros_pkg_name}' does not exist")
        else:
            return package_path

    @staticmethod
    def get_path_to_file_in_ros_package(ros_pkg_name: str, relative_path: str) -> Path:
        """
        Get a Path object to a given the filename inside a ROS package. Please note, that relative_path should NOT
        start with '/', because then it would be considered as an absolute path from pathlib.Path.joinpath()

        Throws OSError if ROS package can't be found or relative_path in ROS package doesn't exist.

        :param ros_pkg_name: name of a ros package
        :param relative_path: the filename of the desired file, relative to the path of ros_pkg_name
        :return: path object to the desired file relative the ros package
        :raises OSError: If package or file cannot be found

        :Example:

        .. code-block:: python

            file_path = loader.get_path_to_file_in_ros_package('my_package', 'config/params.yaml')
            assert file_path.exists()

        .. note::
            The relative_path should NOT start with '/' to avoid being treated
            as an absolute path by pathlib.Path.joinpath()
        """
        ros_pkg_path = FileLoader.get_ros_pkg_path(ros_pkg_name=ros_pkg_name)

        full_path = ros_pkg_path.joinpath(relative_path)
        if not full_path.exists():
            raise OSError(f"The desired path {full_path} doesn't exist.")

        return full_path
