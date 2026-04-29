from dataclasses import dataclass

from typing_extensions import Optional, Tuple, ClassVar

from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig


@dataclass(slots=True)
class FileReaderCameraConfig(BaseCameraConfig):
    """Configuration class for file-based camera data playback.

    This class defines the configuration parameters for reading camera data from files,
    typically used for offline processing or testing. It supports reading from both
    ROS package directories and regular filesystem paths.

    .. note::
        If target_ros_package is None, target_dir is used as an absolute or relative
        path. Otherwise, target_dir is appended to the ROS package path.
    """

    registry_name: ClassVar[str] = "file_reader"

    interface_type: str = "FileReader"

    loop: bool = True
    """Shall we loop after iterating over a directory?"""

    target_ros_package: Optional[str] = None
    """
    Files for the FileReaderInterface should be in a ROS package
    If you don't want that, leave target_ros_package to None and define target_dir
    either absolute or relative to your executable.
    """

    target_dir: str = "/tmp"
    """
    If target_ros_package is None, try to load target_dir directly.
    Otherwise, append target_dir to target_ros_package and load the files from there.
    """

    color2depth_ratio: Tuple[float, float] = (0.5, 0.5)
    """
    If you have depth data to read from, please set the ratio in x,y here
    If it's not set, (1,1) will be the default. Set to None if you don't have that.
    """

    filename_prefix: str = "rk_"
    """Define the prefix of all the files that shall be loaded into the FileReaderInterface"""

    kinect_height_fix_mode: bool = True
    """Apply kinect hack"""
