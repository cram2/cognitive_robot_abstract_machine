from __future__ import annotations

# %% Imports
from krrood.ormatic.utils import classproperty
from dataclasses import dataclass, field


from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
from robokudo.descriptors.camera_configs.components import (
    StaticCameraTransformComponent,
)
from robokudo.types.cv import ColorToDepthRatio


@dataclass(slots=True)
class FileReaderCameraConfig(BaseCameraConfig, StaticCameraTransformComponent):
    """
    Configuration class for file-based camera data playback.

    This class defines the configuration parameters for reading camera data from files,
    typically used for offline processing or testing. It supports reading from both
    ROS package directories and regular filesystem paths.

    .. note::
        If target_ros_package is None, target_dir is used as an absolute or relative
        path. Otherwise, target_dir is appended to the ROS package path.
    """

    @classproperty
    def registry_name(cls) -> str:
        """
        Identifier selecting this camera configuration.
        """
        return "file_reader"

    interface_type: str = "FileReader"

    lookup_viewpoint: bool = False
    """
    File reader does not support live TF lookup; use static_camera_transform_enabled
    instead.
    """
    tf_from: str = "camera"
    """
    Camera frame name accepted for compatibility with transform-aware configs.
    """
    tf_to: str = "map"
    """
    World frame name accepted for compatibility with transform-aware configs.
    """
    loop: bool = True
    """
    Shall we loop after iterating over a directory?
    """
    target_ros_package: str | None = None
    """
    Files for the FileReaderInterface should be in a ROS package If you don't want that,
    leave target_ros_package to None and define target_dir either absolute or relative
    to your executable.
    """
    target_dir: str = "/tmp"
    """
    If target_ros_package is None, try to load target_dir directly.

    Otherwise, append target_dir to target_ros_package and load the files from there.
    """
    color_to_depth_ratio: ColorToDepthRatio | None = field(
        default_factory=lambda: ColorToDepthRatio(horizontal=0.5, vertical=0.5)
    )
    """
    Size of the recorded depth image relative to the recorded color image, or None
    without depth data.
    """
    filename_prefix: str = "rk_"
    """
    Define the prefix of all the files that shall be loaded into the
    FileReaderInterface.
    """
