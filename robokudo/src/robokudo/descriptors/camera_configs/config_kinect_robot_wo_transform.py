from dataclasses import dataclass

from typing_extensions import ClassVar

from robokudo.descriptors.camera_configs.config_kinect_robot import KinectCameraConfig


@dataclass(slots=True)
class KinectWoTfCameraConfig(KinectCameraConfig):
    """Configuration class for a Kinect camera setup without transform lookup.

    This class defines the configuration parameters for a Kinect camera in a robotic
    setup, similar to config_kinect_robot.py but with viewpoint lookup disabled.
    It includes interface settings, topic names, and transformation settings.

    .. note::
        This configuration is identical to config_kinect_robot.py except that
        lookup_viewpoint is set to False by default.
    """

    registry_name: ClassVar[str] = "kinect_wo_tf"
    """Overwrite KinectCameraConfig"""

    lookup_viewpoint: bool = False
    """Overwrite KinectCameraConfig"""
