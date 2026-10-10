"""Native robot camera properties exposed in scene bundles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.robots.robot_parts import Camera

if TYPE_CHECKING:
    from semantic_digital_twin.robots.robot_parts import AbstractRobot


# %% camera metadata
class CameraField(StrEnum):
    """Keys identifying cameras and their projection in scene metadata."""

    CAMERAS = "cameras"
    """Camera descriptors belonging to the scene's robot."""

    NAME = "name"
    """Native camera annotation name shown in the source selector."""

    LINK = "link"
    """Exact name of the camera's root body in the bundled robot."""

    FORWARD = "forward"
    """Viewing direction expressed in the camera root body's frame."""

    HORIZONTAL_ANGLE = "horizontalAngle"
    """Horizontal field of view in radians."""

    VERTICAL_ANGLE = "verticalAngle"
    """Vertical field of view in radians."""

    DEFAULT = "default"
    """Whether the native robot annotation selects this camera by default."""


@dataclass
class RobotCamera:
    """A native camera's frame and projection at the scene serialization boundary."""

    camera: Camera
    """The semantic annotation owning the camera's direction and projection."""

    @classmethod
    def of_robot(cls, robot: AbstractRobot) -> list[RobotCamera]:
        """Collect the robot's annotated cameras.

        :param robot: The robot whose camera sensors are published.
        :return: Camera descriptors in the robot's native sensor order.
        """
        return [
            cls(sensor) for sensor in robot.all_sensors if isinstance(sensor, Camera)
        ]

    def to_payload(self) -> dict[CameraField, str | float | bool | list[float]]:
        """Serialize the camera's identity, frame and native projection.

        :return: JSON-compatible camera metadata with angular values in radians.
        """
        return {
            CameraField.NAME: str(self.camera.name),
            CameraField.LINK: str(self.camera.root.name),
            CameraField.FORWARD: self.camera.forward_facing_axis.to_np()[:3].tolist(),
            CameraField.HORIZONTAL_ANGLE: self.camera.field_of_view.horizontal_angle,
            CameraField.VERTICAL_ANGLE: self.camera.field_of_view.vertical_angle,
            CameraField.DEFAULT: self.camera.default_camera,
        }
