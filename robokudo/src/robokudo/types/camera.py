"""
ROS-independent camera metadata for perception frames.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, replace
from typing_extensions import TYPE_CHECKING

from robokudo.exceptions import CameraPoseMissing, InvalidCameraObservation
from robokudo.utils.ros_frames import normalize_ros_frame_id
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from semantic_digital_twin.datastructures.camera_resolution import CameraResolution

if TYPE_CHECKING:
    from semantic_digital_twin.datastructures.field_of_view import FieldOfView

# %% Camera definitions


@dataclass(frozen=True)
class CameraObservation:
    """
    Describe the camera state used to produce one perception frame.

    :attr:`world_T_camera` alone transforms optical-frame points into world coordinates.
    The camera root is the optical frame; its sampled pose is also its replay pose. A
    supplied pose must have a reference frame and identify this camera's optical root.
    Frame metadata and the camera root must remain unchanged after construction; use
    :meth:`with_world_T_camera` to replace the sampled pose.
    """

    camera: Camera
    """
    Semantic camera that produced the frame.
    """

    effective_camera_model: PinholeCameraModel
    """
    Effective projection model of the delivered image.
    """

    world_T_camera: HomogeneousTransformationMatrix | None
    """
    Sampled optical-frame pose in world coordinates, if available.

    This transform alone places optical-frame points in the world.
    """

    timestamp_nanoseconds: int
    """
    Acquisition time as nanoseconds since the Unix epoch.
    """

    def __post_init__(self) -> None:
        """
        Validate the sampled pose's reference frame and optical camera identity.

        :raises InvalidCameraObservation: If a supplied pose lacks a reference frame or
            identifies another camera.
        """
        pose = self.world_T_camera
        if pose is None:
            return
        if pose.reference_frame is None:
            raise InvalidCameraObservation(
                reason="sampled camera pose has no reference frame"
            )
        pose_child = self.pose_child
        camera_root = self.camera.root
        if pose_child is camera_root:
            return
        if (
            not isinstance(pose_child, Body)
            or pose_child.name.prefix != camera_root.name.prefix
            or pose_child.name.name == camera_root.name.name
            or normalize_ros_frame_id(pose_child.name.name)
            != normalize_ros_frame_id(camera_root.name.name)
        ):
            raise InvalidCameraObservation(
                reason="sampled pose child frame differs from the camera root"
            )

    @property
    def pose_child(self) -> Body:
        """
        Return the sampled child, or the camera root when no child is specified.

        ROS aliases differing only by leading slashes are accepted within the same name
        prefix. Validation occurs when the observation is constructed.

        :return: Optical camera root or its explicitly sampled ROS alias.
        """
        pose = self.world_T_camera
        if pose is None or pose.child_frame is None:
            return self.camera.root
        return pose.child_frame

    @property
    def resolution(self) -> CameraResolution:
        """
        Return the effective image resolution.
        """
        return self.effective_camera_model.resolution

    @property
    def field_of_view(self) -> FieldOfView:
        """
        Return the effective image field of view.
        """
        return self.effective_camera_model.field_of_view

    def world_T_camera_or_raise(self) -> HomogeneousTransformationMatrix:
        """
        Return the sampled pose or raise when the producer had none.

        :return: Transform from the optical frame into the world frame.
        :raises CameraPoseMissing: If the observation has no sampled pose.
        """
        if self.world_T_camera is None:
            raise CameraPoseMissing()
        return self.world_T_camera

    @property
    def camera_T_world(self) -> HomogeneousTransformationMatrix:
        """
        Return the inverse sampled camera pose.

        :raises CameraPoseMissing: If the observation has no sampled pose.
        """
        return self.world_T_camera_or_raise().inverse()

    def with_world_T_camera(
        self, world_T_camera: HomogeneousTransformationMatrix
    ) -> CameraObservation:
        """
        Return a copy carrying the supplied sampled camera pose.

        :param world_T_camera: Transform from the optical frame into the world frame.
        :return: Observation containing the supplied pose.
        :raises InvalidCameraObservation: If the pose lacks a reference frame or
            identifies another camera.
        """
        return replace(self, world_T_camera=world_T_camera)
