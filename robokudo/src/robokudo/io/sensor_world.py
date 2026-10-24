"""
Restore recorded camera context for sensor replay.
"""

from __future__ import annotations

# %% Imports
from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import pairwise

from robokudo.exceptions import InvalidCameraObservation
from robokudo.types.camera import CameraObservation
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body

# %% Camera definitions


@dataclass(frozen=True)
class RecordedCameraWorld:
    """
    Restore the recorded camera and its pose for reprocessing sensor data.

    Camera identities are retained while unrelated entities from the recorded world,
    including previous perception results, are excluded.
    """

    observation: CameraObservation
    """
    Recorded camera state that identifies the sensor context.
    """

    def project(self) -> World:
        """
        Project an isolated optical camera and its sampled pose into a sensor world.

        :return: World containing only the camera and its direct pose frames.
        :raises InvalidCameraObservation: If sampled pose frames are inconsistent.
        """
        observation = self._copy_observation()
        sensor_world = self._create_world(observation)
        self.apply_pose(sensor_world, observation)
        return sensor_world

    def _copy_observation(self) -> CameraObservation:
        """
        Copy camera metadata and pose frames while preserving their recorded identities.

        :return: Observation detached from the recorded world's bodies and geometry.
        :raises InvalidCameraObservation: If the sampled pose cannot describe the
            camera.
        """
        source_camera = self.observation.camera
        source_root = source_camera.root
        source_pose = self.observation.world_T_camera
        source_child = self.observation.pose_child
        source_reference = (
            source_pose.reference_frame if source_pose is not None else None
        )
        if source_reference is source_root and source_child is not source_root:
            raise InvalidCameraObservation(
                reason="aliased camera pose has no separate reference frame"
            )
        root = Body(name=deepcopy(source_root.name), id=source_root.id)
        camera = Camera(
            id=source_camera.id,
            name=deepcopy(source_camera.name),
            root=root,
            camera_model=deepcopy(source_camera.camera_model),
            camera_range=deepcopy(source_camera.camera_range),
            modalities=tuple(source_camera.modalities),
        )
        if source_pose is None:
            return replace(self.observation, camera=camera)
        pose_child = (
            Body(name=deepcopy(source_child.name), id=source_child.id)
            if source_child is not source_root
            else root
        )
        reference = (
            root
            if source_reference is source_root
            else Body(name=deepcopy(source_reference.name), id=source_reference.id)
        )
        pose = HomogeneousTransformationMatrix(
            data=source_pose, reference_frame=reference, child_frame=pose_child
        )
        return replace(self.observation, camera=camera, world_T_camera=pose)

    @staticmethod
    def _create_world(observation: CameraObservation) -> World:
        """
        Attach the observation's camera and direct pose frames to a new world.

        :param observation: Isolated camera metadata and sampled frame references.
        :return: Sensor world with identity mounts ready to receive the sampled pose.
        """
        camera = observation.camera
        root = camera.root
        pose = observation.world_T_camera
        pose_child = observation.pose_child
        reference = pose.reference_frame if pose is not None else None
        sensor_world = World()
        with sensor_world.modify_world():
            if reference is not None and reference is not root:
                sensor_world.add_body(reference)
            if pose_child is not root:
                sensor_world.add_body(pose_child)
            sensor_world.add_body(root)
            for parent, child in pairwise((reference, pose_child, root)):
                if parent is None or parent is child:
                    continue
                sensor_world.add_connection(
                    Connection6DoF.create_with_dofs(
                        parent=parent,
                        child=child,
                        world=sensor_world,
                        name=PrefixedName(
                            name=f"{child.name.name}_T_{parent.name.name}"
                        ),
                    )
                )
            sensor_world.add_semantic_annotation(camera)
        return sensor_world

    def install(self, reasoning_world: World) -> Camera:
        """
        Merge the sensor context and bind the camera to its merged root body.
        """
        reasoning_world.merge_world(self.project())
        camera = reasoning_world.get_semantic_annotation_by_id(
            self.observation.camera.id
        )
        root = reasoning_world.get_kinematic_structure_entity_by_id(
            self.observation.camera.root.id
        )
        if not isinstance(camera, Camera) or not isinstance(root, Body):
            raise InvalidCameraObservation(
                reason="sensor world did not contain a camera and root body"
            )
        with reasoning_world.modify_world():
            camera.root = root
        return camera

    @staticmethod
    def apply_pose(world: World, observation: CameraObservation) -> None:
        """
        Apply the sampled optical pose directly to the camera root.
        """
        pose = observation.world_T_camera
        pose_child = observation.pose_child
        if pose is None:
            return
        if pose.reference_frame is pose_child:
            return
        connection = pose_child.parent_connection
        if connection is None or connection.parent is not pose.reference_frame:
            raise InvalidCameraObservation(
                reason="camera pose is not connected to its reference frame"
            )
        with world.modify_world():
            connection.origin = HomogeneousTransformationMatrix(
                data=observation.world_T_camera,
                reference_frame=pose.reference_frame,
                child_frame=pose_child,
            )
