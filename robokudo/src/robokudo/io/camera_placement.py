"""
Placement of the camera that recorded a pipeline pass, in the world perception perceives
into.

A camera interface measures where its camera was; a :class:`CameraPlacement` puts the
camera there in the world and binds the pose of the pass to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import trimesh.creation
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.cas import CAS
from robokudo.exceptions import UnknownWorldFrameError

# %% measured camera pose


@dataclass
class MeasuredCameraPose:
    """
    Where a camera interface measured its camera to be when it recorded its latest data.
    """

    world_frame: str
    """
    The name of the frame the pose was measured against.
    """

    camera_frame: str
    """
    The name of the frame of the camera.
    """

    world_T_camera: HomogeneousTransformationMatrix
    """
    The pose of the camera relative to :attr:`world_frame`, without frames bound to a
    world.
    """


# %% camera placement


@dataclass
class CameraPlacement:
    """
    Puts the camera of a pipeline pass into :attr:`world`.

    A camera :attr:`world` already holds under the measured camera frame, such as the
    camera of a robot whose joints another process keeps current, is located by the
    world. Any other camera is located by its measured pose relative to its world frame.
    In a world of its own, perception hangs such a camera below its world frame on a
    connection perception owns and moves it to where it was measured. A world other
    processes share gets no camera body.
    """

    world: World
    """
    The world the camera is placed in.
    """

    def place(self, cas: CAS, measured_pose: MeasuredCameraPose) -> None:
        """
        Put the camera of ``cas`` into :attr:`world` and store its pose, bound to the
        bodies of :attr:`world`, and the names of its frames in ``cas``.

        :param cas: The CAS of the pass the camera recorded.
        :param measured_pose: Where the camera was measured to be.
        :raises UnknownWorldFrameError: If :attr:`world` holds bodies but none of the
            measured world frame, and does not locate the camera itself.
        """
        world_T_camera = self._bound_camera_pose(measured_pose)
        cas.world_frame = world_T_camera.reference_frame.name.name
        cas.camera_frame = measured_pose.camera_frame
        cas.camera_to_world_transform = world_T_camera

    def _bound_camera_pose(
        self, measured_pose: MeasuredCameraPose
    ) -> HomogeneousTransformationMatrix:
        """
        :param measured_pose: Where the camera was measured to be.
        :return: The pose of the camera, bound to the bodies of :attr:`world`: the one
            :attr:`world` holds for a camera it locates itself, otherwise the measured
            one relative to the world frame body, with the camera moved there unless
            :attr:`world` is shared.
        """
        camera_bodies = self.world.get_bodies_by_name(name=measured_pose.camera_frame)
        if camera_bodies and not self._is_attached_by_perception(camera_bodies[0]):
            camera_body = camera_bodies[0]
            return HomogeneousTransformationMatrix(
                data=camera_body.global_transform.to_np(),
                reference_frame=self.world.root,
                child_frame=camera_body,
            )

        world_body = self._world_frame_body(measured_pose.world_frame)
        if WorldSynchronizer.all_callbacks_of_this_type_from_world(self.world):
            return HomogeneousTransformationMatrix(
                data=measured_pose.world_T_camera, reference_frame=world_body
            )
        camera_body = (
            camera_bodies[0]
            if camera_bodies
            else self._attached_camera_body(measured_pose.camera_frame, world_body)
        )
        world_T_camera = HomogeneousTransformationMatrix(
            data=measured_pose.world_T_camera,
            reference_frame=world_body,
            child_frame=camera_body,
        )
        camera_body.parent_connection.origin = world_T_camera
        return world_T_camera

    def _world_frame_body(self, frame: str) -> Body:
        """
        :param frame: The name of the world frame.
        :return: The body of the world frame ``frame``, added as the root of
            :attr:`world` if :attr:`world` is empty.
        :raises UnknownWorldFrameError: If :attr:`world` holds bodies but none named
            ``frame``.
        """
        existing_bodies = self.world.get_bodies_by_name(name=frame)
        if existing_bodies:
            return existing_bodies[0]
        if not self.world.is_empty():
            raise UnknownWorldFrameError(world_frame=frame)

        body = self._frame_body(frame)
        with self.world.modify_world():
            self.world.add_body(body)
        return body

    def _attached_camera_body(self, frame: str, world_body: Body) -> Body:
        """
        :param frame: The name of the camera frame.
        :param world_body: The body of the frame the camera pose was measured against.
        :return: A new body for the camera frame ``frame``, attached to ``world_body``
            through a connection perception owns.
        """
        body = self._frame_body(frame)
        with self.world.modify_world():
            self.world.add_connection(
                Connection6DoF.create_with_dofs(
                    parent=world_body,
                    child=body,
                    world=self.world,
                    name=self._connection_name(world_body, body),
                )
            )
        return body

    @staticmethod
    def _frame_body(frame: str) -> Body:
        """
        :param frame: The name of the frame.
        :return: A new body for ``frame``, shown as coordinate axes.
        """
        return Body(
            name=PrefixedName(name=frame),
            visual=ShapeCollection([Mesh.from_trimesh(mesh=trimesh.creation.axis())]),
        )

    def _is_attached_by_perception(self, body: Body) -> bool:
        """
        :param body: The body.
        :return: Whether ``body`` hangs on its parent through a connection perception
            owns.
        """
        connection = body.parent_connection
        return connection is not None and connection.name == self._connection_name(
            connection.parent, body
        )

    @staticmethod
    def _connection_name(parent: Body, body: Body) -> PrefixedName:
        """
        :param parent: The parent body.
        :param body: The attached body.
        :return: The name of the connection perception attaches ``body`` to ``parent``
            with.
        """
        return PrefixedName(name=f"{body.name.name}_T_{parent.name.name}")
