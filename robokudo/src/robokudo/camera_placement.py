"""
Placement of the camera that recorded a pipeline pass, in the world perception perceives
into.

A camera interface measures where its camera was; a :class:`CameraPlacement` puts the
camera there in the world and binds the pose of the pass to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import trimesh.creation
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
    Puts the camera of a pipeline pass into :attr:`world`, a world perception keeps to
    itself, adding the frames it does not know yet.
    """

    world: World
    """
    The world the camera is placed in.
    """

    def place(self, cas: CAS, measured_pose: MeasuredCameraPose) -> None:
        """
        Put the camera of ``cas`` where it was measured to be and store its pose, bound
        to the bodies of :attr:`world`, and the names of its frames in ``cas``.

        A camera that hangs in a kinematic structure of its own, such as the head of a
        robot, is left where it is.

        :param cas: The CAS of the pass the camera recorded.
        :param measured_pose: Where the camera was measured to be.
        """
        world_T_camera = self._bound_camera_pose(measured_pose)
        cas.world_frame = world_T_camera.reference_frame.name.name
        cas.camera_frame = world_T_camera.child_frame.name.name
        cas.camera_to_world_transform = world_T_camera

    def _bound_camera_pose(
        self, measured_pose: MeasuredCameraPose
    ) -> HomogeneousTransformationMatrix:
        """
        :param measured_pose: Where the camera was measured to be.
        :return: The measured pose, bound to the bodies of its frames, with a camera
            perception attached moved there.
        """
        world_body = self._world_frame_body(measured_pose.world_frame)
        camera_body = self._camera_body(measured_pose.camera_frame, world_body)
        world_T_camera = HomogeneousTransformationMatrix(
            data=measured_pose.world_T_camera,
            reference_frame=world_body,
            child_frame=camera_body,
        )
        if self._is_attached_by_perception(camera_body):
            camera_body.parent_connection.origin = world_T_camera
        return world_T_camera

    def _world_frame_body(self, frame: str) -> Body:
        """
        The body of the world frame ``frame``, added to :attr:`world` if it does not
        exist yet: as the root of an empty world, or attached to the root otherwise.

        :param frame: The name of the world frame.
        :return: The body of the world frame.
        """
        existing_bodies = self.world.get_bodies_by_name(name=frame)
        if existing_bodies:
            return existing_bodies[0]

        body = self._frame_body(frame)
        with self.world.modify_world():
            if self.world.is_empty():
                self.world.add_body(body)
                return body
            self._attach(body, parent=self.world.root)
        return body

    def _camera_body(self, frame: str, world_body: Body) -> Body:
        """
        The body of the camera frame ``frame``, attached to ``world_body`` if it does
        not exist yet.

        :param frame: The name of the camera frame.
        :param world_body: The body of the frame the camera pose was measured against.
        :return: The body of the camera frame.
        """
        existing_bodies = self.world.get_bodies_by_name(name=frame)
        if existing_bodies:
            return existing_bodies[0]

        body = self._frame_body(frame)
        with self.world.modify_world():
            self._attach(body, parent=world_body)
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

    def _attach(self, body: Body, parent: Body) -> None:
        """
        Attach ``body`` to ``parent`` through a connection perception owns.

        :param body: The body to attach.
        :param parent: The body it is attached to.
        """
        self.world.add_connection(
            Connection6DoF.create_with_dofs(
                parent=parent,
                child=body,
                world=self.world,
                name=self._connection_name(parent, body),
            )
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


@dataclass
class SharedWorldCameraPlacement(CameraPlacement):
    """
    Puts the camera of a pipeline pass into :attr:`world`, a world other processes
    share, without adding world frames to it.

    A camera hanging in a kinematic structure of the world, such as the head of a robot
    whose joints another process keeps current, is located by that structure: its pose
    is read from the world instead of the measurement.
    """

    def _bound_camera_pose(
        self, measured_pose: MeasuredCameraPose
    ) -> HomogeneousTransformationMatrix:
        camera_bodies = self.world.get_bodies_by_name(name=measured_pose.camera_frame)
        if not camera_bodies or self._is_attached_by_perception(camera_bodies[0]):
            return super()._bound_camera_pose(measured_pose)
        camera_body = camera_bodies[0]
        return HomogeneousTransformationMatrix(
            data=camera_body.global_transform.to_np(),
            reference_frame=self.world.root,
            child_frame=camera_body,
        )

    def _world_frame_body(self, frame: str) -> Body:
        """
        :meth:`CameraPlacement._world_frame_body`, except that a frame the world does
        not hold is never added.

        :raises UnknownWorldFrameError: If :attr:`world` holds no body ``frame``.
        """
        existing_bodies = self.world.get_bodies_by_name(name=frame)
        if not existing_bodies:
            raise UnknownWorldFrameError(world_frame=frame)
        return existing_bodies[0]
