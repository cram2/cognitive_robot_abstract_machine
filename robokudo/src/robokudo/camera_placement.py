"""
Placement of the camera that recorded a pipeline pass, in the world perception perceives
into.
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
from typing_extensions import Optional

# %% camera placement


@dataclass
class CameraPlacement:
    """
    Puts the camera of a pipeline pass into :attr:`world`, adding the frames it does not
    know yet.
    """

    world: World
    """
    The world the camera is placed in.
    """

    def place(
        self,
        world_frame: str,
        camera_frame: str,
        world_T_camera: HomogeneousTransformationMatrix,
    ) -> HomogeneousTransformationMatrix:
        """
        Put the camera at ``world_T_camera`` in :attr:`world`.

        Frames the world does not know yet are added. A camera that already hangs in a
        kinematic structure, such as the head of a robot in a shared world, is located
        by that structure and therefore left where it is.

        :param world_frame: Name of the frame the camera pose is given in.
        :param camera_frame: Name of the camera frame.
        :param world_T_camera: Pose of the camera in the world frame.
        :return: The camera pose with its frames bound to the bodies of :attr:`world`.
        """
        world_body = self._frame_body(world_frame, parent=None)
        camera_body = self._frame_body(camera_frame, parent=world_body)
        bound_world_T_camera = HomogeneousTransformationMatrix(
            data=world_T_camera,
            reference_frame=world_body,
            child_frame=camera_body,
        )
        camera_connection = camera_body.parent_connection
        if camera_connection is not None and camera_connection.name == (
            self._frame_connection_name(world_frame, camera_frame)
        ):
            camera_connection.origin = bound_world_T_camera
        return bound_world_T_camera

    def _frame_body(self, frame: str, parent: Optional[Body]) -> Body:
        """
        The body of ``frame``, added to :attr:`world` if it does not exist yet.

        A new body is attached to ``parent`` through a connection perception owns.
        Without a parent it becomes the root of an empty world or is attached to the
        root of a populated one.

        :param frame: Name of the frame.
        :param parent: Body a new frame body is attached to.
        :return: The body of the frame.
        """
        existing_bodies = self.world.get_bodies_by_name(name=frame)
        if existing_bodies:
            return existing_bodies[0]

        body = Body(
            name=PrefixedName(name=frame),
            visual=ShapeCollection([Mesh.from_trimesh(mesh=trimesh.creation.axis())]),
        )
        with self.world.modify_world():
            if parent is None and self.world.is_empty():
                self.world.add_body(body)
                return body
            parent = parent if parent is not None else self.world.root
            self.world.add_connection(
                Connection6DoF.create_with_dofs(
                    parent=parent,
                    child=body,
                    world=self.world,
                    name=self._frame_connection_name(str(parent.name.name), frame),
                )
            )
        return body

    @staticmethod
    def _frame_connection_name(parent_frame: str, frame: str) -> PrefixedName:
        """
        Name of the connection perception attaches a frame to its parent frame with.

        :param parent_frame: Name of the parent frame.
        :param frame: Name of the attached frame.
        :return: The connection name.
        """
        return PrefixedName(name=f"{frame}_T_{parent_frame}")
