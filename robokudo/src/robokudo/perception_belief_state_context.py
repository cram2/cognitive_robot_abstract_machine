"""
The semantic digital twin RoboKudo perceives into.

Perception reads its prior knowledge from this world and writes what it perceives back
into it. When :class:`~robokudo.process.RoboKudo` joined the world other processes
share, every change perception makes to it is published to them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import trimesh.creation
from py_trees.blackboard import Blackboard
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
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

from robokudo.exceptions import SharedWorldReplacementError
from robokudo.identifier import BBIdentifier
from robokudo.perceived_objects import PerceivedObjects

# %% belief state context


@dataclass
class PerceptionBeliefStateContext:
    """
    The world RoboKudo perceives into and the objects it believes in, in that world.

    Annotators and camera interfaces find the context of the running pipeline on the
    blackboard through :meth:`from_blackboard`.
    """

    world: World = field(default_factory=World)
    """
    The world perception reads its prior knowledge from and writes its beliefs into.
    """

    perceived_objects: PerceivedObjects = field(init=False)
    """
    The objects perception believes in, kept as bodies in :attr:`world`.
    """

    entity_tracker: WorldEntityWithIDKwargsTracker = field(init=False)
    """
    Resolves the world entities serialized CAS content refers to.
    """

    def __post_init__(self) -> None:
        self.perceived_objects = PerceivedObjects(world=self.world)
        self.entity_tracker = WorldEntityWithIDKwargsTracker.from_world(self.world)

    @classmethod
    def from_blackboard(cls) -> PerceptionBeliefStateContext:
        """
        The context of the running pipeline.

        A pipeline started without one perceives into a fresh world of its own, which is
        stored on the blackboard on first access.

        :return: The context stored on the blackboard.
        """
        blackboard = Blackboard()
        if not blackboard.exists(BBIdentifier.PERCEPTION_BELIEF_STATE_CONTEXT):
            cls().store_on_blackboard()
        return blackboard.get(BBIdentifier.PERCEPTION_BELIEF_STATE_CONTEXT)

    def store_on_blackboard(self) -> None:
        """
        Make this the context of the running pipeline.
        """
        Blackboard().set(BBIdentifier.PERCEPTION_BELIEF_STATE_CONTEXT, self)

    @property
    def is_synchronized(self) -> bool:
        """
        Whether :attr:`world` is shared with other processes.
        """
        return bool(WorldSynchronizer.all_callbacks_of_this_type_from_world(self.world))

    def replace_world(self, world: World) -> None:
        """
        Perceive into ``world`` from now on.

        The objects perceived in the previous world are dropped with it.

        :param world: The world to perceive into.
        :raises SharedWorldReplacementError: If the current world is shared with other
            processes.
        """
        if self.is_synchronized:
            raise SharedWorldReplacementError()
        self.world = world
        self.perceived_objects = PerceivedObjects(world=world)
        self.entity_tracker = WorldEntityWithIDKwargsTracker.from_world(world)

    # %% camera placement

    def place_camera(
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
