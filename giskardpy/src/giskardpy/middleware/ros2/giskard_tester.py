"""
A standalone Giskard with a motion server and a client in one process.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import sleep

from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.giskard_client_executor import GiskardClientExecutorNode
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Scale,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

# %% standalone giskard


@dataclass(kw_only=True)
class GiskardTester(ABC):
    """
    A standalone Giskard with a motion server running in the background and a client
    talking to it, for tests that go through the action server.
    """

    api: GiskardClientExecutorNode = field(init=False)
    """
    The client that sends goals to :attr:`giskard` and keeps its own copy of the world.
    """

    giskard: Giskard = field(init=False)
    """
    The Giskard under test, built by :meth:`setup_giskard`.
    """

    def __post_init__(self):
        self.giskard = self.setup_giskard()
        self.giskard.setup()
        self.default_root = self.world.root
        self.giskard.motion_server.start_in_background()
        self.wait_for_cycles(number_of_cycles=1)
        self.api = GiskardClientExecutorNode(node_name="tests")

    @abstractmethod
    def setup_giskard(self) -> Giskard:
        """
        :return: The Giskard to test, not yet set up.
        """

    @property
    def world(self) -> World:
        """
        The world of the Giskard under test, as opposed to the client's copy in
        :attr:`api`.
        """
        return self.giskard.executor.context.world

    def has_odometry_joint(self) -> bool:
        """
        :return: Whether the robot drives with an omnidirectional drive.
        """
        robot = self.world.get_semantic_annotations_by_type(AbstractRobot)[0]
        return isinstance(robot.drive, OmniDrive)

    def wait_for_cycles(self, *, number_of_cycles: int = 5) -> None:
        """
        Block until the motion server completed that many more cycles.

        Control cycles count too, so this also returns while a goal is being executed.

        :param number_of_cycles: How many cycles to wait for.
        """
        cycle_counter = self.giskard.motion_server.cycle_counter
        first_cycle = cycle_counter.completed_cycles
        while cycle_counter.completed_cycles < first_cycle + number_of_cycles:
            sleep(0.001)

    def close(self) -> None:
        """
        Stop the motion server and the client, and detach Giskard from the world.

        The motion server's background thread is stopped first: it is the only thing
        that could still touch the ros node or the world once this method returns. The
        ros node is destroyed between tests while worlds are kept alive, so a callback
        left registered here would publish on a node that is already gone.

        :raises MotionServerThreadStillRunningError: If the background thread does not
            stop in time.
        """
        self.giskard.motion_server.stop()
        self.giskard.close_world_model_ros_interface()
        self.api.close()

    # %% world modification through the client

    def add_box_to_world(
        self,
        *,
        name: str,
        size: tuple[float, float, float],
        pose: HomogeneousTransformationMatrix,
        parent_link: KinematicStructureEntity | None = None,
    ) -> None:
        """
        Adds a box to the client's world, attached to ``parent_link`` at ``pose``, and
        waits for the motion server to pick it up.

        :param size: Edge lengths along x, y and z.
        :param parent_link: The entity the box is fixed to, the world's root if
            ``None``.
        """
        parent_link = parent_link or self.api.world.root

        parent_T_pose = self.api.world.transform(
            spatial_object=pose,
            target_frame=parent_link,
        )
        with self.api.world.modify_world():
            box = Body(name=PrefixedName(name))
            box_shape = Box(scale=Scale(*size))
            box.collision.append(box_shape)
            box.visual.append(box_shape)

            connection = FixedConnection(
                parent=parent_link,
                child=box,
                parent_T_connection_expression=parent_T_pose,
            )
            self.api.world.add_connection(connection)
        self.wait_for_cycles()

    def add_urdf_to_world(
        self,
        *,
        name: str,
        file_path: str,
        pose: HomogeneousTransformationMatrix,
        parent_link: str | PrefixedName | None = None,
    ) -> None:
        """
        Merges the bodies described by the URDF or xacro file at ``file_path`` into the
        client's world at ``pose`` and waits for the motion server to pick them up.

        :param name: Prefix of the merged bodies' names.
        :param file_path: The path to the URDF or xacro file, ``package://`` paths
            included.
        :param parent_link: Name of the entity the URDF's root is fixed to, the world's
            root if ``None``.
        """
        if parent_link is None:
            parent_link = self.api.world.root
        else:
            parent_link = self.api.world.get_kinematic_structure_entity_by_name(
                parent_link
            )
        urdf_parser = URDFParser.from_file(file_path, prefix=name)
        world_from_urdf = urdf_parser.create_world()
        with self.api.world.modify_world():
            parent_C_urdf_root = FixedConnection(
                parent=parent_link,
                child=world_from_urdf.root,
                parent_T_connection_expression=pose,
            )
            self.api.world.merge_world(
                world_from_urdf, root_connection=parent_C_urdf_root
            )

        self.wait_for_cycles()
