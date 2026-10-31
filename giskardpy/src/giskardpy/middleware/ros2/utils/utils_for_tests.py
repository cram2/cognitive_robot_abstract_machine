from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import sleep

import numpy as np

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.giskard_client_executor import GiskardClientExecutorNode
from giskardpy.middleware.ros2.scripts.iai_robots.stretch.configs import (
    WorldWithStretchConfigDiffDrive,
)
from giskardpy.middleware.ros2.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
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

# %% comparing spatial types


def compare_poses(
    actual_pose: cas.HomogeneousTransformationMatrix,
    desired_pose: cas.HomogeneousTransformationMatrix,
    decimal: int = 2,
) -> None:
    """
    Assert that two transformations describe the same position and orientation.

    :param decimal: Number of decimal places the two have to agree on.
    """
    compare_points(
        actual_point=actual_pose.position,
        desired_point=desired_pose.position,
        decimal=decimal,
    )
    compare_orientations(
        actual_orientation=actual_pose.quaternion,
        desired_orientation=desired_pose.quaternion,
        decimal=decimal,
    )


def compare_points(
    actual_point: cas.Point3,
    desired_point: cas.Point3,
    decimal: int = 2,
) -> None:
    """
    Assert that two points are the same up to an absolute tolerance.

    .. note:: The tolerance is absolute because coordinates near the origin of their
        reference frame are common, and a relative one degenerates into an exact
        comparison for them.

    :param decimal: Number of decimal places the two have to agree on.
    """
    np.testing.assert_array_almost_equal(actual_point, desired_point, decimal=decimal)


def compare_orientations(
    actual_orientation: cas.Quaternion,
    desired_orientation: cas.Quaternion,
    decimal: int = 2,
) -> None:
    """
    Assert that two quaternions describe the same orientation.

    A quaternion and its negation are the same orientation, so the desired one is
    flipped onto the hemisphere of the actual one before they are compared.

    :param decimal: Number of decimal places the two have to agree on.
    """
    actual_quaternion = actual_orientation.to_np()
    desired_quaternion = desired_orientation.to_np()
    if np.dot(actual_quaternion, desired_quaternion) < 0:
        desired_quaternion = -desired_quaternion
    np.testing.assert_array_almost_equal(
        actual_quaternion, desired_quaternion, decimal=decimal
    )


@dataclass
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
        self.wait_for_cycles(1)
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

    def wait_for_cycles(self, number_of_cycles: int = 5) -> None:
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
        name: str,
        urdf: str,
        pose: HomogeneousTransformationMatrix,
        parent_link: str | PrefixedName | None = None,
    ) -> None:
        """
        Merges the bodies described by ``urdf`` into the client's world at ``pose`` and
        waits for the motion server to pick them up.

        :param name: Prefix of the merged bodies' names.
        :param urdf: The URDF document.
        :param parent_link: Name of the entity the URDF's root is fixed to, the world's
            root if ``None``.
        """
        if parent_link is None:
            parent_link = self.api.world.root
        else:
            parent_link = self.api.world.get_kinematic_structure_entity_by_name(
                parent_link
            )
        urdf_parser = URDFParser(urdf=urdf, prefix=name)
        world_from_urdf = urdf_parser.parse()
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


@dataclass
class StretchTester(GiskardTester):
    """
    A standalone Giskard driving Stretch through the same configuration the robot's own
    ``stretch_standalone`` launcher uses.

    Lives here rather than beside one test module because both the controller tests and
    the demo's integration test drive the same standalone setup.
    """

    tool_frame: KinematicStructureEntity = field(init=False)
    """
    The frame the demo's pick and place actions drive.
    """

    def __post_init__(self):
        super().__post_init__()
        self.tool_frame = self.api.world.get_kinematic_structure_entity_by_name(
            "link_grasp_center"
        )

    def setup_giskard(self) -> Giskard:
        return Giskard(
            world_config=WorldWithStretchConfigDiffDrive(
                urdf=load_xacro(Stretch.get_ros_file_path())
            ),
            robot_interface_config=StandAloneRobotInterfaceConfig(),
            server_config=GiskardServerConfig(debug_mode=True),
            qp_controller_config=QPControllerConfig.create_with_simulation_defaults(),
        )

    @property
    def robot(self) -> AbstractRobot:
        return self.world.get_semantic_annotations_by_type(Stretch)[0]
