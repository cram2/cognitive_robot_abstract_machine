from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import ClassVar

import rclpy

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import underspecified, variable
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.grasp import GraspDescription
from pycram.motion_executor import simulated_robot
from pycram.plans.factories import execute_single
from pycram.plans.failures import (
    PlanFailure,
)
from pycram.plans.plan import Plan
from pycram.plans.plan_node import UnderspecifiedNode
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.actions.core.misc import MoveToReach
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import (
    Manipulator,
    AbstractRobot,
)
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    OmniDrive,
)
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class TrainingEnvironment(ABC):
    """
    A training environment for generating data for the parameterization of actions.
    """

    action_type: ClassVar[type[ActionDescription]]
    """
    The type of action that is trained.
    """

    executed_plans: list[Plan] = field(default_factory=list)
    """
    The executed plans during training.
    """

    visualize: bool = False
    """
    Rather create a visualization of the executed plans or not.
    """

    @abstractmethod
    def setup_world(self, *kwargs) -> World:
        """
        :return: A world containing everything thats needed for training, including the robot.
        """

    @abstractmethod
    def setup_plan(self, limit: int = 10, **kwargs) -> Plan:
        """
        Create a plan with an underspecified node as a root.
        This plan is used to generate variants of the actions.

        :param limit: The maximum number of actions that should be executed.
        :return: The plan
        """

    def generate_episodes(self, number_of_actions: int = 10):
        """
        Generate episodes until `number_of_actions` have been executed.

        :param number_of_actions: The number of action executions that should be generated.
        """

        remaining_actions = number_of_actions

        while remaining_actions > 0:
            executed_actions = self.generate_episode(remaining_actions)
            remaining_actions -= executed_actions

    def generate_episode(self, limit: int) -> int:
        """
        Generate a single episode.

        :param limit: The maximum number of actions that should be executed.
        :return: The number of actions executed in the episode.
        """

        plan = self.setup_plan(limit)

        if self.visualize:
            pub = VizMarkerPublisher(
                _world=plan.context.world,
                node=rclpy.create_node("test_node"),
            )
            pub.with_tf_publisher()

        with simulated_robot:
            try:
                plan.perform()
            except (PlanFailure, StopIteration):
                ...
            finally:
                self.executed_plans.append(plan)

        number_of_executed_variants = len(plan.root.children)

        if self.visualize:
            pub.stop()
            pub.remove_from_world()

        return number_of_executed_variants


@dataclass
class MoveToReachTrainingEnvironment(TrainingEnvironment):
    """
    Training environment for MoveToReach actions in an empty world using the PR2.
    """

    action_type = MoveToReach

    def setup_world(self, **kwargs) -> World:
        pr2_file = "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"

        urdf_parser = URDFParser.from_file(file_path=pr2_file)
        world_with_urdf = urdf_parser.parse()
        pr2 = PR2.from_world(world_with_urdf)

        with world_with_urdf.modify_world():
            old_root = world_with_urdf.root
            map = Body(name=PrefixedName("map"))
            localization_body = Body(name=PrefixedName("odom_combined"))

            map_C_localization = Connection6DoF.create_with_dofs(
                world_with_urdf, map, localization_body
            )
            world_with_urdf.add_connection(map_C_localization)

            c_root_bf = OmniDrive.create_with_dofs(
                parent=localization_body,
                child=old_root,
                world=world_with_urdf,
            )
            world_with_urdf.add_connection(c_root_bf)
            c_root_bf.has_hardware_interface = True

        return world_with_urdf

    def setup_plan(self, limit: int = 10, **kwargs) -> UnderspecifiedNode:

        world = self.setup_world()
        [robot] = world.get_semantic_annotations_by_type(AbstractRobot)
        target_pose = Pose.from_xyz_rpy(
            x=0,
            y=0,
            z=1,
            reference_frame=world.root,
        )

        move_to_reach = underspecified(MoveToReach)(
            target_pose=target_pose,
            standing_position=underspecified(Point3)(
                x=...,
                y=...,
                z=float(robot.root.global_pose.z),
                reference_frame=target_pose.reference_frame,
            ),
            hip_rotation=...,
            manipulator=variable(Manipulator, world.semantic_annotations),
            grasp_description=underspecified(GraspDescription)(
                approach_direction=...,
                vertical_alignment=...,
                manipulator=variable(Manipulator, world.semantic_annotations),
                rotate_gripper=...,
                manipulation_offset=...,
            ),
        )

        move_to_reach.expression.limit(limit)

        context = Context(
            world=world, robot=robot, query_backend=ProbabilisticBackend()
        )

        return execute_single(move_to_reach, context=context).plan
