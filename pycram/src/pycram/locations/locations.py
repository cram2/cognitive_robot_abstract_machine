import logging
from abc import abstractmethod, ABC
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import List, Optional, Iterator

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
)
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.qp.exceptions import InfeasibleException
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood.adapters.json_serializer import list_like_classes
from pycram.config.action_conf import ActionConfig
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
)
from pycram.datastructures.grasp import GraspDescription, GraspPose
from pycram.locations.base import Location
from pycram.plans.failures import RobotInCollision
from pycram.locations.costmaps import (
    OccupancyCostmap,
    VisibilityCostmap,
    GaussianCostmap,
    Costmap,
    OrientationGenerator,
    RingCostmap,
)
from pycram.locations.pose_validator import (
    visibility_validator,
    pose_sequence_reachability_validator,
)
from pycram.utils import link_pose_for_joint_config
from pycram.view_manager import ViewManager
from semantic_digital_twin.adapters.ros.visualization.pose_publisher import (
    PosePublisher,
)
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
)
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger("pycram")


@dataclass
class GiskardLocation(Location):
    """
    Finds a standing pose for a robot such that the TCP of the given arm can reach the target_pose. This Location
    Designator uses Giskard and full body control to find a pose for the robot base.
    """

    target_pose: Pose
    """
    Target pose for which a standing pose should be found. 
    """

    arm: Arms = None
    """
    Arm which should read the target pose
    """

    grasp_description: Optional[GraspDescription] = None
    """
    The grasp description which should be used to grasp the target pose, used if there is a body at the pose
    """

    threshold: float = field(default=0.02)
    """
    Threshold between the TCP of the arm and the target pose at which a stand pose if deemed successfull
    """

    def setup_costmap(self, pose: Pose) -> Costmap:
        """
        Setup the reachability costmap for initial pose estimation.
        """
        ground_pose = deepcopy(pose)
        ground_pose.z = 0.0

        base_bb = self.robot.base.bounding_box

        occupancy_map = OccupancyCostmap(
            resolution=0.02,
            height=200,
            width=200,
            world=self.world,
            robot_view=self.robot,
            origin=ground_pose,
            distance_to_obstacle=(base_bb.width / 2 + base_bb.depth / 2) / 2,
        )
        gaussian_map = GaussianCostmap(
            resolution=0.02,
            origin=ground_pose,
            mean=200,
            sigma=15,
            world=self.world,
        )

        reachability_map = occupancy_map + gaussian_map
        reachability_map.number_of_samples = 10

        return reachability_map

    def setup_giskard_executor(
        self,
        pose_sequence: List[Pose],
        world: World,
        robot_view: AbstractRobot,
        end_effector: Body,
    ) -> Executor:
        """
        Setup the Giskard executor for a specific pose sequence and a given world.

        :param pose_sequence: The pose sequence which the end_effector should follow
        :param world: The world in which the pose sequence should be executed
        :param robot_view: The robot view of the robot which should be used for the execution, needs to fit the world
        :param end_effector: The end effector which should be controlled by Giskard
        :return: The Giskard executor for the pose sequence
        """
        pose_seq = Sequence(
            nodes=[
                CartesianPose(
                    root_link=world.root,
                    tip_link=end_effector,
                    goal_pose=pose,
                )
                for pose in pose_sequence
            ]
        )
        with world.modify_world():
            world.collision_manager.clear_temporary_rules()
            world.collision_manager.add_temporary_rule(
                AvoidExternalCollisions(
                    robot=robot_view, buffer_zone_distance=0.1, violated_distance=0.0
                )
            )
        msc = MotionStatechart()
        msc.add_nodes(
            [
                pose_seq,
                ExternalCollisionAvoidance(
                    robot=robot_view, cancel_if_collision_violated=False
                ),
            ]
        )
        msc.add_node(EndMotion.when_true(pose_seq))

        executor = Executor(
            MotionStatechartContext(
                world=world,
                qp_controller_config=QPControllerConfig(
                    target_frequency=50, prediction_horizon=4, verbose=False
                ),
            ),
        )
        executor.compile(msc)

        return executor

    def __iter__(self) -> Iterator[GraspPose]:

        reachability_map = self.setup_costmap(self.target_pose)

        ee = ViewManager.get_arm_view(self.arm, self.robot)

        test_world = deepcopy(self.world)
        test_world.name = "Test World"

        test_robot = self.robot.__class__.from_world(test_world)
        test_ee = test_world._get_world_entity_by_hash(hash(ee.manipulator.tool_frame))
        with test_world.modify_world():
            test_robot._setup_collision_rules()

        for candidate in reachability_map:

            grasp_descriptions = (
                [self.grasp_description]
                if self.grasp_description
                else GraspDescription.calculate_grasp_descriptions(
                    (
                        test_robot.left_arm.manipulator
                        if self.arm == Arms.LEFT
                        else test_robot.right_arm.manipulator
                    ),
                    self.target_pose,
                )
            )

            for grasp_desc in grasp_descriptions:

                target_sequence = grasp_desc._pose_sequence(self.target_pose)

                test_robot.root.parent_connection.origin = candidate

                executor = self.setup_giskard_executor(
                    target_sequence, test_world, test_robot, test_ee
                )

                try:
                    executor.tick_until_end()
                except (TimeoutError, InfeasibleException) as e:
                    pass

                dist = test_ee.global_pose.to_position().euclidean_distance(
                    target_sequence[-1].to_position()
                )

                if dist > 0.02:
                    continue

                ret = GraspPose.from_pose(
                    test_robot.root.global_pose,
                    grasp_description=grasp_desc,
                    arm=self.arm,
                )
                yield ret
