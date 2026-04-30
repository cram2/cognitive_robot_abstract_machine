from copy import deepcopy
from dataclasses import dataclass

from typing_extensions import List, Union, Iterable

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.qp.exceptions import InfeasibleException
from giskardpy.qp.qp_controller_config import QPControllerConfig
from pycram.datastructures.enums import Arms
from pycram.datastructures.grasp import GraspDescription, GraspPose
from pycram.locations.base import Location, PoseGeneratorBackend
from pycram.locations.costmaps import Costmap, OccupancyCostmap, GaussianCostmap
from pycram.view_manager import ViewManager
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
    AllowCollisionRule,
    AllowCollisionBetweenGroups,
)
from semantic_digital_twin.robots.abstract_robot import AbstractRobot, Manipulator
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class GiskardLocationBackend(PoseGeneratorBackend):
    """
    Pose generator backend that uses full-body control to steer the robot to a base pose from which the target should be
    reachable.

    .. warning:: This backend uses collision avoidance, so if you use the global_pose of a body instead of the body itself
    the backend will fail because the collision avoidance will keep the gripper and body apart.
    """

    target: Union[Pose, Body]
    """
    The target pose or body which should be reachable by the end effector 
    """

    arm: Arms
    """
    Arm of the which should be used 
    """

    grasp_description: GraspDescription
    """
    Grasp description of how to approach the target
    """

    robot: AbstractRobot
    """
    Robot for which base poses should be found  
    """

    world: World
    """
    The world in which to sample 
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
            distance_to_obstacle=(base_bb.width / 2 + base_bb.depth / 2) / 2 + 0.5,
        )
        gaussian_map = GaussianCostmap(
            resolution=0.02,
            origin=ground_pose,
            mean=200,
            sigma=15,
            world=self.world,
        )

        reachability_map = occupancy_map + gaussian_map
        reachability_map.number_of_samples = 5

        return reachability_map

    def setup_giskard_executor(
        self,
        pose_sequence: List[Pose],
        world: World,
        robot_view: AbstractRobot,
        end_effector: Manipulator,
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
                    tip_link=end_effector.tool_frame,
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
                UpdateTemporaryCollisionRules(
                    temporary_rules=[
                        AllowCollisionBetweenGroups(
                            body_group_a=end_effector.bodies_with_collision,
                            body_group_b=(
                                [self.target] if isinstance(self.target, Body) else []
                            ),
                        )
                    ]
                ),
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

    def __iter__(self):
        with self.world.modify_world():
            self.robot._setup_collision_rules()

        target_pose = (
            self.target if isinstance(self.target, Pose) else self.target.global_pose
        )

        test_ee = ViewManager.get_end_effector_view(self.arm, self.robot)
        target_sequence = self.grasp_description._pose_sequence(target_pose)

        executor = self.setup_giskard_executor(
            target_sequence, self.world, self.robot, test_ee
        )

        for pose_candidate in self.setup_costmap(target_pose):
            self.robot.root.parent_connection.origin = pose_candidate

            try:
                executor.tick_until_end(3_000)
            except (TimeoutError, InfeasibleException) as e:
                pass

            yield self.robot.root.global_pose


@dataclass
class GraspPoseGenerator(PoseGeneratorBackend):
    """
    A PoseGeneratorBackend that wraps another backend and creates GraspPoses from the samples poses of the backend.
    """

    generator: PoseGeneratorBackend
    """
    Pose generator from which to sample
    """

    arm: Arms
    """
    Arm that should be used for the GraspPose
    """

    grasp_description: GraspDescription
    """
    Grasp Description that should be used for the GraspPose
    """

    def __iter__(self) -> Iterable[GraspPose]:
        for pose_candidate in self.generator:
            yield pose_candidate
