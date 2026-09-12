from copy import deepcopy
from dataclasses import dataclass, field
from itertools import islice

from typing_extensions import List

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
from coraplex.datastructures.enums import Arms
from coraplex.robot_plans.mixins import HasApproachesGraspPoses
from coraplex.locations.base import Location, PoseGeneratorBackend
from coraplex.locations.sampling import HighestRatedFirst
from coraplex.locations.costmaps import Costmap, OccupancyCostmap, GaussianCostmap
from coraplex.view_manager import ViewManager
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
    AllowCollisionRule,
    AllowCollisionBetweenGroups,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot, EndEffector
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class GiskardLocationBackend(PoseGeneratorBackend, HasApproachesGraspPoses):
    """
    Pose generator backend that uses full-body control to steer the robot to a base pose
    from which the target should be reachable.
    """

    target_pose: Pose
    """
    The pose the base poses are searched around.
    """

    number_of_candidates: int = field(default=5, kw_only=True)
    """
    How many base poses to draw and drive to.

    Each one costs a full simulated run, so far fewer than a map is usually drawn for.
    """

    arm: Arms
    """
    Arm of the which should be used.
    """

    grasp_pose: Pose
    """
    The grasp frame the end effector should reach on the target.
    """

    robot: AbstractRobot
    """
    Robot for which base poses should be found.
    """

    world: World
    """
    The world in which to sample.
    """

    body_T_grasp: Pose = field(default_factory=Pose, kw_only=True)
    """
    The same grasp in the frame of the body the approach must avoid, which sets how far
    ahead of the grasp the approach begins.

    An identity grasp with no reference frame when there is no such body.
    """

    contact_bodies: List[Body] = field(default_factory=list, kw_only=True)
    """
    The bodies the gripper may touch while reaching, since collision avoidance would
    otherwise keep it from closing on them.
    """

    reverse: bool = field(default=False, kw_only=True)
    """
    Whether the gripper withdraws from :attr:`grasp_pose` rather than moving onto it,
    which is how a body is released where it is placed.
    """

    distance_to_obstacle: float = 0.1
    """
    Distance by which the obstacles should be inflated, is set to the radius of the
    mobile base by default.
    """

    def __post_init__(self):
        base_bb = self.robot.mobile_base.bounding_box
        self.distance_to_obstacle = (base_bb.width / 2 + base_bb.depth / 2) / 2 + 0.5

    def setup_costmap(self, pose: Pose) -> Costmap:
        """
        Setup the reachability costmap for initial pose estimation.
        """
        ground_pose = deepcopy(pose)
        ground_pose.z = 0.0

        base_bb = self.robot.mobile_base.bounding_box

        occupancy_map = OccupancyCostmap(
            resolution=0.02,
            height=200,
            width=200,
            world=self.world,
            robot_view=self.robot,
            origin=ground_pose,
            distance_to_obstacle=self.distance_to_obstacle,
        )
        gaussian_map = GaussianCostmap(
            resolution=0.02,
            origin=ground_pose,
            mean=200,
            sigma=15,
            world=self.world,
        )

        reachability_map = occupancy_map + gaussian_map

        return reachability_map

    def setup_giskard_executor(
        self,
        pose_sequence: List[Pose],
        world: World,
        robot: AbstractRobot,
        end_effector: EndEffector,
    ) -> Executor:
        """
        Setup the Giskard executor for a specific pose sequence and a given world.

        :param pose_sequence: The pose sequence which the end_effector should follow
        :param world: The world in which the pose sequence should be executed
        :param robot: The robot view of the robot which should be used for the
            execution, needs to fit the world
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
                    robot=robot, buffer_zone_distance=0.1, violated_distance=0.0
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
                            body_group_b=self.contact_bodies,
                        )
                    ]
                ),
                ExternalCollisionAvoidance(
                    robot=robot, cancel_if_collision_violated=False
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

        test_ee = ViewManager.get_end_effector_view(self.arm, self.robot)
        target_sequence = self.grasp_pose_sequence(
            self.grasp_pose, test_ee, self.body_T_grasp, reverse=self.reverse
        )

        executor = self.setup_giskard_executor(
            target_sequence, self.world, self.robot, test_ee
        )

        for pose_candidate in islice(
            self.setup_costmap(self.target_pose).candidates(HighestRatedFirst()),
            self.number_of_candidates,
        ):
            self.robot.set_root_pose(pose_candidate)

            try:
                executor.tick_until_end(3_000)
            except (TimeoutError, InfeasibleException) as e:
                pass

            yield self.robot.root.global_pose
