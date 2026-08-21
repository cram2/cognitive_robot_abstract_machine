from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import cast

from typing_extensions import List, Optional, Union, Iterable, Iterator

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.exceptions import CollisionViolatedError
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
from coraplex.datastructures.dataclasses import MotionToleranceConfig
from coraplex.plans.failures import MOTION_DID_NOT_WORK_OUT
from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription, GraspPose
from coraplex.locations.base import Location, PoseGeneratorBackend
from coraplex.locations.costmaps import Costmap, OccupancyCostmap, GaussianCostmap
from coraplex.view_manager import ViewManager
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot, EndEffector
from semantic_digital_twin.spatial_types.spatial_types import Point3, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger("coraplex")


@dataclass
class GiskardLocationBackend(PoseGeneratorBackend):
    """
    Pose generator backend that hands out the base poses full-body control managed to
    reach the target from.

    Every candidate is tried by driving the robot to its target from it, under the same
    collision rules the manipulation itself runs under, so a candidate stands for a reach
    that has been performed rather than one that looks plausible.

    ..note:: Passing a body reaches for it the way a grasp does, from a stand-off
        pre-pose; passing a pose reaches for that pose directly; passing a point reaches
        for it turned the way the robot stands.
    """

    target: Union[Point3, Pose, Body]
    """
    The target point, pose or body which should be reachable by the end effector.

    A point says where something has to end up but not which way it ends up turned; see
    :meth:`target_facing_the_robot`.
    """

    arm: Arms
    """
    Arm of the which should be used.
    """

    grasp_description: Optional[GraspDescription]
    """
    How the gripper comes at the target.

    ``None`` leaves the side open; see :meth:`grasps_to_try`.
    """

    robot: AbstractRobot
    """
    Robot for which base poses should be found.
    """

    world: World
    """
    The world in which to sample.
    """

    distance_to_obstacle: Optional[float] = None
    """
    Distance by which the obstacles should be inflated, in meters.

    Defaults to the clearance the robot's base needs, see
    :meth:`~coraplex.locations.costmaps.OccupancyCostmap.default_distance_to_obstacle`.
    """

    candidate_seeds: int = 50
    """
    How many starting poses full-body control is offered before this backend gives up.
    """

    candidate_tick_budget: int = 1_500
    """
    How many control ticks one candidate may take to bring the target within reach.

    A candidate that works takes well under half of this, so the budget mostly decides
    how long a hopeless one is allowed to hold up the search.
    """

    def __post_init__(self):
        if self.distance_to_obstacle is None:
            self.distance_to_obstacle = OccupancyCostmap.default_distance_to_obstacle(
                self.robot
            )

    def copy_for_world(self, world: World) -> GiskardLocationBackend:
        """
        :param world: The world the candidates should be generated in.
        :return: A backend that steers the same robot towards the same target, in
            ``world``.
        """
        robot = cast(AbstractRobot, world.get_semantic_annotation_by_id(self.robot.id))
        return GiskardLocationBackend(
            target=(
                self.target
                if isinstance(self.target, (Point3, Pose))
                else cast(Body, world.get_world_entity_with_id_by_id(self.target.id))
            ),
            arm=self.arm,
            grasp_description=(
                None
                if self.grasp_description is None
                else replace(
                    self.grasp_description,
                    end_effector=ViewManager.get_end_effector_view(self.arm, robot),
                )
            ),
            robot=robot,
            world=world,
            distance_to_obstacle=self.distance_to_obstacle,
            candidate_seeds=self.candidate_seeds,
            candidate_tick_budget=self.candidate_tick_budget,
        )

    def setup_costmap(self, pose: Pose) -> Costmap:
        """
        Setup the reachability costmap for initial pose estimation.

        Seeds are drawn at random rather than by cost, because the highest-valued cells
        of the gaussian all sit on top of each other and would put every seed within a
        few centimeters of the same spot.
        """
        ground_pose = deepcopy(pose)
        ground_pose.z = 0.0

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
        reachability_map.number_of_samples = self.candidate_seeds
        reachability_map.sample_randomly = True

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
        tolerances = MotionToleranceConfig()
        pose_seq = Sequence(
            nodes=[
                CartesianPose(
                    root_link=world.root,
                    tip_link=end_effector.tool_frame,
                    goal_pose=pose,
                    translation_threshold=tolerances.default_tcp_position_threshold,
                    orientation_threshold=tolerances.tool_orientation_threshold,
                )
                for pose in pose_sequence
            ]
        )
        msc = MotionStatechart()
        msc.add_nodes(
            [
                pose_seq,
                UpdateTemporaryCollisionRules(
                    temporary_rules=[
                        AllowCollisionBetweenGroups(
                            body_group_a=world.bodies_with_collision,
                            body_group_b=end_effector.bodies_with_collision,
                        )
                    ]
                ),
                ExternalCollisionAvoidance(robot=robot),
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

    def target_facing_the_robot(self) -> Union[Pose, Body]:
        """
        Work out what to reach for, given where the robot is standing.

        A point says where something has to end up but not which way it ends up turned,
        so it takes the robot's own heading. That is what lets a standing pose be looked
        for without fixing the heading first: every candidate is judged against the
        heading it would put the object down at, rather than against one chosen before
        anywhere to stand was known.

        :return: The target to reach for, which is the target itself unless it is a
            point.
        """
        if not isinstance(self.target, Point3):
            return self.target
        target_point = self.world.transform(self.target, self.world.root)
        return Pose.from_xyz_rpy(
            target_point.x,
            target_point.y,
            target_point.z,
            yaw=self.robot.root.global_pose.yaw,
            reference_frame=self.world.root,
        )

    def grasps_to_try(self) -> List[GraspDescription]:
        """
        Work out which grasps a standing pose has to allow.

        A backend given no grasp offers one per side the gripper could come from, so
        which side works is settled against the pose the robot is standing at rather
        than before anywhere to stand was known. Which side that turns out to be depends
        on where the robot ends up, so choosing it first is choosing it too early.

        :return: The grasp this backend was given, or every side when it was given none.
        """
        if self.grasp_description is not None:
            return [self.grasp_description]
        end_effector = ViewManager.get_end_effector_view(self.arm, self.robot)
        return [
            GraspDescription(
                approach_direction, VerticalAlignment.NoAlignment, end_effector
            )
            for approach_direction in ApproachDirection
        ]

    def reach_sequence(self, grasp: GraspDescription) -> List[Pose]:
        """
        :param grasp: How the gripper comes at the target.
        :return: The poses the gripper moves through to manipulate the target, which is
            the reach a standing pose has to allow. A body is approached from a stand-off
            pre-pose and closed in on, the way
            :class:`~coraplex.robot_plans.actions.core.pick_up.ReachAction` does. A pose
            is where whatever the gripper holds has to end up, so the hand stops short of
            it by that body, the way
            :class:`~coraplex.robot_plans.actions.core.placing.PlaceAction` does; an
            empty hand goes there itself.
        """
        target = self.target_facing_the_robot()
        if isinstance(target, Body):
            pre_pose, grasp_pose, _ = grasp.grasp_pose_sequence(target)
            return [pre_pose, grasp_pose]
        tool_frame = grasp.end_effector.tool_frame
        if not tool_frame.child_kinematic_structure_entities:
            return [target]
        transport_pose, placing_pose, _ = grasp.place_pose_sequence(target)
        return [transport_pose, placing_pose]

    def can_reach_target(self) -> bool:
        """
        Whether full-body control brings the target within reach for a robot that starts
        out standing at the current pose.

        The robot is free to drive while it reaches, exactly as it is when the
        manipulation itself runs, so what this answers is whether standing here is a
        workable place to start from -- not whether the arm alone spans the distance.

        Every grasp still open is tried from this one pose, which is far cheaper than
        looking for a pose per grasp: the poses are what cost a solve each.
        """
        end_effector = ViewManager.get_end_effector_view(self.arm, self.robot)
        for grasp in self.grasps_to_try():
            executor = self.setup_giskard_executor(
                self.reach_sequence(grasp), self.world, self.robot, end_effector
            )
            try:
                executor.tick_until_end(self.candidate_tick_budget)
            except MOTION_DID_NOT_WORK_OUT:
                continue
            return True
        return False

    def __iter__(self) -> Iterator[Pose]:
        resolved = self.target_facing_the_robot()
        target_pose = resolved.global_pose if isinstance(resolved, Body) else resolved

        for pose_candidate in self.setup_costmap(target_pose):
            with self.world.reset_state_context():
                self.robot.set_root_pose(pose_candidate)
                if Location.stands_in_collision(self.world, self.robot):
                    continue
                if isinstance(self.target, Body) and not self.robot.can_see_body(
                    self.target
                ):
                    continue
                if not self.can_reach_target():
                    continue
            yield pose_candidate


@dataclass
class GraspPoseGenerator(PoseGeneratorBackend):
    """
    A PoseGeneratorBackend that wraps another backend and creates GraspPoses from the
    samples poses of the backend.
    """

    generator: PoseGeneratorBackend
    """
    Pose generator from which to sample.
    """

    arm: Arms
    """
    Arm that should be used for the GraspPose.
    """

    grasp_description: GraspDescription
    """
    Grasp Description that should be used for the GraspPose.
    """

    def __iter__(self) -> Iterable[GraspPose]:
        for pose_candidate in self.generator:
            yield pose_candidate
