from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from giskardpy.motion_statechart.binding_policy import GoalBindingPolicy
from cramph.data_types import ObservationStateValues, LifeCycleValues
from giskardpy.motion_statechart.weights import DefaultWeights
from giskardpy.motion_statechart.goals.base_pose_goals import DifferentialDriveBaseGoal
from giskardpy.motion_statechart.goals.cartesian_goals import CartesianPoseStraight
from cramph.composites import Sequence, Parallel
from giskardpy.motion_statechart.graph_node import EndMotion, MotionStatechartNode
from cramph.node import CancelStatechart, StatechartNode
from giskardpy.motion_statechart.monitors.cartesian_monitors import PositionReached
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetSeedConfiguration,
)
from cramph.statechart import (
    Statechart,
)
from giskardpy.motion_statechart.monitors.progress_monitors import StillProgressing
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    HoldOrientation,
    HoldPose,
    CartesianOrientation,
    CartesianPosition,
    CartesianPositionStraight,
    CartesianVelocityLimit,
    CartesianPositionVelocityLimit,
    CartesianRotationVelocityLimit,
    CartesianPositionTrajectory,
    DistanceAlongAxis,
)
from giskardpy.motion_statechart.tasks.drive_tasks import DrivenBaseTask
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from semantic_digital_twin.robots.robot_parts import MobileBase
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.spatial_types import (
    AxisAngle,
    HomogeneousTransformationMatrix,
    Vector3,
    Point3,
    RotationMatrix,
)
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    DifferentialDrive,
    FixedConnection,
    PrismaticConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Scale,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)
from semantic_digital_twin.world_description.world_state import WorldStateTrajectory
from semantic_digital_twin.world_description.world_state_trajectory_plotter import (
    WorldStateTrajectoryPlotter,
)
from test.giskardpy_test.test_motion_statechart.debug_expression_helpers import (
    CURRENT_COLOR,
    GOAL_COLOR,
    debug_expression_by_name,
)
from semantic_digital_twin.robots.pr2 import PR2Joint
from giskardpy.motion_control import MotionControl
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor, TickingExecutorExtension
from cramph.world_modification_nodes import MoveBranch
from giskardpy.motion_control import WorldStateTrajectoryRecording
from ..motion_control_context import add_to_statechart_with_motion_control

# %% holding a pose


class TestHoldPose:
    """
    Tests for holding a link where it is, rather than moving it to a goal.
    """

    @staticmethod
    def executor_for(world: World) -> StatechartExecutor:
        """
        :return: An executor driving `world` with motion control.
        """
        return StatechartExecutor(
            context=StatechartContext(world=world), extensions=[MotionControl()]
        )

    def test_a_held_link_stays_where_it_was(self, cylinder_bot_world: World):
        """
        A link is held against a task that pulls at it, so the constraints have to be
        real ones rather than a preference the solver can trade away.
        """
        bot = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        start = cylinder_bot_world.compute_forward_kinematics_np(
            cylinder_bot_world.root, bot
        ).copy()
        executor = self.executor_for(cylinder_bot_world)
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_nodes(
            [
                CartesianPosition(
                    root_link=cylinder_bot_world.root,
                    tip_link=bot,
                    goal_point=Point3(1, 0, 0, reference_frame=cylinder_bot_world.root),
                    weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
                ),
                HoldPose(
                    root_link=cylinder_bot_world.root,
                    tip_link=bot,
                    weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
                ),
            ]
        )
        motion_statechart.add_node(EndMotion())

        executor.compile(statechart=motion_statechart)
        for _ in range(100):
            executor.tick()

        reached = cylinder_bot_world.compute_forward_kinematics_np(
            cylinder_bot_world.root, bot
        )
        assert np.linalg.norm(reached[:3, 3] - start[:3, 3]) < 1e-3

    def test_a_held_pose_is_not_watched_for_progress(self, cylinder_bot_world: World):
        """
        Holding a link is an invariant, not a goal being approached, so a stall monitor
        must not count it among the tasks whose progress decides whether a motion is
        getting anywhere.
        """
        bot = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        goal = CartesianPosition(
            root_link=cylinder_bot_world.root,
            tip_link=bot,
            goal_point=Point3(1, 0, 0, reference_frame=cylinder_bot_world.root),
        )
        hold = HoldPose(root_link=cylinder_bot_world.root, tip_link=bot)
        together = Parallel(nodes=[goal, hold])
        progressing = StillProgressing(monitored_node=together)
        executor = self.executor_for(cylinder_bot_world)
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_nodes([together, progressing])
        motion_statechart.add_node(EndMotion())

        executor.compile(statechart=motion_statechart)

        assert progressing.monitored_tasks == [goal]

    def test_a_held_pose_does_not_hold_up_the_motion_it_runs_alongside(
        self, cylinder_bot_world: World
    ):
        """
        A held pose has no goal to reach, so it must report itself satisfied: a motion
        that ends once everything under it reached its goal would otherwise never end.
        """
        bot = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        hold = HoldPose(root_link=cylinder_bot_world.root, tip_link=bot)
        executor = self.executor_for(cylinder_bot_world)
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_node(hold)
        motion_statechart.add_node(EndMotion())

        executor.compile(statechart=motion_statechart)
        executor.tick()

        assert hold.observation_state == ObservationStateValues.TRUE

    def test_a_link_moved_away_from_its_held_pose_is_observed_as_not_held(
        self, cylinder_bot_world: World
    ):
        """
        A hold that a stronger task overrules no longer keeps its link in place, and has
        to say so rather than report an invariant that does not hold.
        """
        bot = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        hold = HoldPose(
            root_link=cylinder_bot_world.root,
            tip_link=bot,
            weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
        )
        executor = self.executor_for(cylinder_bot_world)
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_nodes(
            [
                CartesianPosition(
                    root_link=cylinder_bot_world.root,
                    tip_link=bot,
                    goal_point=Point3(1, 0, 0, reference_frame=cylinder_bot_world.root),
                    weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
                ),
                hold,
            ]
        )
        motion_statechart.add_node(EndMotion())

        executor.compile(statechart=motion_statechart)
        for _ in range(100):
            executor.tick()

        assert hold.observation_state == ObservationStateValues.FALSE

    def test_a_link_held_in_its_orientation_still_moves_but_does_not_turn(
        self, cylinder_bot_world: World
    ) -> None:
        """
        Tasks pull the link both to another position and into another orientation, and
        only the turning is held off.
        """
        world = cylinder_bot_world
        bot = world.get_kinematic_structure_entity_by_name("bot")
        start = world.compute_forward_kinematics_np(world.root, bot).copy()
        goal_point = Point3(1, 0, 0, reference_frame=world.root)
        executor = self.executor_for(world)
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_nodes(
            [
                moving := CartesianPosition(
                    root_link=world.root,
                    tip_link=bot,
                    goal_point=goal_point,
                    weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
                ),
                CartesianOrientation(
                    root_link=world.root,
                    tip_link=bot,
                    goal_orientation=RotationMatrix.from_rpy(
                        yaw=np.pi / 2, reference_frame=world.root
                    ),
                    weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
                ),
                HoldOrientation(
                    root_link=world.root,
                    tip_link=bot,
                    weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
                ),
            ]
        )
        motion_statechart.add_node(EndMotion.when_true(moving))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        reached = world.compute_forward_kinematics_np(world.root, bot)
        np.testing.assert_allclose(
            reached[:3, 3], goal_point.to_np()[:3], atol=moving.threshold
        )
        np.testing.assert_allclose(reached[:3, :3], start[:3, :3], atol=1e-3)


# %% straight line paths

STRAIGHT_LINE_TOLERANCE = 0.02
"""
How far the tip may stray from the line it is supposed to travel along, in meters.

Some deviation is unavoidable: the controller has to accelerate out of the pose the
motion starts at, and it can only correct a lateral offset on the tick after it
appeared.
"""

REAIMED_LINE_TOLERANCE = 0.025
"""
How far the tip of a point goal may stray from the straight line to the goal, in meters.

A point goal aims at the goal again from wherever the tip is on every tick, instead of
pulling the tip back onto the line the motion started on, so a lateral offset is not
corrected but only stops growing. On the PR2 arm this accumulates to about 22 mm.
"""


@dataclass
class StraightLine:
    """
    The line a straight Cartesian motion is supposed to travel along.
    """

    start: np.ndarray
    """
    Position the motion starts at.
    """

    end: np.ndarray
    """
    Position the motion ends at.
    """

    def distance_from(self, position: np.ndarray) -> float:
        """
        :return: Distance between `position` and this line.
        """
        direction = self.end - self.start
        direction = direction / np.linalg.norm(direction)
        offset = position - self.start
        return float(np.linalg.norm(offset - np.dot(offset, direction) * direction))

    def maximum_distance_from(self, positions: list[np.ndarray]) -> float:
        """
        :return: Distance of the position that strayed furthest from this line.
        """
        return max(self.distance_from(position) for position in positions)


def peak_angle_per_tick_towards_roll_goal(world: World, angle: float) -> float:
    """
    Rolls the PR2's right gripper about its own x-axis and measures how fast it turns.

    :param world: A PR2 world.
    :param angle: How far the goal is rolled away from the current orientation.
    :return: The largest angle the gripper turned within one control cycle.
    """
    tip = world.get_kinematic_structure_entity_by_name("r_gripper_tool_frame")
    root = world.get_kinematic_structure_entity_by_name("base_link")
    executor = StatechartExecutor(
        context=StatechartContext(world=world), extensions=[MotionControl()]
    )
    motion_statechart = Statechart(context=executor.context)
    orientation_goal = CartesianOrientation(
        root_link=root,
        tip_link=tip,
        goal_orientation=RotationMatrix.from_axis_angle(
            AxisAngle(Vector3.X(), angle, reference_frame=tip)
        ),
    )
    motion_statechart.add_node(orientation_goal)
    motion_statechart.add_node(EndMotion.when_true(orientation_goal))
    executor.compile(statechart=motion_statechart)

    rotations = [world.compute_forward_kinematics_np(root, tip)[:3, :3].copy()]
    while not executor.statechart.is_ended():
        executor.tick()
        rotations.append(world.compute_forward_kinematics_np(root, tip)[:3, :3].copy())
    return max(
        np.arccos(np.clip((np.trace(before.T @ after) - 1) / 2, -1, 1))
        for before, after in zip(rotations, rotations[1:])
    )


def record_tip_path(
    executor: StatechartExecutor,
    task: MotionStatechartNode,
    root_link: KinematicStructureEntity,
    tip_link: KinematicStructureEntity,
    maximum_ticks: int = 2000,
) -> list[np.ndarray]:
    """
    Tick until the motion ends and collect where the tip was while `task` was running.

    The first entry is the position the tip had when `task` started, which is where a
    straight motion is supposed to begin.

    :raises TimeoutError: if the motion does not end within `maximum_ticks`.
    """
    world = executor.context.world
    life_cycle_state = executor.statechart.life_cycle_state

    def tip_position() -> np.ndarray:
        return world.compute_forward_kinematics_np(root_link, tip_link)[:3, 3].copy()

    path = []
    if life_cycle_state[task] == LifeCycleValues.RUNNING:
        path.append(tip_position())
    for _ in range(maximum_ticks):
        position_before_tick = tip_position()
        executor.tick()
        if not path and life_cycle_state[task] == LifeCycleValues.RUNNING:
            path.append(position_before_tick)
        if path:
            path.append(tip_position())
        if executor.statechart.is_ended():
            return path
    raise TimeoutError(f"{task.name} did not finish within {maximum_ticks} ticks")


class TestCartesianPositionTrajectory:
    def _points_to_np(self, positions: list[Point3] | np.ndarray) -> np.ndarray:
        """
        Convert a sequence of `Point3` or an `ndarray` of shape (N, 3) into an `ndarray`
        of shape (N, 3).
        """
        if isinstance(positions, np.ndarray):
            if positions.ndim != 2 or positions.shape[1] != 3:
                raise ValueError("positions ndarray must have shape (N, 3)")
            return positions.astype(float)
        pts = [
            p.to_np()[:-1] if isinstance(p, Point3) else np.asarray(p, dtype=float)
            for p in positions
        ]
        arr = np.vstack(pts).astype(float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("positions must convert to shape (N, 3)")
        return arr

    def compare_trajectories(
        self,
        positions: list[Point3] | np.ndarray,
        world_state_trajectory: WorldStateTrajectory,
        root_link: KinematicStructureEntity,
        tip_link: KinematicStructureEntity,
        tolerance: float = 0.01,
    ) -> None:
        """
        Compare an executed Cartesian path against a reference list of positions.

        The executed path is reconstructed from the `world_state_trajectory` by
        computing forward kinematics for `tip_link` in the `root_link` frame at each
        recorded state. For each executed point, the minimum Euclidean distance to the
        reference path is computed and asserted to be within `tolerance`.

        :param positions: Reference path as `Point3` iterable or an array of shape (N,
            3). All points are with respect to root_link
        :param world_state_trajectory: Recorded joint-space trajectory with access to
            the world.
        :param root_link: Root kinematic frame for forward kinematics.
        :param tip_link: Tip kinematic frame for forward kinematics.
        :param tolerance: Maximum allowed distance to the reference path for all
            samples.
        """
        ref_np = self._points_to_np(positions)

        world = world_state_trajectory.world
        executed_points = []

        # Reconstruct executed Cartesian path by FK at each recorded state
        for state_view in world_state_trajectory.values():
            # Temporarily set the world's state to the recorded one
            for derivative in Derivatives:
                world.state.set_derivative(derivative, state_view.data[derivative, :])
            world.notify_state_change()
            p = (
                world.compute_forward_kinematics(root_link, tip_link)
                .position.evaluate()[:-1]
                .astype(float)
            )
            executed_points.append(p.copy())

        executed_np = np.vstack(executed_points)

        # Distance of each executed point to the nearest reference point
        def _min_dist_to_ref(p: np.ndarray) -> float:
            return float(np.min(np.linalg.norm(executed_np - p, axis=1)))

        distances = np.apply_along_axis(_min_dist_to_ref, 1, ref_np)

        assert np.max(distances) <= tolerance

    def test_cartesian_position_trajectory_spiral(self, cylinder_bot_world: World):
        points = []
        growth_factor = 0.05  # tunes how fast the spiral radius grows

        for step in range(10000):
            angle = step * np.pi / 5000.0
            radius = growth_factor * angle  # radius grows linearly with the angle
            points.append(
                Point3(
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    0,
                    reference_frame=cylinder_bot_world.root,
                )
            )
        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_world),
            extensions=[
                MotionControl(),
                WorldStateTrajectoryRecording(plotter=WorldStateTrajectoryPlotter()),
            ],
        )
        motion_statechart = Statechart(context=executor.context)
        cartesian_trajectory = CartesianPositionTrajectory(
            root_link=cylinder_bot_world.root,
            tip_link=cylinder_bot_world.get_kinematic_structure_entity_by_name("bot"),
            goal_points=points,
        )
        motion_statechart.add_node(cartesian_trajectory)
        motion_statechart.add_node(EndMotion.when_true(cartesian_trajectory))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()
        self.compare_trajectories(
            points,
            executor.require_extension(
                WorldStateTrajectoryRecording
            ).plotter.world_state_trajectory,
            cartesian_trajectory.root_link,
            cartesian_trajectory.tip_link,
        )

    def test_cartesian_position_trajectory_circle(self, cylinder_bot_world: World):
        points = []
        radius = 0.1

        for step in range(5000):
            angle = step * np.pi / 500.0
            points.append(
                Point3(
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    0,
                    reference_frame=cylinder_bot_world.root,
                )
            )
        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_world),
            extensions=[
                MotionControl(),
                WorldStateTrajectoryRecording(plotter=WorldStateTrajectoryPlotter()),
            ],
        )
        motion_statechart = Statechart(context=executor.context)
        cartesian_trajectory = CartesianPositionTrajectory(
            root_link=cylinder_bot_world.root,
            tip_link=cylinder_bot_world.get_kinematic_structure_entity_by_name("bot"),
            goal_points=points,
            maximum_skip_ahead=20,
        )
        motion_statechart.add_node(cartesian_trajectory)
        motion_statechart.add_node(EndMotion.when_true(cartesian_trajectory))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()
        self.compare_trajectories(
            points,
            executor.require_extension(
                WorldStateTrajectoryRecording
            ).plotter.world_state_trajectory,
            cartesian_trajectory.root_link,
            cartesian_trajectory.tip_link,
        )

    def test_trajectory_follows_a_goal_frame_that_moved_before_it_started(
        self, cylinder_bot_world: World
    ):
        """
        A trajectory bound on start must track its goal frame where that frame is when
        the trajectory starts, even if a task built after the trajectory moved the frame
        first.
        """
        root = cylinder_bot_world.root
        tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        points = [Point3(0, y, 0, reference_frame=tip) for y in np.linspace(0, 0.2, 21)]

        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_world),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        cartesian_trajectory = CartesianPositionTrajectory(
            root_link=root, tip_link=tip, goal_points=points
        )
        motion_statechart.add_node(cartesian_trajectory)
        move_away = CartesianPosition(
            root_link=root,
            tip_link=tip,
            goal_point=Point3(-0.5, 0, 0, reference_frame=root),
        )
        motion_statechart.add_node(move_away)
        move_away.success_condition = move_away.observes_true
        cartesian_trajectory.start_condition = move_away.is_succeeded
        motion_statechart.add_node(EndMotion.when_true(cartesian_trajectory))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        root_P_tip_start = move_away.goal_point.to_np()[:3]
        tip_start_P_last_point = points[-1].to_np()[:3]
        assert np.allclose(
            cylinder_bot_world.compute_forward_kinematics_np(root, tip)[:3, 3],
            root_P_tip_start + tip_start_P_last_point,
            atol=move_away.threshold + cartesian_trajectory.threshold,
        )

    def test_cartesian_position_trajectory_spiral_pr2(
        self, pr2_world_state_reset: World, better_pr2_pose
    ):
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )

        seed = SetSeedConfiguration(
            seed_configuration=JointState.from_str_dict(
                better_pr2_pose, world=pr2_world_state_reset
            )
        )
        add_to_statechart_with_motion_control(seed, pr2_world_state_reset)
        seed.on_start()

        points = []
        root_points = []
        a = 0.05  # spiral growth factor (tunes how fast radius grows)
        for i in range(10000):
            t = (
                i * np.pi / 5000.0
            )  # angle parameter; adjust divisor for tighter/looser turns
            r = a * t  # radius grows linearly with t
            point = Point3(
                r * np.cos(t),
                r * np.sin(t),
                0,
                reference_frame=tip,
            )
            points.append(point)
            root_points.append(pr2_world_state_reset.transform(point, root))
        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[
                MotionControl(),
                WorldStateTrajectoryRecording(plotter=WorldStateTrajectoryPlotter()),
            ],
        )
        motion_statechart = Statechart(context=executor.context)

        motion_statechart.add_node(
            cartesian_trajectory := CartesianPositionTrajectory(
                root_link=root,
                tip_link=tip,
                goal_points=points,
            )
        )
        motion_statechart.add_node(EndMotion.when_true(cartesian_trajectory))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()
        self.compare_trajectories(
            root_points,
            executor.require_extension(
                WorldStateTrajectoryRecording
            ).plotter.world_state_trajectory,
            cartesian_trajectory.root_link,
            cartesian_trajectory.tip_link,
        )


class TestCartesianTasks:
    """
    Test suite for all Cartesian motion tasks.
    """

    def test_simple_cartesian_pose(self, cylinder_bot_world: World):
        tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")

        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_world),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_nodes(
            [
                goal := CartesianPose(
                    root_link=cylinder_bot_world.root,
                    tip_link=tip,
                    goal_pose=Pose.from_xyz_rpy(
                        x=1, reference_frame=cylinder_bot_world.root
                    ),
                ),
            ]
        )
        motion_statechart.add_node(EndMotion.when_true(goal))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        assert np.allclose(
            cylinder_bot_world.compute_forward_kinematics(cylinder_bot_world.root, tip),
            goal.goal_pose,
            atol=goal.translation_threshold,
        )

    def test_orientation_threshold_decouples_rotation_tolerance(
        self, cylinder_bot_world: World
    ):
        """
        A residual orientation error above ``threshold`` but below
        ``orientation_threshold`` must count as goal reached -- a physically tracked arm
        settles with a small orientation error that a shared position/rotation threshold
        (meant as a position tolerance in meters) wrongly rejects, leaving the task
        running forever.

        The goal pose here only differs from the start pose by a 0.05 rad yaw, so on the
        very first tick the position error is exactly zero while the rotation error is
        0.05 rad -- isolating the rotation half of the observation.

        The orientation sub-tasks are inspected directly, because the observation of the
        enclosing :class:`Parallel` only reflects its children on the following tick.
        """
        tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        goal_pose = Pose.from_xyz_rpy(yaw=0.05, reference_frame=cylinder_bot_world.root)

        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_world),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_nodes(
            [
                strict := CartesianPose(
                    root_link=cylinder_bot_world.root,
                    tip_link=tip,
                    goal_pose=goal_pose,
                    translation_threshold=0.01,
                    name="strict",
                ),
                loose := CartesianPose(
                    root_link=cylinder_bot_world.root,
                    tip_link=tip,
                    goal_pose=goal_pose,
                    translation_threshold=0.01,
                    orientation_threshold=0.1,
                    name="loose",
                ),
            ]
        )
        motion_statechart.add_node(EndMotion.when_true(loose))

        executor.compile(statechart=motion_statechart)
        executor.tick()

        strict_orientation = next(
            node for node in strict.nodes if isinstance(node, CartesianOrientation)
        )
        loose_orientation = next(
            node for node in loose.nodes if isinstance(node, CartesianOrientation)
        )
        assert strict_orientation.observation_state == ObservationStateValues.FALSE
        assert loose_orientation.observation_state == ObservationStateValues.TRUE

    def test_end_motion_waits_for_convergence(self, cylinder_bot_world: World):
        """
        EndMotion.when_true(goal) must not end the motion the instant the task's own
        goal-error threshold is crossed; it must wait until the robot's DOF velocities
        have actually settled.

        A loose ``threshold`` makes the task report "reached" long before the base has
        slowed down, so the base is still moving fast when that happens.
        """
        tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")

        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_world),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_nodes(
            [
                goal := CartesianPose(
                    root_link=cylinder_bot_world.root,
                    tip_link=tip,
                    goal_pose=Pose.from_xyz_rpy(
                        x=1, reference_frame=cylinder_bot_world.root
                    ),
                    translation_threshold=0.5,
                ),
            ]
        )
        motion_statechart.add_node(EndMotion.when_true(goal))

        executor.compile(statechart=motion_statechart)

        goal_reached_tick = None
        for i in range(1000):
            executor.tick()
            if (
                goal_reached_tick is None
                and motion_statechart.observation_state[goal]
                == ObservationStateValues.TRUE
            ):
                goal_reached_tick = i
            if motion_statechart.is_ended():
                break
        else:
            raise TimeoutError("motion never ended")

        assert goal_reached_tick is not None
        assert i > goal_reached_tick, (
            "EndMotion ended on the same tick the task's own threshold was crossed, "
            "before the controller had a chance to decelerate"
        )
        max_velocity = max(
            abs(dof.variables.velocity.resolve())
            for dof in cylinder_bot_world.active_degrees_of_freedom
        )
        assert max_velocity < 0.06, (
            f"EndMotion ended while a DOF was still moving at {max_velocity} m/s or "
            f"rad/s"
        )

    def test_long_goal(self, pr2_world_state_reset: World):
        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_nodes(
            [
                cart_goal := CartesianPose(
                    root_link=pr2_world_state_reset.root,
                    tip_link=pr2_world_state_reset.get_kinematic_structure_entity_by_name(
                        "base_footprint"
                    ),
                    goal_pose=Pose.from_xyz_rpy(
                        x=50, reference_frame=pr2_world_state_reset.root
                    ),
                ),
                JointPositionList(
                    goal_state=JointState.from_str_dict(
                        {
                            PR2Joint.TORSO_LIFT: 0.2999225173357618,
                            PR2Joint.HEAD_PAN: 0.042,
                            PR2Joint.HEAD_TILT: -0.37,
                            PR2Joint.RIGHT_UPPER_ARM_ROLL: -0.9487714747527726,
                            PR2Joint.RIGHT_SHOULDER_PAN: -1.0047307505973626,
                            PR2Joint.RIGHT_SHOULDER_LIFT: 0.48736790658811985,
                            PR2Joint.RIGHT_FOREARM_ROLL: -14.895833882874182,
                            PR2Joint.RIGHT_ELBOW_FLEX: -1.392377908925028,
                            PR2Joint.RIGHT_WRIST_FLEX: -0.4548695149411013,
                            PR2Joint.RIGHT_WRIST_ROLL: 0.11426798984097819,
                            PR2Joint.LEFT_UPPER_ARM_ROLL: 1.7383062350263658,
                            PR2Joint.LEFT_SHOULDER_PAN: 1.8799810286792007,
                            PR2Joint.LEFT_SHOULDER_LIFT: 0.011627231224188975,
                            PR2Joint.LEFT_FOREARM_ROLL: 312.67276414458695,
                            PR2Joint.LEFT_ELBOW_FLEX: -2.0300928925694675,
                            PR2Joint.LEFT_WRIST_FLEX: -0.1,
                            PR2Joint.LEFT_WRIST_ROLL: -6.062015047706399,
                        },
                        world=pr2_world_state_reset,
                    )
                ),
            ]
        )
        motion_statechart.add_node(EndMotion.when_true(cart_goal))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end(1_000_000)

        assert cart_goal.observation_state == ObservationStateValues.TRUE

    def test_cart_goal_1eef(self, pr2_world_state_reset: World):
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        tip_goal = Pose.from_xyz_quaternion(pos_x=-0.2, reference_frame=tip)
        expected = pr2_world_state_reset.transform(tip_goal, root)

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)

        motion_statechart.add_nodes(
            [
                cart_goal := CartesianPose(
                    root_link=root,
                    tip_link=tip,
                    goal_pose=tip_goal,
                ),
                EndMotion.when_true(cart_goal),
            ]
        )

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        assert np.allclose(
            executor.context.world.compute_forward_kinematics(root, tip),
            expected,
            atol=cart_goal.translation_threshold,
        )

    def test_unreachable_goal_ends_in_a_local_minimum(
        self, pr2_world_state_reset: World
    ) -> None:
        """
        A base pose below the floor cannot be reached, so the motion settles in a local
        minimum instead of reaching it.
        """
        base_footprint = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_nodes(
            [
                cart_goal := CartesianPose(
                    root_link=pr2_world_state_reset.root,
                    tip_link=base_footprint,
                    goal_pose=Pose.from_xyz_rpy(
                        z=-1, reference_frame=pr2_world_state_reset.root
                    ),
                ),
                local_min := LocalMinimumReached(),
            ]
        )
        motion_statechart.add_node(EndMotion.when_true(local_min))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        assert local_min.observation_state == ObservationStateValues.TRUE
        assert cart_goal.observation_state == ObservationStateValues.FALSE

    def test_cart_goal_orientation_singularity(
        self, pr2_world_state_reset: World, default_pr2_pose: dict[str, float]
    ) -> None:
        """
        Goals that keep the orientation of both grippers are reached, although an
        orientation error of zero is a singularity of the rotation error.
        """
        JointState.from_str_dict(default_pr2_pose, pr2_world_state_reset).apply_to(
            pr2_world_state_reset
        )
        base_link = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_link"
        )
        right_tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        left_tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "l_gripper_tool_frame"
        )
        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_node(
            both_grippers := Parallel(
                nodes=[
                    CartesianPose(
                        root_link=base_link,
                        tip_link=right_tip,
                        goal_pose=Pose.from_xyz_quaternion(
                            pos_x=-0.1, reference_frame=right_tip
                        ),
                    ),
                    CartesianPose(
                        root_link=base_link,
                        tip_link=left_tip,
                        goal_pose=Pose.from_xyz_quaternion(
                            pos_x=-0.05, reference_frame=left_tip
                        ),
                    ),
                ]
            )
        )
        motion_statechart.add_node(EndMotion.when_true(both_grippers))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        assert both_grippers.observation_state == ObservationStateValues.TRUE

    def test_cart_goal_left_right_chain(
        self, pr2_world_state_reset: World, default_pr2_pose: dict[str, float]
    ) -> None:
        """
        A goal for one gripper relative to the other moves both arms.
        """
        JointState.from_str_dict(default_pr2_pose, pr2_world_state_reset).apply_to(
            pr2_world_state_reset
        )
        left_tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "l_gripper_tool_frame"
        )
        right_tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        goal_pose = HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=Point3(0.2, 0.0, reference_frame=left_tip),
            rotation_matrix=RotationMatrix(
                [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]],
                reference_frame=left_tip,
            ),
            reference_frame=left_tip,
        )
        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_node(
            cart_goal := CartesianPose(
                root_link=left_tip, tip_link=right_tip, goal_pose=goal_pose
            )
        )
        motion_statechart.add_node(EndMotion.when_true(cart_goal))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        assert np.allclose(
            pr2_world_state_reset.compute_forward_kinematics(left_tip, right_tip),
            goal_pose,
            atol=cart_goal.translation_threshold,
        )

    def test_root_link_not_equal_chain_root(self, pr2_world_state_reset: World) -> None:
        """
        A goal whose root link is in the middle of the kinematic chain moves only the
        joints below it.
        """
        torso_lift_link = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "torso_lift_link"
        )
        right_tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        base_footprint = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        goal_pose = Pose.from_xyz_rpy(
            x=0.8, y=-0.5, z=1.0, reference_frame=base_footprint
        )
        expected = pr2_world_state_reset.transform(goal_pose, torso_lift_link)
        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_node(
            cart_goal := CartesianPose(
                root_link=torso_lift_link, tip_link=right_tip, goal_pose=goal_pose
            )
        )
        motion_statechart.add_node(EndMotion.when_true(cart_goal))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        assert np.allclose(
            pr2_world_state_reset.compute_forward_kinematics(
                torso_lift_link, right_tip
            ),
            expected,
            atol=cart_goal.translation_threshold,
        )

    @pytest.mark.parametrize(
        "goal_x, goal_y, goal_yaw",
        [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 0.0, np.pi / 3),
            (1.0, 0.0, np.pi / 3),
        ],
        ids=["forward", "left", "diagonal", "rotate", "forward_and_rotate"],
    )
    def test_omni_drive_base_reaches_its_goal(
        self, hsr_world_copy: World, goal_x: float, goal_y: float, goal_yaw: float
    ) -> None:
        """
        An omnidirectional base drives sideways, diagonally and turning to its goal.
        """
        base_footprint = hsr_world_copy.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        goal_pose = Pose.from_xyz_rpy(
            x=goal_x, y=goal_y, yaw=goal_yaw, reference_frame=hsr_world_copy.root
        )
        executor = StatechartExecutor(
            context=StatechartContext(world=hsr_world_copy),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_node(
            cart_goal := CartesianPose(
                root_link=hsr_world_copy.root,
                tip_link=base_footprint,
                goal_pose=goal_pose,
            )
        )
        motion_statechart.add_node(EndMotion.when_true(cart_goal))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        assert np.allclose(
            hsr_world_copy.compute_forward_kinematics(
                hsr_world_copy.root, base_footprint
            ),
            goal_pose,
            atol=cart_goal.translation_threshold,
        )

    def test_daisy_finger_tip_reaches_its_goal(
        self, daisy_world_copy: World, default_daisy_pose: dict[str, float]
    ) -> None:
        JointState.from_str_dict(default_daisy_pose, daisy_world_copy).apply_to(
            daisy_world_copy
        )
        base = daisy_world_copy.get_kinematic_structure_entity_by_name("left_base_link")
        finger_tip = daisy_world_copy.get_kinematic_structure_entity_by_name(
            "left_gripper_left_finger_tip_link"
        )
        goal_pose = Pose.from_xyz_axis_angle(z=0.2, reference_frame=finger_tip)
        expected = daisy_world_copy.transform(goal_pose, base)
        executor = StatechartExecutor(
            context=StatechartContext(world=daisy_world_copy),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_node(
            cart_goal := CartesianPose(
                root_link=base, tip_link=finger_tip, goal_pose=goal_pose
            )
        )
        motion_statechart.add_node(EndMotion.when_true(cart_goal))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        assert np.allclose(
            daisy_world_copy.compute_forward_kinematics(base, finger_tip),
            expected,
            atol=cart_goal.translation_threshold,
        )

    def test_grasp_frame_orientation(self, _hsr_world_setup: World):
        """
        Test combined position and orientation control in parallel.
        """
        with _hsr_world_setup.modify_world():
            box = Body(
                name=PrefixedName("muh"),
                collision=ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.1))]),
            )
            connection = FixedConnection(
                parent=_hsr_world_setup.root,
                child=box,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=2, z=0.5
                ),
            )
            _hsr_world_setup.add_connection(connection)

        hsr = _hsr_world_setup.get_semantic_annotations_by_type(HSRB)[0]
        hand = _hsr_world_setup.get_semantic_annotations_by_type(EndEffector)[0]
        executor = StatechartExecutor(
            context=StatechartContext(world=_hsr_world_setup),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        orientation_goal = hand.tool_R_grasp.inverse()
        orientation_goal.reference_frame = _hsr_world_setup.get_body_by_name(
            "base_footprint"
        )
        motion_statechart.add_node(
            goal := Parallel(
                nodes=[
                    CartesianOrientation(
                        root_link=_hsr_world_setup.root,
                        tip_link=hand.tool_frame,
                        goal_orientation=orientation_goal,
                    ),
                    CartesianPosition(
                        root_link=_hsr_world_setup.root,
                        tip_link=hand.tool_frame,
                        goal_point=_hsr_world_setup.bodies[
                            -1
                        ].global_transform.position,
                    ),
                ]
            )
        )
        motion_statechart.add_node(EndMotion.when_true(goal))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        assert goal.observation_state == ObservationStateValues.TRUE

    def test_cart_goal_sequence_at_build(self, pr2_world_state_reset: World):
        """
        Test CartesianPose sequence with Bind_at_build policy.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        tip_goal1 = Pose.from_xyz_quaternion(pos_x=-2, reference_frame=tip)
        tip_goal2 = Pose.from_xyz_quaternion(pos_x=0.2, reference_frame=tip)

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        cart_goal1 = CartesianPose(
            root_link=root,
            tip_link=tip,
            goal_pose=tip_goal1,
        )
        motion_statechart.add_node(cart_goal1)

        cart_goal2 = CartesianPose(
            root_link=root,
            tip_link=tip,
            goal_pose=tip_goal2,
            binding_policy=GoalBindingPolicy.Bind_at_build,
        )
        motion_statechart.add_node(cart_goal2)

        cart_goal1.success_condition = cart_goal1.observes_true
        cart_goal2.start_condition = cart_goal1.is_succeeded

        motion_statechart.add_node(EndMotion.when_all_true([cart_goal1, cart_goal2]))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        forward_kinematics = pr2_world_state_reset.compute_forward_kinematics_np(
            root, tip
        )
        assert np.allclose(
            forward_kinematics, tip_goal2.to_np(), atol=cart_goal2.translation_threshold
        )

    def test_cart_goal_sequence_on_start(self, pr2_world_state_reset: World):
        """
        Test CartesianPose sequence with Bind_on_start policy (default).
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        tip_goal1 = Pose.from_xyz_quaternion(pos_x=-0.2, reference_frame=tip)
        tip_goal2 = Pose.from_xyz_quaternion(pos_x=0.2, reference_frame=tip)

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        cart_goal1 = CartesianPose(
            root_link=root,
            tip_link=tip,
            goal_pose=tip_goal1,
        )
        motion_statechart.add_node(cart_goal1)

        cart_goal2 = CartesianPose(
            root_link=root,
            tip_link=tip,
            goal_pose=tip_goal2,
        )
        motion_statechart.add_node(cart_goal2)

        cart_goal1.success_condition = cart_goal1.observes_true
        cart_goal2.start_condition = cart_goal1.is_succeeded

        motion_statechart.add_node(EndMotion.when_all_true([cart_goal1, cart_goal2]))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        forward_kinematics = pr2_world_state_reset.compute_forward_kinematics_np(
            root, tip
        )
        expected = np.eye(4)
        assert np.allclose(
            forward_kinematics, expected, atol=cart_goal2.translation_threshold
        )

    def test_CartesianOrientation(self, pr2_world_state_reset: World):
        """
        Test basic CartesianOrientation goal.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        tip_goal = RotationMatrix.from_axis_angle(
            AxisAngle(Vector3.Z(), 4.0, reference_frame=tip)
        )

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        cart_goal = CartesianOrientation(
            root_link=root,
            tip_link=tip,
            goal_orientation=tip_goal,
        )
        motion_statechart.add_node(cart_goal)
        end = EndMotion()
        motion_statechart.add_node(end)
        end.start_condition = cart_goal.observes_true

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        forward_kinematics = pr2_world_state_reset.compute_forward_kinematics_np(
            root, tip
        )
        assert np.allclose(
            forward_kinematics, tip_goal.to_np(), atol=cart_goal.threshold
        )

    def test_cartesian_position_sequence_at_build(self, pr2_world_state_reset: World):
        """
        Test CartesianPosition with Bind_at_build policy.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        tip_goal1 = Point3(-0.2, 0, 0, reference_frame=tip)
        tip_goal2 = Point3(0.2, 0, 0, reference_frame=tip)

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        cart_goal1 = CartesianPosition(
            root_link=root,
            tip_link=tip,
            goal_point=tip_goal1,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        motion_statechart.add_node(cart_goal1)

        cart_goal2 = CartesianPosition(
            root_link=root,
            tip_link=tip,
            goal_point=tip_goal2,
            binding_policy=GoalBindingPolicy.Bind_at_build,
        )
        motion_statechart.add_node(cart_goal2)

        cart_goal1.success_condition = cart_goal1.observes_true
        cart_goal2.start_condition = cart_goal1.is_succeeded

        motion_statechart.add_node(EndMotion.when_all_true([cart_goal1, cart_goal2]))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        forward_kinematics = pr2_world_state_reset.compute_forward_kinematics_np(
            root, tip
        )
        # goal2 was captured at build time, so should end at that absolute position
        expected = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=0.2, reference_frame=pr2_world_state_reset.root
        ).to_np()
        assert np.allclose(
            forward_kinematics[:3, 3], expected[:3, 3], atol=cart_goal2.threshold
        )

    def test_cartesian_position_sequence_on_start(self, pr2_world_state_reset: World):
        """
        Test CartesianPosition with Bind_on_start policy (default).
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        tip_goal1 = Point3(-0.2, 0, 0, reference_frame=tip)
        tip_goal2 = Point3(0.2, 0, 0, reference_frame=tip)

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        cart_goal1 = CartesianPosition(
            root_link=root,
            tip_link=tip,
            goal_point=tip_goal1,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        motion_statechart.add_node(cart_goal1)

        cart_goal2 = CartesianPosition(
            root_link=root,
            tip_link=tip,
            goal_point=tip_goal2,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        motion_statechart.add_node(cart_goal2)

        cart_goal1.success_condition = cart_goal1.observes_true
        cart_goal2.start_condition = cart_goal1.is_succeeded

        motion_statechart.add_node(EndMotion.when_all_true([cart_goal1, cart_goal2]))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        forward_kinematics = pr2_world_state_reset.compute_forward_kinematics_np(
            root, tip
        )
        # Both goals captured when tasks start, so should return near origin
        expected = np.eye(4)
        assert np.allclose(
            forward_kinematics[:3, 3], expected[:3, 3], atol=cart_goal2.threshold
        )

    def test_cartesian_position_with_sequence_node(self, pr2_world_state_reset: World):
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        tip_goal1 = Point3(-0.2, 0, 0, reference_frame=tip)
        tip_goal2 = Point3(0.2, 0, 0, reference_frame=tip)

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        cart_goal1 = CartesianPosition(
            root_link=root,
            tip_link=tip,
            goal_point=tip_goal1,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )

        cart_goal2 = CartesianPosition(
            root_link=root,
            tip_link=tip,
            goal_point=tip_goal2,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        motion_statechart.add_node(seq := Sequence(nodes=[cart_goal1, cart_goal2]))

        motion_statechart.add_node(EndMotion.when_true(seq))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        forward_kinematics = pr2_world_state_reset.compute_forward_kinematics_np(
            root, tip
        )
        # Both goals captured when tasks start, so should return near origin
        expected = np.eye(4)
        assert np.allclose(
            forward_kinematics[:3, 3], expected[:3, 3], atol=cart_goal2.threshold
        )

    def test_cartesian_orientation_sequence_at_build(
        self, pr2_world_state_reset: World
    ):
        """
        Test CartesianOrientation with Bind_at_build policy.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        # Store initial orientation for comparison
        initial_fk = pr2_world_state_reset.compute_forward_kinematics_np(root, tip)

        tip_rot1 = RotationMatrix.from_axis_angle(
            AxisAngle(Vector3.Z(), np.pi / 6, reference_frame=tip)
        )
        tip_rot2 = RotationMatrix.from_axis_angle(
            AxisAngle(Vector3.Z(), -np.pi / 6, reference_frame=tip)
        )

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        cart_goal1 = CartesianOrientation(
            root_link=root,
            tip_link=tip,
            goal_orientation=tip_rot1,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        motion_statechart.add_node(cart_goal1)

        cart_goal2 = CartesianOrientation(
            root_link=root,
            tip_link=tip,
            goal_orientation=tip_rot2,
            binding_policy=GoalBindingPolicy.Bind_at_build,
        )
        motion_statechart.add_node(cart_goal2)

        cart_goal1.success_condition = cart_goal1.observes_true
        cart_goal2.start_condition = cart_goal1.is_succeeded

        motion_statechart.add_node(EndMotion.when_all_true([cart_goal1, cart_goal2]))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        forward_kinematics = pr2_world_state_reset.compute_forward_kinematics_np(
            root, tip
        )

        # goal2 captured at build, so ends at -pi/6 from original
        expected = tip_rot2.to_np()
        assert np.allclose(forward_kinematics, expected, atol=cart_goal2.threshold)

    def test_cartesian_orientation_sequence_on_start(
        self, pr2_world_state_reset: World
    ):
        """
        Test CartesianOrientation with Bind_on_start policy (default).
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        tip_rot1 = RotationMatrix.from_axis_angle(
            AxisAngle(Vector3.Z(), np.pi / 6, reference_frame=tip)
        )
        tip_rot2 = RotationMatrix.from_axis_angle(
            AxisAngle(Vector3.Z(), -np.pi / 6, reference_frame=tip)
        )

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        cart_goal1 = CartesianOrientation(
            root_link=root,
            tip_link=tip,
            goal_orientation=tip_rot1,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        motion_statechart.add_node(cart_goal1)

        cart_goal2 = CartesianOrientation(
            root_link=root,
            tip_link=tip,
            goal_orientation=tip_rot2,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        motion_statechart.add_node(cart_goal2)

        cart_goal1.success_condition = cart_goal1.observes_true
        cart_goal2.start_condition = cart_goal1.is_succeeded

        motion_statechart.add_node(EndMotion.when_all_true([cart_goal1, cart_goal2]))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        forward_kinematics = pr2_world_state_reset.compute_forward_kinematics_np(
            root, tip
        )
        # Both goals captured when tasks start, so rotates +pi/6 then -pi/6 = back to origin
        expected = np.eye(4)
        assert np.allclose(forward_kinematics, expected, atol=cart_goal2.threshold)

    def test_cartesian_position_straight(self, pr2_world_state_reset: World):
        """
        Test CartesianPositionStraight basic functionality.

        Verifies that the tip reaches the goal and (ideally) follows a straight path.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        goal_point = Point3(0.1, 0, 0, reference_frame=tip)

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        cart_straight = CartesianPositionStraight(
            root_link=root,
            tip_link=tip,
            goal_point=goal_point,
            binding_policy=GoalBindingPolicy.Bind_on_start,
            threshold=0.015,
        )
        motion_statechart.add_node(cart_straight)
        end = EndMotion()
        motion_statechart.add_node(end)
        end.start_condition = cart_straight.observes_true

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        # Verify goal was achieved
        assert cart_straight.observation_state == ObservationStateValues.TRUE

    def test_cartesian_pose_straight(self, pr2_world_state_reset: World):
        """
        Test CartesianPositionStraight basic functionality.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        goal_pose = Pose.from_xyz_rpy(0.1, 2, 0, reference_frame=tip)

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        cart_straight = CartesianPoseStraight(
            root_link=root,
            tip_link=tip,
            goal_pose=goal_pose,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        motion_statechart.add_node(cart_straight)
        motion_statechart.add_node(EndMotion.when_true(cart_straight))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        # Verify task detected completion
        assert cart_straight.observation_state == ObservationStateValues.TRUE

        assert np.allclose(
            cart_straight.goal_pose.to_np(), goal_pose.to_np(), atol=0.015
        )

    def test_cartesian_pose_straight_holds_its_tasks_to_its_thresholds(
        self, pr2_world_state_reset: World
    ) -> None:
        """
        The position task must arrive within the translation threshold and the
        orientation task within the orientation threshold of the goal.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )
        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        goal = CartesianPoseStraight(
            root_link=root,
            tip_link=tip,
            goal_pose=Pose.from_xyz_rpy(0.1, 0, 0, reference_frame=tip),
            translation_threshold=0.03,
            orientation_threshold=0.2,
        )
        motion_statechart.add_node(goal)
        executor.compile(statechart=motion_statechart)

        position_task, orientation_task = goal.nodes
        assert position_task.threshold == goal.translation_threshold
        assert orientation_task.threshold == goal.orientation_threshold

    def test_straight_path_while_the_orientation_changes(
        self, pr2_world_state_reset: World
    ):
        """
        The tip must stay on the line to the goal while the very same goal rotates it.

        The orientation half of :class:`CartesianPoseStraight` runs in parallel with the
        position half, so a goal that also reorients the tip turns the tip frame while
        the straight line motion is under way.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )
        start = pr2_world_state_reset.compute_forward_kinematics_np(root, tip)[:3, 3]
        goal_pose = Pose.from_xyz_rpy(
            x=start[0] + 0.3,
            y=start[1],
            z=start[2],
            pitch=np.pi / 2,
            reference_frame=root,
        )

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        goal = CartesianPoseStraight(
            root_link=root,
            tip_link=tip,
            goal_pose=goal_pose,
        )
        motion_statechart.add_node(goal)
        motion_statechart.add_node(EndMotion.when_true(goal))

        executor.compile(statechart=motion_statechart)
        straight = next(
            node for node in goal.nodes if isinstance(node, CartesianPositionStraight)
        )
        path = record_tip_path(executor, straight, root, tip)

        line = StraightLine(start=path[0], end=goal_pose.to_np()[:3, 3])
        assert line.maximum_distance_from(path) <= STRAIGHT_LINE_TOLERANCE

    def test_orientation_goal_turns_as_fast_far_away_as_close_by(
        self, pr2_world_state_reset: World, orientation_speed_ratio_tolerance: float
    ):
        """
        The angular speed towards an orientation goal is set by its reference velocity,
        not by how far away the goal is.

        Capping the half-angle rows directly makes the angular speed grow with
        1/cos(angle/2), so a goal at 150 degrees would turn more than twice as fast as
        one at 30 degrees.
        """
        initial_state = pr2_world_state_reset.state._data.copy()
        peak_speeds = []
        for angle in (np.radians(30), np.radians(150)):
            pr2_world_state_reset.state._data[:] = initial_state
            pr2_world_state_reset.notify_state_change()
            peak_speeds.append(
                peak_angle_per_tick_towards_roll_goal(pr2_world_state_reset, angle)
            )

        assert peak_speeds[1] <= orientation_speed_ratio_tolerance * peak_speeds[0]

    def test_cartesian_position_straight_weighs_its_rows_by_its_weight(
        self, cylinder_bot_world: World
    ):
        """
        Progress along the line costs the task's weight, and leaving the line costs that
        weight times the lateral weight factor.
        """
        task = CartesianPositionStraight(
            root_link=cylinder_bot_world.root,
            tip_link=cylinder_bot_world.get_kinematic_structure_entity_by_name("bot"),
            goal_point=Point3(x=1, reference_frame=cylinder_bot_world.root),
            weight=DefaultWeights.WEIGHT_COLLISION_AVOIDANCE,
            name="straight",
        )

        add_to_statechart_with_motion_control(task, cylinder_bot_world)
        artifacts = task.build()

        weights = {
            constraint.name: constraint.quadratic_weight
            for constraint in artifacts.constraints.equality_constraints
        }
        lateral_weight = task.weight * task.LATERAL_WEIGHT_FACTOR
        assert weights == {
            "line/progress": task.weight,
            "line/y": lateral_weight,
            "line/z": lateral_weight,
        }

    def test_cartesian_position_moves_straight_towards_a_diagonal_goal(
        self, pr2_world_state_reset: World
    ):
        """
        A point goal whose direction is not aligned with an axis of the root frame must
        still be approached along the straight line to it.

        Capping each axis of the error on its own would point the motion along the
        diagonal of the cap instead.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )
        start = pr2_world_state_reset.compute_forward_kinematics_np(root, tip)[:3, 3]
        goal_point = Point3(
            start[0] + 0.25, start[1] + 0.15, start[2] + 0.1, reference_frame=root
        )

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        position_goal = CartesianPosition(
            root_link=root, tip_link=tip, goal_point=goal_point
        )
        motion_statechart.add_node(position_goal)
        motion_statechart.add_node(EndMotion.when_true(position_goal))
        executor.compile(statechart=motion_statechart)
        path = record_tip_path(executor, position_goal, root, tip)

        line = StraightLine(start=path[0], end=goal_point.to_np()[:3])
        assert line.maximum_distance_from(path) <= REAIMED_LINE_TOLERANCE

    def test_straight_line_starts_where_the_task_starts(
        self, pr2_world_state_reset: World
    ):
        """
        The line must start at the pose the tip has when the task starts running, not at
        the pose it had when the statechart was compiled.

        Every node is built during compilation, while a task that waits for another one
        starts after that motion has moved the tip somewhere else.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )
        start = pr2_world_state_reset.compute_forward_kinematics_np(root, tip)[:3, 3]
        goal_point = Point3(start[0] + 0.2, start[1], start[2], reference_frame=root)

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        wrist_goal = JointPositionList(
            goal_state=JointState.from_str_dict(
                {PR2Joint.RIGHT_WRIST_FLEX: -np.pi / 2},
                world=pr2_world_state_reset,
            )
        )
        straight = CartesianPositionStraight(
            root_link=root,
            tip_link=tip,
            goal_point=goal_point,
        )
        motion_statechart.add_nodes([wrist_goal, straight])
        wrist_goal.success_condition = wrist_goal.observes_true
        straight.start_condition = wrist_goal.observes_true
        motion_statechart.add_node(EndMotion.when_true(straight))

        executor.compile(statechart=motion_statechart)
        path = record_tip_path(executor, straight, root, tip)

        line = StraightLine(start=path[0], end=goal_point.to_np()[:3])
        assert line.maximum_distance_from(path) <= STRAIGHT_LINE_TOLERANCE

    def test_soft_trunk_cartesian_position(self):
        """
        Verifies that Giskardpy can solve and execute a CartesianPosition task for the
        procedurally built Piecewise Constant Curvature SoftTrunk robot.
        """
        from semantic_digital_twin.datastructures.soft_trunk import (
            SoftTrunk,
            SoftTrunkSection,
        )

        world = World()
        # Define 3 identical sections of 0.3m, 0.02m radius, and 10 resolution
        sections = [SoftTrunkSection(length=0.3, radius=0.02, resolution=10)] * 3
        trunk = SoftTrunk.build_piecewise_constant_curvature(world, sections)

        # Define a reachable Cartesian target point relative to the base root
        goal_point = Point3(0.3, 0.0, 0.6, reference_frame=world.root)

        executor = StatechartExecutor(
            context=StatechartContext(world=world), extensions=[MotionControl()]
        )
        statechart = Statechart(context=executor.context)
        goal = CartesianPosition(
            root_link=world.root,
            tip_link=trunk.arms[0].tip,
            goal_point=goal_point,
        )
        statechart.add_node(goal)
        statechart.add_node(EndMotion.when_true(goal))

        executor.compile(statechart=statechart)
        executor.tick_until_end()

        # Retrieve the final tip pose
        tip_body = trunk.arms[0].tip
        fk = world.compute_forward_kinematics_np(world.root, tip_body)

        # Verify that the tip reached the target point within the goal threshold
        actual_position = fk[:3, 3]
        target_position = goal_point.to_np()[:3]
        distance_error = np.linalg.norm(actual_position - target_position)

        assert distance_error <= goal.threshold


# %% driving a differential drive


@dataclass
class PositionWhileRunning(TickingExecutorExtension):
    """
    Records where a body is after every tick in which a node is running.
    """

    body: Body = field(kw_only=True)
    """
    The body whose position is recorded.
    """

    node: StatechartNode | None = field(default=None, kw_only=True)
    """
    The node during whose run the position is recorded.
    """

    positions: list[np.ndarray] = field(default_factory=list, init=False)
    """
    The positions of :attr:`body` relative to the world root, one per tick.
    """

    def after_tick(self, executor: StatechartExecutor) -> None:
        if self.node.life_cycle_state != LifeCycleValues.RUNNING:
            return
        world = executor.context.world
        self.positions.append(
            world.compute_forward_kinematics_np(world.root, self.body)[:3, 3]
        )


class TestDiffDriveBaseGoal:
    @pytest.mark.parametrize(
        "goal_pose",
        [
            Pose.from_xyz_rpy(x=0.489, y=-0.598, z=0.000),
            Pose.from_xyz_quaternion(
                pos_x=-0.026,
                pos_y=0.569,
                pos_z=0.0,
                quat_x=0.0,
                quat_y=0.0,
                quat_z=0.916530200374776,
                quat_w=0.3999654882623912,
            ),
            Pose.from_xyz_rpy(x=1, y=1, yaw=np.pi / 4),
            Pose.from_xyz_rpy(x=2, y=0, yaw=-np.pi / 4),
            Pose.from_xyz_rpy(yaw=-np.pi / 4),
            Pose.from_xyz_rpy(x=-1, y=-1, yaw=np.pi / 4),
            Pose.from_xyz_rpy(x=-2, y=-1, yaw=-np.pi / 4),
            Pose.from_xyz_rpy(x=0.01, y=0.5, yaw=np.pi / 8),
            Pose.from_xyz_rpy(x=-0.01, y=-0.5, yaw=np.pi / 5),
            Pose.from_xyz_rpy(x=1.1, y=2.0, yaw=-np.pi),
            Pose.from_xyz_rpy(y=1),
        ],
    )
    def test_drive(
        self,
        cylinder_bot_diff_world,
        goal_pose: Pose,
    ):
        bot = cylinder_bot_diff_world.get_body_by_name("bot")
        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_diff_world),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        goal_pose.reference_frame = cylinder_bot_diff_world.root
        motion_statechart.add_node(
            goal := DifferentialDriveBaseGoal(
                mobile_base=cylinder_bot_diff_world.get_semantic_annotations_by_type(
                    MobileBase
                )[0],
                goal_pose=goal_pose,
            )
        )
        motion_statechart.add_node(EndMotion.when_true(goal))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        assert np.allclose(
            cylinder_bot_diff_world.compute_forward_kinematics(
                cylinder_bot_diff_world.root, bot
            ),
            goal_pose,
            atol=1e-2,
        )

    def test_custom_threshold_applies_to_every_task(
        self, cylinder_bot_diff_world
    ) -> None:
        """
        DifferentialDriveBaseGoal exposes a single ``threshold`` field for its callers,
        so it must reach the position and the orientation tasks of every step --
        otherwise a caller raising ``threshold`` relaxes only some of the tolerances.
        """
        goal_pose = Pose.from_xyz_rpy(
            x=1, y=1, yaw=np.pi / 4, reference_frame=cylinder_bot_diff_world.root
        )
        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_diff_world),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_node(
            goal := DifferentialDriveBaseGoal(
                mobile_base=cylinder_bot_diff_world.get_semantic_annotations_by_type(
                    MobileBase
                )[0],
                goal_pose=goal_pose,
                threshold=0.3,
            )
        )
        motion_statechart.add_node(EndMotion.when_true(goal))

        executor.compile(statechart=motion_statechart)

        tasks = [node for node in goal.descendants if isinstance(node, DrivenBaseTask)]
        assert {task.threshold for task in tasks} == {goal.threshold}

    def test_second_goal_drives_from_where_the_first_one_ended(
        self, cylinder_bot_diff_world
    ):
        """
        Every goal in a statechart is expanded before any of them runs, so a goal that
        reads the base pose while expanding reads where the base started rather than
        where its own leg begins.

        The two legs turn a corner, so the direction the second leg has to drive in
        differs from the direction it would have had from the start pose -- a heading
        taken at expansion time asks a differential drive to translate sideways, which
        it cannot do.
        """
        first_goal_pose = Pose.from_xyz_rpy(
            x=1, y=0, reference_frame=cylinder_bot_diff_world.root
        )
        second_goal_pose = Pose.from_xyz_rpy(
            x=1, y=1, reference_frame=cylinder_bot_diff_world.root
        )
        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_diff_world),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_node(
            first_leg := DifferentialDriveBaseGoal(
                mobile_base=cylinder_bot_diff_world.get_semantic_annotations_by_type(
                    MobileBase
                )[0],
                goal_pose=first_goal_pose,
            )
        )
        motion_statechart.add_node(
            second_leg := DifferentialDriveBaseGoal(
                mobile_base=cylinder_bot_diff_world.get_semantic_annotations_by_type(
                    MobileBase
                )[0],
                goal_pose=second_goal_pose,
            )
        )
        second_leg.start_condition = first_leg.observes_true
        motion_statechart.add_node(EndMotion.when_true(second_leg))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        assert np.allclose(
            cylinder_bot_diff_world.compute_forward_kinematics(
                cylinder_bot_diff_world.root,
                cylinder_bot_diff_world.get_body_by_name("bot"),
            ),
            second_goal_pose,
            atol=1e-2,
        )

    @pytest.mark.parametrize(
        "binding_policy, expected_x",
        [
            (GoalBindingPolicy.Bind_on_start, 1.5),
            (GoalBindingPolicy.Bind_at_build, 0.5),
        ],
    )
    def test_a_goal_relative_to_a_moving_frame_is_bound_by_its_policy(
        self, cylinder_bot_diff_world, binding_policy, expected_x
    ) -> None:
        """
        The second leg is given relative to the base, which the first leg moves before
        the second starts, so where the second leg ends depends on when its goal frame
        is bound.
        """
        bot = cylinder_bot_diff_world.get_body_by_name("bot")
        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_diff_world),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_node(
            first_leg := DifferentialDriveBaseGoal(
                mobile_base=cylinder_bot_diff_world.get_semantic_annotations_by_type(
                    MobileBase
                )[0],
                goal_pose=Pose.from_xyz_rpy(
                    x=1, reference_frame=cylinder_bot_diff_world.root
                ),
            )
        )
        motion_statechart.add_node(
            second_leg := DifferentialDriveBaseGoal(
                mobile_base=cylinder_bot_diff_world.get_semantic_annotations_by_type(
                    MobileBase
                )[0],
                goal_pose=Pose.from_xyz_rpy(x=0.5, reference_frame=bot),
                binding_policy=binding_policy,
            )
        )
        second_leg.start_condition = first_leg.observes_true
        motion_statechart.add_node(EndMotion.when_true(second_leg))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        assert np.allclose(
            cylinder_bot_diff_world.compute_forward_kinematics_np(
                cylinder_bot_diff_world.root, bot
            ),
            Pose.from_xyz_rpy(x=expected_x).to_np(),
            atol=1e-2,
        )

    def test_a_base_moved_onto_a_lift_drives_relative_to_the_lift(
        self, cylinder_bot_diff_world
    ) -> None:
        """
        The base is moved onto the lift and the lift is raised while the statechart
        runs, after the drive goal was built, so the goal only arrives if it drives
        relative to the drive's parent as it is then, and leaves the lift where it is.
        """
        world = cylinder_bot_diff_world
        base = world.get_connections_by_type(DifferentialDrive)[0].child
        lift = Body(name=PrefixedName("lift"))
        with world.modify_world():
            lift_connection = PrismaticConnection.create_with_dofs(
                world=world,
                parent=world.root,
                child=lift,
                axis=Vector3.Z(),
                dof_limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                    maximum_speed=1
                ),
            )
            world.add_connection(lift_connection)
        lift_connection.has_hardware_interface = True
        lift_height = 1.0
        executor = StatechartExecutor(
            context=StatechartContext(world=world), extensions=[MotionControl()]
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_node(
            ride := Sequence(
                nodes=[
                    MoveBranch(branch_root=base, new_parent=lift),
                    raising := JointPositionList(
                        goal_state=JointState.from_mapping(
                            {lift_connection: lift_height}
                        )
                    ),
                    DifferentialDriveBaseGoal(
                        mobile_base=world.get_semantic_annotations_by_type(MobileBase)[
                            0
                        ],
                        goal_pose=Pose.from_xyz_rpy(x=1, reference_frame=lift),
                    ),
                ]
            )
        )
        motion_statechart.add_node(EndMotion.when_true(ride))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        assert lift_connection.position == pytest.approx(
            lift_height, abs=raising.threshold
        )
        assert np.allclose(
            world.compute_forward_kinematics_np(lift, base),
            Pose.from_xyz_rpy(x=1).to_np(),
            atol=1e-2,
        )

    def test_a_goal_bound_to_the_base_stays_where_it_was_bound_when_the_base_is_moved(
        self, cylinder_bot_diff_world
    ) -> None:
        """
        The goal is given relative to the base and bound when the drive starts, so
        moving the base under another parent halfway there must not carry the goal along
        with the base.
        """
        world = cylinder_bot_diff_world
        base = world.get_connections_by_type(DifferentialDrive)[0].child
        platform = Body(name=PrefixedName("platform"))
        with world.modify_world():
            world.add_connection(FixedConnection(parent=world.root, child=platform))
        executor = StatechartExecutor(
            context=StatechartContext(world=world), extensions=[MotionControl()]
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_node(
            drive := DifferentialDriveBaseGoal(
                mobile_base=world.get_semantic_annotations_by_type(MobileBase)[0],
                goal_pose=Pose.from_xyz_rpy(x=1, reference_frame=base),
            )
        )
        motion_statechart.add_node(
            halfway := PositionReached(
                root_link=world.root,
                tip_link=base,
                goal_point=Point3(x=0.5, reference_frame=world.root),
                threshold=0.1,
            )
        )
        motion_statechart.add_node(
            moving := MoveBranch(branch_root=base, new_parent=platform)
        )
        moving.start_condition = halfway.observes_true
        motion_statechart.add_node(EndMotion.when_all_true([drive, moving]))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        assert base.parent_kinematic_structure_entity is platform
        assert np.allclose(
            world.compute_forward_kinematics_np(world.root, base),
            Pose.from_xyz_rpy(x=1).to_np(),
            atol=2 * drive.threshold,
        )

    def test_the_first_step_only_turns_the_base(self, cylinder_bot_diff_world) -> None:
        """
        The goal lies to the side of the base, so the base could face it sooner by also
        driving in an arc; the first step must only turn it.
        """
        world = cylinder_bot_diff_world
        base = world.get_connections_by_type(DifferentialDrive)[0].child
        start = world.compute_forward_kinematics_np(world.root, base)[:3, 3]
        recording = PositionWhileRunning(body=base)
        executor = StatechartExecutor(
            context=StatechartContext(world=world),
            extensions=[MotionControl(), recording],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_node(
            drive := DifferentialDriveBaseGoal(
                mobile_base=world.get_semantic_annotations_by_type(MobileBase)[0],
                goal_pose=Pose.from_xyz_rpy(y=1, reference_frame=world.root),
            )
        )
        recording.node = drive.sequence.nodes[0]
        motion_statechart.add_node(EndMotion.when_true(drive))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        assert recording.positions
        np.testing.assert_allclose(
            recording.positions, [start] * len(recording.positions), atol=1e-3
        )


class TestVelocityTasks:
    def _build_statechart(
        self, executor: StatechartExecutor, goal_node, limit_node
    ) -> Statechart:
        """
        Build a small MSC: goal_node -> limit_node -> EndMotion(when_true=goal_node)
        Returns the Statechart but does not compile or run it.
        """
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_node(goal_node)
        motion_statechart.add_node(limit_node)
        motion_statechart.add_node(EndMotion.when_true(goal_node))
        return motion_statechart

    def _compile_statechart_and_run_until_end(
        self, world: World, goal_node, limit_node
    ):
        """
        Build the MSC (no extra nodes), compile into a StatechartExecutor, run until end
        and return (control_cycles, executor)
        """
        executor = StatechartExecutor(
            context=StatechartContext(world=world), extensions=[MotionControl()]
        )
        motion_statechart = self._build_statechart(
            executor=executor, goal_node=goal_node, limit_node=limit_node
        )
        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()
        return executor.tick_count, executor

    @pytest.mark.parametrize(
        "goal_type, limit_cls",
        [
            ("position", CartesianVelocityLimit),
            ("position", CartesianPositionVelocityLimit),
            ("rotation", CartesianVelocityLimit),
            ("rotation", CartesianRotationVelocityLimit),
        ],
        ids=["pos/generic", "pos/position-only", "rot/generic", "rot/rotation-only"],
    )
    def test_observation_variable(
        self, pr2_world_state_reset: World, goal_type: str, limit_cls: type
    ):
        """
        Tests that velocity limit's observation variable can trigger a CancelStatechart
        when the optimizer chooses to violate the limit.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        if goal_type == "position":
            goal = CartesianPosition(
                root_link=root,
                tip_link=tip,
                goal_point=Point3(1, 0, 0, reference_frame=tip),
                weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
            )
        else:
            goal = CartesianOrientation(
                root_link=root,
                tip_link=tip,
                goal_orientation=RotationMatrix.from_rpy(
                    yaw=np.pi / 2, reference_frame=tip
                ),
                weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
            )

        low_weight_limit = limit_cls(
            root_link=root,
            tip_link=tip,
            weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
        )
        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        motion_statechart = self._build_statechart(
            executor=executor, goal_node=goal, limit_node=low_weight_limit
        )
        cancel_motion = CancelStatechart(exception=Exception("test"))
        cancel_motion.start_condition = low_weight_limit.observes_false
        motion_statechart.add_node(cancel_motion)

        executor.compile(statechart=motion_statechart)

        with pytest.raises(Exception):
            executor.tick_until_end()

    @pytest.mark.parametrize(
        "limit_cls",
        [CartesianVelocityLimit, CartesianPositionVelocityLimit],
        ids=["generic_linear", "position_only_linear"],
    )
    def test_cartesian_position_velocity_limit(
        self, pr2_world_state_reset: World, limit_cls: type
    ):
        """
        Position velocity limit: check slower limit increases cycles.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        point = Point3(1, 0, 0, reference_frame=tip)
        position_goal = CartesianPosition(
            root_link=root, tip_link=tip, goal_point=point
        )

        usual_limit = limit_cls(root_link=root, tip_link=tip, max_linear_velocity=0.1)
        half_velocity_limit = limit_cls(
            root_link=root,
            tip_link=tip,
            max_linear_velocity=(usual_limit.max_linear_velocity / 2.1),
        )

        loose_cycles, _ = self._compile_statechart_and_run_until_end(
            world=pr2_world_state_reset, goal_node=position_goal, limit_node=usual_limit
        )
        tight_cycles, _ = self._compile_statechart_and_run_until_end(
            world=pr2_world_state_reset,
            goal_node=position_goal,
            limit_node=half_velocity_limit,
        )

        assert tight_cycles >= 2 * loose_cycles, (
            f"tight ({tight_cycles}) should take >= loose ({2 * loose_cycles}) control cycles"
        )

    @pytest.mark.parametrize(
        "limit_cls",
        [CartesianVelocityLimit, CartesianRotationVelocityLimit],
        ids=["generic_angular", "rotation_only_angular"],
    )
    def test_cartesian_rotation_velocity_limit(
        self, pr2_world_state_reset: World, limit_cls: type
    ):
        """
        Rotation velocity limit: check slower limit increases cycles.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        rotation = RotationMatrix.from_rpy(yaw=np.pi / 2, reference_frame=tip)
        orientation = CartesianOrientation(
            root_link=root, tip_link=tip, goal_orientation=rotation
        )

        usual_limit = limit_cls(root_link=root, tip_link=tip, max_angular_velocity=0.3)
        half_velocity_limit = limit_cls(
            root_link=root,
            tip_link=tip,
            max_angular_velocity=(usual_limit.max_angular_velocity / 2.1),
        )

        loose_cycles, _ = self._compile_statechart_and_run_until_end(
            world=pr2_world_state_reset, goal_node=orientation, limit_node=usual_limit
        )
        tight_cycles, _ = self._compile_statechart_and_run_until_end(
            world=pr2_world_state_reset,
            goal_node=orientation,
            limit_node=half_velocity_limit,
        )

        assert tight_cycles >= 2 * loose_cycles, (
            f"tight ({tight_cycles}) should take >= loose ({2 * loose_cycles}) control cycles"
        )


class TestDebugExpressions:
    """
    Cartesian tasks register a goal and a current debug expression, named with the task
    name and colored green (goal) and red (current).
    """

    def test_cartesian_position(self, cylinder_bot_world: World):
        root = cylinder_bot_world.root
        tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        task = CartesianPosition(
            root_link=root,
            tip_link=tip,
            goal_point=Point3(x=1, reference_frame=root),
            name="cart_pos",
        )

        add_to_statechart_with_motion_control(task, cylinder_bot_world)
        artifacts = task.build()

        goal = debug_expression_by_name(artifacts.debug_expressions, "cart_pos/goal")
        current = debug_expression_by_name(
            artifacts.debug_expressions, "cart_pos/current"
        )
        assert isinstance(goal.expression, Point3)
        assert isinstance(current.expression, Point3)
        assert goal.color == GOAL_COLOR
        assert current.color == CURRENT_COLOR

    def test_cartesian_position_straight(self, cylinder_bot_world: World):
        root = cylinder_bot_world.root
        tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        task = CartesianPositionStraight(
            root_link=root,
            tip_link=tip,
            goal_point=Point3(x=1, reference_frame=root),
            name="straight",
        )

        add_to_statechart_with_motion_control(task, cylinder_bot_world)
        artifacts = task.build()

        goal = debug_expression_by_name(artifacts.debug_expressions, "straight/goal")
        current = debug_expression_by_name(
            artifacts.debug_expressions, "straight/current"
        )
        assert isinstance(goal.expression, Point3)
        assert isinstance(current.expression, Point3)
        assert goal.color == GOAL_COLOR
        assert current.color == CURRENT_COLOR

    def test_cartesian_orientation(self, cylinder_bot_world: World):
        root = cylinder_bot_world.root
        tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        task = CartesianOrientation(
            root_link=root,
            tip_link=tip,
            goal_orientation=RotationMatrix.from_rpy(
                yaw=np.pi / 2, reference_frame=root
            ),
            name="orient",
        )

        add_to_statechart_with_motion_control(task, cylinder_bot_world)
        artifacts = task.build()

        goal = debug_expression_by_name(artifacts.debug_expressions, "orient/goal")
        current = debug_expression_by_name(
            artifacts.debug_expressions, "orient/current"
        )
        assert isinstance(goal.expression, RotationMatrix)
        assert isinstance(current.expression, RotationMatrix)
        assert goal.color == GOAL_COLOR
        assert current.color == CURRENT_COLOR

    def test_cartesian_position_trajectory(self, cylinder_bot_world: World):
        root = cylinder_bot_world.root
        tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        task = CartesianPositionTrajectory(
            root_link=root,
            tip_link=tip,
            goal_points=[
                Point3(x=0, reference_frame=root),
                Point3(x=1, reference_frame=root),
            ],
            name="traj",
        )

        add_to_statechart_with_motion_control(task, cylinder_bot_world)
        artifacts = task.build()

        goal = debug_expression_by_name(artifacts.debug_expressions, "traj/goal")
        current = debug_expression_by_name(artifacts.debug_expressions, "traj/current")
        assert isinstance(goal.expression, Point3)
        assert isinstance(current.expression, Point3)
        assert goal.color == GOAL_COLOR
        assert current.color == CURRENT_COLOR

    def test_cartesian_pose_uses_prefixed_names(self, cylinder_bot_world: World):
        """
        CartesianPose is a Parallel over a position and an orientation task, so its
        debug expressions are registered by those children, each prefixed with its own
        name.
        """
        root = cylinder_bot_world.root
        tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        task = CartesianPose(
            root_link=root,
            tip_link=tip,
            goal_pose=Pose.from_xyz_rpy(x=1, reference_frame=root),
            name="pose",
        )
        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_world),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_node(task)
        motion_statechart.add_node(EndMotion.when_true(task))

        executor.compile(statechart=motion_statechart)

        names = {
            debug_expression.name
            for child in task.nodes
            for debug_expression in child.debug_expressions
        }
        assert names == {
            "pose/position/goal",
            "pose/position/current",
            "pose/orientation/goal",
            "pose/orientation/current",
        }


# %% two links a distance apart along an axis


class TestDistanceAlongAxis:
    def test_points_on_two_links_end_the_distance_apart_along_the_axis(
        self, prismatic_bot2: World
    ) -> None:
        reference = prismatic_bot2.get_body_by_name("robot")
        tip = prismatic_bot2.get_body_by_name("robot2")
        executor = StatechartExecutor(
            context=StatechartContext(world=prismatic_bot2),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_node(
            goal := DistanceAlongAxis(
                root_link=prismatic_bot2.root,
                reference_point=Point3(reference_frame=reference),
                tip_point=Point3(z=0.1, reference_frame=tip),
                axis=Vector3.Z(reference_frame=prismatic_bot2.root),
                distance=0.3,
                threshold=0.001,
            )
        )
        motion_statechart.add_node(EndMotion.when_true(goal))

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end()

        root_P_reference, root_P_tip = (
            prismatic_bot2.transform(
                target_frame=prismatic_bot2.root, spatial_object=point
            ).to_np()
            for point in (goal.reference_point, goal.tip_point)
        )
        assert root_P_tip[2] - root_P_reference[2] == pytest.approx(
            goal.distance, abs=goal.threshold
        )
