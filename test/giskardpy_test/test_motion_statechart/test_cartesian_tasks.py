from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from giskardpy.executor import Executor
from giskardpy.motion_statechart.binding_policy import GoalBindingPolicy
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    ObservationStateValues,
    DefaultWeights,
    LifeCycleValues,
)
from giskardpy.motion_statechart.goals.cartesian_goals import (
    DifferentialDriveBaseGoal,
    CartesianPoseStraight,
)
from giskardpy.motion_statechart.goals.templates import Sequence, Parallel
from giskardpy.motion_statechart.graph_node import (
    EndMotion,
    CancelMotion,
    MotionStatechartNode,
)
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetSeedConfiguration,
)
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianOrientation,
    CartesianPosition,
    CartesianPositionStraight,
    CartesianVelocityLimit,
    CartesianPositionVelocityLimit,
    CartesianRotationVelocityLimit,
    CartesianPositionTrajectory,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from krrood.symbolic_math.symbolic_math import (
    trinary_logic_and,
    trinary_logic_not,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
    Point3,
    RotationMatrix,
)
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
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

# %% straight line paths

STRAIGHT_LINE_TOLERANCE = 0.02
"""
How far the tip may stray from the line it is supposed to travel along, in meters.

Some deviation is unavoidable: the controller has to accelerate out of the pose the
motion starts at, and it can only correct a lateral offset on the tick after it appeared.
"""


@dataclass
class StraightLine:
    """
    The line a straight Cartesian motion is supposed to travel along.
    """

    start: np.ndarray
    """Position the motion starts at."""

    end: np.ndarray
    """Position the motion ends at."""

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


def record_tip_path(
    executor: Executor,
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
    life_cycle_state = executor.motion_statechart.life_cycle_state

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
        if executor.motion_statechart.is_end_motion():
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
                .to_position()
                .evaluate()[:-1]
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
        motion_statechart = MotionStatechart()
        cartesian_trajectory = CartesianPositionTrajectory(
            root_link=cylinder_bot_world.root,
            tip_link=cylinder_bot_world.get_kinematic_structure_entity_by_name("bot"),
            goal_points=points,
        )
        motion_statechart.add_node(cartesian_trajectory)
        motion_statechart.add_node(EndMotion.when_true(cartesian_trajectory))

        executor = Executor(
            context=MotionStatechartContext(
                world=cylinder_bot_world,
            ),
            trajectory_plotter=WorldStateTrajectoryPlotter(),
        )
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()
        self.compare_trajectories(
            points,
            executor.trajectory_plotter.world_state_trajectory,
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
        motion_statechart = MotionStatechart()
        cartesian_trajectory = CartesianPositionTrajectory(
            root_link=cylinder_bot_world.root,
            tip_link=cylinder_bot_world.get_kinematic_structure_entity_by_name("bot"),
            goal_points=points,
            maximum_skip_ahead=20,
        )
        motion_statechart.add_node(cartesian_trajectory)
        motion_statechart.add_node(EndMotion.when_true(cartesian_trajectory))

        executor = Executor(
            context=MotionStatechartContext(
                world=cylinder_bot_world,
            ),
            trajectory_plotter=WorldStateTrajectoryPlotter(),
        )
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()
        self.compare_trajectories(
            points,
            executor.trajectory_plotter.world_state_trajectory,
            cartesian_trajectory.root_link,
            cartesian_trajectory.tip_link,
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

        SetSeedConfiguration(
            seed_configuration=JointState.from_str_dict(
                better_pr2_pose, world=pr2_world_state_reset
            )
        ).on_start(MotionStatechartContext(world=pr2_world_state_reset))

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
        motion_statechart = MotionStatechart()

        motion_statechart.add_node(
            cartesian_trajectory := CartesianPositionTrajectory(
                root_link=root,
                tip_link=tip,
                goal_points=points,
            )
        )
        motion_statechart.add_node(EndMotion.when_true(cartesian_trajectory))

        executor = Executor(
            context=MotionStatechartContext(
                world=pr2_world_state_reset,
            ),
            trajectory_plotter=WorldStateTrajectoryPlotter(),
        )
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()
        self.compare_trajectories(
            root_points,
            executor.trajectory_plotter.world_state_trajectory,
            cartesian_trajectory.root_link,
            cartesian_trajectory.tip_link,
        )


class TestCartesianTasks:
    """
    Test suite for all Cartesian motion tasks.
    """

    def test_simple_cartesian_pose(self, cylinder_bot_world: World):
        tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")

        motion_statechart = MotionStatechart()
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

        executor = Executor(MotionStatechartContext(world=cylinder_bot_world))
        executor.compile(motion_statechart=motion_statechart)
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

        motion_statechart = MotionStatechart()
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

        executor = Executor(MotionStatechartContext(world=cylinder_bot_world))
        executor.compile(motion_statechart=motion_statechart)
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

        motion_statechart = MotionStatechart()
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

        executor = Executor(MotionStatechartContext(world=cylinder_bot_world))
        executor.compile(motion_statechart=motion_statechart)

        goal_reached_tick = None
        for i in range(1000):
            executor.tick()
            if (
                goal_reached_tick is None
                and motion_statechart.observation_state[goal]
                == ObservationStateValues.TRUE
            ):
                goal_reached_tick = i
            if motion_statechart.is_end_motion():
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
        motion_statechart = MotionStatechart()
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

        executor = Executor(
            MotionStatechartContext(
                world=pr2_world_state_reset,
            )
        )
        executor.compile(motion_statechart=motion_statechart)
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

        motion_statechart = MotionStatechart()

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

        executor = Executor(
            MotionStatechartContext(
                world=pr2_world_state_reset,
            )
        )
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()

        assert np.allclose(
            executor.context.world.compute_forward_kinematics(root, tip),
            expected,
            atol=cart_goal.translation_threshold,
        )

    def test_front_facing_orientation(self, _hsr_world_setup: World):
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
        motion_statechart = MotionStatechart()
        orientation_goal = hand.front_facing_orientation.to_rotation_matrix()
        orientation_goal.reference_frame = _hsr_world_setup.get_body_by_name(
            "base_footprint"
        )
        motion_statechart.add_node(
            goal := Parallel(
                [
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
                        ].global_transform.to_position(),
                    ),
                ]
            )
        )
        motion_statechart.add_node(EndMotion.when_true(goal))

        executor = Executor(MotionStatechartContext(world=_hsr_world_setup))
        executor.compile(motion_statechart=motion_statechart)
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

        motion_statechart = MotionStatechart()
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

        cart_goal1.end_condition = cart_goal1.observation_variable
        cart_goal2.start_condition = cart_goal1.observation_variable

        end = EndMotion()
        motion_statechart.add_node(end)
        end.start_condition = trinary_logic_and(
            cart_goal1.observation_variable, cart_goal2.observation_variable
        )

        executor = Executor(
            MotionStatechartContext(
                world=pr2_world_state_reset,
            )
        )

        executor.compile(motion_statechart=motion_statechart)
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

        motion_statechart = MotionStatechart()
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

        cart_goal1.end_condition = cart_goal1.observation_variable
        cart_goal2.start_condition = cart_goal1.observation_variable

        end = EndMotion()
        motion_statechart.add_node(end)
        end.start_condition = trinary_logic_and(
            cart_goal1.observation_variable, cart_goal2.observation_variable
        )

        executor = Executor(
            MotionStatechartContext(
                world=pr2_world_state_reset,
            )
        )
        executor.compile(motion_statechart=motion_statechart)
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

        tip_goal = RotationMatrix.from_axis_angle(Vector3.Z(), 4.0, reference_frame=tip)

        motion_statechart = MotionStatechart()
        cart_goal = CartesianOrientation(
            root_link=root,
            tip_link=tip,
            goal_orientation=tip_goal,
        )
        motion_statechart.add_node(cart_goal)
        end = EndMotion()
        motion_statechart.add_node(end)
        end.start_condition = cart_goal.observation_variable

        executor = Executor(
            MotionStatechartContext(
                world=pr2_world_state_reset,
            )
        )
        executor.compile(motion_statechart=motion_statechart)
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

        motion_statechart = MotionStatechart()
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

        cart_goal1.end_condition = cart_goal1.observation_variable
        cart_goal2.start_condition = cart_goal1.observation_variable

        end = EndMotion()
        motion_statechart.add_node(end)
        end.start_condition = trinary_logic_and(
            cart_goal1.observation_variable, cart_goal2.observation_variable
        )

        executor = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        executor.compile(motion_statechart=motion_statechart)
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

        motion_statechart = MotionStatechart()
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

        cart_goal1.end_condition = cart_goal1.observation_variable
        cart_goal2.start_condition = cart_goal1.observation_variable

        end = EndMotion()
        motion_statechart.add_node(end)
        end.start_condition = trinary_logic_and(
            cart_goal1.observation_variable, cart_goal2.observation_variable
        )

        executor = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        executor.compile(motion_statechart=motion_statechart)
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

        motion_statechart = MotionStatechart()
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

        executor = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        executor.compile(motion_statechart=motion_statechart)
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
            Vector3.Z(), np.pi / 6, reference_frame=tip
        )
        tip_rot2 = RotationMatrix.from_axis_angle(
            Vector3.Z(), -np.pi / 6, reference_frame=tip
        )

        motion_statechart = MotionStatechart()
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

        cart_goal1.end_condition = cart_goal1.observation_variable
        cart_goal2.start_condition = cart_goal1.observation_variable

        end = EndMotion()
        motion_statechart.add_node(end)
        end.start_condition = trinary_logic_and(
            cart_goal1.observation_variable, cart_goal2.observation_variable
        )

        executor = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        executor.compile(motion_statechart=motion_statechart)
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
            Vector3.Z(), np.pi / 6, reference_frame=tip
        )
        tip_rot2 = RotationMatrix.from_axis_angle(
            Vector3.Z(), -np.pi / 6, reference_frame=tip
        )

        motion_statechart = MotionStatechart()
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

        cart_goal1.end_condition = cart_goal1.observation_variable
        cart_goal2.start_condition = cart_goal1.observation_variable

        end = EndMotion()
        motion_statechart.add_node(end)
        end.start_condition = trinary_logic_and(
            cart_goal1.observation_variable, cart_goal2.observation_variable
        )

        executor = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        executor.compile(motion_statechart=motion_statechart)
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

        motion_statechart = MotionStatechart()
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
        end.start_condition = cart_straight.observation_variable

        executor = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        executor.compile(motion_statechart=motion_statechart)
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

        motion_statechart = MotionStatechart()
        cart_straight = CartesianPoseStraight(
            root_link=root,
            tip_link=tip,
            goal_pose=goal_pose,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        motion_statechart.add_node(cart_straight)
        motion_statechart.add_node(EndMotion.when_true(cart_straight))

        executor = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()

        # Verify task detected completion
        assert cart_straight.observation_state == ObservationStateValues.TRUE

        assert np.allclose(
            cart_straight.goal_pose.to_np(), goal_pose.to_np(), atol=0.015
        )

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

        motion_statechart = MotionStatechart()
        goal = CartesianPoseStraight(
            root_link=root,
            tip_link=tip,
            goal_pose=goal_pose,
        )
        motion_statechart.add_node(goal)
        motion_statechart.add_node(EndMotion.when_true(goal))

        executor = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        executor.compile(motion_statechart=motion_statechart)
        straight = next(
            node for node in goal.nodes if isinstance(node, CartesianPositionStraight)
        )
        path = record_tip_path(executor, straight, root, tip)

        line = StraightLine(start=path[0], end=goal_pose.to_np()[:3, 3])
        assert line.maximum_distance_from(path) <= STRAIGHT_LINE_TOLERANCE

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

        motion_statechart = MotionStatechart()
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
        wrist_goal.end_condition = wrist_goal.observation_variable
        straight.start_condition = wrist_goal.observation_variable
        motion_statechart.add_node(EndMotion.when_true(straight))

        executor = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        executor.compile(motion_statechart=motion_statechart)
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

        msc = MotionStatechart()
        goal = CartesianPosition(
            root_link=world.root,
            tip_link=trunk.arms[0].tip,
            goal_point=goal_point,
        )
        msc.add_node(goal)
        msc.add_node(EndMotion.when_true(goal))

        kin_sim = Executor(MotionStatechartContext(world=world))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        # Retrieve the final tip pose
        tip_body = trunk.arms[0].tip
        fk = world.compute_forward_kinematics_np(world.root, tip_body)

        # Verify that the tip reached the target point within the goal threshold
        actual_position = fk[:3, 3]
        target_position = goal_point.to_np()[:3]
        distance_error = np.linalg.norm(actual_position - target_position)

        assert distance_error <= goal.threshold


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
        motion_statechart = MotionStatechart()
        goal_pose.reference_frame = cylinder_bot_diff_world.root
        motion_statechart.add_node(
            goal := DifferentialDriveBaseGoal(goal_pose=goal_pose)
        )
        motion_statechart.add_node(EndMotion.when_true(goal))

        executor = Executor(MotionStatechartContext(world=cylinder_bot_diff_world))
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()

        assert np.allclose(
            cylinder_bot_diff_world.compute_forward_kinematics(
                cylinder_bot_diff_world.root, bot
            ),
            goal_pose,
            atol=1e-2,
        )

    def test_custom_threshold_applies_to_both_translation_and_orientation(
        self, cylinder_bot_diff_world
    ):
        """
        DifferentialDriveBaseGoal exposes a single ``threshold`` field for its callers,
        so it must feed both of CartesianPose's translation_threshold and
        orientation_threshold -- otherwise a caller raising ``threshold`` only relaxes
        the position tolerance while the rotation tolerance silently stays at
        CartesianPose's own hardcoded default.
        """
        goal_pose = Pose.from_xyz_rpy(
            x=1, y=1, yaw=np.pi / 4, reference_frame=cylinder_bot_diff_world.root
        )
        motion_statechart = MotionStatechart()
        motion_statechart.add_node(
            goal := DifferentialDriveBaseGoal(goal_pose=goal_pose, threshold=0.3)
        )
        motion_statechart.add_node(EndMotion.when_true(goal))

        executor = Executor(MotionStatechartContext(world=cylinder_bot_diff_world))
        executor.compile(motion_statechart=motion_statechart)

        for step in goal.nodes[1:]:
            assert step.translation_threshold == 0.3
            assert step.orientation_threshold == 0.3


class TestVelocityTasks:
    def _build_msc(self, goal_node, limit_node) -> MotionStatechart:
        """
        Build a small MSC: goal_node -> limit_node -> EndMotion(when_true=goal_node)
        Returns the MotionStatechart but does not compile or run it.
        """
        motion_statechart = MotionStatechart()
        motion_statechart.add_node(goal_node)
        motion_statechart.add_node(limit_node)
        motion_statechart.add_node(EndMotion.when_true(goal_node))
        return motion_statechart

    def _compile_msc_and_run_until_end(self, world: World, goal_node, limit_node):
        """
        Build the MSC (no extra nodes), compile into an Executor, run until end and
        return (control_cycles, executor)
        """
        motion_statechart = self._build_msc(goal_node=goal_node, limit_node=limit_node)
        executor = Executor(MotionStatechartContext(world=world))
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()
        return executor.control_cycles, executor

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
        Tests that velocity limit's observation variable can trigger a CancelMotion when
        the optimizer chooses to violate the limit.
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
        motion_statechart = self._build_msc(goal_node=goal, limit_node=low_weight_limit)
        cancel_motion = CancelMotion(exception=Exception("test"))
        cancel_motion.start_condition = trinary_logic_not(
            low_weight_limit.observation_variable
        )
        motion_statechart.add_node(cancel_motion)

        executor = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        executor.compile(motion_statechart=motion_statechart)

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

        loose_cycles, _ = self._compile_msc_and_run_until_end(
            world=pr2_world_state_reset, goal_node=position_goal, limit_node=usual_limit
        )
        tight_cycles, _ = self._compile_msc_and_run_until_end(
            world=pr2_world_state_reset,
            goal_node=position_goal,
            limit_node=half_velocity_limit,
        )

        assert (
            tight_cycles >= 2 * loose_cycles
        ), f"tight ({tight_cycles}) should take >= loose ({2 * loose_cycles}) control cycles"

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

        loose_cycles, _ = self._compile_msc_and_run_until_end(
            world=pr2_world_state_reset, goal_node=orientation, limit_node=usual_limit
        )
        tight_cycles, _ = self._compile_msc_and_run_until_end(
            world=pr2_world_state_reset,
            goal_node=orientation,
            limit_node=half_velocity_limit,
        )

        assert (
            tight_cycles >= 2 * loose_cycles
        ), f"tight ({tight_cycles}) should take >= loose ({2 * loose_cycles}) control cycles"


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

        artifacts = task.build(MotionStatechartContext(world=cylinder_bot_world))

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

        artifacts = task.build(MotionStatechartContext(world=cylinder_bot_world))

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

        artifacts = task.build(MotionStatechartContext(world=cylinder_bot_world))

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

        artifacts = task.build(MotionStatechartContext(world=cylinder_bot_world))

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
        motion_statechart = MotionStatechart()
        motion_statechart.add_node(task)
        motion_statechart.add_node(EndMotion.when_true(task))

        executor = Executor(MotionStatechartContext(world=cylinder_bot_world))
        executor.compile(motion_statechart=motion_statechart)

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
