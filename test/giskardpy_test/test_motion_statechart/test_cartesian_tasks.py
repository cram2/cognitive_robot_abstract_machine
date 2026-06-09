import time
from typing import Type

import numpy as np
import pytest

from giskardpy.executor import Executor
from giskardpy.motion_statechart.binding_policy import GoalBindingPolicy
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    ObservationStateValues,
    DefaultWeights,
)
from giskardpy.motion_statechart.goals.cartesian_goals import (
    DifferentialDriveBaseGoal,
    CartesianPoseStraight,
)
from giskardpy.motion_statechart.goals.templates import Sequence, Parallel
from giskardpy.motion_statechart.graph_node import (
    EndMotion,
    CancelMotion,
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


class TestCartesianPositionTrajectory:

    def _points_to_np(self, positions: list[Point3] | np.ndarray) -> np.ndarray:
        """
        Convert a sequence of `Point3` or an `ndarray` of shape (N, 3) into an `ndarray` of shape (N, 3).
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
    ):
        """
        Compare an executed Cartesian path against a reference list of positions.

        The executed path is reconstructed from the `world_state_trajectory` by computing
        forward kinematics for `tip_link` in the `root_link` frame at each recorded state.
        For each executed point, the minimum Euclidean distance to the reference path is
        computed. Aggregate statistics and pass/fail against `tolerance` are returned.

        :param positions: Reference path as `Point3` iterable or an array of shape (N, 3). All points are with respect to root_link
        :param world_state_trajectory: Recorded joint-space trajectory with access to the world.
        :param root_link: Root kinematic frame for forward kinematics.
        :param tip_link: Tip kinematic frame for forward kinematics.
        :param tolerance: Maximum allowed distance to the reference path for all samples.
        :return: Dictionary containing distances per sample and summary metrics.
        """
        ref_np = self._points_to_np(positions)

        world = world_state_trajectory.world
        executed_points = []

        # Reconstruct executed Cartesian path by FK at each recorded state
        for state_view in world_state_trajectory.values():
            # Temporarily set the world's state to the recorded one
            world.state._data[:] = state_view.data
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
        a = 0.05  # spiral growth factor (tunes how fast radius grows)

        for i in range(10000):
            t = (
                i * np.pi / 5000.0
            )  # angle parameter; adjust divisor for tighter/looser turns
            r = a * t  # radius grows linearly with t
            points.append(
                Point3(
                    r * np.cos(t),
                    r * np.sin(t),
                    0,
                    reference_frame=cylinder_bot_world.root,
                )
            )
        msc = MotionStatechart()
        cart_traj = CartesianPositionTrajectory(
            root_link=cylinder_bot_world.root,
            tip_link=cylinder_bot_world.get_kinematic_structure_entity_by_name("bot"),
            goal_points=points,
        )
        msc.add_node(cart_traj)
        msc.add_node(EndMotion.when_true(cart_traj))

        kin_sim = Executor(
            context=MotionStatechartContext(
                world=cylinder_bot_world,
            ),
            trajectory_plotter=WorldStateTrajectoryPlotter(),
        )
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()
        self.compare_trajectories(
            points,
            kin_sim.trajectory_plotter.world_state_trajectory,
            cart_traj.root_link,
            cart_traj.tip_link,
        )

    def test_cartesian_position_trajectory_circle(self, cylinder_bot_world: World):
        points = []
        a = 0.1

        for i in range(5000):
            t = (
                i * np.pi / 500.0
            )  # angle parameter; adjust divisor for tighter/looser turns
            points.append(
                Point3(
                    a * np.cos(t),
                    a * np.sin(t),
                    0,
                    reference_frame=cylinder_bot_world.root,
                )
            )
        msc = MotionStatechart()
        cart_traj = CartesianPositionTrajectory(
            root_link=cylinder_bot_world.root,
            tip_link=cylinder_bot_world.get_kinematic_structure_entity_by_name("bot"),
            goal_points=points,
            maximum_skip_ahead=20,
        )
        msc.add_node(cart_traj)
        msc.add_node(EndMotion.when_true(cart_traj))

        kin_sim = Executor(
            context=MotionStatechartContext(
                world=cylinder_bot_world,
            ),
            trajectory_plotter=WorldStateTrajectoryPlotter(),
        )
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()
        self.compare_trajectories(
            points,
            kin_sim.trajectory_plotter.world_state_trajectory,
            cart_traj.root_link,
            cart_traj.tip_link,
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
        msc = MotionStatechart()

        msc.add_node(
            cart_traj := CartesianPositionTrajectory(
                root_link=root,
                tip_link=tip,
                goal_points=points,
            )
        )
        msc.add_node(EndMotion.when_true(cart_traj))

        kin_sim = Executor(
            context=MotionStatechartContext(
                world=pr2_world_state_reset,
            ),
            trajectory_plotter=WorldStateTrajectoryPlotter(),
        )
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()
        self.compare_trajectories(
            root_points,
            kin_sim.trajectory_plotter.world_state_trajectory,
            cart_traj.root_link,
            cart_traj.tip_link,
        )


class TestCartesianTasks:
    """Test suite for all Cartesian motion tasks."""

    def test_simple_cartesian_pose(self, cylinder_bot_world: World):
        tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")

        msc = MotionStatechart()
        msc.add_nodes(
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
        msc.add_node(EndMotion.when_true(goal))

        kin_sim = Executor(MotionStatechartContext(world=cylinder_bot_world))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

    def test_long_goal(self, pr2_world_state_reset: World):
        msc = MotionStatechart()
        msc.add_nodes(
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
                            "torso_lift_joint": 0.2999225173357618,
                            "head_pan_joint": 0.042,
                            "head_tilt_joint": -0.37,
                            "r_upper_arm_roll_joint": -0.9487714747527726,
                            "r_shoulder_pan_joint": -1.0047307505973626,
                            "r_shoulder_lift_joint": 0.48736790658811985,
                            "r_forearm_roll_joint": -14.895833882874182,
                            "r_elbow_flex_joint": -1.392377908925028,
                            "r_wrist_flex_joint": -0.4548695149411013,
                            "r_wrist_roll_joint": 0.11426798984097819,
                            "l_upper_arm_roll_joint": 1.7383062350263658,
                            "l_shoulder_pan_joint": 1.8799810286792007,
                            "l_shoulder_lift_joint": 0.011627231224188975,
                            "l_forearm_roll_joint": 312.67276414458695,
                            "l_elbow_flex_joint": -2.0300928925694675,
                            "l_wrist_flex_joint": -0.1,
                            "l_wrist_roll_joint": -6.062015047706399,
                        },
                        world=pr2_world_state_reset,
                    )
                ),
            ]
        )
        msc.add_node(EndMotion.when_true(cart_goal))

        kin_sim = Executor(
            MotionStatechartContext(
                world=pr2_world_state_reset,
            )
        )
        kin_sim.compile(motion_statechart=msc)
        t = time.perf_counter()
        kin_sim.tick_until_end(1_000_000)
        after = time.perf_counter()
        diff = after - t
        print(diff / kin_sim.control_cycles)

    def test_cart_goal_1eef(self, pr2_world_state_reset: World):
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        tip_goal = Pose.from_xyz_quaternion(pos_x=-0.2, reference_frame=tip)
        expected = pr2_world_state_reset.transform(tip_goal, root)

        msc = MotionStatechart()
        cart_goal = CartesianPose(
            root_link=root,
            tip_link=tip,
            goal_pose=tip_goal,
        )
        msc.add_node(cart_goal)
        end = EndMotion()
        msc.add_node(end)
        end.start_condition = cart_goal.observation_variable

        kin_sim = Executor(
            MotionStatechartContext(
                world=pr2_world_state_reset,
            )
        )
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        assert np.allclose(
            kin_sim.context.world.compute_forward_kinematics(root, tip),
            expected,
            atol=cart_goal.threshold,
        )

    def test_front_facing_orientation(self, hsr_world_setup: World):
        """Test combined position and orientation control in parallel."""
        with hsr_world_setup.modify_world():
            box = Body(
                name=PrefixedName("muh"),
                collision=ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.1))]),
            )
            connection = FixedConnection(
                parent=hsr_world_setup.root,
                child=box,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=2, z=0.5
                ),
            )
            hsr_world_setup.add_connection(connection)

        hsr = hsr_world_setup.get_semantic_annotations_by_type(HSRB)[0]
        hand = hsr_world_setup.get_semantic_annotations_by_type(EndEffector)[0]
        msc = MotionStatechart()
        orientation_goal = hand.front_facing_orientation.to_rotation_matrix()
        orientation_goal.reference_frame = hsr_world_setup.get_body_by_name(
            "base_footprint"
        )
        msc.add_node(
            goal := Parallel(
                [
                    CartesianOrientation(
                        root_link=hsr_world_setup.root,
                        tip_link=hand.tool_frame,
                        goal_orientation=orientation_goal,
                    ),
                    CartesianPosition(
                        root_link=hsr_world_setup.root,
                        tip_link=hand.tool_frame,
                        goal_point=hsr_world_setup.bodies[
                            -1
                        ].global_transform.to_position(),
                    ),
                ]
            )
        )
        msc.add_node(EndMotion.when_true(goal))

        kin_sim = Executor(MotionStatechartContext(world=hsr_world_setup))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

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

        msc = MotionStatechart()
        cart_goal1 = CartesianPose(
            root_link=root,
            tip_link=tip,
            goal_pose=tip_goal1,
        )
        msc.add_node(cart_goal1)

        cart_goal2 = CartesianPose(
            root_link=root,
            tip_link=tip,
            goal_pose=tip_goal2,
            binding_policy=GoalBindingPolicy.Bind_at_build,
        )
        msc.add_node(cart_goal2)

        cart_goal1.end_condition = cart_goal1.observation_variable
        cart_goal2.start_condition = cart_goal1.observation_variable

        end = EndMotion()
        msc.add_node(end)
        end.start_condition = trinary_logic_and(
            cart_goal1.observation_variable, cart_goal2.observation_variable
        )

        kin_sim = Executor(
            MotionStatechartContext(
                world=pr2_world_state_reset,
            )
        )

        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        fk = pr2_world_state_reset.compute_forward_kinematics_np(root, tip)
        assert np.allclose(fk, tip_goal2.to_np(), atol=cart_goal2.threshold)

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

        msc = MotionStatechart()
        cart_goal1 = CartesianPose(
            root_link=root,
            tip_link=tip,
            goal_pose=tip_goal1,
        )
        msc.add_node(cart_goal1)

        cart_goal2 = CartesianPose(
            root_link=root,
            tip_link=tip,
            goal_pose=tip_goal2,
        )
        msc.add_node(cart_goal2)

        cart_goal1.end_condition = cart_goal1.observation_variable
        cart_goal2.start_condition = cart_goal1.observation_variable

        end = EndMotion()
        msc.add_node(end)
        end.start_condition = trinary_logic_and(
            cart_goal1.observation_variable, cart_goal2.observation_variable
        )

        kin_sim = Executor(
            MotionStatechartContext(
                world=pr2_world_state_reset,
            )
        )
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        fk = pr2_world_state_reset.compute_forward_kinematics_np(root, tip)
        expected = np.eye(4)
        assert np.allclose(fk, expected, atol=cart_goal2.threshold)

    def test_CartesianOrientation(self, pr2_world_state_reset: World):
        """Test basic CartesianOrientation goal."""
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        tip_goal = RotationMatrix.from_axis_angle(Vector3.Z(), 4.0, reference_frame=tip)

        msc = MotionStatechart()
        cart_goal = CartesianOrientation(
            root_link=root,
            tip_link=tip,
            goal_orientation=tip_goal,
        )
        msc.add_node(cart_goal)
        end = EndMotion()
        msc.add_node(end)
        end.start_condition = cart_goal.observation_variable

        kin_sim = Executor(
            MotionStatechartContext(
                world=pr2_world_state_reset,
            )
        )
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        fk = pr2_world_state_reset.compute_forward_kinematics_np(root, tip)
        assert np.allclose(fk, tip_goal.to_np(), atol=cart_goal.threshold)

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

        msc = MotionStatechart()
        cart_goal1 = CartesianPosition(
            root_link=root,
            tip_link=tip,
            goal_point=tip_goal1,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        msc.add_node(cart_goal1)

        cart_goal2 = CartesianPosition(
            root_link=root,
            tip_link=tip,
            goal_point=tip_goal2,
            binding_policy=GoalBindingPolicy.Bind_at_build,
        )
        msc.add_node(cart_goal2)

        cart_goal1.end_condition = cart_goal1.observation_variable
        cart_goal2.start_condition = cart_goal1.observation_variable

        end = EndMotion()
        msc.add_node(end)
        end.start_condition = trinary_logic_and(
            cart_goal1.observation_variable, cart_goal2.observation_variable
        )

        kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        fk = pr2_world_state_reset.compute_forward_kinematics_np(root, tip)
        # goal2 was captured at build time, so should end at that absolute position
        expected = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=0.2, reference_frame=pr2_world_state_reset.root
        ).to_np()
        assert np.allclose(fk[:3, 3], expected[:3, 3], atol=cart_goal2.threshold)

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

        msc = MotionStatechart()
        cart_goal1 = CartesianPosition(
            root_link=root,
            tip_link=tip,
            goal_point=tip_goal1,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        msc.add_node(cart_goal1)

        cart_goal2 = CartesianPosition(
            root_link=root,
            tip_link=tip,
            goal_point=tip_goal2,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        msc.add_node(cart_goal2)

        cart_goal1.end_condition = cart_goal1.observation_variable
        cart_goal2.start_condition = cart_goal1.observation_variable

        end = EndMotion()
        msc.add_node(end)
        end.start_condition = trinary_logic_and(
            cart_goal1.observation_variable, cart_goal2.observation_variable
        )

        kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        fk = pr2_world_state_reset.compute_forward_kinematics_np(root, tip)
        # Both goals captured when tasks start, so should return near origin
        expected = np.eye(4)
        assert np.allclose(fk[:3, 3], expected[:3, 3], atol=cart_goal2.threshold)

    def test_cartesian_position_with_sequence_node(self, pr2_world_state_reset: World):
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        tip_goal1 = Point3(-0.2, 0, 0, reference_frame=tip)
        tip_goal2 = Point3(0.2, 0, 0, reference_frame=tip)

        msc = MotionStatechart()
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
        msc.add_node(seq := Sequence(nodes=[cart_goal1, cart_goal2]))

        msc.add_node(EndMotion.when_true(seq))

        kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        fk = pr2_world_state_reset.compute_forward_kinematics_np(root, tip)
        # Both goals captured when tasks start, so should return near origin
        expected = np.eye(4)
        assert np.allclose(fk[:3, 3], expected[:3, 3], atol=cart_goal2.threshold)

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

        msc = MotionStatechart()
        cart_goal1 = CartesianOrientation(
            root_link=root,
            tip_link=tip,
            goal_orientation=tip_rot1,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        msc.add_node(cart_goal1)

        cart_goal2 = CartesianOrientation(
            root_link=root,
            tip_link=tip,
            goal_orientation=tip_rot2,
            binding_policy=GoalBindingPolicy.Bind_at_build,
        )
        msc.add_node(cart_goal2)

        cart_goal1.end_condition = cart_goal1.observation_variable
        cart_goal2.start_condition = cart_goal1.observation_variable

        end = EndMotion()
        msc.add_node(end)
        end.start_condition = trinary_logic_and(
            cart_goal1.observation_variable, cart_goal2.observation_variable
        )

        kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        fk = pr2_world_state_reset.compute_forward_kinematics_np(root, tip)

        # goal2 captured at build, so ends at -pi/6 from original
        expected = tip_rot2.to_np()
        assert np.allclose(fk, expected, atol=cart_goal2.threshold)

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

        msc = MotionStatechart()
        cart_goal1 = CartesianOrientation(
            root_link=root,
            tip_link=tip,
            goal_orientation=tip_rot1,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        msc.add_node(cart_goal1)

        cart_goal2 = CartesianOrientation(
            root_link=root,
            tip_link=tip,
            goal_orientation=tip_rot2,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        msc.add_node(cart_goal2)

        cart_goal1.end_condition = cart_goal1.observation_variable
        cart_goal2.start_condition = cart_goal1.observation_variable

        end = EndMotion()
        msc.add_node(end)
        end.start_condition = trinary_logic_and(
            cart_goal1.observation_variable, cart_goal2.observation_variable
        )

        kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        fk = pr2_world_state_reset.compute_forward_kinematics_np(root, tip)
        # Both goals captured when tasks start, so rotates +pi/6 then -pi/6 = back to origin
        expected = np.eye(4)
        assert np.allclose(fk, expected, atol=cart_goal2.threshold)

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

        msc = MotionStatechart()
        cart_straight = CartesianPositionStraight(
            root_link=root,
            tip_link=tip,
            goal_point=goal_point,
            binding_policy=GoalBindingPolicy.Bind_on_start,
            threshold=0.015,
        )
        msc.add_node(cart_straight)
        end = EndMotion()
        msc.add_node(end)
        end.start_condition = cart_straight.observation_variable

        kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        # Verify goal was achieved
        assert cart_straight.observation_state == ObservationStateValues.TRUE

    def test_cartesian_pose_straight(self, pr2_world_state_reset: World):
        """Test CartesianPositionStraight basic functionality."""
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        goal_pose = Pose.from_xyz_rpy(0.1, 2, 0, reference_frame=tip)

        msc = MotionStatechart()
        cart_straight = CartesianPoseStraight(
            root_link=root,
            tip_link=tip,
            goal_pose=goal_pose,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        msc.add_node(cart_straight)
        msc.add_node(EndMotion.when_true(cart_straight))

        kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        # Verify task detected completion
        assert cart_straight.observation_state == ObservationStateValues.TRUE

        assert np.allclose(
            cart_straight.goal_pose.to_np(), goal_pose.to_np(), atol=0.015
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
        msc = MotionStatechart()
        goal_pose.reference_frame = cylinder_bot_diff_world.root
        msc.add_node(goal := DifferentialDriveBaseGoal(goal_pose=goal_pose))
        msc.add_node(EndMotion.when_true(goal))

        kin_sim = Executor(MotionStatechartContext(world=cylinder_bot_diff_world))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        assert np.allclose(
            cylinder_bot_diff_world.compute_forward_kinematics(
                cylinder_bot_diff_world.root, bot
            ),
            goal_pose,
            atol=1e-2,
        )


class TestVelocityTasks:
    def _build_msc(self, goal_node, limit_node) -> MotionStatechart:
        """
        Build a small MSC: goal_node -> limit_node -> EndMotion(when_true=goal_node)
        Returns the MotionStatechart but does not compile or run it.
        """
        msc = MotionStatechart()
        msc.add_node(goal_node)
        msc.add_node(limit_node)
        msc.add_node(EndMotion.when_true(goal_node))
        return msc

    def _compile_msc_and_run_until_end(self, world: World, goal_node, limit_node):
        """
        Build the MSC (no extra nodes), compile into an Executor,
        run until end and return (control_cycles, executor)
        """
        msc = self._build_msc(goal_node=goal_node, limit_node=limit_node)
        kin_sim = Executor(MotionStatechartContext(world=world))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()
        return kin_sim.control_cycles, kin_sim

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
        self, pr2_world_state_reset: World, goal_type: str, limit_cls: Type
    ):
        """
        Tests that velocity limit's observation variable can trigger a CancelMotion
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
                weight=DefaultWeights.WEIGHT_ABOVE_CA,
            )
        else:
            goal = CartesianOrientation(
                root_link=root,
                tip_link=tip,
                goal_orientation=RotationMatrix.from_rpy(
                    yaw=np.pi / 2, reference_frame=tip
                ),
                weight=DefaultWeights.WEIGHT_ABOVE_CA,
            )

        low_weight_limit = limit_cls(
            root_link=root, tip_link=tip, weight=DefaultWeights.WEIGHT_BELOW_CA
        )
        msc = self._build_msc(goal_node=goal, limit_node=low_weight_limit)
        cancel_motion = CancelMotion(exception=Exception("test"))
        cancel_motion.start_condition = trinary_logic_not(
            low_weight_limit.observation_variable
        )
        msc.add_node(cancel_motion)

        kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        kin_sim.compile(motion_statechart=msc)

        with pytest.raises(Exception):
            kin_sim.tick_until_end()

    @pytest.mark.parametrize(
        "limit_cls",
        [CartesianVelocityLimit, CartesianPositionVelocityLimit],
        ids=["generic_linear", "position_only_linear"],
    )
    def test_cartesian_position_velocity_limit(
        self, pr2_world_state_reset: World, limit_cls: Type
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
        self, pr2_world_state_reset: World, limit_cls: Type
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
