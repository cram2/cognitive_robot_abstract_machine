import time
from dataclasses import dataclass

import numpy as np
import pytest

from giskardpy.data_types.exceptions import DuplicateNameException
from giskardpy.motion_statechart.constraint_builders import GeometricConstraintBuilder
from cramph.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.weights import DefaultWeights
from giskardpy.motion_statechart.exceptions import (
    EmptyDegreesOfFreedomError,
    MissingErrorSignalError,
)
from cramph.composites import Sequence, Parallel
from giskardpy.motion_statechart.graph_node import (
    ConvergingTask,
    EndMotion,
)
from cramph.node import (
    NodeArtifacts,
)
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from cramph.monitors import (
    Pulse,
    CountTicks,
    CheckTickCount,
)
from cramph.statechart import (
    Statechart,
)
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.motion_statechart.tasks.weight_scaling_goals import MaxManipulability
from giskardpy.qp.constraint import GiskardEqualityConstraint
from giskardpy.qp.constraint_collection import ConstraintCollection
from giskardpy.qp.enforcement_strategy import IntegralStrategy
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Quaternion,
    RotationMatrix,
    Vector3,
    Point3,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.robots.pr2 import PR2Joint

from ...semantic_digital_twin_test.test_orm.test_orm import hsr_world_state_reset
from giskardpy.motion_control import MotionControl
from cramph.context import StatechartContext
from cramph.exceptions import TickLimitReachedError
from cramph.executor import StatechartExecutor
from ..motion_control_context import add_to_statechart_with_motion_control

# %% motion nodes in a statechart


def _create_executor() -> StatechartExecutor:
    """
    :return: An executor with motion control acting in an empty world.
    """
    return StatechartExecutor(
        context=StatechartContext(world=World()), extensions=[MotionControl()]
    )


@dataclass(eq=False, repr=False, kw_only=True)
class _ConvergingTaskWithoutErrorSignal(ConvergingTask):
    """
    Converging task whose artifacts leave the error unset.
    """

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts()


def test_converging_task_without_error_signal_is_rejected():
    executor = _create_executor()
    statechart = Statechart(context=executor.context)
    task = _ConvergingTaskWithoutErrorSignal()
    statechart.add_node(task)
    statechart.add_node(EndMotion.when_true(task))

    with pytest.raises(MissingErrorSignalError):
        executor.compile(statechart=statechart)


def test_two_goals(pr2_world_state_reset: World):
    torso_joint = pr2_world_state_reset.get_connection_by_name(PR2Joint.TORSO_LIFT)
    r_wrist_roll_joint = pr2_world_state_reset.get_connection_by_name(
        PR2Joint.RIGHT_WRIST_ROLL
    )
    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_nodes(
        [
            JointPositionList(goal_state=JointState.from_mapping({torso_joint: 0.1})),
            local_min := LocalMinimumReached(),
        ]
    )
    statechart.add_node(EndMotion.when_true(local_min))

    executor.compile(statechart=statechart)

    executor.tick_until_end()
    assert np.isclose(torso_joint.position, 0.1, atol=1e-4)

    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        joint_goal := JointPositionList(
            goal_state=JointState.from_mapping({r_wrist_roll_joint: 1})
        )
    )
    statechart.add_node(EndMotion.when_true(joint_goal))

    executor.compile(statechart=statechart)

    executor.tick_until_end()
    assert np.isclose(torso_joint.position, 0.1, atol=1e-4)
    assert np.allclose(pr2_world_state_reset.state.velocities, 0)
    assert np.allclose(pr2_world_state_reset.state.accelerations, 0)
    assert np.allclose(pr2_world_state_reset.state.jerks, 0)


def test_parallel_local_minimum_reached_tolerates_stall(pr2_world_state_reset: World):
    """
    A :class:`JointPositionList` goal and a :class:`LocalMinimumReached` monitor
    combined via ``Parallel(..., minimum_success=1)`` must finish once the commanded
    joint's velocity has settled near zero, even though the goal task itself never
    reaches its nominal target -- simulating a joint that got physically stopped (e.g.
    a gripper finger against a grasped object) before arriving. Capping the joint's own
    velocity limit to a tiny value makes it provably unable to traverse the requested
    distance within the tick budget, while its tracked velocity still settles below the
    stall threshold almost immediately.

    This is the pattern that replaces baking stall-tolerance into a task's own
    observation: the goal's observation still means "goal reached", nothing else, and
    the ``Parallel`` node is what tolerates the stall.
    """
    torso_joint = pr2_world_state_reset.get_connection_by_name(PR2Joint.TORSO_LIFT)
    torso_joint.raw_dof.limits.lower.velocity = -1e-3
    torso_joint.raw_dof.limits.upper.velocity = 1e-3

    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        combined := Parallel(
            nodes=[
                joint_goal := JointPositionList(
                    goal_state=JointState.from_mapping({torso_joint: 1.0}),
                ),
                LocalMinimumReached(
                    degrees_of_freedom=[torso_joint.raw_dof],
                    minimum_time=0.2,
                    measure_from_own_start=True,
                ),
            ],
            minimum_success=1,
        )
    )
    statechart.add_node(EndMotion.when_true(combined))

    executor.compile(statechart=statechart)
    executor.tick_until_end(timeout=1000)

    assert not np.isclose(torso_joint.position, 1.0, atol=1e-2), (
        "the joint should not have reached its nominal target -- its velocity "
        "was capped far too low to traverse the distance within the timeout, "
        "this test is only meaningful if it stayed far away"
    )
    assert statechart.observation_state[joint_goal] == ObservationStateValues.FALSE, (
        "the goal task's own observation must still mean 'goal reached' -- it must "
        "not be the thing that turned true here, only the surrounding Parallel"
    )


def test_joint_position_list_alone_times_out_on_stall(
    pr2_world_state_reset: World,
):
    """
    Regression control for test_parallel_local_minimum_reached_tolerates_stall: a bare
    :class:`JointPositionList`, without the surrounding ``Parallel`` +
    :class:`LocalMinimumReached`, must never reach EndMotion in the same stalled
    scenario -- proving the monitor is what unblocks it, not some unrelated change.
    """
    torso_joint = pr2_world_state_reset.get_connection_by_name(PR2Joint.TORSO_LIFT)
    torso_joint.raw_dof.limits.lower.velocity = -1e-3
    torso_joint.raw_dof.limits.upper.velocity = 1e-3

    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        joint_goal := JointPositionList(
            goal_state=JointState.from_mapping({torso_joint: 1.0}),
        )
    )
    statechart.add_node(EndMotion.when_true(joint_goal))

    executor.compile(statechart=statechart)

    with pytest.raises(TickLimitReachedError):
        executor.tick_until_end(timeout=200)


def test_local_minimum_reached_only_depends_on_given_degrees_of_freedom(
    pr2_world_state_reset: World,
):
    """
    LocalMinimumReached(degrees_of_freedom=[...]) must only depend on the given subset
    of degrees of freedom, not every active one in the world -- otherwise passing a
    subset to tolerate a stall on one joint could be defeated by unrelated motion
    elsewhere in the robot.
    """
    torso_joint = pr2_world_state_reset.get_connection_by_name(PR2Joint.TORSO_LIFT)
    moving_joint = pr2_world_state_reset.get_connection_by_name(
        PR2Joint.RIGHT_WRIST_ROLL
    )

    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_nodes(
        [
            JointPositionList(goal_state=JointState.from_mapping({moving_joint: 2.0})),
            local_min := LocalMinimumReached(
                degrees_of_freedom=[torso_joint.raw_dof], minimum_time=0.1
            ),
        ]
    )
    statechart.add_node(EndMotion.when_true(local_min))

    executor.compile(statechart=statechart)

    # Checked directly against local_min's own observation state, not against
    # is_ended()/tick_until_end(): EndMotion additionally waits for every active
    # DOF's velocity to settle before actually ending the whole motion (see
    # test_end_motion_waits_for_convergence), so it is no longer a reliable proxy for
    # "when did this specific monitor become true".
    for _ in range(10):
        executor.tick()
    assert statechart.observation_state[local_min] == ObservationStateValues.TRUE, (
        "the scoped monitor should have settled almost immediately -- torso_joint "
        "never moves"
    )
    assert not np.isclose(moving_joint.position, 2.0, atol=1e-2), (
        "r_wrist_roll_joint should still be mid-motion at this point -- otherwise "
        "this test doesn't prove the monitor ignored it"
    )


def test_local_minimum_reached_measures_minimum_time_from_own_start_by_default():
    """
    measure_from_own_start must default to True: LocalMinimumReached is normally used to
    detect a stall on one specific, possibly late-starting motion (e.g. wrapped in a
    Parallel around it), so minimum_time should count from when the monitor itself
    started running, not from the start of the whole motion chart, even if the caller
    never sets the flag explicitly.
    """
    assert LocalMinimumReached().measure_from_own_start is True


def test_local_minimum_reached_raises_on_explicitly_empty_degrees_of_freedom(
    pr2_world_state_reset,
):
    """
    Passing an explicit empty degrees_of_freedom list is a caller misconfiguration (e.g.
    an empty set of connections upstream) and must raise, rather than silently turning
    the monitor into a constant-true observation that could mask the bug.
    """
    monitor = LocalMinimumReached(degrees_of_freedom=[])
    add_to_statechart_with_motion_control(monitor, pr2_world_state_reset)

    with pytest.raises(EmptyDegreesOfFreedomError):
        monitor.build()


def test_long_goal(pr2_world_state_reset: World):
    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_nodes(
        [
            cart_goal := CartesianPose(
                root_link=pr2_world_state_reset.root,
                tip_link=pr2_world_state_reset.get_kinematic_structure_entity_by_name(
                    "base_footprint"
                ),
                goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
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
    statechart.add_node(EndMotion.when_true(cart_goal))

    executor.compile(statechart=statechart)
    t = time.perf_counter()
    executor.tick_until_end(1_000_000)
    after = time.perf_counter()
    diff = after - t
    print(diff / executor.tick_count)


class TestTemplates:
    def test_hsr_cutting(self, hsr_world_state_reset: World, rclpy_node):
        """
        The HSR cuts a loaf with a knife: down, up, then a sideways shift, repeated
        until five seconds have passed and paused while a human is close.
        """
        map_link = hsr_world_state_reset.root
        gripper = hsr_world_state_reset.get_body_by_name("hand_gripper_tool_frame")
        with hsr_world_state_reset.modify_world():
            knife = Body(
                name=PrefixedName("knife"),
                visual=ShapeCollection(
                    [
                        Box(
                            scale=Scale(0.05, 0.01, 0.15),
                            color=Color.from_rgba(red=0.0, green=0.588, blue=0.784),
                        )
                    ]
                ),
            )
            hsr_world_state_reset.add_connection(
                FixedConnection(
                    parent=gripper,
                    child=knife,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=0.06, reference_frame=gripper
                    ),
                )
            )
            loaf = Body(
                name=PrefixedName("loaf"),
                visual=ShapeCollection(
                    [
                        Box(
                            scale=Scale(0.1, 0.2, 0.06),
                            color=Color.from_rgba(red=0.784, green=0.588, blue=0.0),
                        )
                    ]
                ),
            )
            hsr_world_state_reset.add_connection(
                FixedConnection(
                    parent=map_link,
                    child=loaf,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=0.91, y=0.25, z=0.62, reference_frame=map_link
                    ),
                )
            )

        depth = 0.1
        right_shift = -0.1
        # The knife's x axis points up in map, so its -x stroke cuts downwards.
        pre_cut_pose = Pose(
            position=Point3(x=0.85, y=0.2, z=0.75, reference_frame=map_link),
            orientation=Quaternion.from_rotation_matrix(
                RotationMatrix.from_vectors(
                    x=Vector3(z=1, reference_frame=map_link),
                    y=Vector3(y=-1, reference_frame=map_link),
                )
            ),
            reference_frame=map_link,
        )

        executor = StatechartExecutor(
            context=StatechartContext(world=hsr_world_state_reset),
            extensions=[MotionControl()],
        )
        statechart = Statechart(context=executor.context)
        position_knife = CartesianPose(
            name="Position Knife",
            root_link=map_link,
            tip_link=knife,
            goal_pose=pre_cut_pose,
        )
        cut = Sequence(
            nodes=[
                CartesianPose(
                    name="Down",
                    root_link=map_link,
                    tip_link=knife,
                    goal_pose=Pose(
                        position=Point3(x=-depth, reference_frame=knife),
                        reference_frame=knife,
                    ),
                ),
                CartesianPose(
                    name="Up",
                    root_link=map_link,
                    tip_link=knife,
                    goal_pose=Pose(
                        position=Point3(x=depth, reference_frame=knife),
                        reference_frame=knife,
                    ),
                ),
                CartesianPose(
                    name="Move Right",
                    root_link=map_link,
                    tip_link=knife,
                    goal_pose=Pose(
                        position=Point3(y=right_shift, reference_frame=knife),
                        reference_frame=knife,
                    ),
                ),
            ],
            name="Cut",
        )
        # A human blocks the cut 50 cycles after the knife is in place, for 50 cycles.
        wait_for_human = CountTicks(name="Human Approaching", ticks=30)
        human_close = Pulse(name="Human Close?", length=50)
        done = CheckTickCount(name="Done?", threshold=200)
        statechart.add_nodes([position_knife, cut, wait_for_human, human_close, done])

        position_knife.success_condition = position_knife.observes_true
        cut.start_condition = position_knife.is_succeeded
        cut.success_condition = cut.observes_true
        wait_for_human.start_condition = position_knife.is_succeeded
        human_close.start_condition = wait_for_human.observes_true
        human_close.interrupt_condition = done.observes_true
        done.start_condition = cut.is_succeeded
        done.reset_condition = done.observes_false
        cut.pause_condition = human_close.observes_true
        # Each finished pass restarts the cut, until the five seconds are up.
        cut.reset_condition = done.observes_false
        statechart.add_node(EndMotion.when_true(done))

        executor.compile(statechart=statechart)

        executor.tick_until_end()
        statechart.draw("/tmp/muh.pdf")

        assert done.observation_state == ObservationStateValues.TRUE
        cut_life_cycle = statechart.state_history.get_life_cycle_history_of_node(cut)
        assert LifeCycleValues.PAUSED in cut_life_cycle
        restarts = sum(
            1
            for previous, current in zip(cut_life_cycle, cut_life_cycle[1:])
            if current == LifeCycleValues.NOT_STARTED
            and previous != LifeCycleValues.NOT_STARTED
        )
        assert restarts >= 1

    def test_parallel_with_tasks(self, pr2_world_state_reset: World):
        map = pr2_world_state_reset.root
        r_tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        statechart = Statechart(context=executor.context)
        statechart.add_node(
            parallel := Parallel(
                nodes=[
                    AlignPlanes(
                        root_link=map,
                        tip_link=r_tip,
                        tip_normal=Vector3.X(reference_frame=r_tip),
                        goal_normal=Vector3.X(reference_frame=map),
                    ),
                    AlignPlanes(
                        root_link=map,
                        tip_link=r_tip,
                        tip_normal=Vector3.Y(reference_frame=r_tip),
                        goal_normal=Vector3.Z(reference_frame=map),
                    ),
                ]
            )
        )
        statechart.add_node(EndMotion.when_true(parallel))

        executor.compile(statechart=statechart)
        executor.tick_until_end()


def test_constraint_collection(pr2_world_state_reset: World):
    """
    Test the constraint collection naming behavior.

    Expected behavior is: - Not naming constraints should result in automatically generated unique names
    - Manually naming constraints the same name should result in an Exception
    - Merging constraint collections should handle duplicates via prefix if they are in different collections
    - Merge raises an Exception if a collection contains duplicates in itself
    """
    col = ConstraintCollection()
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")

    expr = Vector3.X(tip).angle_between(Vector3.Y(root))

    GeometricConstraintBuilder(col).add_point_goal_constraints(
        frame_P_current=Point3(0, 0, 0, reference_frame=tip),
        frame_P_goal=Point3(0, 0, 0, reference_frame=tip),
        reference_velocity=0.1,
        quadratic_weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
    )
    assert len(col.equality_constraints) >= 3

    for i in range(3):
        col.add_equality_constraint(
            reference_velocity=0.1 * i,
            equality_bound=0.0,
            quadratic_weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
            task_expression=expr,
        )

    col.add_inequality_constraint(
        name="same_name",
        reference_velocity=0.2,
        quadratic_weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
        task_expression=expr,
        lower_error=0.1,
        upper_error=0.2,
    )

    with pytest.raises(DuplicateNameException):
        col.add_equality_constraint(
            name="same_name",
            reference_velocity=0.2,
            equality_bound=0.0,
            quadratic_weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
            task_expression=expr,
        )

    col2 = ConstraintCollection()
    col2.add_equality_constraint(
        name="same_name",
        reference_velocity=0.2,
        equality_bound=0.0,
        quadratic_weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
        task_expression=expr,
    )

    col.merge("prefix", col2)
    assert any(c.name.startswith("prefix/") for c in col._constraints)

    with pytest.raises(DuplicateNameException):
        col.merge("", col2)

    col3 = ConstraintCollection()
    col3.add_equality_constraint(
        name="same_name",
        reference_velocity=0.2,
        equality_bound=0.0,
        quadratic_weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
        task_expression=expr,
    )
    constraint = GiskardEqualityConstraint(
        name="same_name",
        expression=expr,
        normalization_factor=0.1,
        quadratic_weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
        lower_slack_limit=-float("inf"),
        upper_slack_limit=float("inf"),
        linear_weight=0,
        enforcement_strategy=IntegralStrategy,
        bound=0.0,
    )
    col3._constraints.append(constraint)

    with pytest.raises(DuplicateNameException):
        col3._are_names_unique()

    with pytest.raises(DuplicateNameException):
        col2.merge("", col3)


class TestMaxManipulability:
    def test_MaxManipulability(self, pr2_world_state_reset: World) -> None:
        root = pr2_world_state_reset.get_body_by_name("base_footprint")
        tip = pr2_world_state_reset.get_body_by_name("r_gripper_tool_frame")

        goal_pose = Pose.from_xyz_rpy(
            x=0.8, y=-0.3, z=1.0, reference_frame=pr2_world_state_reset.root
        )
        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        statechart = Statechart(context=executor.context)
        cart_goal = CartesianPose(
            root_link=pr2_world_state_reset.root,
            tip_link=tip,
            goal_pose=goal_pose,
        )
        statechart.add_nodes(
            [
                cart_goal,
                manipulability := MaxManipulability(root_link=root, tip_link=tip),
            ]
        )
        manipulability.interrupt_condition = cart_goal.observes_true
        statechart.add_node(EndMotion.when_true(cart_goal))

        executor.compile(statechart=statechart)
        executor.tick_until_end()

        fk = pr2_world_state_reset.compute_forward_kinematics_np(
            pr2_world_state_reset.root, tip
        )
        assert np.allclose(fk, goal_pose.to_np(), atol=cart_goal.translation_threshold)
