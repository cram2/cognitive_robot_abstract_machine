import numpy as np
import pytest

from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.composites import Parallel, Sequence
from giskardpy.motion_statechart.graph_node import (
    EndMotion,
)
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetSeedConfiguration,
    SetOdometry,
)
from cramph.statechart import (
    Statechart,
)
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointPositionList,
    JointState,
    JointVelocityLimit,
)
from cramph.nodes_for_testing import ConstTrueNode
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood.symbolic_math.symbolic_math import (
    shortest_angular_distance,
    logic_and,
)
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    ActiveConnection1DOF,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
)
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.pr2 import PR2Joint
from giskardpy.motion_control import MotionControl
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor


def test_set_seed_configuration(pr2_world_state_reset):
    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    goal = 0.1

    connection: ActiveConnection1DOF = pr2_world_state_reset.get_connection_by_name(
        PR2Joint.TORSO_LIFT
    )

    node1 = SetSeedConfiguration(
        seed_configuration=JointState.from_mapping({connection: goal})
    )
    end = EndMotion()
    statechart.add_node(node1)
    statechart.add_node(end)
    node1.success_condition = node1.observes_true
    end.start_condition = node1.observes_true

    executor.compile(statechart=statechart)

    executor.tick_until_end()
    assert node1.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert end.observation_state == ObservationStateValues.TRUE
    assert end.life_cycle_state == LifeCycleValues.RUNNING

    assert np.isclose(connection.position, goal)


def test_set_seed_odometry(pr2_world_state_reset):
    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)

    goal = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1,
        y=-1,
        z=1,
        roll=1,
        pitch=1,
        yaw=1,
        reference_frame=pr2_world_state_reset.root,
    )
    expected = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1, y=-1, yaw=1, reference_frame=pr2_world_state_reset.root
    )

    node1 = SetOdometry(
        base_pose=goal,
    )
    end = EndMotion()
    statechart.add_node(node1)
    statechart.add_node(end)
    node1.success_condition = node1.observes_true
    end.start_condition = node1.observes_true

    executor.compile(statechart=statechart)

    executor.tick_until_end()
    assert node1.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert end.observation_state == ObservationStateValues.TRUE
    assert end.life_cycle_state == LifeCycleValues.RUNNING

    assert np.allclose(
        expected.to_np(),
        pr2_world_state_reset.compute_forward_kinematics_np(
            pr2_world_state_reset.root, node1.odom_connection.child
        ),
    )


def test_joint_goal(tmp_path):
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        tip = Body(name=PrefixedName("tip"))
        tip2 = Body(name=PrefixedName("tip2"))
        limits = DegreeOfFreedomLimits.from_position_range_and_speed(maximum_speed=1)
        dof = DegreeOfFreedom(
            name=PrefixedName("dof", "a"),
            limits=limits,
        )
        world.add_degree_of_freedom(dof)
        root_C_tip = RevoluteConnection(
            parent=root, child=tip, axis=Vector3.Z(), raw_dof=dof
        )
        world.add_connection(root_C_tip)

        dof = DegreeOfFreedom(
            name=PrefixedName("dof", "b"),
            limits=limits,
        )
        world.add_degree_of_freedom(dof)
        root_C_tip2 = RevoluteConnection(
            parent=root, child=tip2, axis=Vector3.Z(), raw_dof=dof
        )
        world.add_connection(root_C_tip2)

    executor = StatechartExecutor(
        context=StatechartContext(world=world),
        extensions=[
            MotionControl(
                qp_controller_config=QPControllerConfig(
                    target_frequency=20,
                    prediction_horizon=7,
                )
            )
        ],
    )
    statechart = Statechart(context=executor.context)

    task1 = JointPositionList(goal_state=JointState.from_mapping({root_C_tip: 1}))
    always_true = ConstTrueNode()
    statechart.add_node(always_true)
    statechart.add_node(task1)
    end = EndMotion()
    statechart.add_node(end)

    task1.start_condition = always_true.observes_true
    end.start_condition = logic_and(task1.observes_true, always_true.observes_true)

    executor.compile(statechart=statechart)

    assert task1.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN
    assert task1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    statechart.draw(str(tmp_path / "muh.pdf"))
    executor.tick_until_end()
    statechart.draw(str(tmp_path / "muh.pdf"))
    # +1 compared to the tick count without a moving dof: EndMotion now waits for the
    # dof's velocity to settle below its convergence threshold before observing True.
    assert len(statechart.state_history) == 7
    assert (
        statechart.state_history.get_observation_history_of_node(task1)[-1]
        == ObservationStateValues.TRUE
    )
    assert (
        statechart.state_history.get_observation_history_of_node(end)[-1]
        == ObservationStateValues.TRUE
    )
    assert (
        statechart.state_history.get_life_cycle_history_of_node(task1)[-1]
        == LifeCycleValues.RUNNING
    )
    assert (
        statechart.state_history.get_life_cycle_history_of_node(end)[-1]
        == LifeCycleValues.RUNNING
    )


def test_continuous_joint(pr2_world_state_reset):
    r_wrist_roll_joint = pr2_world_state_reset.get_connection_by_name(
        PR2Joint.RIGHT_WRIST_ROLL
    )
    l_wrist_roll_joint = pr2_world_state_reset.get_connection_by_name(
        PR2Joint.LEFT_WRIST_ROLL
    )
    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    joint_goal = JointPositionList(
        goal_state=JointState.from_mapping(
            {
                r_wrist_roll_joint: -np.pi,
                l_wrist_roll_joint: -2.1 * np.pi,
            },
        ),
    )
    statechart.add_node(joint_goal)
    end = EndMotion()
    statechart.add_node(end)
    end.start_condition = joint_goal.observes_true

    executor.compile(statechart=statechart)
    executor.tick_until_end()
    assert np.isclose(
        shortest_angular_distance(r_wrist_roll_joint.position, -np.pi),
        0,
        atol=0.005,
    )
    assert np.isclose(
        shortest_angular_distance(l_wrist_roll_joint.position, -2.1 * np.pi),
        0,
        atol=0.005,
    )


def test_revolute_joint(pr2_world_state_reset):
    head_pan_joint = pr2_world_state_reset.get_connection_by_name(PR2Joint.HEAD_PAN)
    head_tilt_joint = pr2_world_state_reset.get_connection_by_name(PR2Joint.HEAD_TILT)
    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    joint_goal = JointPositionList(
        goal_state=JointState.from_mapping(
            {
                head_pan_joint: 0.042,
                head_tilt_joint: -0.37,
            },
        ),
    )
    statechart.add_node(joint_goal)
    end = EndMotion()
    statechart.add_node(end)
    end.start_condition = joint_goal.observes_true

    executor.compile(statechart=statechart)
    executor.tick_until_end()
    assert np.isclose(head_pan_joint.position, 0.042, atol=1e-3)
    assert np.isclose(head_tilt_joint.position, -0.37, atol=1e-2)


def test_joint_velocity_limit_caps_a_fast_goal(pr2_world_state_reset):
    """
    JointVelocityLimit must genuinely slow down a joint's approach towards a distant
    target, unlike a reference/normalization velocity which is only a QP-optimization
    hint -- the joint's velocity must stay within max_velocity on every single tick,
    all the way until the goal is actually reached.
    """
    head_pan_joint = pr2_world_state_reset.get_connection_by_name(PR2Joint.HEAD_PAN)
    max_velocity = 0.05

    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_nodes(
        [
            joint_goal := JointPositionList(
                goal_state=JointState.from_mapping({head_pan_joint: 0.5}),
            ),
            JointVelocityLimit(connections=[head_pan_joint], max_velocity=max_velocity),
        ]
    )

    executor.compile(statechart=statechart)

    for i in range(400):
        executor.tick()
        observed_velocity = abs(head_pan_joint.raw_dof.variables.velocity.resolve())
        assert observed_velocity <= max_velocity + 1e-3, (
            f"head_pan_joint's velocity {observed_velocity} exceeded the "
            f"{max_velocity} rad/s cap on tick {i}"
        )
        if statechart.observation_state[joint_goal] == ObservationStateValues.TRUE:
            break
    else:
        raise TimeoutError(
            "joint_goal never reached its target while under the velocity limit"
        )

    assert np.isclose(head_pan_joint.position, 0.5, atol=1e-2)


def test_joint_sequence(pr2_world_state_reset):
    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        sequence := Sequence(
            nodes=[
                JointPositionList(
                    goal_state=JointState.from_str_dict(
                        {PR2Joint.TORSO_LIFT: 0.1}, world=pr2_world_state_reset
                    )
                ),
                JointPositionList(
                    goal_state=JointState.from_str_dict(
                        {PR2Joint.TORSO_LIFT: 0.2}, world=pr2_world_state_reset
                    )
                ),
            ]
        )
    )
    statechart.add_node(EndMotion.when_true(sequence))

    executor.compile(statechart=statechart)
    executor.tick_until_end()


# %% static joint states


@pytest.mark.parametrize("arm_index", [0, 1], ids=["left", "right"])
def test_daisy_arm_reaches_its_park_state(
    daisy_world_copy: World, default_daisy_pose: dict[str, float], arm_index: int
) -> None:
    JointState.from_str_dict(default_daisy_pose, daisy_world_copy).apply_to(
        daisy_world_copy
    )
    robot = daisy_world_copy.get_semantic_annotations_by_type(DAiSy)[0]
    park_state = robot.arms[arm_index].get_joint_state_by_type(StaticJointState.PARK)
    executor = StatechartExecutor(
        context=StatechartContext(world=daisy_world_copy),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(joint_goal := JointPositionList(goal_state=park_state))
    statechart.add_node(EndMotion.when_true(joint_goal))

    executor.compile(statechart=statechart)
    executor.tick_until_end()

    assert park_state.is_achieved()
