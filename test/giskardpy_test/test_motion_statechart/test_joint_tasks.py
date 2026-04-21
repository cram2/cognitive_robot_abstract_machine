import numpy as np

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import (
    EndMotion,
)
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetSeedConfiguration,
    SetOdometry,
)
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.motion_statechart.test_nodes.test_nodes import (
    ConstTrueNode,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood.symbolic_math.symbolic_math import (
    trinary_logic_and,
    shortest_angular_distance,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
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


def test_set_seed_configuration(pr2_world_state_reset):
    msc = MotionStatechart()
    goal = 0.1

    connection: ActiveConnection1DOF = pr2_world_state_reset.get_connection_by_name(
        "torso_lift_joint"
    )

    node1 = SetSeedConfiguration(
        seed_configuration=JointState.from_mapping({connection: goal})
    )
    end = EndMotion()
    msc.add_node(node1)
    msc.add_node(end)
    node1.end_condition = node1.observation_variable
    end.start_condition = node1.observation_variable

    kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
    kin_sim.compile(motion_statechart=msc)

    kin_sim.tick_until_end()
    assert node1.observation_state == ObservationStateValues.TRUE
    assert node1.life_cycle_state == LifeCycleValues.DONE
    assert end.observation_state == ObservationStateValues.TRUE
    assert end.life_cycle_state == LifeCycleValues.RUNNING

    assert np.isclose(connection.position, goal)


def test_set_seed_odometry(pr2_world_state_reset):
    msc = MotionStatechart()

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
    msc.add_node(node1)
    msc.add_node(end)
    node1.end_condition = node1.observation_variable
    end.start_condition = node1.observation_variable

    kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
    kin_sim.compile(motion_statechart=msc)

    kin_sim.tick_until_end()
    assert node1.observation_state == ObservationStateValues.TRUE
    assert node1.life_cycle_state == LifeCycleValues.DONE
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
        ul = DerivativeMap()
        ul.velocity = 1
        ll = DerivativeMap()
        ll.velocity = -1
        dof = DegreeOfFreedom(
            name=PrefixedName("dof", "a"),
            limits=DegreeOfFreedomLimits(lower=ll, upper=ul),
        )
        world.add_degree_of_freedom(dof)
        root_C_tip = RevoluteConnection(
            parent=root, child=tip, axis=Vector3.Z(), dof_id=dof.id
        )
        world.add_connection(root_C_tip)

        dof = DegreeOfFreedom(
            name=PrefixedName("dof", "b"),
            limits=DegreeOfFreedomLimits(lower=ll, upper=ul),
        )
        world.add_degree_of_freedom(dof)
        root_C_tip2 = RevoluteConnection(
            parent=root, child=tip2, axis=Vector3.Z(), dof_id=dof.id
        )
        world.add_connection(root_C_tip2)

    msc = MotionStatechart()

    task1 = JointPositionList(goal_state=JointState.from_mapping({root_C_tip: 1}))
    always_true = ConstTrueNode()
    msc.add_node(always_true)
    msc.add_node(task1)
    end = EndMotion()
    msc.add_node(end)

    task1.start_condition = always_true.observation_variable
    end.start_condition = trinary_logic_and(
        task1.observation_variable, always_true.observation_variable
    )

    kin_sim = Executor(
        MotionStatechartContext(
            world=world,
            qp_controller_config=QPControllerConfig(
                target_frequency=20,
                prediction_horizon=7,
            ),
        )
    )
    kin_sim.compile(motion_statechart=msc)

    assert task1.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN
    assert task1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    msc.draw(str(tmp_path / "muh.pdf"))
    kin_sim.tick_until_end()
    msc.draw(str(tmp_path / "muh.pdf"))
    assert len(msc.history) == 6
    assert (
        msc.history.get_observation_history_of_node(task1)[-1]
        == ObservationStateValues.TRUE
    )
    assert (
        msc.history.get_observation_history_of_node(end)[-1]
        == ObservationStateValues.TRUE
    )
    assert (
        msc.history.get_life_cycle_history_of_node(task1)[-1] == LifeCycleValues.RUNNING
    )
    assert (
        msc.history.get_life_cycle_history_of_node(end)[-1] == LifeCycleValues.RUNNING
    )


def test_continuous_joint(pr2_world_state_reset):
    r_wrist_roll_joint = pr2_world_state_reset.get_connection_by_name(
        "r_wrist_roll_joint"
    )
    l_wrist_roll_joint = pr2_world_state_reset.get_connection_by_name(
        "l_wrist_roll_joint"
    )
    msc = MotionStatechart()
    joint_goal = JointPositionList(
        goal_state=JointState.from_mapping(
            {
                r_wrist_roll_joint: -np.pi,
                l_wrist_roll_joint: -2.1 * np.pi,
            },
        ),
    )
    msc.add_node(joint_goal)
    end = EndMotion()
    msc.add_node(end)
    end.start_condition = joint_goal.observation_variable

    kin_sim = Executor(
        MotionStatechartContext(
            world=pr2_world_state_reset,
        )
    )
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()
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
    head_pan_joint = pr2_world_state_reset.get_connection_by_name("head_pan_joint")
    head_tilt_joint = pr2_world_state_reset.get_connection_by_name("head_tilt_joint")
    msc = MotionStatechart()
    joint_goal = JointPositionList(
        goal_state=JointState.from_mapping(
            {
                head_pan_joint: 0.042,
                head_tilt_joint: -0.37,
            },
        ),
    )
    msc.add_node(joint_goal)
    end = EndMotion()
    msc.add_node(end)
    end.start_condition = joint_goal.observation_variable

    kin_sim = Executor(
        MotionStatechartContext(
            world=pr2_world_state_reset,
        )
    )
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()
    assert np.isclose(head_pan_joint.position, 0.042, atol=1e-3)
    assert np.isclose(head_tilt_joint.position, -0.37, atol=1e-2)


def test_joint_sequence(pr2_world_state_reset):
    msc = MotionStatechart()
    msc.add_node(
        sequence := Sequence(
            [
                JointPositionList(
                    goal_state=JointState.from_str_dict(
                        {"torso_lift_joint": 0.1}, world=pr2_world_state_reset
                    )
                ),
                JointPositionList(
                    goal_state=JointState.from_str_dict(
                        {"torso_lift_joint": 0.2}, world=pr2_world_state_reset
                    )
                ),
            ]
        )
    )
    msc.add_node(EndMotion.when_true(sequence))

    kin_sim = Executor(
        MotionStatechartContext(
            world=pr2_world_state_reset,
        )
    )
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()
