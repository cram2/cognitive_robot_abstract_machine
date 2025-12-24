import json
import time
from dataclasses import dataclass
from math import radians
from typing import Type

import numpy as np
import pytest

from giskardpy.data_types.exceptions import DuplicateNameException
from giskardpy.executor import Executor, SimulationPacer
from giskardpy.model.collision_matrix_manager import CollisionRequest
from giskardpy.model.collision_world_syncer import CollisionCheckerLib
from giskardpy.motion_statechart.binding_policy import GoalBindingPolicy
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
    DefaultWeights,
)
from giskardpy.motion_statechart.exceptions import (
    NotInMotionStatechartError,
    EndMotionInGoalError,
    InputNotExpressionError,
    SelfInStartConditionError,
    NonObservationVariableError,
    NodeAlreadyBelongsToDifferentNodeError,
)
from giskardpy.motion_statechart.goals.collision_avoidance import (
    CollisionAvoidance,
)
from giskardpy.motion_statechart.goals.open_close import Open, Close
from giskardpy.motion_statechart.goals.templates import Sequence, Parallel
from giskardpy.motion_statechart.graph_node import (
    EndMotion,
    CancelMotion,
)
from giskardpy.motion_statechart.graph_node import ThreadPayloadMonitor
from giskardpy.motion_statechart.monitors.joint_monitors import JointPositionReached
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetSeedConfiguration,
    SetOdometry,
)
from giskardpy.motion_statechart.monitors.payload_monitors import (
    Print,
    Pulse,
    CountSeconds,
    CountTicks,
)
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
)
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianOrientation,
    CartesianPosition,
    CartesianPositionStraight,
    CartesianVelocityLimit,
    CartesianPositionVelocityLimit,
    CartesianRotationVelocityLimit,
)
from giskardpy.motion_statechart.tasks.feature_functions import (
    AngleGoal,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.motion_statechart.tasks.pointing import Pointing, PointingCone
from giskardpy.motion_statechart.test_nodes.test_nodes import (
    ChangeStateOnEvents,
    ConstTrueNode,
    TestGoal,
    TestNestedGoal,
    ConstFalseNode,
    TestRunAfterStop,
    TestRunAfterStopFromPause,
    TestEndBeforeStart,
    TestUnpauseUnknownFromParentPause,
)
from giskardpy.qp.constraint import EqualityConstraint
from giskardpy.qp.constraint_collection import ConstraintCollection
from giskardpy.qp.exceptions import HardConstraintsViolatedException
from giskardpy.utils.math import angle_between_vector
from krrood.symbolic_math.symbolic_math import (
    trinary_logic_and,
    trinary_logic_not,
    trinary_logic_or,
    FloatVariable,
)
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    KinematicStructureEntityKwargsTracker,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.factories import (
    DoorFactory,
    SemanticPositionDescription,
    HandleFactory,
    HorizontalSemanticDirection,
    VerticalSemanticDirection,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
    Point3,
    RotationMatrix,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    ActiveConnection1DOF,
    FixedConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import Cylinder
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


def test_condition_to_str():
    msc = MotionStatechart()
    node1 = ConstTrueNode()
    msc.add_node(node1)
    node2 = ConstTrueNode()
    msc.add_node(node2)
    node3 = ConstTrueNode()
    msc.add_node(node3)
    end = EndMotion()
    msc.add_node(end)

    end.start_condition = trinary_logic_and(
        node1.observation_variable,
        trinary_logic_or(
            node2.observation_variable,
            trinary_logic_not(node3.observation_variable),
        ),
    )
    a = str(end._start_condition)
    assert a == '("ConstTrueNode#0" and ("ConstTrueNode#1" or not "ConstTrueNode#2"))'


def test_motion_statechart_to_dot():
    msc = MotionStatechart()
    node1 = ConstTrueNode()
    msc.add_node(node1)
    node2 = ConstTrueNode()
    msc.add_node(node2)
    end = EndMotion()
    msc.add_node(end)
    node1.end_condition = node2.observation_variable
    end.start_condition = trinary_logic_and(
        node1.observation_variable, node2.observation_variable
    )
    msc.draw("muh.pdf")


def test_motion_statechart():
    msc = MotionStatechart()

    node1 = ConstTrueNode()
    msc.add_node(node1)
    node2 = ConstTrueNode()
    msc.add_node(node2)
    node3 = ConstTrueNode()
    msc.add_node(node3)
    end = EndMotion()
    msc.add_node(end)

    node1.start_condition = trinary_logic_or(
        node3.observation_variable, node2.observation_variable
    )
    end.start_condition = node1.observation_variable

    kin_sim = Executor(world=World())
    kin_sim.compile(motion_statechart=msc)

    assert len(msc.nodes) == 4
    assert len(msc.edges) == 3
    kin_sim.tick_until_end()

    assert len(msc.history) == 5
    # %% node1
    assert msc.history.get_life_cycle_history_of_node(node1) == [
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
    ]
    assert msc.history.get_observation_history_of_node(node1) == [
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
    ]
    # %% node2
    assert msc.history.get_life_cycle_history_of_node(node2) == [
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
    ]
    assert msc.history.get_observation_history_of_node(node2) == [
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
    ]
    # %% node3
    assert msc.history.get_life_cycle_history_of_node(node3) == [
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
    ]
    assert msc.history.get_observation_history_of_node(node3) == [
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
    ]
    # %% end
    assert msc.history.get_life_cycle_history_of_node(end) == [
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
    ]
    assert msc.history.get_observation_history_of_node(end) == [
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.TRUE,
    ]


def test_sequence_goal():
    msc = MotionStatechart()
    node = Sequence(
        nodes=[
            ConstTrueNode(),
            ConstTrueNode(),
            ConstTrueNode(),
            ConstTrueNode(),
        ]
    )
    msc.add_node(node)
    msc.add_node(EndMotion.when_true(node))

    kin_sim = Executor(world=World())
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()
    msc.draw("muh.pdf")
    assert kin_sim.control_cycles == 7
    assert msc.nodes[0].life_cycle_state == LifeCycleValues.RUNNING
    assert msc.nodes[1].life_cycle_state == LifeCycleValues.RUNNING
    assert msc.nodes[2].life_cycle_state == LifeCycleValues.DONE
    assert msc.nodes[3].life_cycle_state == LifeCycleValues.DONE
    assert msc.nodes[4].life_cycle_state == LifeCycleValues.DONE
    assert msc.nodes[5].life_cycle_state == LifeCycleValues.DONE


def test_print():
    msc = MotionStatechart()
    print_node1 = Print(name="cow", message="muh")
    msc.add_node(print_node1)
    print_node2 = Print(name="cow2", message="muh")
    msc.add_node(print_node2)

    node1 = ConstTrueNode()
    msc.add_node(node1)
    end = EndMotion()
    msc.add_node(end)

    node1.start_condition = print_node1.observation_variable
    print_node2.start_condition = node1.observation_variable
    end.start_condition = print_node2.observation_variable

    kin_sim = Executor(world=World())
    kin_sim.compile(motion_statechart=msc)

    assert len(msc.nodes) == 4
    assert len(msc.edges) == 3

    assert print_node1.observation_state == ObservationStateValues.UNKNOWN
    assert node1.observation_state == ObservationStateValues.UNKNOWN
    assert print_node2.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert print_node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert print_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    kin_sim.tick()
    assert print_node1.observation_state == ObservationStateValues.UNKNOWN
    assert node1.observation_state == ObservationStateValues.UNKNOWN
    assert print_node2.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert print_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    kin_sim.tick()
    assert print_node1.observation_state == ObservationStateValues.TRUE
    assert node1.observation_state == ObservationStateValues.UNKNOWN
    assert print_node2.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert print_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    kin_sim.tick()
    assert print_node1.observation_state == ObservationStateValues.TRUE
    assert node1.observation_state == ObservationStateValues.TRUE
    assert print_node2.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert print_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    kin_sim.tick()
    assert print_node1.observation_state == ObservationStateValues.TRUE
    assert node1.observation_state == ObservationStateValues.TRUE
    assert print_node2.observation_state == ObservationStateValues.TRUE
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert print_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert not msc.is_end_motion()

    kin_sim.tick()
    assert print_node1.observation_state == ObservationStateValues.TRUE
    assert node1.observation_state == ObservationStateValues.TRUE
    assert print_node2.observation_state == ObservationStateValues.TRUE
    assert end.observation_state == ObservationStateValues.TRUE

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert print_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert msc.is_end_motion()


def test_cancel_motion():
    msc = MotionStatechart()
    node1 = ConstTrueNode()
    msc.add_node(node1)
    cancel = CancelMotion(exception=Exception("krrood_test"))
    msc.add_node(cancel)
    cancel.start_condition = node1.observation_variable

    kin_sim = Executor(world=World())
    kin_sim.compile(motion_statechart=msc)

    kin_sim.tick()  # first tick, cancel motion node1 turns true
    kin_sim.tick()  # second tick, cancel goes into running
    with pytest.raises(Exception):
        kin_sim.tick()  # third tick, cancel goes true and triggers
    msc.draw("muh.pdf")


def test_draw_with_invisible_node():
    msc = MotionStatechart()
    msc.add_nodes(
        [
            sequence := Sequence(
                nodes=[s1n1 := ConstTrueNode(), s1n2 := ConstTrueNode()]
            ),
            sequence2 := Sequence(
                nodes=[s2n1 := ConstTrueNode(), s2n2 := ConstTrueNode()]
            ),
        ]
    )
    msc.add_node(EndMotion.when_all_true(msc.nodes))

    sequence.plot_specs.visible = False
    s1n2.plot_specs.visible = False
    s2n2.plot_specs.visible = False

    kin_sim = Executor(world=World())
    kin_sim.compile(motion_statechart=msc)
    msc.draw("muh.pdf")


def test_joint_goal():
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
            name=PrefixedName("dof", "a"), lower_limits=ll, upper_limits=ul
        )
        world.add_degree_of_freedom(dof)
        root_C_tip = RevoluteConnection(
            parent=root, child=tip, axis=Vector3.Z(), dof_id=dof.id
        )
        world.add_connection(root_C_tip)

        dof = DegreeOfFreedom(
            name=PrefixedName("dof", "b"), lower_limits=ll, upper_limits=ul
        )
        world.add_degree_of_freedom(dof)
        root_C_tip2 = RevoluteConnection(
            parent=root, child=tip2, axis=Vector3.Z(), dof_id=dof.id
        )
        world.add_connection(root_C_tip2)

    msc = MotionStatechart()

    task1 = JointPositionList(goal_state=JointState({root_C_tip: 1}))
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
        world=world,
    )
    kin_sim.compile(motion_statechart=msc)

    assert task1.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN
    assert task1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    msc.draw("muh.pdf")
    kin_sim.tick_until_end()
    msc.draw("muh.pdf")
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


class TestConditions:
    def test_InvalidConditionError(self):
        node = ConstTrueNode()
        with pytest.raises(InputNotExpressionError):
            node.end_condition = node

    def test_nodes_cannot_have_themselves_as_start_condition(self):
        msc = MotionStatechart()
        node1 = ConstTrueNode()
        msc.add_node(node1)
        with pytest.raises(SelfInStartConditionError):
            node1.start_condition = node1.observation_variable

    def test_non_observation_variable_in_condition(self):
        msc = MotionStatechart()
        msc.add_node(node := ConstTrueNode())
        with pytest.raises(NonObservationVariableError):
            node.start_condition = FloatVariable(name="muh")

    def test_add_node_to_multiple_goals(self):
        msc = MotionStatechart()
        node = ConstTrueNode()
        msc.add_node(Sequence([node]))
        msc.add_node(Sequence([node]))

        kin_sim = Executor(
            world=World(),
        )
        with pytest.raises(NodeAlreadyBelongsToDifferentNodeError):
            kin_sim.compile(motion_statechart=msc)

    def test_add_node_to_multiple_goals2(self):
        msc = MotionStatechart()
        node = ConstTrueNode()
        msc.add_node(node)
        msc.add_node(Sequence([node]))

        kin_sim = Executor(
            world=World(),
        )
        with pytest.raises(NodeAlreadyBelongsToDifferentNodeError):
            kin_sim.compile(motion_statechart=msc)


def test_two_goals(pr2_world_state_reset: World):
    torso_joint = pr2_world_state_reset.get_connection_by_name("torso_lift_joint")
    r_wrist_roll_joint = pr2_world_state_reset.get_connection_by_name(
        "r_wrist_roll_joint"
    )
    msc = MotionStatechart()
    msc.add_nodes(
        [
            JointPositionList(goal_state=JointState({torso_joint: 0.1})),
            local_min := LocalMinimumReached(),
        ]
    )
    msc.add_node(EndMotion.when_true(local_min))

    kin_sim = Executor(
        world=pr2_world_state_reset,
    )
    kin_sim.compile(motion_statechart=msc)

    kin_sim.tick_until_end()
    assert np.isclose(torso_joint.position, 0.1, atol=1e-4)

    msc = MotionStatechart()
    msc.add_node(
        joint_goal := JointPositionList(goal_state=JointState({r_wrist_roll_joint: 1}))
    )
    msc.add_node(EndMotion.when_true(joint_goal))

    kin_sim = Executor(
        world=pr2_world_state_reset,
    )
    kin_sim.compile(motion_statechart=msc)

    kin_sim.tick_until_end()
    assert np.isclose(torso_joint.position, 0.1, atol=1e-4)
    assert np.allclose(pr2_world_state_reset.state.velocities, 0)
    assert np.allclose(pr2_world_state_reset.state.accelerations, 0)
    assert np.allclose(pr2_world_state_reset.state.jerks, 0)


def test_reset():
    msc = MotionStatechart()
    node1 = ConstTrueNode()
    msc.add_node(node1)
    node2 = ConstTrueNode()
    msc.add_node(node2)
    node3 = ConstTrueNode()
    msc.add_node(node3)
    end = EndMotion()
    msc.add_node(end)
    node1.reset_condition = node2.observation_variable
    node2.start_condition = node1.observation_variable
    node3.start_condition = node2.observation_variable
    node2.end_condition = node2.observation_variable
    end.start_condition = trinary_logic_and(
        node1.observation_variable,
        node2.observation_variable,
        node3.observation_variable,
    )

    kin_sim = Executor(world=World())
    kin_sim.compile(motion_statechart=msc)
    msc.draw("muh.pdf")

    kin_sim.tick()
    assert node1.observation_state == ObservationStateValues.UNKNOWN
    assert node2.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    kin_sim.tick()
    assert node1.observation_state == ObservationStateValues.TRUE
    assert node2.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node2.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    kin_sim.tick()
    assert node1.observation_state == ObservationStateValues.TRUE
    assert node2.observation_state == ObservationStateValues.TRUE
    assert end.observation_state == ObservationStateValues.UNKNOWN
    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert node2.life_cycle_state == LifeCycleValues.DONE
    assert node3.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    kin_sim.tick()
    assert node1.observation_state == ObservationStateValues.UNKNOWN
    assert node2.observation_state == ObservationStateValues.TRUE
    assert node3.observation_state == ObservationStateValues.TRUE
    assert end.observation_state == ObservationStateValues.UNKNOWN
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node2.life_cycle_state == LifeCycleValues.DONE
    assert node3.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    kin_sim.tick()
    assert node1.observation_state == ObservationStateValues.TRUE
    assert node2.observation_state == ObservationStateValues.TRUE
    assert node3.observation_state == ObservationStateValues.TRUE
    assert end.observation_state == ObservationStateValues.UNKNOWN
    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert node2.life_cycle_state == LifeCycleValues.DONE
    assert node3.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert not msc.is_end_motion()

    kin_sim.tick()
    assert node1.observation_state == ObservationStateValues.UNKNOWN
    assert node2.observation_state == ObservationStateValues.TRUE
    assert node3.observation_state == ObservationStateValues.TRUE
    assert end.observation_state == ObservationStateValues.TRUE
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node2.life_cycle_state == LifeCycleValues.DONE
    assert node3.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert msc.is_end_motion()


def test_nested_goals():
    msc = MotionStatechart()

    node1 = ConstTrueNode()
    msc.add_node(node1)

    outer = TestNestedGoal()
    msc.add_node(outer)
    outer.start_condition = node1.observation_variable

    end = EndMotion()
    msc.add_node(end)
    end.start_condition = outer.observation_variable

    json_data = msc.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    msc_copy = MotionStatechart.from_json(new_json_data)

    for node in msc.nodes:
        assert node.index == msc_copy.get_node_by_index(node.index).index

    kin_sim = Executor(world=World())
    node1 = msc_copy.get_nodes_by_type(ConstTrueNode)[0]
    outer = msc_copy.get_nodes_by_type(TestNestedGoal)[0]
    end = msc_copy.get_nodes_by_type(EndMotion)[0]
    kin_sim.compile(motion_statechart=msc_copy)
    msc_copy.draw("muh.pdf")
    assert node1.observation_state == ObservationStateValues.UNKNOWN
    assert outer.inner.sub_node1.observation_state == ObservationStateValues.UNKNOWN
    assert outer.inner.sub_node2.observation_state == ObservationStateValues.UNKNOWN
    assert outer.inner.observation_state == ObservationStateValues.UNKNOWN
    assert outer.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert outer.inner.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert outer.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc_copy.is_end_motion()

    kin_sim.tick()
    msc_copy.draw("muh.pdf")
    assert node1.observation_state == ObservationStateValues.UNKNOWN
    assert outer.inner.sub_node1.observation_state == ObservationStateValues.UNKNOWN
    assert outer.inner.sub_node2.observation_state == ObservationStateValues.UNKNOWN
    assert outer.inner.observation_state == ObservationStateValues.UNKNOWN
    assert outer.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert outer.inner.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert outer.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc_copy.is_end_motion()

    kin_sim.tick()
    msc_copy.draw("muh.pdf")
    assert node1.observation_state == ObservationStateValues.TRUE
    assert outer.inner.sub_node1.observation_state == ObservationStateValues.UNKNOWN
    assert outer.inner.sub_node2.observation_state == ObservationStateValues.UNKNOWN
    assert outer.inner.observation_state == ObservationStateValues.UNKNOWN
    assert outer.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc_copy.is_end_motion()

    kin_sim.tick()
    msc_copy.draw("muh.pdf")
    assert node1.observation_state == ObservationStateValues.TRUE
    assert outer.inner.sub_node1.observation_state == ObservationStateValues.TRUE
    assert outer.inner.sub_node2.observation_state == ObservationStateValues.UNKNOWN
    assert outer.inner.observation_state == ObservationStateValues.UNKNOWN
    assert outer.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc_copy.is_end_motion()

    kin_sim.tick()
    msc_copy.draw("muh.pdf")
    assert node1.observation_state == ObservationStateValues.TRUE
    assert outer.inner.sub_node1.observation_state == ObservationStateValues.TRUE
    assert outer.inner.sub_node2.observation_state == ObservationStateValues.TRUE
    assert outer.inner.observation_state == ObservationStateValues.UNKNOWN
    assert outer.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc_copy.is_end_motion()

    kin_sim.tick()
    msc_copy.draw("muh.pdf")
    assert node1.observation_state == ObservationStateValues.TRUE
    assert outer.inner.sub_node1.observation_state == ObservationStateValues.TRUE
    assert outer.inner.sub_node2.observation_state == ObservationStateValues.TRUE
    assert outer.inner.observation_state == ObservationStateValues.TRUE
    assert outer.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc_copy.is_end_motion()

    kin_sim.tick()
    msc_copy.draw("muh.pdf")
    assert node1.observation_state == ObservationStateValues.TRUE
    assert outer.inner.sub_node1.observation_state == ObservationStateValues.TRUE
    assert outer.inner.sub_node2.observation_state == ObservationStateValues.TRUE
    assert outer.inner.observation_state == ObservationStateValues.TRUE
    assert outer.observation_state == ObservationStateValues.TRUE
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert not msc_copy.is_end_motion()

    kin_sim.tick()
    msc_copy.draw("muh.pdf")
    assert node1.observation_state == ObservationStateValues.TRUE
    assert outer.inner.sub_node1.observation_state == ObservationStateValues.TRUE
    assert outer.inner.sub_node2.observation_state == ObservationStateValues.TRUE
    assert outer.inner.observation_state == ObservationStateValues.TRUE
    assert outer.observation_state == ObservationStateValues.TRUE
    assert end.observation_state == ObservationStateValues.TRUE

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert msc_copy.is_end_motion()


@dataclass(eq=False, repr=False)
class _TestThreadMonitor(ThreadPayloadMonitor):
    delay: float = 0.05
    return_value: float = ObservationStateValues.TRUE

    def _compute_observation(self):
        time.sleep(self.delay)
        return self.return_value


def test_thread_payload_monitor_non_blocking_and_caching():
    msc = MotionStatechart()
    mon = _TestThreadMonitor(
        delay=0.05,
        return_value=ObservationStateValues.TRUE,
    )
    msc.add_node(mon)
    # First call should be non-blocking and return Unknown until worker completes at least once
    start = time.perf_counter()
    val0 = mon.compute_observation()
    elapsed = time.perf_counter() - start
    assert elapsed < mon.delay / 4.0
    assert val0 == ObservationStateValues.UNKNOWN
    # Wait for worker to finish and cache
    time.sleep(mon.delay * 2)
    val1 = mon.compute_observation()
    assert val1 == ObservationStateValues.TRUE


def test_goal():
    msc = MotionStatechart()

    node1 = ConstTrueNode()
    msc.add_node(node1)

    goal = TestGoal()
    msc.add_node(goal)

    goal.start_condition = node1.observation_variable

    end = EndMotion()
    msc.add_node(end)
    end.start_condition = goal.observation_variable

    kin_sim = Executor(world=World())

    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()
    assert len(msc.history) == 7
    # %% goal
    assert msc.history.get_life_cycle_history_of_node(goal) == [
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
    ]
    assert msc.history.get_observation_history_of_node(goal) == [
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
    ]
    # %% node1
    assert msc.history.get_life_cycle_history_of_node(node1) == [
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
    ]
    assert msc.history.get_observation_history_of_node(node1) == [
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
    ]
    # %% sub_node1
    assert msc.history.get_life_cycle_history_of_node(goal.sub_node1) == [
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.RUNNING,
        LifeCycleValues.DONE,
        LifeCycleValues.DONE,
        LifeCycleValues.DONE,
        LifeCycleValues.DONE,
    ]
    assert msc.history.get_observation_history_of_node(goal.sub_node1) == [
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
    ]
    # %% sub_node2
    assert msc.history.get_life_cycle_history_of_node(goal.sub_node2) == [
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
    ]
    assert msc.history.get_observation_history_of_node(goal.sub_node2) == [
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
    ]
    # %% sub_node2
    assert msc.history.get_life_cycle_history_of_node(end) == [
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
    ]
    assert msc.history.get_observation_history_of_node(end) == [
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.TRUE,
    ]
    msc.draw("muh.pdf")


def test_set_seed_configuration(pr2_world_state_reset):
    msc = MotionStatechart()
    goal = 0.1

    connection: ActiveConnection1DOF = pr2_world_state_reset.get_connection_by_name(
        "torso_lift_joint"
    )

    node1 = SetSeedConfiguration(seed_configuration=JointState({connection: goal}))
    end = EndMotion()
    msc.add_node(node1)
    msc.add_node(end)
    node1.end_condition = node1.observation_variable
    end.start_condition = node1.observation_variable

    kin_sim = Executor(world=pr2_world_state_reset)
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

    kin_sim = Executor(world=pr2_world_state_reset)
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


def test_continuous_joint(pr2_world_state_reset):
    msc = MotionStatechart()
    joint_goal = JointPositionList(
        goal_state=JointState.from_str_dict(
            {
                "r_wrist_roll_joint": -np.pi,
                "l_wrist_roll_joint": -2.1 * np.pi,
            },
            world=pr2_world_state_reset,
        ),
    )
    msc.add_node(joint_goal)
    end = EndMotion()
    msc.add_node(end)
    end.start_condition = joint_goal.observation_variable

    kin_sim = Executor(
        world=pr2_world_state_reset,
    )
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()


def test_revolute_joint(pr2_world_state_reset):
    msc = MotionStatechart()
    joint_goal = JointPositionList(
        goal_state=JointState.from_str_dict(
            {
                "head_pan_joint": 0.041880780651479044,
                "head_tilt_joint": -0.37,
            },
            world=pr2_world_state_reset,
        ),
    )
    msc.add_node(joint_goal)
    end = EndMotion()
    msc.add_node(end)
    end.start_condition = joint_goal.observation_variable

    kin_sim = Executor(
        world=pr2_world_state_reset,
    )
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()


def test_cart_goal_1eef(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "base_footprint"
    )
    tip_goal = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=-0.2, reference_frame=tip
    )

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
        world=pr2_world_state_reset,
    )
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()


def test_long_goal(pr2_world_state_reset: World):
    msc = MotionStatechart()
    msc.add_nodes(
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
                        "torso_lift_joint": 0.2999225173357618,
                        "head_pan_joint": 0.041880780651479044,
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
        world=pr2_world_state_reset,
    )
    kin_sim.compile(motion_statechart=msc)
    t = time.perf_counter()
    kin_sim.tick_until_end(1_000_000)
    after = time.perf_counter()
    diff = after - t
    print(diff / kin_sim.control_cycles)


class TestCartesianTasks:
    """Test suite for all Cartesian motion tasks."""

    # ===== EXISTING TESTS (moved from standalone functions into this class) =====

    def test_cart_goal_1eef(self, pr2_world_state_reset: World):
        """Single CartesianPose goal test."""
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        tip_goal = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=-0.2, reference_frame=tip
        )

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

        kin_sim = Executor(world=pr2_world_state_reset)
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()


def test_cart_goal_sequence_at_build(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name("base_footprint")
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")

    tip_goal1 = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=-0.2, reference_frame=tip
    )
    tip_goal2 = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=0.2, reference_frame=tip
    )

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
        world=pr2_world_state_reset,
    )

    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()

    fk = pr2_world_state_reset.compute_forward_kinematics_np(root, tip)
    assert np.allclose(fk, tip_goal2.to_np(), atol=cart_goal2.threshold)


def test_cart_goal_sequence_on_start(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name("base_footprint")
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")

    tip_goal1 = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=-0.2, reference_frame=tip
    )
    tip_goal2 = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=0.2, reference_frame=tip
    )

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
        world=pr2_world_state_reset,
    )
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()

    fk = pr2_world_state_reset.compute_forward_kinematics_np(root, tip)
    expected = np.eye(4)
    assert np.allclose(fk, expected, atol=cart_goal2.threshold)


def test_CartesianOrientation(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name("base_footprint")
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")

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
        world=pr2_world_state_reset,
    )
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()

    fk = pr2_world_state_reset.compute_forward_kinematics_np(root, tip)
    assert np.allclose(fk, tip_goal.to_np(), atol=cart_goal.threshold)

    def test_cartesian_position_sequence_at_build(self, pr2_world: World):
        """Test CartesianPosition with Bind_at_build policy."""
        tip = pr2_world.get_kinematic_structure_entity_by_name("base_footprint")
        root = pr2_world.get_kinematic_structure_entity_by_name("odom_combined")

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

        kin_sim = Executor(world=pr2_world)
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        fk = pr2_world.compute_forward_kinematics_np(root, tip)
        # goal2 was captured at build time, so should end at that position
        expected = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=0.2, reference_frame=pr2_world.root
        ).to_np()
        assert np.allclose(fk[:3, 3], expected[:3, 3], atol=cart_goal2.threshold)

    def test_cartesian_position_sequence_on_start(self, pr2_world: World):
        """Test CartesianPosition with Bind_on_start policy (default)."""
        tip = pr2_world.get_kinematic_structure_entity_by_name("base_footprint")
        root = pr2_world.get_kinematic_structure_entity_by_name("odom_combined")

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

        kin_sim = Executor(world=pr2_world)
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        fk = pr2_world.compute_forward_kinematics_np(root, tip)
        # Both goals captured when tasks start, so should return near origin
        expected = np.eye(4)
        assert np.allclose(fk[:3, 3], expected[:3, 3], atol=cart_goal2.threshold)

    def test_cartesian_orientation_sequence_at_build(self, pr2_world: World):
        """Test CartesianOrientation with Bind_at_build policy."""
        tip = pr2_world.get_kinematic_structure_entity_by_name("base_footprint")
        root = pr2_world.get_kinematic_structure_entity_by_name("odom_combined")

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

        kin_sim = Executor(world=pr2_world)
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        fk = pr2_world.compute_forward_kinematics_np(root, tip)
        # goal2 captured at build, so ends at -pi/6 from original
        expected = tip_rot2.to_np()
        assert np.allclose(fk, expected, atol=cart_goal2.threshold)

    def test_cartesian_orientation_sequence_on_start(self, pr2_world: World):
        """Test CartesianOrientation with Bind_on_start policy (default)."""
        tip = pr2_world.get_kinematic_structure_entity_by_name("base_footprint")
        root = pr2_world.get_kinematic_structure_entity_by_name("odom_combined")

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

        kin_sim = Executor(world=pr2_world)
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        fk = pr2_world.compute_forward_kinematics_np(root, tip)
        # Both goals captured when tasks start, so rotates +pi/6 then -pi/6 = back to origin
        expected = np.eye(4)
        assert np.allclose(fk, expected, atol=cart_goal2.threshold)

    def test_cartesian_position_straight(self, pr2_world: World):
        """Test CartesianPositionStraight basic functionality."""
        tip = pr2_world.get_kinematic_structure_entity_by_name("base_footprint")
        root = pr2_world.get_kinematic_structure_entity_by_name("odom_combined")

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

        kin_sim = Executor(world=pr2_world)
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        # Verify task detected completion
        assert cart_straight.observation_state == ObservationStateValues.TRUE


def test_pointing(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")

    msc = MotionStatechart()

    goal_point = Point3(2, 0, 0, reference_frame=root)
    pointing_axis = Vector3.X(reference_frame=tip)

    pointing = Pointing(
        root_link=root,
        tip_link=tip,
        goal_point=goal_point,
        pointing_axis=pointing_axis,
    )
    msc.add_node(pointing)
    end = EndMotion()
    msc.add_node(end)
    end.start_condition = pointing.observation_variable

    kin_sim = Executor(
        world=pr2_world_state_reset,
    )
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()


def test_pointing_cone(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")

    msc = MotionStatechart()

    goal_point = Point3(-1, 0, 5, reference_frame=root)
    pointing_axis = Vector3.X(tip)
    cone_theta = radians(20)
    pointing_cone = PointingCone(
        root_link=root,
        tip_link=tip,
        goal_point=goal_point,
        pointing_axis=pointing_axis,
        cone_theta=cone_theta,
    )
    msc.add_node(pointing_cone)
    end = EndMotion()
    msc.add_node(end)
    end.start_condition = pointing_cone.observation_variable

    kin_sim = Executor(
        world=pr2_world_state_reset,
    )
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()

    # Check if angle between pointing axis and tip->goal vector is within the cone
    root_V_pointing_axis = pr2_world_state_reset.transform(
        target_frame=root, spatial_object=pointing_axis
    )
    root_V_pointing_axis.scale(1)
    v_pointing = root_V_pointing_axis.to_np()[:3]

    root_P_goal = pr2_world_state_reset.transform(
        target_frame=root, spatial_object=goal_point
    )
    tip_origin_in_root = pr2_world_state_reset.transform(
        target_frame=root, spatial_object=Point3(0, 0, 0, reference_frame=tip)
    )
    root_V_goal_axis = root_P_goal - tip_origin_in_root
    root_V_goal_axis.scale(1)
    v_goal = root_V_goal_axis.to_np()[:3]

    eps = 1e-9
    assert np.linalg.norm(v_pointing) > eps, "pointing axis became zero-length"
    assert np.linalg.norm(v_goal) > eps, "tip->goal vector became zero-length"

    angle = angle_between_vector(v_pointing, v_goal)

    assert (
        angle <= cone_theta + pointing_cone.threshold
    ), f"PointingCone failed: final angle {angle:.6f} rad > cone_theta {cone_theta:.6f} rad"


def test_align_planes(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")

    msc = MotionStatechart()

    goal_normal = Vector3.X(reference_frame=root)
    tip_normal = Vector3.Y(reference_frame=tip)

    align_planes = AlignPlanes(
        root_link=root, tip_link=tip, goal_normal=goal_normal, tip_normal=tip_normal
    )
    msc.add_node(align_planes)

    end = EndMotion()
    msc.add_node(end)
    end.start_condition = align_planes.observation_variable

    kin_sim = Executor(
        world=pr2_world_state_reset,
    )
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()

    # Check if the angle between normal vectors is below the threshold
    root_V_goal_normal = pr2_world_state_reset.transform(
        target_frame=root, spatial_object=goal_normal
    )
    root_V_goal_normal.scale(1)
    root_V_tip_normal = pr2_world_state_reset.transform(
        target_frame=root, spatial_object=tip_normal
    )
    root_V_tip_normal.scale(1)
    v_tip = root_V_tip_normal.to_np()[:3]
    v_goal = root_V_goal_normal.to_np()[:3]

    eps = 1e-9
    assert np.linalg.norm(v_goal) > eps, "goal normal became zero-length"
    assert np.linalg.norm(v_tip) > eps, "tip normal became zero-length"

    angle = angle_between_vector(v_tip, v_goal)

    assert (
        angle <= align_planes.threshold
    ), f"AlignPlanes failed: final angle {angle:.6f} rad > threshold {align_planes.threshold:.6f} rad"


def test_angle_goal(pr2_world_state_reset: World):
    """
    Ensure AngleGoal drives the angle between tip_vector and reference_vector
    into the interval [lower_angle, upper_angle].
    """
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")

    msc = MotionStatechart()

    tip_vector = Vector3.Y(reference_frame=tip)
    reference_vector = Vector3.X(reference_frame=root)

    lower_angle = radians(30)
    upper_angle = radians(32)

    angle_goal = AngleGoal(
        root_link=root,
        tip_link=tip,
        tip_vector=tip_vector,
        reference_vector=reference_vector,
        lower_angle=lower_angle,
        upper_angle=upper_angle,
    )
    msc.add_node(angle_goal)

    msc.add_node(EndMotion.when_true(angle_goal))

    kin_sim = Executor(world=pr2_world_state_reset)
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()

    root_V_tip = pr2_world_state_reset.transform(
        target_frame=root, spatial_object=tip_vector
    )
    root_V_tip.scale(1)
    root_V_ref = pr2_world_state_reset.transform(
        target_frame=root, spatial_object=reference_vector
    )
    root_V_ref.scale(1)

    v_tip = root_V_tip.to_np()[:3]
    v_ref = root_V_ref.to_np()[:3]

    eps = 1e-9
    assert np.linalg.norm(v_tip) > eps, "tip_vector became zero-length"
    assert np.linalg.norm(v_ref) > eps, "reference_vector became zero-length"

    angle = angle_between_vector(v_tip, v_ref)

    assert (
        lower_angle <= angle <= upper_angle
    ), f"AngleGoal failed: final angle {angle:.6f} rad not in [{lower_angle:.6f}, {upper_angle:.6f}]"


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
        kin_sim = Executor(world=world)
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
            )
        else:
            goal = CartesianOrientation(
                root_link=root,
                tip_link=tip,
                goal_orientation=RotationMatrix.from_rpy(
                    yaw=np.pi / 2, reference_frame=tip
                ),
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

        kin_sim = Executor(world=pr2_world_state_reset)
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


def test_transition_triggers():
    msc = MotionStatechart()

    changer = ChangeStateOnEvents()
    msc.add_node(changer)

    node1 = Pulse()
    msc.add_node(node1)

    node2 = Pulse()
    msc.add_node(node2)
    node2.start_condition = node1.observation_variable

    node3 = Pulse()
    msc.add_node(node3)
    node3.start_condition = trinary_logic_and(
        trinary_logic_not(node1.observation_variable),
        trinary_logic_not(node2.observation_variable),
    )

    node4 = Pulse()
    msc.add_node(node4)
    node4.start_condition = node3.observation_variable

    changer.start_condition = node1.observation_variable
    changer.pause_condition = node2.observation_variable
    changer.end_condition = node3.observation_variable
    changer.reset_condition = node4.observation_variable

    kin_sim = Executor(world=World())
    kin_sim.compile(motion_statechart=msc)

    assert changer.state is None

    kin_sim.tick()
    msc.draw("muh.pdf")
    kin_sim.tick()
    msc.draw("muh.pdf")
    assert changer.life_cycle_state == LifeCycleValues.RUNNING
    assert changer.state == "on_start"

    kin_sim.tick()
    msc.draw("muh.pdf")
    assert changer.life_cycle_state == LifeCycleValues.PAUSED
    assert changer.state == "on_pause"

    kin_sim.tick()
    msc.draw("muh.pdf")
    assert changer.life_cycle_state == LifeCycleValues.RUNNING
    assert changer.state == "on_unpause"

    kin_sim.tick()
    msc.draw("muh.pdf")
    assert changer.life_cycle_state == LifeCycleValues.DONE
    assert changer.state == "on_end"

    kin_sim.tick()
    msc.draw("muh.pdf")
    assert changer.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert changer.state == "on_reset"


def test_not_not_in_motion_statechart():
    node = ConstTrueNode()
    with pytest.raises(NotInMotionStatechartError):
        muh = node.observation_variable
    with pytest.raises(NotInMotionStatechartError):
        muh = node.life_cycle_variable
    with pytest.raises(NotInMotionStatechartError):
        node.start_condition = node.observation_variable
    with pytest.raises(NotInMotionStatechartError):
        node.pause_condition = node.observation_variable
    with pytest.raises(NotInMotionStatechartError):
        node.end_condition = node.observation_variable
    with pytest.raises(NotInMotionStatechartError):
        node.reset_condition = node.observation_variable


def test_counting():
    class FakeClock:
        def __init__(self):
            self.t = 0.0

        def time(self):
            return self.t

        def advance(self, dt):
            self.t += dt

    clock = FakeClock()

    msc = MotionStatechart()
    seconds = 1
    msc.add_nodes(
        [counter := CountSeconds(seconds=seconds, _now=clock.time), pulse := Pulse()]
    )

    pulse.start_condition = counter.observation_variable
    counter.reset_condition = pulse.observation_variable

    msc.add_node(end := EndMotion())

    end.start_condition = trinary_logic_and(
        counter.observation_variable, trinary_logic_not(pulse.observation_variable)
    )

    kin_sim = Executor(
        world=World(),
        pacer=SimulationPacer(real_time_factor=None),
    )
    kin_sim.compile(motion_statechart=msc)

    # Advance fake time deterministically without wall-clock sleeps
    step = 0.1
    while not msc.is_end_motion():
        kin_sim.tick()
        clock.advance(step)
        if kin_sim.control_cycles > 1000:
            raise TimeoutError("test stuck")

    # it takes 2 * seconds to finish the counters
    # + 1 for pulse to trigger
    # + 1 for reset
    # + 1 for EndMotion to transition to RUNNING
    # + 1 for EndMotion to observe True
    assert np.allclose(seconds * 2 + 0.4, clock.time())


def test_count_ticks():
    msc = MotionStatechart()
    msc.add_node(counter := CountTicks(ticks=3))
    msc.add_node(EndMotion.when_true(counter))
    kin_sim = Executor(world=World())
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()
    # ending tacks 2 ticks, one to turn EndMotion to Running and one more to turn it to true
    assert kin_sim.control_cycles == 3 + 2


class TestEndMotion:
    def test_end_motion_when_all_done1(self):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                ConstTrueNode(),
                ConstTrueNode(),
            ]
        )
        end = EndMotion.when_all_true(msc.nodes)
        msc.add_node(end)

        kin_sim = Executor(
            world=World(),
        )
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()
        msc.draw("muh.pdf")
        assert end.life_cycle_state == LifeCycleValues.RUNNING

    def test_end_motion_when_all_done2(self):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                ConstTrueNode(),
                ConstFalseNode(),
            ]
        )
        end = EndMotion.when_all_true(msc.nodes)
        msc.add_node(end)

        kin_sim = Executor(
            world=World(),
        )
        kin_sim.compile(motion_statechart=msc)
        with pytest.raises(TimeoutError):
            kin_sim.tick_until_end()
        msc.draw("muh.pdf")
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED

    def test_end_motion_when_any_done1(self):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                ConstTrueNode(),
                ConstFalseNode(),
            ]
        )
        end = EndMotion.when_any_true(msc.nodes)
        msc.add_node(end)

        kin_sim = Executor(
            world=World(),
        )
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()
        msc.draw("muh.pdf")
        assert end.life_cycle_state == LifeCycleValues.RUNNING

    def test_end_motion_when_any_done2(self):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                ConstFalseNode(),
                ConstFalseNode(),
            ]
        )
        end = EndMotion.when_any_true(msc.nodes)
        msc.add_node(end)

        kin_sim = Executor(
            world=World(),
        )
        kin_sim.compile(motion_statechart=msc)
        with pytest.raises(TimeoutError):
            kin_sim.tick_until_end()
        msc.draw("muh.pdf")
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED

    def test_goals_cannot_have_end_motion(self):
        msc = MotionStatechart()
        msc.add_node(Sequence([ConstTrueNode(), EndMotion()]))
        with pytest.raises(EndMotionInGoalError):
            kin_sim = Executor(
                world=World(),
            )
            kin_sim.compile(motion_statechart=msc)


class TestParallel:
    def test_parallel(self):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                parallel := Parallel([CountTicks(ticks=3), CountTicks(ticks=5)]),
            ]
        )
        msc.add_node(EndMotion.when_true(parallel))

        kin_sim = Executor(
            world=World(),
        )
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()
        # 5 (longest ticker) + 1 (for parallel to turn True) + 2 (for end to trigger)
        assert kin_sim.control_cycles == 8

    def test_parallel_with_tasks(self, pr2_world_state_reset: World):
        map = pr2_world_state_reset.root
        r_tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        msc = MotionStatechart()
        msc.add_node(
            parallel := Parallel(
                [
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
        msc.add_node(EndMotion.when_true(parallel))

        kin_sim = Executor(
            world=pr2_world_state_reset,
        )
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()


class TestOpenClose:
    def test_open(self, pr2_world_state_reset):
        factory = DoorFactory(
            name=PrefixedName("door"),
            handle_factory=HandleFactory(name=PrefixedName("handle")),
            semantic_position=SemanticPositionDescription(
                horizontal_direction_chain=[
                    HorizontalSemanticDirection.RIGHT,
                    HorizontalSemanticDirection.FULLY_CENTER,
                ],
                vertical_direction_chain=[VerticalSemanticDirection.FULLY_CENTER],
            ),
        )
        door_world = factory.create()
        with pr2_world_state_reset.modify_world():
            lower_limits = DerivativeMap()
            lower_limits.position = -np.pi / 2
            lower_limits.velocity = -1
            upper_limits = DerivativeMap()
            upper_limits.position = np.pi / 2
            upper_limits.velocity = 1
            dof = DegreeOfFreedom(
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                name=PrefixedName("hinge"),
            )
            pr2_world_state_reset.add_degree_of_freedom(dof)
            root_T_door = RevoluteConnection(
                dof_id=dof.id,
                parent=pr2_world_state_reset.root,
                child=door_world.root,
                axis=-Vector3.Z(),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1.5, z=1, yaw=np.pi, reference_frame=pr2_world_state_reset.root
                ),
            )
            pr2_world_state_reset.merge_world(door_world, root_connection=root_T_door)

        r_tip = pr2_world_state_reset.get_body_by_name("r_gripper_tool_frame")
        handle = pr2_world_state_reset.get_semantic_annotations_by_type(Handle)[0].body
        open_goal = 1
        close_goal = -1

        msc = MotionStatechart()
        msc.add_nodes(
            [
                Sequence(
                    [
                        CartesianPose(
                            root_link=pr2_world_state_reset.root,
                            tip_link=r_tip,
                            goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                                yaw=np.pi, reference_frame=handle
                            ),
                        ),
                        Parallel(
                            [
                                Open(
                                    tip_link=r_tip,
                                    environment_link=handle,
                                    goal_joint_state=open_goal,
                                ),
                                opened := JointPositionReached(
                                    connection=root_T_door,
                                    position=open_goal,
                                    name="opened",
                                ),
                            ]
                        ),
                        Parallel(
                            [
                                Close(
                                    tip_link=r_tip,
                                    environment_link=handle,
                                    goal_joint_state=close_goal,
                                ),
                                closed := JointPositionReached(
                                    connection=root_T_door,
                                    position=close_goal,
                                    name="closed",
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        )
        msc.add_node(EndMotion.when_true(msc.nodes[0]))

        kin_sim = Executor(
            world=pr2_world_state_reset,
        )
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()
        msc.draw("muh.pdf")

        assert opened.observation_state == ObservationStateValues.TRUE
        assert closed.observation_state == ObservationStateValues.TRUE


class TestCollisionAvoidance:
    def test_collision_avoidance(self, box_bot_world: World):
        tip = box_bot_world.get_kinematic_structure_entity_by_name("bot")

        msc = MotionStatechart()
        msc.add_nodes(
            [
                CartesianPose(
                    root_link=box_bot_world.root,
                    tip_link=tip,
                    goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=1, reference_frame=box_bot_world.root
                    ),
                ),
                CollisionAvoidance(
                    collision_entries=[CollisionRequest.avoid_all_collision()],
                ),
                local_min := LocalMinimumReached(),
            ]
        )
        msc.add_node(EndMotion.when_true(local_min))

        json_data = msc.to_json()
        json_str = json.dumps(json_data)
        new_json_data = json.loads(json_str)

        tracker = KinematicStructureEntityKwargsTracker.from_world(box_bot_world)
        kwargs = tracker.create_kwargs()
        msc_copy = MotionStatechart.from_json(new_json_data, **kwargs)

        kin_sim = Executor(
            world=box_bot_world,
            collision_checker=CollisionCheckerLib.bpb,
        )
        kin_sim.compile(motion_statechart=msc_copy)

        msc_copy.draw("muh.pdf")
        kin_sim.tick_until_end(500)
        kin_sim.collision_scene.check_collisions()
        contact_distance = (
            kin_sim.collision_scene.closest_points.external_collisions[tip]
            .data[0]
            .contact_distance
        )
        assert contact_distance > 0.049

    def test_hard_constraints_violated(self, box_bot_world: World):
        root = box_bot_world.root
        with box_bot_world.modify_world():
            env2 = Body(
                name=PrefixedName("environment2"),
                collision=ShapeCollection(shapes=[Cylinder(width=0.5, height=0.1)]),
            )
            env_connection = FixedConnection(
                parent=root,
                child=env2,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    0.75
                ),
            )
            box_bot_world.add_connection(env_connection)

            env3 = Body(
                name=PrefixedName("environment3"),
                collision=ShapeCollection(shapes=[Cylinder(width=0.5, height=0.1)]),
            )
            env_connection = FixedConnection(
                parent=root,
                child=env3,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    1.25
                ),
            )
            box_bot_world.add_connection(env_connection)
            env4 = Body(
                name=PrefixedName("environment4"),
                collision=ShapeCollection(shapes=[Cylinder(width=0.5, height=0.1)]),
            )
            env_connection = FixedConnection(
                parent=root,
                child=env4,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1, y=-0.25
                ),
            )
            box_bot_world.add_connection(env_connection)
            env5 = Body(
                name=PrefixedName("environment5"),
                collision=ShapeCollection(shapes=[Cylinder(width=0.5, height=0.1)]),
            )
            env_connection = FixedConnection(
                parent=root,
                child=env5,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1, y=0.25
                ),
            )
            box_bot_world.add_connection(env_connection)

        tip = box_bot_world.get_kinematic_structure_entity_by_name("bot")

        msc = MotionStatechart()
        msc.add_node(
            Sequence(
                [
                    SetOdometry(
                        base_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                            x=1, reference_frame=box_bot_world.root
                        )
                    ),
                    Parallel(
                        [
                            CartesianPose(
                                root_link=box_bot_world.root,
                                tip_link=tip,
                                goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                                    x=1, reference_frame=box_bot_world.root
                                ),
                            ),
                            CollisionAvoidance(
                                collision_entries=[
                                    CollisionRequest.avoid_all_collision()
                                ],
                            ),
                        ]
                    ),
                ]
            )
        )
        msc.add_node(local_min := LocalMinimumReached())
        msc.add_node(EndMotion.when_true(local_min))

        json_data = msc.to_json()
        json_str = json.dumps(json_data)
        new_json_data = json.loads(json_str)

        tracker = KinematicStructureEntityKwargsTracker.from_world(box_bot_world)
        kwargs = tracker.create_kwargs()
        msc_copy = MotionStatechart.from_json(new_json_data, **kwargs)

        kin_sim = Executor(
            world=box_bot_world,
            collision_checker=CollisionCheckerLib.bpb,
        )
        kin_sim.compile(motion_statechart=msc_copy)

        msc_copy.draw("muh.pdf")
        with pytest.raises(HardConstraintsViolatedException):
            kin_sim.tick_until_end()


def test_constraint_collection(pr2_world_state_reset: World):
    """
    Test the constraint collection naming behavior. Expected behavior is:
    - Not naming constraints should result in automatically generated unique names
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

    col.add_point_goal_constraints(
        frame_P_current=Point3(0, 0, 0, reference_frame=tip),
        frame_P_goal=Point3(0, 0, 0, reference_frame=tip),
        reference_velocity=0.1,
        weight=DefaultWeights.WEIGHT_BELOW_CA,
    )
    assert len(col.eq_constraints) >= 3

    for i in range(3):
        col.add_equality_constraint(
            reference_velocity=0.1 * i,
            equality_bound=0.0,
            weight=DefaultWeights.WEIGHT_BELOW_CA,
            task_expression=expr,
        )

    col.add_inequality_constraint(
        name="same_name",
        reference_velocity=0.2,
        weight=DefaultWeights.WEIGHT_BELOW_CA,
        task_expression=expr,
        lower_error=0.1,
        upper_error=0.2,
    )

    with pytest.raises(DuplicateNameException):
        col.add_equality_constraint(
            name="same_name",
            reference_velocity=0.2,
            equality_bound=0.0,
            weight=DefaultWeights.WEIGHT_BELOW_CA,
            task_expression=expr,
        )

    col2 = ConstraintCollection()
    col2.add_equality_constraint(
        name="same_name",
        reference_velocity=0.2,
        equality_bound=0.0,
        weight=DefaultWeights.WEIGHT_BELOW_CA,
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
        weight=DefaultWeights.WEIGHT_BELOW_CA,
        task_expression=expr,
    )
    constraint = EqualityConstraint(
        name="same_name",
        expression=expr,
        bound=0.0,
        normalization_factor=0.1,
        quadratic_weight=DefaultWeights.WEIGHT_BELOW_CA,
        lower_slack_limit=-float("inf"),
        upper_slack_limit=float("inf"),
        linear_weight=0,
    )
    col3._constraints.append(constraint)

    with pytest.raises(DuplicateNameException):
        col3._are_names_unique()

    with pytest.raises(DuplicateNameException):
        col2.merge("", col3)


class TestLifeCycleTransitions:
    """
    Tests the LifeCycle transitions of nodes in various edge cases and intended behavior.
    """

    def test_run_after_stop(self):
        """
        Test for node to run after the parent node already stopped.
        """
        msc = MotionStatechart()

        msc.add_node(
            sequence := Sequence(
                [
                    ConstTrueNode(),
                    TestRunAfterStop(),
                    CountTicks(name="delay endmotion", ticks=5),
                ]
            )
        )
        msc.add_node(EndMotion.when_true(sequence))

        kin_sim = Executor(world=World())
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        assert sequence.nodes[1].cancel.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert sequence.nodes[1].ticking1.life_cycle_state == LifeCycleValues.DONE
        assert sequence.nodes[1].ticking2.life_cycle_state == LifeCycleValues.DONE
        assert sequence.nodes[1].life_cycle_state == LifeCycleValues.DONE

    def test_run_after_stop_from_pause(self):
        """
        Test for node to run from paused while the parent node already stopped.
        """
        msc = MotionStatechart()

        msc.add_node(
            sequence := Sequence(
                [
                    ConstTrueNode(),
                    TestRunAfterStopFromPause(),
                    CountTicks(name="delay endmotion", ticks=5),
                ]
            )
        )
        msc.add_node(EndMotion.when_true(sequence))

        kin_sim = Executor(world=World())
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        assert sequence.nodes[1].cancel.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert sequence.nodes[1].ticking1.life_cycle_state == LifeCycleValues.DONE
        assert sequence.nodes[1].ticking2.life_cycle_state == LifeCycleValues.DONE
        assert sequence.nodes[1].ticking3.life_cycle_state == LifeCycleValues.DONE
        assert sequence.nodes[1].pulse.life_cycle_state == LifeCycleValues.DONE
        assert sequence.nodes[1].life_cycle_state == LifeCycleValues.DONE

    def test_end_before_start(self):
        """
        Test for node to start even if it's end condition is met before start condition.
        Node3 should start and run for 1 tick before ending, instead of never starting.
        """
        msc = MotionStatechart()

        node1 = CountTicks(ticks=1)
        node2 = ConstTrueNode()
        node3 = ConstTrueNode()

        msc.add_nodes(nodes=[node1, node2, node3])
        msc.add_node(EndMotion.when_true(node3))

        node3.start_condition = node1.observation_variable
        node3.end_condition = node2.observation_variable

        kin_sim = Executor(world=World())
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick()
        kin_sim.tick()

        assert node3.life_cycle_state == LifeCycleValues.RUNNING
        assert node3.observation_state == ObservationStateValues.UNKNOWN

        kin_sim.tick()

        assert node3.life_cycle_state == LifeCycleValues.DONE
        assert node3.observation_state == ObservationStateValues.TRUE

    def test_end_before_start_in_template(self):
        """
        Test for node to start even if it's end condition is met before start condition,
        when the nodes are inside a template.
        """
        msc = MotionStatechart()

        node = TestEndBeforeStart()
        msc.add_node(node)

        kin_sim = Executor(world=World())
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick()
        kin_sim.tick()

        assert node.node3.life_cycle_state == LifeCycleValues.RUNNING
        assert node.node3.observation_state == ObservationStateValues.UNKNOWN

        kin_sim.tick()

        assert node.node3.life_cycle_state == LifeCycleValues.DONE
        assert node.node3.observation_state == ObservationStateValues.TRUE

    def test_intended_transitions(self):
        """
        Test for intended LifeCycle transitions of nodes.
        """
        msc = MotionStatechart()

        count_node1 = CountTicks(ticks=1, name="node1")
        count_node2 = CountTicks(ticks=2, name="node2")
        end_count_node1 = CountTicks(ticks=11, name="end_node1")
        pulse_node1 = Pulse(name="pulse1")
        pulse_node2 = Pulse(name="pulse2")

        msc.add_nodes(
            nodes=[
                count_node1,
                count_node2,
                end_count_node1,
                pulse_node1,
                pulse_node2,
            ]
        )
        msc.add_node(end_node := EndMotion.when_true(end_count_node1))

        pulse_node1.start_condition = count_node1.observation_variable
        pulse_node2.start_condition = count_node2.observation_variable
        count_node2.start_condition = pulse_node1.observation_variable

        count_node1.pause_condition = pulse_node1.observation_variable

        count_node1.end_condition = count_node2.observation_variable
        pulse_node1.end_condition = count_node2.observation_variable

        count_node1.reset_condition = pulse_node2.observation_variable
        count_node2.reset_condition = pulse_node2.observation_variable
        pulse_node1.reset_condition = pulse_node2.observation_variable

        kin_sim = Executor(world=World())
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        assert len(msc.history) == 14
        # %% count_node1 history
        assert msc.history.get_life_cycle_history_of_node(count_node1) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.PAUSED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.DONE,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.PAUSED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.DONE,
            LifeCycleValues.DONE,
            LifeCycleValues.DONE,
        ]
        assert msc.history.get_observation_history_of_node(count_node1) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
        ]

        # %% count_node2 history
        assert msc.history.get_life_cycle_history_of_node(count_node2) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.history.get_observation_history_of_node(count_node2) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.FALSE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.FALSE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
        ]

        # %% end_count_node1 history
        assert msc.history.get_life_cycle_history_of_node(end_count_node1) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.history.get_observation_history_of_node(end_count_node1) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
        ]

        # %% end_node history
        assert msc.history.get_life_cycle_history_of_node(end_node) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.history.get_observation_history_of_node(end_node) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
        ]

        # %% pulse_node1 history
        assert msc.history.get_life_cycle_history_of_node(pulse_node1) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.DONE,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.DONE,
            LifeCycleValues.DONE,
            LifeCycleValues.DONE,
        ]
        assert msc.history.get_observation_history_of_node(pulse_node1) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
        ]

        # %% pulse_node2 history
        assert msc.history.get_life_cycle_history_of_node(pulse_node2) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.history.get_observation_history_of_node(pulse_node2) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
        ]

    def test_unpause_unknown_from_parent_pause(self):
        """
        Test for child node to unpause when parent node unpauses.
        Child node pause condition is unknown.
        """

        msc = MotionStatechart()

        pulse = Pulse()
        unpause = TestUnpauseUnknownFromParentPause()

        msc.add_nodes(nodes=[pulse, unpause])
        msc.add_node(EndMotion.when_true(unpause))

        unpause.pause_condition = pulse.observation_variable

        kin_sim = Executor(world=World())
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        assert unpause.count_ticks1.life_cycle_state == LifeCycleValues.DONE
        assert unpause.cancel.life_cycle_state == LifeCycleValues.NOT_STARTED

        assert unpause.count_ticks1.observation_state == ObservationStateValues.TRUE
        assert unpause.observation_state == ObservationStateValues.TRUE
