import pytest

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.motion_statechart.data_types import LifeCycleValues
from giskardpy.motion_statechart.goals.goal import Goal
from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    EndMotion,
    CancelMotion,
)
from giskardpy.motion_statechart.monitors.monitors import TrueMonitor
from giskardpy.motion_statechart.monitors.payload_monitors import Print
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import RevoluteConnection
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Body


def test_condition_to_str():
    msg = MotionStatechart(World())
    node1 = TrueMonitor(name=PrefixedName("muh"), motion_statechart=msg)
    node2 = TrueMonitor(name=PrefixedName("muh2"), motion_statechart=msg)
    node3 = TrueMonitor(name=PrefixedName("muh3"), motion_statechart=msg)
    end = EndMotion(name=PrefixedName("done"), motion_statechart=msg)

    end.start_condition = cas.trinary_logic_and(
        node1, cas.trinary_logic_or(node2, cas.trinary_logic_not(node3))
    )
    a = str(end._start_condition)
    assert a == '"muh" and ("muh2" or not "muh3")'


def test_motion_statechart_to_dot():
    msg = MotionStatechart(World())
    node1 = TrueMonitor(name=PrefixedName("muh"), motion_statechart=msg)
    msg.draw()


def test_motion_statechart():
    msg = MotionStatechart(World())

    node1 = TrueMonitor(name=PrefixedName("muh"), motion_statechart=msg)
    node2 = TrueMonitor(name=PrefixedName("muh2"), motion_statechart=msg)
    node3 = TrueMonitor(name=PrefixedName("muh3"), motion_statechart=msg)
    end = EndMotion(name=PrefixedName("done"), motion_statechart=msg)

    node1.start_condition = cas.trinary_logic_or(node3, node2)
    a = str(node1._start_condition)
    end.start_condition = node1
    assert len(msg.nodes) == 4
    assert len(msg.edges) == 3

    msg.compile()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node3] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[node2] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[node3] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[end] == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()
    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node3] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[node2] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[node3] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()
    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node3] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[node2] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[node3] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()
    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node3] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[node2] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[node3] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.RUNNING
    assert not msg.is_end_motion()
    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node3] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryTrue
    assert msg.life_cycle_state[node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[node2] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[node3] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.RUNNING
    assert msg.is_end_motion()


def test_duplicate_name():
    msg = MotionStatechart(World())

    with pytest.raises(ValueError):
        cas.Symbol(name=PrefixedName("muh"))
        MotionStatechartNode(name=PrefixedName("muh"), motion_statechart=msg)
        MotionStatechartNode(name=PrefixedName("muh"), motion_statechart=msg)


def test_print():
    msg = MotionStatechart(World())
    print_node1 = Print(name=PrefixedName("cow"), message="muh", motion_statechart=msg)
    print_node2 = Print(name=PrefixedName("cow2"), message="muh", motion_statechart=msg)

    node1 = TrueMonitor(name=PrefixedName("muh"), motion_statechart=msg)
    node1.start_condition = print_node1
    print_node2.start_condition = node1
    end = EndMotion(name=PrefixedName("done"), motion_statechart=msg)
    end.start_condition = print_node2
    assert len(msg.nodes) == 4
    assert len(msg.edges) == 3

    msg.compile()
    assert msg.observation_state[print_node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[print_node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown

    assert msg.life_cycle_state[print_node1] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[node1] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[print_node2] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[end] == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[print_node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[print_node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown

    assert msg.life_cycle_state[print_node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[node1] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[print_node2] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[end] == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[print_node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[print_node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown

    assert msg.life_cycle_state[print_node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[print_node2] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[end] == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[print_node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[print_node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown

    assert msg.life_cycle_state[print_node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[print_node2] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[print_node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[print_node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown

    assert msg.life_cycle_state[print_node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[print_node2] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.RUNNING
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[print_node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[print_node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryTrue

    assert msg.life_cycle_state[print_node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[print_node2] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.RUNNING
    assert msg.is_end_motion()


def test_cancel_motion():
    msg = MotionStatechart(World())
    node1 = TrueMonitor(name=PrefixedName("muh"), motion_statechart=msg)
    cancel = CancelMotion(
        name=PrefixedName("done"), motion_statechart=msg, exception=Exception("test")
    )
    cancel.start_condition = node1

    msg.compile()
    msg.tick()  # first tick, cancel motion node1 turns true
    msg.tick()  # second tick, cancel goes into running
    with pytest.raises(Exception):
        msg.tick()  # third tick, cancel goes true and triggers


def test_joint_goal():
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        tip = Body(name=PrefixedName("tip"))
        ul = DerivativeMap()
        ul.velocity = 1
        ll = DerivativeMap()
        ll.velocity = -1
        dof = DegreeOfFreedom(
            name=PrefixedName("dof"), lower_limits=ll, upper_limits=ul
        )
        world.add_degree_of_freedom(dof)
        root_C_tip = RevoluteConnection(
            parent=root, child=tip, axis=cas.Vector3.Z(), dof_name=dof.name
        )
        world.add_connection(root_C_tip)

    msg = MotionStatechart(world)

    task1 = JointPositionList(
        name=PrefixedName("task1"), goal_state={root_C_tip: 1}, motion_statechart=msg
    )
    end = EndMotion(name=PrefixedName("done"), motion_statechart=msg)
    end.start_condition = task1

    msg.compile()
    msg.tick()
    assert msg.observation_state[task1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[task1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.NOT_STARTED

    root_C_tip.position = 1

    msg.tick()
    assert msg.observation_state[task1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[task1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.RUNNING
    msg.tick()
    assert msg.observation_state[task1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryTrue
    assert msg.life_cycle_state[task1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.RUNNING


def test_reset():
    msg = MotionStatechart(World())
    node1 = TrueMonitor(name=PrefixedName("muh"), motion_statechart=msg)
    node2 = TrueMonitor(name=PrefixedName("muh2"), motion_statechart=msg)
    node3 = TrueMonitor(name=PrefixedName("muh3"), motion_statechart=msg)
    end = EndMotion(name=PrefixedName("done"), motion_statechart=msg)
    node1.reset_condition = node2
    node2.start_condition = node1
    node3.start_condition = node2
    node2.end_condition = node2
    end.start_condition = cas.trinary_logic_and(node1, node2, node3)

    msg.compile()

    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[node2] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[end] == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[node2] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[node2] == LifeCycleValues.DONE
    assert msg.life_cycle_state[node3] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node3] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[node2] == LifeCycleValues.DONE
    assert msg.life_cycle_state[node3] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node3] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[node2] == LifeCycleValues.DONE
    assert msg.life_cycle_state[node3] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.RUNNING
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node3] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryTrue
    assert msg.life_cycle_state[node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[node2] == LifeCycleValues.DONE
    assert msg.life_cycle_state[node3] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.RUNNING
    assert msg.is_end_motion()


def test_nested_goals():
    msg = MotionStatechart(World())

    node1 = TrueMonitor(name=PrefixedName("start"), motion_statechart=msg)

    # inner goal with two sub-nodes
    inner = Goal(name=PrefixedName("inner"), motion_statechart=msg)
    sub_node1 = TrueMonitor(name=PrefixedName("inner sub 1"), motion_statechart=msg)
    sub_node2 = TrueMonitor(name=PrefixedName("inner sub 2"), motion_statechart=msg)
    inner.add_node(sub_node1)
    inner.add_node(sub_node2)
    sub_node1.end_condition = sub_node1
    sub_node2.start_condition = sub_node1
    inner.observation_expression = sub_node2

    # outer goal that contains the inner goal as a node
    outer = Goal(name=PrefixedName("outer"), motion_statechart=msg)
    outer.add_node(inner)
    outer.observation_expression = inner
    outer.start_condition = node1

    end = EndMotion(name=PrefixedName("done nested"), motion_statechart=msg)
    end.start_condition = outer

    # compile and check initial states
    msg.compile()
    msg.draw()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[sub_node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[sub_node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[inner] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[outer] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown

    assert msg.life_cycle_state[node1] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[sub_node1] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[sub_node2] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[inner] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[outer] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[end] == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    # tick 1: start trigger begins running
    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[sub_node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[sub_node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[inner] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[outer] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown

    assert msg.life_cycle_state[node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[sub_node1] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[sub_node2] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[inner] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[outer] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[end] == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    # tick 2: start trigger turns true; inner sub_node1 already resolves to True and sub_node2 starts
    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[sub_node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[inner] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[outer] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown

    assert msg.life_cycle_state[node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[sub_node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[sub_node2] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[inner] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[outer] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    # tick 3: inner sub_node2 turns true (inner goal still evaluating)
    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[inner] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[outer] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown

    assert msg.life_cycle_state[node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[sub_node1] == LifeCycleValues.DONE
    assert msg.life_cycle_state[sub_node2] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[inner] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[outer] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    # tick 4: inner sub_node2 turns true
    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[inner] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[outer] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown

    assert msg.life_cycle_state[node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[sub_node1] == LifeCycleValues.DONE
    assert msg.life_cycle_state[sub_node2] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[inner] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[outer] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    # tick 5: inner goal becomes true, outer still running, end starts running
    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[inner] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[outer] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown

    assert msg.life_cycle_state[node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[sub_node1] == LifeCycleValues.DONE
    assert msg.life_cycle_state[sub_node2] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[inner] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[outer] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    # tick 6: outer goal becomes true; end still running
    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[inner] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[outer] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown

    assert msg.life_cycle_state[node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[sub_node1] == LifeCycleValues.DONE
    assert msg.life_cycle_state[sub_node2] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[inner] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[outer] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.RUNNING
    assert not msg.is_end_motion()

    # tick 7: end motion becomes true
    msg.tick()
    assert msg.observation_state[end] == msg.observation_state.TrinaryTrue
    assert msg.is_end_motion()


def test_goal():
    msg = MotionStatechart(World())

    node1 = TrueMonitor(name=PrefixedName("muh"), motion_statechart=msg)

    goal = Goal(name=PrefixedName("goal"), motion_statechart=msg)
    sub_node1 = TrueMonitor(name=PrefixedName("sub muh1"), motion_statechart=msg)
    sub_node2 = TrueMonitor(name=PrefixedName("sub muh2"), motion_statechart=msg)
    goal.add_node(sub_node1)
    goal.add_node(sub_node2)
    sub_node1.end_condition = sub_node1
    sub_node2.start_condition = sub_node1
    goal.observation_expression = sub_node2
    goal.start_condition = node1

    end = EndMotion(name=PrefixedName("done"), motion_statechart=msg)
    end.start_condition = goal

    msg.compile()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[sub_node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[sub_node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[goal] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[sub_node1] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[sub_node2] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[goal] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[end] == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[sub_node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[sub_node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[goal] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[sub_node1] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[sub_node2] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[goal] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[end] == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[sub_node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[goal] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[sub_node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[sub_node2] == LifeCycleValues.NOT_STARTED
    assert msg.life_cycle_state[goal] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[goal] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[sub_node1] == LifeCycleValues.DONE
    assert msg.life_cycle_state[sub_node2] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[goal] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[goal] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[sub_node1] == LifeCycleValues.DONE
    assert msg.life_cycle_state[sub_node2] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[goal] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[goal] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[sub_node1] == LifeCycleValues.DONE
    assert msg.life_cycle_state[sub_node2] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[goal] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.RUNNING
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[sub_node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[goal] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryTrue
    assert msg.life_cycle_state[node1] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[sub_node1] == LifeCycleValues.DONE
    assert msg.life_cycle_state[sub_node2] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[goal] == LifeCycleValues.RUNNING
    assert msg.life_cycle_state[end] == LifeCycleValues.RUNNING
    assert msg.is_end_motion()
