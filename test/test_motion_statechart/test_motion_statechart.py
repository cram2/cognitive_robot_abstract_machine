import pytest
import time
from dataclasses import dataclass

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.motion_statechart.data_types import LifeCycleValues
from giskardpy.motion_statechart.graph_node import Goal
from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    EndMotion,
    CancelMotion,
)
from giskardpy.motion_statechart.graph_node import ThreadPayloadMonitor
from giskardpy.motion_statechart.monitors.monitors import TrueMonitor
from giskardpy.motion_statechart.monitors.payload_monitors import Print
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
    ObservationState,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import RevoluteConnection
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Body


def test_condition_to_str():
    msc = MotionStatechart(World())
    node1 = TrueMonitor(name=PrefixedName("muh"))
    msc.add_node(node1)
    node2 = TrueMonitor(name=PrefixedName("muh2"))
    msc.add_node(node2)
    node3 = TrueMonitor(name=PrefixedName("muh3"))
    msc.add_node(node3)
    end = EndMotion(name=PrefixedName("done"))
    msc.add_node(end)

    end.start_condition = cas.trinary_logic_and(
        node1.observation_variable,
        cas.trinary_logic_or(
            node2.observation_variable,
            cas.trinary_logic_not(node3.observation_variable),
        ),
    )
    a = str(end._start_condition)
    assert a == '("muh/observation" and ("muh2/observation" or not "muh3/observation"))'


def test_motion_statechart_to_dot():
    msc = MotionStatechart(World())
    node1 = TrueMonitor(name=PrefixedName("muh"))
    msc.add_node(node1)
    node2 = TrueMonitor(name=PrefixedName("muh2"))
    msc.add_node(node2)
    end = EndMotion(name=PrefixedName("done"))
    msc.add_node(end)
    node1.end_condition = node2.observation_variable
    end.start_condition = cas.trinary_logic_and(
        node1.observation_variable, node2.observation_variable
    )
    msc.draw()


@pytest.mark.skip(reason="not implemented yet")
def test_self_start_condition():
    pass


@pytest.mark.skip(reason="not implemented yet")
def test_all_conditions_with_goals():
    pass


@pytest.mark.skip(reason="not implemented yet")
def test_all_conditions_with_nodes():
    pass


@pytest.mark.skip(reason="not implemented yet")
def test_arrange_in_sequence():
    pass


@pytest.mark.skip(reason="not implemented yet")
def test_transition_hooks():
    pass


@pytest.mark.skip(reason="not implemented yet")
def test_optionality_of_qp_controller_in_compile():
    pass


@pytest.mark.skip(reason="not implemented yet")
def test_state_deletion():
    pass


def test_motion_statechart():
    msc = MotionStatechart(World())

    node1 = TrueMonitor(name=PrefixedName("muh"))
    msc.add_node(node1)
    node2 = TrueMonitor(name=PrefixedName("muh2"))
    msc.add_node(node2)
    node3 = TrueMonitor(name=PrefixedName("muh3"))
    msc.add_node(node3)
    end = EndMotion(name=PrefixedName("done"))
    msc.add_node(end)

    node1.start_condition = cas.trinary_logic_or(
        node3.observation_variable, node2.observation_variable
    )
    end.start_condition = node1.observation_variable
    msc.compile()

    assert len(msc.nodes) == 4
    assert len(msc.edges) == 3

    assert node1.observation_state == msc.observation_state.TrinaryUnknown
    assert node2.observation_state == msc.observation_state.TrinaryUnknown
    assert node3.observation_state == msc.observation_state.TrinaryUnknown
    assert end.observation_state == msc.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert node3.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()
    msc.tick()
    assert node1.observation_state == msc.observation_state.TrinaryUnknown
    assert node2.observation_state == msc.observation_state.TrinaryUnknown
    assert node3.observation_state == msc.observation_state.TrinaryUnknown
    assert end.observation_state == msc.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert node2.life_cycle_state == LifeCycleValues.RUNNING
    assert node3.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()
    msc.tick()
    assert node1.observation_state == msc.observation_state.TrinaryUnknown
    assert node2.observation_state == msc.observation_state.TrinaryTrue
    assert node3.observation_state == msc.observation_state.TrinaryTrue
    assert end.observation_state == msc.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node2.life_cycle_state == LifeCycleValues.RUNNING
    assert node3.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()
    msc.tick()
    assert node1.observation_state == msc.observation_state.TrinaryTrue
    assert node2.observation_state == msc.observation_state.TrinaryTrue
    assert node3.observation_state == msc.observation_state.TrinaryTrue
    assert end.observation_state == msc.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node2.life_cycle_state == LifeCycleValues.RUNNING
    assert node3.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert not msc.is_end_motion()
    msc.tick()
    assert node1.observation_state == msc.observation_state.TrinaryTrue
    assert node2.observation_state == msc.observation_state.TrinaryTrue
    assert node3.observation_state == msc.observation_state.TrinaryTrue
    assert end.observation_state == msc.observation_state.TrinaryTrue
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node2.life_cycle_state == LifeCycleValues.RUNNING
    assert node3.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert msc.is_end_motion()


def test_duplicate_name():
    msc = MotionStatechart(World())

    with pytest.raises(ValueError):
        cas.FloatVariable(name=PrefixedName("muh"))
        msc.add_node(TrueMonitor(name=PrefixedName("muh")))
        msc.add_node(TrueMonitor(name=PrefixedName("muh")))


def test_print():
    msc = MotionStatechart(World())
    print_node1 = Print(name=PrefixedName("cow"), message="muh")
    msc.add_node(print_node1)
    print_node2 = Print(name=PrefixedName("cow2"), message="muh")
    msc.add_node(print_node2)

    node1 = TrueMonitor(name=PrefixedName("muh"))
    msc.add_node(node1)
    end = EndMotion(name=PrefixedName("done"))
    msc.add_node(end)

    node1.start_condition = print_node1.observation_variable
    print_node2.start_condition = node1.observation_variable
    end.start_condition = print_node2.observation_variable
    msc.compile()

    assert len(msc.nodes) == 4
    assert len(msc.edges) == 3

    assert print_node1.observation_state == msc.observation_state.TrinaryUnknown
    assert node1.observation_state == msc.observation_state.TrinaryUnknown
    assert print_node2.observation_state == msc.observation_state.TrinaryUnknown
    assert end.observation_state == msc.observation_state.TrinaryUnknown

    assert print_node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert print_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    msc.tick()
    assert print_node1.observation_state == msc.observation_state.TrinaryUnknown
    assert node1.observation_state == msc.observation_state.TrinaryUnknown
    assert print_node2.observation_state == msc.observation_state.TrinaryUnknown
    assert end.observation_state == msc.observation_state.TrinaryUnknown

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert print_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    msc.tick()
    assert print_node1.observation_state == msc.observation_state.TrinaryTrue
    assert node1.observation_state == msc.observation_state.TrinaryUnknown
    assert print_node2.observation_state == msc.observation_state.TrinaryUnknown
    assert end.observation_state == msc.observation_state.TrinaryUnknown

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert print_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    msc.tick()
    assert print_node1.observation_state == msc.observation_state.TrinaryTrue
    assert node1.observation_state == msc.observation_state.TrinaryTrue
    assert print_node2.observation_state == msc.observation_state.TrinaryUnknown
    assert end.observation_state == msc.observation_state.TrinaryUnknown

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert print_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    msc.tick()
    assert print_node1.observation_state == msc.observation_state.TrinaryTrue
    assert node1.observation_state == msc.observation_state.TrinaryTrue
    assert print_node2.observation_state == msc.observation_state.TrinaryTrue
    assert end.observation_state == msc.observation_state.TrinaryUnknown

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert print_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert not msc.is_end_motion()

    msc.tick()
    assert print_node1.observation_state == msc.observation_state.TrinaryTrue
    assert node1.observation_state == msc.observation_state.TrinaryTrue
    assert print_node2.observation_state == msc.observation_state.TrinaryTrue
    assert end.observation_state == msc.observation_state.TrinaryTrue

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert print_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert msc.is_end_motion()


def test_cancel_motion():
    msc = MotionStatechart(World())
    node1 = TrueMonitor(name=PrefixedName("muh"))
    msc.add_node(node1)
    cancel = CancelMotion(name=PrefixedName("done"), exception=Exception("test"))
    msc.add_node(cancel)
    cancel.start_condition = node1.observation_variable

    msc.compile()
    msc.tick()  # first tick, cancel motion node1 turns true
    msc.tick()  # second tick, cancel goes into running
    with pytest.raises(Exception):
        msc.tick()  # third tick, cancel goes true and triggers
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
            parent=root, child=tip, axis=cas.Vector3.Z(), dof_name=dof.name
        )
        world.add_connection(root_C_tip)

        dof = DegreeOfFreedom(
            name=PrefixedName("dof", "b"), lower_limits=ll, upper_limits=ul
        )
        world.add_degree_of_freedom(dof)
        root_C_tip2 = RevoluteConnection(
            parent=root, child=tip2, axis=cas.Vector3.Z(), dof_name=dof.name
        )
        world.add_connection(root_C_tip2)

    msc = MotionStatechart(world)

    task1 = JointPositionList(name=PrefixedName("task1"), goal_state={root_C_tip: 1})
    always_true = TrueMonitor(name=PrefixedName("muh"))
    msc.add_node(always_true)
    msc.add_node(task1)
    end = EndMotion(name=PrefixedName("done"))
    msc.add_node(end)

    task1.start_condition = always_true.observation_variable
    end.start_condition = cas.trinary_logic_and(
        task1.observation_variable, always_true.observation_variable
    )

    msc.compile(QPControllerConfig.create_default_with_50hz())

    assert task1.observation_state == msc.observation_state.TrinaryUnknown
    assert end.observation_state == msc.observation_state.TrinaryUnknown
    assert task1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    msc.draw("muh.pdf")
    for i in range(100):
        msc.tick()
        if msc.is_end_motion():
            break
    else:
        raise Exception("Did not finish motion")
    msc.draw("muh.pdf")
    assert task1.observation_state == msc.observation_state.TrinaryTrue
    assert end.observation_state == msc.observation_state.TrinaryTrue
    assert task1.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING


def test_reset():
    msc = MotionStatechart(World())
    node1 = TrueMonitor(name=PrefixedName("muh"))
    msc.add_node(node1)
    node2 = TrueMonitor(name=PrefixedName("muh2"))
    msc.add_node(node2)
    node3 = TrueMonitor(name=PrefixedName("muh3"))
    msc.add_node(node3)
    end = EndMotion(name=PrefixedName("done"))
    msc.add_node(end)
    node1.reset_condition = node2.observation_variable
    node2.start_condition = node1.observation_variable
    node3.start_condition = node2.observation_variable
    node2.end_condition = node2.observation_variable
    end.start_condition = cas.trinary_logic_and(
        node1.observation_variable,
        node2.observation_variable,
        node3.observation_variable,
    )

    msc.compile()

    msc.tick()
    assert node1.observation_state == msc.observation_state.TrinaryUnknown
    assert node2.observation_state == msc.observation_state.TrinaryUnknown
    assert end.observation_state == msc.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    msc.tick()
    assert node1.observation_state == msc.observation_state.TrinaryTrue
    assert node2.observation_state == msc.observation_state.TrinaryUnknown
    assert end.observation_state == msc.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node2.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    msc.tick()
    assert node1.observation_state == msc.observation_state.TrinaryTrue
    assert node2.observation_state == msc.observation_state.TrinaryTrue
    assert end.observation_state == msc.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert node2.life_cycle_state == LifeCycleValues.DONE
    assert node3.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    msc.tick()
    assert node1.observation_state == msc.observation_state.TrinaryUnknown
    assert node2.observation_state == msc.observation_state.TrinaryTrue
    assert node3.observation_state == msc.observation_state.TrinaryTrue
    assert end.observation_state == msc.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node2.life_cycle_state == LifeCycleValues.DONE
    assert node3.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    msc.tick()
    assert node1.observation_state == msc.observation_state.TrinaryTrue
    assert node2.observation_state == msc.observation_state.TrinaryTrue
    assert node3.observation_state == msc.observation_state.TrinaryTrue
    assert end.observation_state == msc.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert node2.life_cycle_state == LifeCycleValues.DONE
    assert node3.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert not msc.is_end_motion()

    msc.tick()
    assert node1.observation_state == msc.observation_state.TrinaryUnknown
    assert node2.observation_state == msc.observation_state.TrinaryTrue
    assert node3.observation_state == msc.observation_state.TrinaryTrue
    assert end.observation_state == msc.observation_state.TrinaryTrue
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node2.life_cycle_state == LifeCycleValues.DONE
    assert node3.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert msc.is_end_motion()


def test_nested_goals():
    msc = MotionStatechart(World())

    node1 = TrueMonitor(name=PrefixedName("start"))
    msc.add_node(node1)

    # inner goal with two sub-nodes
    inner = Goal(name=PrefixedName("inner"))
    sub_node1 = TrueMonitor(name=PrefixedName("inner sub 1"))
    msc.add_node(sub_node1)
    sub_node2 = TrueMonitor(name=PrefixedName("inner sub 2"))
    msc.add_node(sub_node2)
    inner.add_node(sub_node1)
    inner.add_node(sub_node2)
    sub_node1.end_condition = sub_node1.observation_variable
    sub_node2.start_condition = sub_node1.observation_variable
    inner._create_observation_expression = lambda: sub_node2.observation_variable

    # outer goal that contains the inner goal as a node
    outer = Goal(name=PrefixedName("outer"))
    msc.add_node(outer)
    outer.add_node(inner)
    outer._create_observation_expression = lambda: inner.observation_variable
    outer.start_condition = node1.observation_variable

    end = EndMotion(name=PrefixedName("done nested"))
    msc.add_node(end)
    end.start_condition = outer.observation_variable

    # compile and check initial states
    msc.compile()
    msc.draw()
    assert node1.observation_state == msc.observation_state.TrinaryUnknown
    assert sub_node1.observation_state == msc.observation_state.TrinaryUnknown
    assert sub_node2.observation_state == msc.observation_state.TrinaryUnknown
    assert inner.observation_state == msc.observation_state.TrinaryUnknown
    assert outer.observation_state == msc.observation_state.TrinaryUnknown
    assert end.observation_state == msc.observation_state.TrinaryUnknown

    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert sub_node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert sub_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert inner.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert outer.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    # tick 1: start trigger begins running
    msc.tick()
    msc.draw()
    assert node1.observation_state == msc.observation_state.TrinaryUnknown
    assert sub_node1.observation_state == msc.observation_state.TrinaryUnknown
    assert sub_node2.observation_state == msc.observation_state.TrinaryUnknown
    assert inner.observation_state == msc.observation_state.TrinaryUnknown
    assert outer.observation_state == msc.observation_state.TrinaryUnknown
    assert end.observation_state == msc.observation_state.TrinaryUnknown

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert sub_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert inner.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert outer.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    # tick 2: start trigger turns true; inner sub_node1 already resolves to True and sub_node2 starts
    msc.tick()
    msc.draw()
    assert node1.observation_state == msc.observation_state.TrinaryTrue
    assert sub_node1.observation_state == msc.observation_state.TrinaryUnknown
    assert sub_node2.observation_state == msc.observation_state.TrinaryUnknown
    assert inner.observation_state == msc.observation_state.TrinaryUnknown
    assert outer.observation_state == msc.observation_state.TrinaryUnknown
    assert end.observation_state == msc.observation_state.TrinaryUnknown

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert inner.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    # tick 3: inner sub_node2 turns true (inner goal still evaluating)
    msc.tick()
    msc.draw()
    assert node1.observation_state == msc.observation_state.TrinaryTrue
    assert sub_node1.observation_state == msc.observation_state.TrinaryTrue
    assert sub_node2.observation_state == msc.observation_state.TrinaryUnknown
    assert inner.observation_state == msc.observation_state.TrinaryUnknown
    assert outer.observation_state == msc.observation_state.TrinaryUnknown
    assert end.observation_state == msc.observation_state.TrinaryUnknown

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert inner.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    # tick 4: inner sub_node2 turns true
    msc.tick()
    msc.draw()
    assert node1.observation_state == msc.observation_state.TrinaryTrue
    assert sub_node1.observation_state == msc.observation_state.TrinaryTrue
    assert sub_node2.observation_state == msc.observation_state.TrinaryTrue
    assert inner.observation_state == msc.observation_state.TrinaryUnknown
    assert outer.observation_state == msc.observation_state.TrinaryUnknown
    assert end.observation_state == msc.observation_state.TrinaryUnknown

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert inner.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    # tick 5: inner goal becomes true, outer still running, end starts running
    msc.tick()
    msc.draw()
    assert node1.observation_state == msc.observation_state.TrinaryTrue
    assert sub_node1.observation_state == msc.observation_state.TrinaryTrue
    assert sub_node2.observation_state == msc.observation_state.TrinaryTrue
    assert inner.observation_state == msc.observation_state.TrinaryTrue
    assert outer.observation_state == msc.observation_state.TrinaryUnknown
    assert end.observation_state == msc.observation_state.TrinaryUnknown

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert inner.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    # tick 6: outer goal becomes true; end still running
    msc.tick()
    msc.draw()
    assert node1.observation_state == msc.observation_state.TrinaryTrue
    assert sub_node1.observation_state == msc.observation_state.TrinaryTrue
    assert sub_node2.observation_state == msc.observation_state.TrinaryTrue
    assert inner.observation_state == msc.observation_state.TrinaryTrue
    assert outer.observation_state == msc.observation_state.TrinaryTrue
    assert end.observation_state == msc.observation_state.TrinaryUnknown

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert inner.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert not msc.is_end_motion()

    # tick 7: end motion becomes true
    msc.tick()
    msc.draw()
    assert end.observation_state == msc.observation_state.TrinaryTrue
    assert msc.is_end_motion()


@dataclass(eq=False, repr=False)
class _TestThreadMonitor(ThreadPayloadMonitor):
    delay: float = 0.05
    return_value: float = ObservationState.TrinaryTrue

    def _compute_observation(self):
        time.sleep(self.delay)
        return self.return_value


def test_thread_payload_monitor_non_blocking_and_caching():
    msc = MotionStatechart(World())
    mon = _TestThreadMonitor(
        name=PrefixedName("thread_mon"),
        delay=0.05,
        return_value=ObservationState.TrinaryTrue,
    )
    msc.add_node(mon)
    # First call should be non-blocking and return Unknown until worker completes at least once
    start = time.perf_counter()
    val0 = mon.compute_observation()
    elapsed = time.perf_counter() - start
    assert elapsed < mon.delay / 4.0
    assert val0 == ObservationState.TrinaryUnknown
    # Wait for worker to finish and cache
    time.sleep(mon.delay * 2)
    val1 = mon.compute_observation()
    assert val1 == ObservationState.TrinaryTrue


def test_thread_payload_monitor_integration():
    msc = MotionStatechart(World())
    mon = _TestThreadMonitor(
        name=PrefixedName("thread_mon2"),
        delay=0.03,
        return_value=ObservationState.TrinaryTrue,
    )
    msc.add_node(mon)
    end = EndMotion(name=PrefixedName("done thread"))
    msc.add_node(end)
    end.start_condition = mon.observation_variable

    msc.compile()

    # tick 1: monitor not started yet becomes RUNNING; end not started
    msc.tick()
    assert mon.observation_state == ObservationState.TrinaryUnknown
    assert mon.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED

    # tick 2: compute_observation is triggered asynchronously; still Unknown immediately
    msc.tick()
    assert mon.observation_state == ObservationState.TrinaryUnknown
    assert mon.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED

    # allow background to finish and propagate on next tick
    time.sleep(mon.delay * 2)
    msc.tick()
    assert mon.observation_state == ObservationState.TrinaryTrue
    assert end.life_cycle_state == LifeCycleValues.RUNNING

    # next tick the EndMotion should turn true
    msc.tick()
    assert end.observation_state == ObservationState.TrinaryTrue


def test_goal():
    msc = MotionStatechart(World())

    node1 = TrueMonitor(name=PrefixedName("muh"))
    msc.add_node(node1)

    goal = Goal(name=PrefixedName("goal"))
    msc.add_node(goal)
    sub_node1 = TrueMonitor(name=PrefixedName("sub muh1"))
    goal.add_node(sub_node1)
    sub_node2 = TrueMonitor(name=PrefixedName("sub muh2"))
    goal.add_node(sub_node2)
    goal.add_node(sub_node1)
    goal.add_node(sub_node2)
    sub_node1.end_condition = sub_node1.observation_variable
    sub_node2.start_condition = sub_node1.observation_variable
    goal._create_observation_expression = lambda: sub_node2.observation_variable
    goal.start_condition = node1.observation_variable

    end = EndMotion(name=PrefixedName("done"))
    msc.add_node(end)
    end.start_condition = goal.observation_variable

    msc.compile()
    assert node1.observation_state == msc.observation_state.TrinaryUnknown
    assert sub_node1.observation_state == msc.observation_state.TrinaryUnknown
    assert sub_node2.observation_state == msc.observation_state.TrinaryUnknown
    assert goal.observation_state == msc.observation_state.TrinaryUnknown
    assert end.observation_state == msc.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert sub_node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert sub_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert goal.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    msc.tick()
    assert node1.observation_state == msc.observation_state.TrinaryUnknown
    assert sub_node1.observation_state == msc.observation_state.TrinaryUnknown
    assert sub_node2.observation_state == msc.observation_state.TrinaryUnknown
    assert goal.observation_state == msc.observation_state.TrinaryUnknown
    assert end.observation_state == msc.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert sub_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert goal.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    msc.tick()
    assert node1.observation_state == msc.observation_state.TrinaryTrue
    assert sub_node1.observation_state == msc.observation_state.TrinaryUnknown
    assert sub_node2.observation_state == msc.observation_state.TrinaryUnknown
    assert goal.observation_state == msc.observation_state.TrinaryUnknown
    assert end.observation_state == msc.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert goal.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    msc.tick()
    assert node1.observation_state == msc.observation_state.TrinaryTrue
    assert sub_node1.observation_state == msc.observation_state.TrinaryTrue
    assert sub_node2.observation_state == msc.observation_state.TrinaryUnknown
    assert goal.observation_state == msc.observation_state.TrinaryUnknown
    assert end.observation_state == msc.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert goal.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    msc.tick()
    assert node1.observation_state == msc.observation_state.TrinaryTrue
    assert sub_node1.observation_state == msc.observation_state.TrinaryTrue
    assert sub_node2.observation_state == msc.observation_state.TrinaryTrue
    assert goal.observation_state == msc.observation_state.TrinaryUnknown
    assert end.observation_state == msc.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert goal.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    msc.tick()
    assert node1.observation_state == msc.observation_state.TrinaryTrue
    assert sub_node1.observation_state == msc.observation_state.TrinaryTrue
    assert sub_node2.observation_state == msc.observation_state.TrinaryTrue
    assert goal.observation_state == msc.observation_state.TrinaryTrue
    assert end.observation_state == msc.observation_state.TrinaryUnknown
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert goal.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert not msc.is_end_motion()

    msc.tick()
    assert node1.observation_state == msc.observation_state.TrinaryTrue
    assert sub_node1.observation_state == msc.observation_state.TrinaryTrue
    assert sub_node2.observation_state == msc.observation_state.TrinaryTrue
    assert goal.observation_state == msc.observation_state.TrinaryTrue
    assert end.observation_state == msc.observation_state.TrinaryTrue
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert goal.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert msc.is_end_motion()
    msc.draw("muh.pdf")
