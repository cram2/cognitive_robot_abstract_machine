"""
Tests for the failures a plan raises about the motions it ran.
"""

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.nodes_for_testing.nodes_for_testing import (
    ConstFalseNode,
)
from semantic_digital_twin.world import World

from coraplex.exceptions import MotionDidNotFinish


def _running_motion() -> MotionStatechartNode:
    """
    :return: A node of a compiled statechart that has been ticked, so it is in a life
        cycle state a failure can report.
    """
    motion = ConstFalseNode(name="motion")
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(motion)
    executor = Executor(MotionStatechartContext(world=World()))
    executor.compile(motion_statechart=motion_statechart)
    executor.tick()
    return motion


def test_the_failure_names_the_state_each_unfinished_motion_is_in():
    """
    A plan reports its motions in the vocabulary the motion statechart itself uses, so
    nothing translates between two sets of state names.
    """
    motion = _running_motion()

    message = MotionDidNotFinish([motion]).error_message()

    assert f"{motion.unique_name} ({motion.life_cycle_state.name})" in message
