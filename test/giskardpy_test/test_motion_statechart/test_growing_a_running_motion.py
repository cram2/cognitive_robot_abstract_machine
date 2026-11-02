import pytest

from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues
from cramph.executor import StatechartExecutor
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from semantic_digital_twin.robots.pr2 import PR2Joint

# %% helpers


def _torso_goal(world, position: float) -> Sequence:
    """
    :return: A sequence moving the torso of the PR2 in `world` to `position`.
    """
    return Sequence(
        nodes=[
            JointPositionList(
                goal_state=JointState.from_str_dict(
                    {PR2Joint.TORSO_LIFT: position}, world=world
                )
            )
        ]
    )


def _torso_position(world) -> float:
    return world.get_connection_by_name(PR2Joint.TORSO_LIFT).position


# %% growing a running motion


def test_a_motion_added_while_running_moves_the_robot(pr2_world_state_reset) -> None:
    world = pr2_world_state_reset
    executor = StatechartExecutor(
        context=StatechartContext(world=world), extensions=[MotionControl()]
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(first := _torso_goal(world, 0.1))
    executor.compile(statechart)
    maximum_ticks = 1_000
    for _ in range(maximum_ticks):
        if first.life_cycle_state == LifeCycleValues.SUCCEEDED:
            break
        executor.tick()
    assert first.life_cycle_state == LifeCycleValues.SUCCEEDED

    second = _torso_goal(world, 0.2)
    second.start_condition = first.is_succeeded
    with statechart.modify():
        statechart.add_node(second)
        statechart.add_node(EndMotion.when_true(second))
    executor.tick_until_end(timeout=maximum_ticks)

    assert first.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert _torso_position(world) == pytest.approx(0.2, abs=0.01)
