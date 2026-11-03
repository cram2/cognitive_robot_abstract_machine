from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import ClassVar

from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.weights import DefaultWeights
from giskardpy.motion_statechart.graph_node import EndMotion, MotionNodeArtifacts, Task
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from ..conftest import _make_prismatic_world, _symmetric_prismatic_limits

# %% helpers


@dataclass(eq=False, repr=False)
class FarAwayFloor(Task):
    """
    Keeps a joint above a floor so far below it that the floor never comes into reach.
    """

    degree_of_freedom: DegreeOfFreedom = field(kw_only=True)
    """
    The joint kept above the floor.
    """

    FLOOR_POSITION: ClassVar[float] = -10.0
    """
    Where the floor is, far outside the joint's position limits.
    """

    REFERENCE_VELOCITY: ClassVar[float] = 0.05
    """
    A reference velocity well below the joint's velocity limit, so that a cap derived
    from it would visibly slow the joint down.
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        artifacts = MotionNodeArtifacts()
        position = self.degree_of_freedom.variables.position
        artifacts.constraints.add_inequality_constraint(
            reference_velocity=self.REFERENCE_VELOCITY,
            lower_error=self.FLOOR_POSITION - position,
            upper_error=float("inf"),
            quadratic_weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
            task_expression=position,
        )
        return artifacts


def single_joint_world() -> World:
    """
    A fresh world with one prismatic joint, limited to 1 m and 1 m/s.
    """
    return _make_prismatic_world([_symmetric_prismatic_limits(1, 1)])


def control_cycles_to_reach_joint_goal(world: World, with_floor: bool) -> int:
    """
    Runs a joint goal on the world's only joint and counts the control cycles it takes.

    :param world: A world with a single active joint.
    :param with_floor: Whether a :class:`FarAwayFloor` runs next to the joint goal.
    :return: The number of control cycles until the joint goal is reached.
    """
    connection = next(c for c in world.connections if isinstance(c, ActiveConnection))
    executor = StatechartExecutor(
        context=StatechartContext(world=world), extensions=[MotionControl()]
    )
    motion_statechart = Statechart(context=executor.context)
    joint_goal = JointPositionList(
        goal_state=JointState.from_mapping({connection: 0.9})
    )
    motion_statechart.add_node(joint_goal)
    if with_floor:
        motion_statechart.add_node(FarAwayFloor(degree_of_freedom=connection.dof))
    motion_statechart.add_node(EndMotion.when_true(joint_goal))

    executor.compile(statechart=motion_statechart)
    executor.tick_until_end()
    return executor.tick_count


# %% inequality caps


def test_an_inequality_far_out_of_reach_does_not_slow_down_a_goal():
    """
    An inequality only permits motion away from a floor that is out of reach, so it must
    not limit how fast the joint moves towards its goal.
    """
    cycles_without_floor = control_cycles_to_reach_joint_goal(
        single_joint_world(), with_floor=False
    )
    cycles_with_floor = control_cycles_to_reach_joint_goal(
        single_joint_world(), with_floor=True
    )

    assert cycles_with_floor == cycles_without_floor
