from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from typing_extensions import ClassVar

from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.weights import DefaultWeights
from giskardpy.motion_statechart.graph_node import EndMotion, MotionNodeArtifacts, Task
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.qp.constraint_collection import ConstraintCollection
from giskardpy.qp.enforcement_strategy import IntegralStrategy
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from ..conftest import _make_prismatic_world, _symmetric_prismatic_limits

# %% helpers


@dataclass(kw_only=True, eq=False, repr=False)
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

    CRUISE_VELOCITY: ClassVar[float] = 0.05
    """
    A cruise velocity well below the joint's velocity limit, so that a cap derived from
    it would visibly slow the joint down.
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        artifacts = MotionNodeArtifacts()
        position = self.degree_of_freedom.variables.position
        artifacts.constraints.add_inequality_constraint(
            cruise_velocity=self.CRUISE_VELOCITY,
            velocity_scale=self.degree_of_freedom.limits.upper.velocity,
            lower_error=self.FLOOR_POSITION - position,
            upper_error=float("inf"),
            quadratic_weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
            task_expression=position,
        )
        return artifacts


@dataclass(kw_only=True, eq=False, repr=False)
class Ceiling(Task):
    """
    Keeps a joint below a ceiling, as a constraint that outweighs a goal.
    """

    degree_of_freedom: DegreeOfFreedom = field(kw_only=True)
    """
    The joint kept below the ceiling.
    """

    ceiling_position: float = field(kw_only=True)
    """
    Where the ceiling is.
    """

    cruise_velocity: float = field(kw_only=True)
    """
    The cruise velocity of the ceiling's constraint.
    """

    weight: float = DefaultWeights.WEIGHT_COLLISION_AVOIDANCE.value

    def build_artifacts(self) -> MotionNodeArtifacts:
        artifacts = MotionNodeArtifacts()
        position = self.degree_of_freedom.variables.position
        artifacts.constraints.add_inequality_constraint(
            cruise_velocity=self.cruise_velocity,
            velocity_scale=self.degree_of_freedom.limits.upper.velocity,
            lower_error=-float("inf"),
            upper_error=self.ceiling_position - position,
            quadratic_weight=self.weight,
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
    motion_statechart.add_node(node=joint_goal)
    if with_floor:
        motion_statechart.add_node(node=FarAwayFloor(degree_of_freedom=connection.dof))
    motion_statechart.add_node(node=EndMotion.when_true(node=joint_goal))

    executor.compile(statechart=motion_statechart)
    executor.tick_until_end()
    return executor.tick_count


# %% executed speed


@pytest.fixture(
    params=[
        QPControllerConfig.create_with_simulation_defaults,
        QPControllerConfig.create_with_physical_simulation_defaults,
    ],
    ids=["simulation", "physical"],
)
def controller_preset(request: pytest.FixtureRequest) -> QPControllerConfig:
    """
    :return: A controller configuration preset.
    """
    config = request.param()
    config.verbose = False
    return config


def cruise_speed_of_a_far_joint_goal(
    cruise_velocity: float, config: QPControllerConfig
) -> float:
    """
    Runs a joint goal so far away that it is never reached on a joint with a velocity
    limit of 1 m/s.

    :param cruise_velocity: The cruise velocity of the joint goal.
    :param config: The controller configuration.
    :return: The mean velocity of the joint over the last of three seconds.
    """
    world = _make_prismatic_world([_symmetric_prismatic_limits(1000, 1)])
    connection = next(c for c in world.connections if isinstance(c, ActiveConnection))
    executor = StatechartExecutor(
        context=StatechartContext(world=world),
        extensions=[MotionControl(qp_controller_config=config)],
    )
    motion_statechart = Statechart(context=executor.context)
    motion_statechart.add_node(
        node=JointPositionList(
            goal_state=JointState.from_mapping({connection: 500.0}),
            cruise_velocity=cruise_velocity,
        )
    )
    executor.compile(statechart=motion_statechart)
    control_cycles_per_second = round(1 / config.control_time_step.total_seconds())
    velocities = []
    for _ in range(3 * control_cycles_per_second):
        executor.tick()
        velocities.append(world.state[connection.dof.id].velocity)
    return float(np.mean(velocities[-control_cycles_per_second:]))


def test_a_far_goal_moves_at_its_cruise_velocity(controller_preset):
    """
    A slow goal is not held back by the jerk limit, so its executed velocity is its
    cruise velocity up to how the jerk limit shapes the plan.
    """
    cruise_velocity = 0.05

    speed = cruise_speed_of_a_far_joint_goal(cruise_velocity, controller_preset)

    assert speed == pytest.approx(cruise_velocity, rel=0.15)


@pytest.mark.parametrize("cruise_velocity", [0.05, 0.2, 0.5])
def test_a_far_goal_never_moves_faster_than_its_cruise_velocity(
    cruise_velocity, controller_preset
):
    speed = cruise_speed_of_a_far_joint_goal(cruise_velocity, controller_preset)

    assert speed <= cruise_velocity


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


# %% priorities


def penetration_into_ceiling(
    ceiling_cruise_velocity: float = 0.2,
    goal_cruise_velocity: float = 0.2,
    goal_above_ceiling: float = 0.6,
    control_cycles: int = 200,
) -> float:
    """
    Pushes a joint goal against a :class:`Ceiling` below it until the joint rests.

    :param ceiling_cruise_velocity: The cruise velocity of the ceiling.
    :param goal_cruise_velocity: The cruise velocity of the joint goal.
    :param goal_above_ceiling: How far above the ceiling the joint goal is.
    :param control_cycles: How many control cycles to run.
    :return: How far above the ceiling the joint rests.
    """
    world = single_joint_world()
    connection = next(c for c in world.connections if isinstance(c, ActiveConnection))
    executor = StatechartExecutor(
        context=StatechartContext(world=world), extensions=[MotionControl()]
    )
    motion_statechart = Statechart(context=executor.context)
    ceiling_position = 0.3
    motion_statechart.add_node(
        node=JointPositionList(
            goal_state=JointState.from_mapping(
                {connection: ceiling_position + goal_above_ceiling}
            ),
            cruise_velocity=goal_cruise_velocity,
        )
    )
    motion_statechart.add_node(
        node=Ceiling(
            degree_of_freedom=connection.dof,
            ceiling_position=ceiling_position,
            cruise_velocity=ceiling_cruise_velocity,
        )
    )

    executor.compile(statechart=motion_statechart)
    for _ in range(control_cycles):
        executor.tick()
    return world.state[connection.dof.id].position - ceiling_position


def test_how_far_a_goal_pushes_into_a_constraint_does_not_depend_on_the_constraints_cruise_velocity():
    """
    Only the weights set which of two competing constraints wins, and by how much.
    """
    slow_ceiling = penetration_into_ceiling(ceiling_cruise_velocity=0.1)
    fast_ceiling = penetration_into_ceiling(ceiling_cruise_velocity=0.5)

    assert fast_ceiling == pytest.approx(slow_ceiling)


def test_how_far_a_goal_within_reach_pushes_into_a_constraint_does_not_depend_on_its_cruise_velocity():
    """
    A goal close enough to demand its whole error within the control horizon pushes with
    a force set by its weight alone.
    """
    slow_goal = penetration_into_ceiling(
        goal_cruise_velocity=0.05, goal_above_ceiling=0.002
    )
    fast_goal = penetration_into_ceiling(
        goal_cruise_velocity=0.5, goal_above_ceiling=0.002
    )

    assert fast_goal == pytest.approx(slow_goal)


def test_how_far_a_goal_out_of_reach_pushes_into_a_constraint_does_not_depend_on_its_cruise_velocity():
    """
    A goal too far away to demand its whole error within the control horizon still
    pushes with a force set by its weight alone, up to a percent: a slower goal is
    stiffer, which slightly changes the motion it plans while it pushes.
    """
    slow_goal = penetration_into_ceiling(goal_cruise_velocity=0.05)
    fast_goal = penetration_into_ceiling(goal_cruise_velocity=0.5)

    assert fast_goal == pytest.approx(slow_goal, rel=0.01)


def pull_of_a_violated_lower_bound(cruise_velocity: float) -> float:
    """
    Evaluates an inequality whose lower bound lies far out of reach above its joint.

    :param cruise_velocity: The cruise velocity of the inequality.
    :return: How hard the inequality pulls on the joint while it is at rest, up to a
        factor every constraint shares.
    """
    world = single_joint_world()
    degree_of_freedom = next(
        c for c in world.connections if isinstance(c, ActiveConnection)
    ).dof
    collection = ConstraintCollection()
    collection.add_inequality_constraint(
        cruise_velocity=cruise_velocity,
        velocity_scale=degree_of_freedom.limits.upper.velocity,
        lower_error=5.0,
        upper_error=float("inf"),
        quadratic_weight=DefaultWeights.WEIGHT_COLLISION_AVOIDANCE,
        task_expression=degree_of_freedom.variables.position,
    )
    strategy = IntegralStrategy(
        degrees_of_freedom=[],
        constraints=collection.inequality_constraints,
        qp_controller_config=QPControllerConfig(
            target_frequency=20, prediction_horizon=10
        ),
    )
    slack_weight = strategy.create_slack_variables().quadratic_weights.to_np()[0]
    capped_lower_bound = strategy.create_lower_bounds().to_np()[0]
    return slack_weight * capped_lower_bound


def test_how_hard_a_violated_inequality_pulls_back_does_not_depend_on_its_cruise_velocity():
    slow = pull_of_a_violated_lower_bound(cruise_velocity=0.05)
    fast = pull_of_a_violated_lower_bound(cruise_velocity=0.5)

    assert fast == pytest.approx(slow)
