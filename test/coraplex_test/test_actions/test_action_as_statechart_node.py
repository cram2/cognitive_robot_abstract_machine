from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field

import pytest

from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import PlanExecutor
from coraplex.robot_plans.actions.base import (
    Action,
    GoalAction,
    ParallelAction,
    SequentialAction,
    TryAllAction,
)
from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.robot_plans.actions.core.pick_up import (
    GraspingAction,
    PickUpAction,
    ReachAction,
)
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
)
from cramph.composites import Parallel, Sequence
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.exceptions import (
    CompositeNodeWithoutChildrenError,
    NodesMissingContextExtensionsError,
    TickLimitReachedError,
)
from cramph.executor import StatechartExecutor
from cramph.node import EndStatechart, StatechartNode
from cramph.nodes_for_testing import (
    NodeFailingOnObservingFalse,
    NodeObservingAFixedValue,
    NodeSucceedingOnObservingTrue,
)
from cramph.statechart import Statechart
from cramph.world_modification_nodes import MoveBranch
from giskardpy.motion_statechart.goals.collision_avoidance import (
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
)
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.collision_checking.collision_matrix import CollisionRule
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForEndEffector,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types.spatial_types import Pose
from ..conftest import nodes_of_type
from ..test_designator.test_multi_robot_action_designator import _handle_annotation
from ...plan_running import (
    simulated_executor,
    statechart_of,
)
from cramph.context import ContextExtension, StatechartContext

# %% a body handed in, so expansion is exercised without a robot


@dataclass(eq=False, repr=False, kw_only=True)
class ActionRunningHandedInSteps(SequentialAction):
    """
    An action whose steps are handed to it rather than derived from a robot, so that
    expanding and running an action can be exercised on its own.
    """

    steps: list[StatechartNode] = field(default_factory=list)
    """
    The children of this action, in order.
    """

    def create_steps(self) -> list[StatechartNode]:
        return list(self.steps)


@dataclass(eq=False, repr=False, kw_only=True)
class ActionHoldingHandedInSteps(ParallelAction):
    """
    An action running the steps handed to it side by side.
    """

    steps: list[StatechartNode] = field(default_factory=list)
    """
    The children of this action.
    """

    def create_steps(self) -> list[StatechartNode]:
        return list(self.steps)


@dataclass(eq=False, repr=False, kw_only=True)
class ActionTryingHandedInSteps(TryAllAction):
    """
    An action trying the steps handed to it side by side as alternatives.
    """

    steps: list[StatechartNode] = field(default_factory=list)
    """
    The alternatives of this action.
    """

    def create_steps(self) -> list[StatechartNode]:
        return list(self.steps)


def _succeeding(name: str) -> NodeSucceedingOnObservingTrue:
    """
    :return: A node that ends itself successfully as soon as it observes.
    """
    return NodeSucceedingOnObservingTrue(
        name=name, observation=ObservationStateValues.TRUE
    )


def _failing(name: str) -> NodeFailingOnObservingFalse:
    """
    :return: A node that declares itself failed as soon as it observes.
    """
    return NodeFailingOnObservingFalse(
        name=name, observation=ObservationStateValues.FALSE
    )


def _expanded(
    action: Action,
    extensions: list[ContextExtension],
    executor: PlanExecutor | None = None,
) -> Statechart:
    """
    Adds `action` to a statechart, which expands it.

    :param executor: The executor whose context the statechart is built in, a simulated
        one by default.
    :return: The statechart holding the expanded action.
    """
    if executor is None:
        executor = simulated_executor(extensions)
    return statechart_of(executor, action)


def _run_until(statechart: Statechart, end: EndStatechart) -> None:
    """
    Compiles `statechart` and ticks it until `end` ends it.
    """
    statechart.add_node(end)
    executor = StatechartExecutor(context=statechart.context)
    executor.compile(statechart)
    executor.tick_until_end(timeout=100)


# %% an action is a statechart node


def test_a_sequential_action_is_the_sequence_of_its_steps(simple_pr2_context) -> None:
    """
    The steps of an action are its own children, without a sequence between them and
    the action.
    """
    _, _, extensions = simple_pr2_context
    steps = [_succeeding("first"), _succeeding("second")]
    action = ActionRunningHandedInSteps(steps=steps)

    _expanded(action, extensions)

    assert isinstance(action, Sequence)
    assert action.nodes == steps
    assert action.children == steps


@dataclass(eq=False, repr=False, kw_only=True)
class ActionRunningOneNode(GoalAction):
    """
    An action reaching a single goal rather than running steps.
    """

    node: StatechartNode
    """
    The child of this action.
    """

    def create_goal(self) -> StatechartNode:
        return self.node


def test_an_action_has_no_goal_before_it_is_expanded() -> None:
    assert ActionRunningOneNode(node=_succeeding("goal")).goal is None


def test_an_action_has_no_steps_before_it_is_expanded() -> None:
    assert ActionRunningHandedInSteps(steps=[_succeeding("step")]).nodes == []


def test_an_action_needs_the_robot_performing_it(simple_pr2_context) -> None:
    world, _, _ = simple_pr2_context
    statechart = Statechart(context=StatechartContext(world=world))
    action = ActionRunningHandedInSteps(steps=[_succeeding("only")])

    with pytest.raises(NodesMissingContextExtensionsError) as raised:
        statechart.add_node(action)

    assert raised.value.nodes_by_missing_extension == {RobotAccess: [action]}


def test_an_action_reaching_a_goal_holds_the_goal_it_created(
    simple_pr2_context,
) -> None:
    _, _, extensions = simple_pr2_context
    action = ActionRunningOneNode(node=_succeeding("goal"))

    _expanded(action, extensions)

    assert action.children == [action.goal]


def test_an_action_reaching_a_goal_succeeds_once_the_goal_did(
    simple_pr2_context,
) -> None:
    _, _, extensions = simple_pr2_context
    step = _succeeding("only step")
    action = ActionRunningOneNode(node=step)
    statechart = _expanded(action, extensions)

    _run_until(statechart, EndStatechart.when_true(action))

    assert action.goal is step
    assert action.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_an_action_whose_body_its_owner_decides_succeeds_once_the_body_arrives(
    simple_pr2_context,
) -> None:
    """
    A single task never ends on its own, so the action ends it once it is at its goal.
    """
    _, _, extensions = simple_pr2_context
    task = NodeObservingAFixedValue(
        name="task", observation=ObservationStateValues.TRUE
    )
    action = ActionRunningOneNode(node=task)
    statechart = _expanded(action, extensions)

    _run_until(statechart, EndStatechart.when_true(action))

    assert action.goal is task
    assert task.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert action.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_an_action_whose_body_its_owner_decides_waits_while_the_body_is_away(
    simple_pr2_context,
) -> None:
    """
    A task that has not arrived yet observes False, which is not a failure of the
    action running it.
    """
    _, _, extensions = simple_pr2_context
    task = NodeObservingAFixedValue(
        name="task", observation=ObservationStateValues.FALSE
    )
    action = ActionRunningOneNode(node=task)
    statechart = _expanded(action, extensions)
    statechart.add_node(EndStatechart.when_true(action))
    executor = StatechartExecutor(context=statechart.context)
    executor.compile(statechart)

    for _ in range(3):
        executor.tick()

    assert action.life_cycle_state == LifeCycleValues.RUNNING


def test_a_parallel_action_ends_itself_once_all_its_steps_are_at_their_goals(
    simple_pr2_context,
) -> None:
    """
    A parallel on its own leaves ending it to its owner; as an action it ends itself,
    so it can be a step of another one.
    """
    _, _, extensions = simple_pr2_context
    steps = [
        NodeObservingAFixedValue(name=name, observation=ObservationStateValues.TRUE)
        for name in ("first", "second")
    ]
    action = ActionHoldingHandedInSteps(steps=steps)
    statechart = _expanded(action, extensions)

    _run_until(statechart, EndStatechart.when_true(action))

    assert action.children == steps
    assert action.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_a_parallel_action_waits_while_a_step_is_away(simple_pr2_context) -> None:
    """
    A step that has not arrived yet makes the parallel observe False, which is not a
    failure of the action.
    """
    _, _, extensions = simple_pr2_context
    action = ActionHoldingHandedInSteps(
        steps=[
            NodeObservingAFixedValue(
                name="arrived", observation=ObservationStateValues.TRUE
            ),
            NodeObservingAFixedValue(
                name="away", observation=ObservationStateValues.FALSE
            ),
        ]
    )
    statechart = _expanded(action, extensions)
    statechart.add_node(EndStatechart.when_true(action))
    executor = StatechartExecutor(context=statechart.context)
    executor.compile(statechart)

    for _ in range(3):
        executor.tick()

    assert action.life_cycle_state == LifeCycleValues.RUNNING


def test_a_parallel_action_fails_once_a_step_can_no_longer_arrive(
    simple_pr2_context,
) -> None:
    _, _, extensions = simple_pr2_context
    action = ActionHoldingHandedInSteps(
        steps=[
            NodeObservingAFixedValue(
                name="arrived", observation=ObservationStateValues.TRUE
            ),
            _failing("gives up"),
        ]
    )
    statechart = _expanded(action, extensions)

    _run_until(statechart, EndStatechart.when_failed(action))

    assert action.life_cycle_state == LifeCycleValues.FAILED


def test_a_parallel_action_takes_no_minimum_of_successes(simple_pr2_context) -> None:
    """
    How many steps have to arrive is up to the action, not one of its parameters.
    """
    assert ActionHoldingHandedInSteps(steps=[]).designator_parameter == {"steps": []}


def test_a_try_all_action_succeeds_once_one_of_its_steps_succeeded(
    simple_pr2_context,
) -> None:
    _, _, extensions = simple_pr2_context
    action = ActionTryingHandedInSteps(
        steps=[_failing("gives up"), _succeeding("arrives")]
    )
    statechart = _expanded(action, extensions)

    _run_until(statechart, EndStatechart.when_true(action))

    assert action.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_action_succeeds_once_its_last_step_succeeded(simple_pr2_context) -> None:
    """
    An action ends successfully when the sequence it runs does.
    """
    _, _, extensions = simple_pr2_context
    action = ActionRunningHandedInSteps(steps=[_succeeding("only step")])
    statechart = _expanded(action, extensions)

    _run_until(statechart, EndStatechart.when_true(action))

    assert action.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_action_fails_once_a_step_ended_without_succeeding(simple_pr2_context) -> None:
    """
    A step that cannot arrive fails the action rather than leaving it waiting.
    """
    _, _, extensions = simple_pr2_context
    action = ActionRunningHandedInSteps(steps=[_failing("step that gives up")])
    statechart = _expanded(action, extensions)

    _run_until(statechart, EndStatechart.when_failed(action))

    assert action.life_cycle_state == LifeCycleValues.FAILED


def test_action_without_steps_is_rejected(simple_pr2_context) -> None:
    """
    An action that names no node to run is a mistake, not an empty success.
    """
    _, _, extensions = simple_pr2_context
    statechart = _expanded(ActionRunningHandedInSteps(steps=[]), extensions)

    with pytest.raises(CompositeNodeWithoutChildrenError):
        statechart.compiler.compile()


def _subclasses_of(cls: type) -> list[type]:
    """
    :return: Every class deriving from `cls`, however indirectly.
    """
    return [
        subclass
        for direct in cls.__subclasses__()
        for subclass in [direct, *_subclasses_of(direct)]
    ]


def _every_action_type() -> list[type[Action]]:
    """
    :return: Every action class of coraplex, the composite ones included.
    """
    importlib.import_module("coraplex.robot_plans.actions.composite.transporting")
    importlib.import_module("coraplex.robot_plans.actions.composite.tool_based")
    importlib.import_module("coraplex.robot_plans.actions.composite.searching")
    return _subclasses_of(Action)


def _members_decided_by_base_order(action_type: type) -> dict[str, list[str]]:
    """
    :return: Each member of `action_type` that several of its classes define, where the
        class defining it first does not derive from the others, mapped to the names of
        those others. Which of them wins then depends only on the order of the bases.
        Comparison and hashing count, every other dunder does not.
    """
    member_names = {
        member
        for owner in action_type.__mro__
        for member in owner.__dict__
        if member in ("__eq__", "__hash__")
        or not (member.startswith("__") or member == "_abc_impl")
    }
    decided_by_order = {}
    for member in member_names:
        winner, *overridden = [
            owner for owner in action_type.__mro__ if member in owner.__dict__
        ]
        beaten_by_order = [
            owner.__name__ for owner in overridden if not issubclass(winner, owner)
        ]
        if beaten_by_order:
            decided_by_order[member] = beaten_by_order
    return decided_by_order


def test_no_member_of_an_action_depends_on_the_order_of_its_bases() -> None:
    """
    Where two bases of an action define the same member, a class deriving from both
    decides it, so reordering the bases cannot change what the action does.
    """
    decided_by_order = {
        action_type.__name__: _members_decided_by_base_order(action_type)
        for action_type in _every_action_type()
    }

    assert {
        name: members for name, members in decided_by_order.items() if members
    } == {}


def test_every_action_is_compared_and_hashed_as_the_node_it_is() -> None:
    """
    The mixins describing an action's parameters are dataclasses comparing by value,
    which must not decide whether two actions are the same node.
    """
    deciding = {
        action_type.__name__: {
            method: next(
                owner for owner in action_type.__mro__ if method in owner.__dict__
            )
            for method in ("__eq__", "__hash__")
        }
        for action_type in _every_action_type()
    }

    assert {
        name: owners
        for name, owners in deciding.items()
        if set(owners.values()) != {StatechartNode}
    } == {}


def test_action_parameters_leave_out_statechart_machinery(simple_pr2_context) -> None:
    """
    Only what the action was parameterized with counts as its parameters.
    """
    action = MoveTorsoAction(torso_state=TorsoState.HIGH)

    assert action.designator_parameter == {"torso_state": TorsoState.HIGH}


@dataclass(eq=False, repr=False, kw_only=True)
class ActionWithKeywordOnlyParameter(ActionRunningHandedInSteps):
    """
    An action taking one of its parameters by keyword only.
    """

    speed: float = 1.0
    """
    A parameter that can only be passed by keyword.
    """


def test_a_keyword_only_field_is_a_parameter() -> None:
    action = ActionWithKeywordOnlyParameter(speed=2.0)

    assert action.designator_parameter == {"steps": [], "speed": 2.0}


def test_the_name_of_an_action_is_not_a_parameter() -> None:
    action = MoveTorsoAction(torso_state=TorsoState.HIGH, name="lift")

    assert action.designator_parameter == {"torso_state": TorsoState.HIGH}


# %% an action keeps its collision allowances in force while it runs


@dataclass(eq=False, repr=False, kw_only=True)
class ActionLettingAGripperTouch(ActionRunningHandedInSteps):
    """
    An action running the steps handed to it while a gripper may touch its
    surroundings.
    """

    end_effector: EndEffector
    """
    The end effector that may touch its surroundings while this action runs.
    """

    def create_temporary_collision_rules(self) -> list[CollisionRule]:
        return [AllowCollisionForEndEffector(end_effector=self.end_effector)]


def _in_force(statechart: Statechart, action: Action) -> bool:
    """
    :return: Whether the collision rules of `action` are in force in the world of
        `statechart` now. Compared by identity, because the shared world may hold equal
        rules of its own.
    """
    in_force = statechart.context.world.collision_manager.temporary_rules
    return bool(action.temporary_collision_rules) and all(
        any(rule is rule_in_force for rule_in_force in in_force)
        for rule in action.temporary_collision_rules
    )


def _away(name: str) -> NodeObservingAFixedValue:
    """
    :return: A step that never arrives, so the action running it keeps running.
    """
    return NodeObservingAFixedValue(name=name, observation=ObservationStateValues.FALSE)


def test_an_action_keeps_its_collision_allowances_in_force_while_it_runs(
    simple_pr2_context,
) -> None:
    _, robot, extensions = simple_pr2_context
    end_effector = robot.left_arm.end_effector
    action = ActionLettingAGripperTouch(
        steps=[_away("step")], end_effector=end_effector
    )
    statechart = _expanded(action, extensions)
    statechart.add_node(EndStatechart.when_true(action))
    executor = StatechartExecutor(context=statechart.context)

    assert not _in_force(statechart, action)
    executor.compile(statechart)
    for _ in range(3):
        executor.tick()

    assert action.life_cycle_state == LifeCycleValues.RUNNING
    assert _in_force(statechart, action)


def test_an_action_takes_its_collision_allowances_out_once_it_ended(
    simple_pr2_context,
) -> None:
    _, robot, extensions = simple_pr2_context
    end_effector = robot.left_arm.end_effector
    action = ActionLettingAGripperTouch(
        steps=[_succeeding("step")], end_effector=end_effector
    )
    statechart = _expanded(action, extensions)

    _run_until(statechart, EndStatechart.when_true(action))

    assert action.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert not _in_force(statechart, action)


def test_a_reset_action_takes_its_collision_allowances_out(simple_pr2_context) -> None:
    """
    A reset ends a run without the action ending, so the allowances go then as well.
    """
    _, robot, extensions = simple_pr2_context
    end_effector = robot.left_arm.end_effector
    action = ActionLettingAGripperTouch(
        steps=[_away("step")], end_effector=end_effector
    )
    action.reset_condition = action.is_running
    statechart = _expanded(action, extensions)
    statechart.add_node(EndStatechart.when_true(action))
    executor = StatechartExecutor(context=statechart.context)
    executor.compile(statechart)
    assert action.life_cycle_state == LifeCycleValues.RUNNING
    assert _in_force(statechart, action)

    executor.tick()

    assert action.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not _in_force(statechart, action)


def test_an_aborted_run_takes_the_collision_allowances_of_a_running_action_out(
    simple_pr2_context,
) -> None:
    """
    A run that raises while an action runs ends it neither by ending nor by resetting it.
    """
    _, robot, extensions = simple_pr2_context
    end_effector = robot.left_arm.end_effector
    action = ActionLettingAGripperTouch(
        steps=[_away("step")], end_effector=end_effector
    )
    statechart = _expanded(action, extensions)
    statechart.add_node(EndStatechart.when_true(action))
    executor = StatechartExecutor(context=statechart.context)
    executor.compile(statechart)

    with pytest.raises(TickLimitReachedError):
        executor.tick_until_end(timeout=2)

    assert action.life_cycle_state == LifeCycleValues.RUNNING
    assert not _in_force(statechart, action)


def test_an_action_is_sent_with_its_collision_allowances(simple_pr2_context) -> None:
    """
    A receiver does not expand an action again, so its allowances have to arrive with
    it.
    """
    world, robot, extensions = simple_pr2_context
    end_effector = robot.left_arm.end_effector
    sent = _expanded(
        action := ActionLettingAGripperTouch(
            steps=[_succeeding("step")], end_effector=end_effector
        ),
        extensions,
    )

    received = Statechart.from_json(
        json.loads(json.dumps(sent.to_json())),
        context=simulated_executor(extensions).context,
        **WorldEntityWithIDKwargsTracker.from_world(world).create_kwargs(),
    )

    [rule] = received.get_node_by_index(action.index).temporary_collision_rules
    assert isinstance(rule, AllowCollisionForEndEffector)
    assert rule.end_effector is end_effector


# %% actions touching their surroundings with a gripper allow it themselves


def test_a_pick_up_and_its_grasp_and_reach_let_the_gripper_touch(
    pr2_apartment_context,
) -> None:
    """
    The allowance is the action's own rather than a node beside each of its goals.
    """
    world, robot, extensions = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    pick_up = PickUpAction(grasp=milk.grasp_candidates()[0], arm=robot.left_arm)

    _expanded(pick_up, extensions)

    [grasping] = nodes_of_type(pick_up, GraspingAction)
    [reach] = nodes_of_type(pick_up, ReachAction)
    for action in (pick_up, grasping, reach):
        [rule] = action.temporary_collision_rules
        assert rule.end_effector is robot.left_arm.end_effector
    assert nodes_of_type(pick_up, UpdateTemporaryCollisionRules) == []


def test_a_place_lets_the_gripper_holding_its_object_touch(
    pr2_apartment_context,
) -> None:
    world, robot, extensions = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    pick_up = PickUpAction(grasp=milk.grasp_candidates()[0], arm=robot.left_arm)
    place = PlaceAction(
        object_designator=milk,
        target_location=Pose.from_xyz_rpy(0.8, -1.9, 0.7, reference_frame=world.root),
    )

    statechart_of(simulated_executor(extensions), Sequence(nodes=[pick_up, place]))

    [rule] = place.temporary_collision_rules
    assert rule.end_effector is robot.left_arm.end_effector
    assert nodes_of_type(place, UpdateTemporaryCollisionRules) == []


def test_opening_lets_the_gripper_on_the_handle_touch(pr2_apartment_context) -> None:
    world, robot, extensions = pr2_apartment_context
    handle = _handle_annotation(world, "handle_cab10_m")
    action = OpenAction(grasp=handle.grasp_candidates()[0], arm=robot.right_arm)

    _expanded(action, extensions)

    [rule] = action.temporary_collision_rules
    assert rule.end_effector is robot.right_arm.end_effector
    assert nodes_of_type(action, UpdateTemporaryCollisionRules) == []


def test_an_action_lets_nothing_touch_unless_it_says_so(simple_pr2_context) -> None:
    _, _, extensions = simple_pr2_context
    action = ActionRunningHandedInSteps(steps=[_succeeding("step")])

    _expanded(action, extensions)

    assert action.temporary_collision_rules == []


# %% a pick up and a place run their steps without grouping them further


def _goal_moving_the_tool(step: StatechartNode) -> StatechartNode:
    """
    :return: The goal moving the tool center point in `step`, which runs in a parallel
        only where something accompanies it.
    """
    return step.nodes[0] if type(step) is Parallel else step


def test_a_pick_up_grasps_reattaches_and_lifts_as_its_own_steps(
    pr2_apartment_context,
) -> None:
    world, robot, extensions = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    pick_up = PickUpAction(grasp=milk.grasp_candidates()[0], arm=robot.left_arm)

    _expanded(pick_up, extensions)

    grasping, reattaching, lifting = pick_up.nodes
    assert type(grasping) is GraspingAction
    assert type(reattaching) is MoveBranch
    assert reattaching.new_parent is robot.left_arm.end_effector.tool_frame
    assert type(_goal_moving_the_tool(lifting)) is CartesianPose


def test_a_place_reattaches_and_retracts_as_its_own_steps(
    pr2_apartment_context,
) -> None:
    world, robot, extensions = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    pick_up = PickUpAction(grasp=milk.grasp_candidates()[0], arm=robot.left_arm)
    place = PlaceAction(
        object_designator=milk,
        target_location=Pose.from_xyz_rpy(0.8, -1.9, 0.7, reference_frame=world.root),
    )

    statechart_of(simulated_executor(extensions), Sequence(nodes=[pick_up, place]))

    *_, reattaching, retracting = place.nodes
    assert type(reattaching) is MoveBranch
    assert reattaching.new_parent is world.root
    assert type(_goal_moving_the_tool(retracting)) is CartesianPose
    assert not any(type(node) is Sequence for node in place.descendants)
