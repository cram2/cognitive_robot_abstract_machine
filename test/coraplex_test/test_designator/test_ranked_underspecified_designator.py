from krrood.entity_query_language.factories import a, variable_from
from coraplex.action_belief.action_belief_query import ActionBeliefQuery
from coraplex.datastructures.enums import Arms, TaskStatus
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.robot_plans.actions.core.container import OpenAction


def test_rank_grounded_actions_orders_the_feasible_open_action_first(
    immutable_model_world,
):
    """
    ``ActionBeliefQuery.rank_grounded_actions`` reorders already-grounded actions
    without dropping any -- unit-level, below the plan/execution machinery.
    """
    world, view, context = immutable_model_world
    handle = world.get_body_by_name("handle_cab10_m")
    infeasible_first = [
        OpenAction(handle, Arms.RIGHT),
        OpenAction(handle, Arms.LEFT),
    ]

    ranked = list(
        ActionBeliefQuery(
            action_type=OpenAction, fixed_kwargs={}, context=context
        ).rank_grounded_actions(infeasible_first)
    )

    assert len(ranked) == 2
    assert {action.arm for action in ranked} == {Arms.LEFT, Arms.RIGHT}
    assert ranked[0].arm == Arms.LEFT


def test_underspecified_open_action_grounds_the_feasible_arm_first(
    immutable_model_world,
):
    """
    ``OpenAction`` is registered in ``ACTION_BELIEF_SPACES``, so grounding an
    underspecified ``arm`` must try the reachable one first: this handle is only
    reachable by the left arm (see test_action_belief_query.py), so listing the right
    arm first in the domain must not cost a failed attempt.
    """
    world, view, context = immutable_model_world
    handle = world.get_body_by_name("handle_cab10_m")
    action = a(OpenAction)(
        object_designator=handle,
        arm=variable_from([Arms.RIGHT, Arms.LEFT]),
    )

    plan = execute_single(action_like=action, context=context).plan
    with simulated_robot:
        plan.perform()

    assert plan.root.status == TaskStatus.SUCCEEDED
    assert len(plan.root.children) == 1
    assert plan.root.children[0].designator.arm == Arms.LEFT


def test_underspecified_unregistered_action_keeps_the_query_backend_order(
    immutable_model_world,
):
    """
    ``ParkArmsAction`` is not registered in ``ACTION_BELIEF_SPACES``, so grounding must
    fall through to the query backend's own order unchanged -- the first candidate must
    be the first domain value, not a ranked one.
    """
    world, view, context = immutable_model_world
    action = a(ParkArmsAction)(arm=variable_from([Arms.LEFT, Arms.RIGHT]))

    plan = execute_single(action_like=action, context=context).plan
    with simulated_robot:
        plan.perform()

    assert plan.root.status == TaskStatus.SUCCEEDED
    assert plan.root.children[0].designator.arm == Arms.LEFT
