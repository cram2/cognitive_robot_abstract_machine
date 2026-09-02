from __future__ import annotations

from typing import Callable

from typing_extensions import List, assert_never, Optional, TYPE_CHECKING, TypeVar

from coraplex.datastructures.dataclasses import Context
from coraplex.plans.plan import Plan
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from krrood.entity_query_language.query.match import Match

if TYPE_CHECKING:
    from coraplex.language import (
        SequentialNode,
        ParallelNode,
        TryInOrderNode,
        TryAllNode,
        CancelMonitor,
        PauseMonitor,
        PauseUntilMonitor,
        RepeatNode,
        CodeNode,
    )
    from coraplex.plans.plan_node import ActionLike, PlanNode


def execute_single(
    action_like: ActionLike,
    context: Optional[Context] = None,
) -> PlanNode:
    """
    Executes the given action like on its own.

    :param action_like: The action like to execute
    :param context: Plan context to pass to the action
    :return: The root node of the plan in which the action is build
    """
    node = make_node(action_like)
    plan = Plan(context=context)
    plan.add_node(node)
    return node


def sequential(
    children: List[ActionLike],
    context: Optional[Context] = None,
) -> SequentialNode:
    """
    Executes the given actions or motions in order, one after the other.

    If any of them fails, the plan fails.
    :param children: The actions to be executed
    :param context: The plan context to pass to the actions
    :return: The root node of the plan in which the actions are built
    """
    from coraplex.language import SequentialNode

    return _make_plan_from_type_and_children(SequentialNode(), children, context)


def parallel(
    children: List[ActionLike],
    context: Optional[Context] = None,
) -> ParallelNode:
    """
    Executes the given actions in parallel as far as possible, this uses the Giskard
    Template for parallel.

    .. warning:: If there are model changes during the execution unexpected behaviour can occur, so avoid using manipulation actions such as Pick and Place in this

    :param children: Actions or motions to be executed in parallel
    :param context: The plan context to pass to the actions
    :return: The root node of the plan in which the actions are built
    """
    from coraplex.language import ParallelNode

    return _make_plan_from_type_and_children(ParallelNode(), children, context)


def try_in_order(
    children: List[ActionLike],
    context: Optional[Context] = None,
) -> TryInOrderNode:
    """
    Executes the given actions or motions one after another, if one fails the next one
    is executed until one succeeds or all fail.

    :param children: The actions or motions to be executed
    :param context: The plan context that should be passed to the build plan
    :return: The root node of the plan that is constructed
    """
    from coraplex.language import TryInOrderNode

    return _make_plan_from_type_and_children(TryInOrderNode(), children, context)


def try_all(
    children: List[ActionLike],
    context: Optional[Context] = None,
) -> TryAllNode:
    """
    Tries all given actions or motions, similar to try_in_order but executes them in
    parallel.

    Succeeded if at least one child succeeds.

    :param children: Actions or motions to be executed
    :param context: the plan context that should be passed to the children
    :return: The root node of the plan that is constructed
    """
    from coraplex.language import TryAllNode

    return _make_plan_from_type_and_children(TryAllNode(), children, context)


def pause_while(
    children: List[ActionLike],
    monitor: MotionStatechartNode,
    context: Optional[Context] = None,
) -> PauseMonitor:
    """
    Hold `children` for as long as `monitor` observes True.

    :param children: Actions or motions that should be under the monitor, meaning they
        are paused when the monitor is true
    :param monitor: The motion state chart node observed while the children run.
    :param context: The context of the plan to be passed to the children
    """
    from coraplex.language import PauseMonitor

    return _make_plan_from_type_and_children(
        PauseMonitor(monitor=monitor), children, context
    )


def pause_until(
    children: List[ActionLike],
    monitor: MotionStatechartNode,
    context: Optional[Context] = None,
) -> PauseUntilMonitor:
    """
    Hold `children` until `monitor` observes True.

    :param children: Actions or motions that should be paused until the monitor turns
        true
    :param monitor: The motion state chart node observed while the children run.
    :param context: The plan context to be passed to the children
    """
    from coraplex.language import PauseUntilMonitor

    return _make_plan_from_type_and_children(
        PauseUntilMonitor(monitor=monitor), children, context
    )


def cancel_when(
    children: List[ActionLike],
    monitor: MotionStatechartNode,
    context: Optional[Context] = None,
) -> CancelMonitor:
    """
    Stop `children` once `monitor` observes True and cancel the rest of the plan.

    :param monitor: The motion state chart node observed while the children run.
    :param children: Actions or motions that should be cancelled when the monitor turns
        true
    :param context: The plan context to be passed to the children
    """
    from coraplex.language import CancelMonitor

    return _make_plan_from_type_and_children(
        CancelMonitor(monitor=monitor), children, context
    )


def repeat(
    children: List[ActionLike],
    maximum_repetitions: int,
    context: Optional[Context] = None,
    **repeat_arguments,
) -> RepeatNode:
    """
    Attempt `children` until they succeed, at most `maximum_repetitions` times.

    :param children: Actions or motions that should be repeated according to the repeat
        arguments
    :param maximum_repetitions: How many attempts before the repeating is given up on.
    :param repeat_arguments: Passed to :class:`~coraplex.language.RepeatNode`, for
        instance a `repeat_template` or a `failure_monitor`.
    :param context: The plan context to be passed to the children
    """
    from coraplex.language import RepeatNode

    root = RepeatNode(maximum_repetitions=maximum_repetitions, **repeat_arguments)
    return _make_plan_from_type_and_children(root, children, context)


def code(function: Callable, context: Optional[Context] = None) -> CodeNode:
    from coraplex.language import CodeNode

    root = CodeNode(code=function)
    return execute_single(root, context=context)


T = TypeVar("T")


def _make_plan_from_type_and_children(
    root: T, children: List[ActionLike], context: Optional[Context]
) -> T:
    plan = Plan(context=context)
    plan.add_node(root)

    for action_like in children:
        child = make_node(action_like)
        if child.plan:
            root.mount_subplan(child)
        else:
            root.add_child(child)
    plan.simplify()
    return root


def make_node(action_like: ActionLike) -> PlanNode:
    from coraplex.plans.plan_node import (
        PlanNode,
        UnderspecifiedNode,
        ActionNode,
        MotionNode,
    )
    from coraplex.robot_plans.actions.base import ActionDescription
    from coraplex.robot_plans import BaseMotion

    if isinstance(action_like, PlanNode):
        return action_like
    elif isinstance(action_like, Match):
        underspecified_action = UnderspecifiedNode(underspecified_action=action_like)
        return underspecified_action
    elif isinstance(action_like, ActionDescription):
        return ActionNode(designator=action_like)
    elif isinstance(action_like, BaseMotion):
        return MotionNode(designator=action_like)
    else:
        assert_never(action_like)
