from __future__ import annotations

from typing import Union, Callable

from typing_extensions import List, assert_never, Optional, TYPE_CHECKING, Type, TypeVar

from krrood.entity_query_language.query.match import is_underspecified
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import MonitorBehavior
from coraplex.fluent import Fluent
from coraplex.plans.plan import Plan

if TYPE_CHECKING:
    from coraplex.language import (
        SequentialNode,
        LanguageNode,
        ParallelNode,
        TryInOrderNode,
        TryAllNode,
        MonitorNode,
        RepeatNode,
        CodeNode,
    )
    from coraplex.plans.plan_node import ActionLike, PlanNode


def execute_single(
    action_like: ActionLike,
    context: Optional[Context] = None,
) -> PlanNode:

    node = make_node(action_like)
    plan = Plan(context=context)
    plan.add_node(node)
    return node


def sequential(
    children: List[ActionLike],
    context: Optional[Context] = None,
) -> SequentialNode:
    from coraplex.language import SequentialNode

    return _make_plan_from_type_and_children(SequentialNode(), children, context)


def parallel(
    children: List[ActionLike],
    context: Optional[Context] = None,
) -> ParallelNode:
    from coraplex.language import ParallelNode

    return _make_plan_from_type_and_children(ParallelNode(), children, context)


def try_in_order(
    children: List[ActionLike],
    context: Optional[Context] = None,
) -> TryInOrderNode:
    from coraplex.language import TryInOrderNode

    return _make_plan_from_type_and_children(TryInOrderNode(), children, context)


def try_all(
    children: List[ActionLike],
    context: Optional[Context] = None,
) -> TryAllNode:
    from coraplex.language import TryAllNode

    return _make_plan_from_type_and_children(TryAllNode(), children, context)


def monitor(
    children: List[ActionLike],
    condition: Union[Callable, Fluent],
    behavior: MonitorBehavior = MonitorBehavior.INTERRUPT,
    context: Optional[Context] = None,
) -> MonitorNode:
    from coraplex.language import MonitorNode

    return _make_plan_from_type_and_children(
        MonitorNode(condition=condition, behavior=behavior), children, context
    )


def repeat(
    children: List[ActionLike],
    repetitions: int,
    context: Optional[Context] = None,
) -> RepeatNode:
    from coraplex.language import RepeatNode

    root = RepeatNode(repetitions=repetitions)
    return _make_plan_from_type_and_children(root, children, context)


def code(function: Callable, context: Optional[Context] = None) -> CodeNode:
    from coraplex.language import CodeNode

    root = CodeNode(code=function)
    return execute_single(root, context=context)


T = TypeVar("T")


def _make_plan_from_type_and_children(
    root: T, children: List[ActionLike], context: Optional[Context]
) -> T:
    from coraplex.language import LanguageNode

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
    elif is_underspecified(action_like):
        underspecified_action = UnderspecifiedNode(underspecified_action=action_like)
        return underspecified_action
    elif isinstance(action_like, ActionDescription):
        return ActionNode(designator=action_like)
    elif isinstance(action_like, BaseMotion):
        return MotionNode(designator=action_like)
    else:
        assert_never(action_like)
