from __future__ import annotations

from giskardpy.motion_statechart.graph_node import DebugExpression
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianTask

GOAL_COLOR = CartesianTask.GOAL_COLOR
"""The color cartesian tasks use for the goal debug expression."""

CURRENT_COLOR = CartesianTask.CURRENT_COLOR
"""The color cartesian tasks use for the current debug expression."""


def debug_expression_by_name(
    debug_expressions: list[DebugExpression], name: str
) -> DebugExpression:
    """
    Return the single debug expression with the given name, or fail.
    """
    matches = [
        debug_expression
        for debug_expression in debug_expressions
        if debug_expression.name == name
    ]
    assert len(matches) == 1, f"expected exactly one {name}, got {len(matches)}"
    return matches[0]
