from __future__ import annotations

from typing_extensions import Union, TYPE_CHECKING

from .conclusion_selector import ExceptIf, Alternative, Next
from .enums import RDREdge
from .symbolic import (
    SymbolicExpression,
    chained_logic,
    AND,
    BinaryOperator,
)
from .utils import T

if TYPE_CHECKING:
    from .entity import ConditionType


def refinement(*conditions: ConditionType) -> SymbolicExpression[T]:
    """
    Add a refinement branch (ExceptIf node with its right the new conditions and its left the base/parent rule/query)
     to the current condition tree.

    Each provided condition is chained with AND, and the resulting branch is
    connected via ExceptIf to the current node, representing a refinement/specialization path.

    :param conditions: The refinement conditions. They are chained with AND.
    :returns: The newly created branch node for further chaining.
    """
    new_branch = chained_logic(AND, *conditions)
    current_node = SymbolicExpression._current_parent_()
    prev_parent = current_node._parent_
    current_node._parent_ = None
    new_conditions_root = ExceptIf(SymbolicExpression._current_parent_(), new_branch)
    new_branch._node_.weight = RDREdge.Refinement
    new_conditions_root._parent_ = prev_parent
    return new_conditions_root.right


def alternative(*conditions: ConditionType) -> SymbolicExpression[T]:
    """
    Add an alternative branch (logical ElseIf) to the current condition tree.

    Each provided condition is chained with AND, and the resulting branch is
    connected via ElseIf to the current node, representing an alternative path.

    :param conditions: Conditions to chain with AND and attach as an alternative.
    :returns: The newly created branch node for further chaining.
    """
    return alternative_or_next(RDREdge.Alternative, *conditions)


def next_rule(*conditions: ConditionType) -> SymbolicExpression[T]:
    """
    Add a consequent rule that gets always executed after the current rule.

    Each provided condition is chained with AND, and the resulting branch is
    connected via Next to the current node, representing the next path.

    :param conditions: Conditions to chain with AND and attach as an alternative.
    :returns: The newly created branch node for further chaining.
    """
    return alternative_or_next(RDREdge.Next, *conditions)


def alternative_or_next(
    type_: Union[RDREdge.Alternative, RDREdge.Next],
    *conditions: ConditionType,
) -> SymbolicExpression[T]:
    """
    Add an alternative/next branch to the current condition tree.

    Each provided condition is chained with AND, and the resulting branch is
    connected via ElseIf/Next to the current node, representing an alternative/next path.

    :param type_: The type of the branch, either alternative or next.
    :param conditions: Conditions to chain with AND and attach as an alternative.
    :returns: The newly created branch node for further chaining.
    """
    new_branch = chained_logic(AND, *conditions)
    current_node = SymbolicExpression._current_parent_()
    if isinstance(current_node._parent_, (Alternative, Next)):
        current_node = current_node._parent_
    elif (
        isinstance(current_node._parent_, ExceptIf)
        and current_node is current_node._parent_.left
    ):
        current_node = current_node._parent_
    prev_parent = current_node._parent_
    current_node._parent_ = None
    if type_ == RDREdge.Alternative:
        new_conditions_root = Alternative(current_node, new_branch)
    elif type_ == RDREdge.Next:
        new_conditions_root = Next(current_node, new_branch)
    else:
        raise ValueError(
            f"Invalid type: {type_}, expected one of: {RDREdge.Alternative}, {RDREdge.Next}"
        )
    new_branch._node_.weight = type_
    new_conditions_root._parent_ = prev_parent
    if isinstance(prev_parent, BinaryOperator):
        prev_parent.right = new_conditions_root
    return new_conditions_root.right
