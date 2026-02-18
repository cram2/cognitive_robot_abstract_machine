"""
User interface (grammar & vocabulary) for entity query language.
"""

from __future__ import annotations

import operator

from typing_extensions import Union, Iterable

from .core.base_expressions import SymbolicExpression
from .enums import RDREdge
from .failures import UnsupportedExpressionTypeForDistinct
from .query.match import Match, MatchVariable
from .operators.aggregators import Max, Min, Sum, Average, Count
from .operators.comparator import Comparator
from .operators.core_logical_operators import chained_logic, AND, OR, LogicalOperator
from .operators.logical_quantifiers import ForAll, Exists
from .operators.concatenation import Concatenation
from .query.result_quantifiers import (
    ResultQuantificationConstraint,
    An,
    The,
    ResultQuantifier,
)
from .rules.conclusion_selector import ExceptIf, Alternative, Next
from .query.query_descriptor import Entity, SetOf, Query
from .utils import is_iterable
from .core.variable import (
    DomainType,
    Literal,
)
from .core.domain_mapping import Flatten, CanBehaveLikeAVariable
from .predicate import *  # type: ignore
from ..symbol_graph.symbol_graph import Symbol, SymbolGraph

ConditionType = Union[SymbolicExpression, bool, Predicate]
"""
The possible types for conditions.
"""

# %% Query Construction


def entity(selected_variable: T) -> Entity[T]:
    """
    Create an entity descriptor for a selected variable.

    :param selected_variable: The variable to select in the result.
    :return: Entity descriptor.
    """
    return Entity(_selected_variables_=(selected_variable,))


def set_of(*selected_variables: Union[Selectable[T], Any]) -> SetOf:
    """
    Create a set descriptor for the selected variables.

    :param selected_variables: The variables to select in the result set.
    :return: Set descriptor.
    """
    return SetOf(_selected_variables_=selected_variables)


# %% Match


def match(
    type_: Optional[Union[Type[T], Selectable[T]]] = None,
) -> Union[Type[T], CanBehaveLikeAVariable[T], Match[T]]:
    """
    Create a symbolic variable matching the type and the provided keyword arguments. This is used for easy variable
     definitions when there are structural constraints.

    :param type_: The type of the variable (i.e., The class you want to instantiate).
    :return: The Match instance.
    """
    return Match(type_=type_)


def match_variable(
    type_: Union[Type[T], Selectable[T]], domain: DomainType
) -> Union[Type[T], An[T], CanBehaveLikeAVariable[T], MatchVariable[T]]:
    """
    Same as :py:func:`krrood.entity_query_language.match.match` but with a domain to use for the variable created
     by the match.

    :param type_: The type of the variable (i.e., The class you want to instantiate).
    :param domain: The domain used for the variable created by the match.
    :return: The Match instance.
    """
    return MatchVariable(type_=type_, domain=domain)


# %% Variable Declaration


def variable(
    type_: Type[T],
    domain: DomainType,
    name: Optional[str] = None,
    inferred: bool = False,
) -> Union[T, Selectable[T]]:
    """
    Declare a symbolic variable that can be used inside queries.

    Filters the domain to elements that are instances of T.

    .. warning::

        If no domain is provided, and the type_ is a Symbol type, then the domain will be inferred from the SymbolGraph,
         which may contain unnecessarily many elements.

    :param type_: The type of variable.
    :param domain: Iterable of potential values for the variable or None.
     If None, the domain will be inferred from the SymbolGraph for Symbol types, else should not be evaluated by EQL
      but by another evaluator (e.g., EQL To SQL converter in Ormatic).
    :param name: The variable name, only required for pretty printing.
    :param inferred: Whether the variable is inferred or not.
    :return: A Variable that can be queried for.
    """
    # Determine the domain source
    if is_iterable(domain):
        domain_source = filter(lambda x: isinstance(x, type_), domain)
    elif domain is None and not inferred and issubclass(type_, Symbol):
        domain_source = SymbolGraph().get_instances_of_type(type_)
    else:
        domain_source = domain

    if name is None:
        name = type_.__name__

    result = Variable(
        _type_=type_,
        _domain_source_=domain_source,
        _name__=name,
        _is_inferred_=inferred,
    )

    return result


def variable_from(
    domain: Union[Iterable[T], Selectable[T]],
    name: Optional[str] = None,
) -> Union[T, Selectable[T]]:
    """
    Similar to `variable` but constructed from a domain directly wihout specifying its type.
    """
    return Literal(data=domain, name=name, wrap_in_iterator=False)


# %% Operators on Variables


def concatenation(
    *variables: Union[Iterable[T], Selectable[T]],
) -> Union[T, Selectable[T]]:
    """
    Concatenation of two or more variables.
    """
    return Concatenation(_operation_children_=variables)


def contains(
    container: Union[Iterable, CanBehaveLikeAVariable[T]], item: Any
) -> Comparator:
    """
    Check whether a container contains an item.

    :param container: The container expression.
    :param item: The item to look for.
    :return: A comparator expression equivalent to ``item in container``.
    :rtype: SymbolicExpression
    """
    return in_(item, container)


def in_(item: Any, container: Union[Iterable, CanBehaveLikeAVariable[T]]):
    """
    Build a comparator for membership: ``item in container``.

    :param item: The candidate item.
    :param container: The container expression.
    :return: Comparator expression for membership.
    :rtype: Comparator
    """
    return Comparator(container, item, operator.contains)


def flatten(
    var: Union[CanBehaveLikeAVariable[T], Iterable[T]],
) -> Union[CanBehaveLikeAVariable[T], T]:
    """
    Flatten a nested iterable domain into individual items while preserving the parent bindings.
    This returns a DomainMapping that, when evaluated, yields one solution per inner element
    (similar to SQL UNNEST), keeping existing variable bindings intact.
    """
    return Flatten(var)


# %% Logical Operators


def and_(*conditions: ConditionType):
    """
    Logical conjunction of conditions.

    :param conditions: One or more conditions to combine.
    :type conditions: SymbolicExpression | bool
    :return: An AND operator joining the conditions.
    :rtype: SymbolicExpression
    """
    return chained_logic(AND, *conditions)


def or_(*conditions):
    """
    Logical disjunction of conditions.

    :param conditions: One or more conditions to combine.
    :type conditions: SymbolicExpression | bool
    :return: An OR operator joining the conditions.
    :rtype: SymbolicExpression
    """
    return chained_logic(OR, *conditions)


def not_(operand: ConditionType) -> SymbolicExpression:
    """
    A symbolic NOT operation that can be used to negate symbolic expressions.
    """
    if not isinstance(operand, SymbolicExpression):
        operand = Literal(operand)
    return operand._invert_()


def for_all(
    universal_variable: Union[CanBehaveLikeAVariable[T], T],
    condition: ConditionType,
):
    """
    A universal on variable that finds all sets of variable bindings (values) that satisfy the condition for **every**
     value of the universal_variable.

    :param universal_variable: The universal on variable that the condition must satisfy for all its values.
    :param condition: A SymbolicExpression or bool representing a condition that must be satisfied.
    :return: A SymbolicExpression that can be evaluated producing every set that satisfies the condition.
    """
    return ForAll(universal_variable, condition)


def exists(
    universal_variable: Union[CanBehaveLikeAVariable[T], T],
    condition: ConditionType,
):
    """
    A universal on variable that finds all sets of variable bindings (values) that satisfy the condition for **any**
     value of the universal_variable.

    :param universal_variable: The universal on variable that the condition must satisfy for any of its values.
    :param condition: A SymbolicExpression or bool representing a condition that must be satisfied.
    :return: A SymbolicExpression that can be evaluated producing every set that satisfies the condition.
    """
    return Exists(universal_variable, condition)


# %% Result Quantifiers


def an(
    entity_: Union[T, Query],
    quantification: Optional[ResultQuantificationConstraint] = None,
) -> Union[T, Query]:
    """
    Select all values satisfying the given entity description.

    :param entity_: An entity or a set expression to quantify over.
    :param quantification: Optional quantification constraint.
    :return: The entity with the quantifier applied.
    """
    return entity_._quantify_(An, quantification_constraint=quantification)


a = an
"""
This is an alias to accommodate for words not starting with vowels.
"""


def the(
    entity_: Union[T, Query],
) -> Union[T, Query]:
    """
    Select the unique value satisfying the given entity description.

    :param entity_: An entity or a set expression to quantify over.
    :return: The entity with the quantifier applied.
    """
    return entity_._quantify_(The)


# %% Rules


def inference(
    type_: Type[T],
) -> Union[Type[T], Callable[[Any], Variable[T]]]:
    """
    This returns a factory function that creates a new variable of the given type and takes keyword arguments for the
    type constructor.

    :param type_: The type of the variable (i.e., The class you want to instantiate).
    :return: The factory function for creating a new variable.
    """
    return lambda **kwargs: InstantiatedVariable(
        _type_=type_,
        _name__=type_.__name__,
        _kwargs_=kwargs,
    )


def refinement(*conditions: ConditionType) -> SymbolicExpression:
    """
    Add a refinement branch (ExceptIf node with its right the new conditions and its left the base/parent rule/query)
     to the current condition tree.

    Each provided condition is chained with AND, and the resulting branch is
    connected via ExceptIf to the current node, representing a refinement/specialization path.

    :param conditions: The refinement conditions. They are chained with AND.
    :returns: The newly created branch node for further chaining.
    """
    new_branch = chained_logic(AND, *conditions)
    current_node = SymbolicExpression._current_parent_in_context_stack_()
    prev_parent = current_node._parent_
    new_conditions_root = ExceptIf(current_node, new_branch)
    prev_parent._replace_child_(current_node, new_conditions_root)
    return new_branch


def alternative(*conditions: ConditionType) -> SymbolicExpression:
    """
    Add an alternative branch (logical ElseIf) to the current condition tree.

    Each provided condition is chained with AND, and the resulting branch is
    connected via ElseIf to the current node, representing an alternative path.

    :param conditions: Conditions to chain with AND and attach as an alternative.
    :returns: The newly created branch node for further chaining.
    """
    return alternative_or_next(RDREdge.Alternative, *conditions)


def next_rule(*conditions: ConditionType) -> SymbolicExpression:
    """
    Add a consequent rule that gets always executed after the current rule.

    Each provided condition is chained with AND, and the resulting branch is
    connected via Next to the current node, representing the next path.

    :param conditions: Conditions to chain with AND and attach as an alternative.
    :returns: The newly created branch node for further chaining.
    """
    return alternative_or_next(RDREdge.Next, *conditions)


def alternative_or_next(
    condition_edge_type: Union[RDREdge.Alternative, RDREdge.Next],
    *conditions: ConditionType,
) -> SymbolicExpression:
    """
    Add an alternative/next branch to the current condition tree.

    Each provided condition is chained with AND, and the resulting branch is
    connected via ElseIf/Next to the current node, representing an alternative/next path.

    :param condition_edge_type: The type of the branch, either alternative or next.
    :param conditions: Conditions to chain with AND and attach as an alternative.
    :returns: The newly created branch node for further chaining.
    """
    new_condition = chained_logic(AND, *conditions)

    current_conditions_root = get_current_conditions_root_for_alternative_or_next()

    prev_parent = current_conditions_root._parent_

    new_conditions_root = construct_new_conditions_root_for_alternative_or_next(
        condition_edge_type, current_conditions_root, new_condition
    )

    if new_conditions_root is not current_conditions_root:
        prev_parent._replace_child_(current_conditions_root, new_conditions_root)

    return new_condition


def get_current_conditions_root_for_alternative_or_next() -> ConditionType:
    """
    :return: the current conditions root to use for creating a new condition connected via alternative or next edge.
    """
    current_node = SymbolicExpression._current_parent_in_context_stack_()
    if isinstance(current_node._parent_, (Alternative, Next)):
        current_node = current_node._parent_
    elif (
        isinstance(current_node._parent_, ExceptIf)
        and current_node is current_node._parent_.left
    ):
        current_node = current_node._parent_
    return current_node


def construct_new_conditions_root_for_alternative_or_next(
    condition_edge_type: Union[RDREdge.Next, RDREdge.Alternative],
    current_conditions_root: SymbolicExpression,
    new_condition: LogicalOperator,
) -> Union[Next, Alternative]:
    """
    Constructs a new conditions root for alternative or next condition edge types.

    :param condition_edge_type: The type of the edge connecting the current node to the new branch.
    :param current_conditions_root: The current conditions root in the expression tree.
    :param new_condition: The new condition to be added to the rule tree.
    """
    match condition_edge_type:
        case RDREdge.Alternative:
            new_conditions_root = Alternative(current_conditions_root, new_condition)
        case RDREdge.Next:
            match current_conditions_root:
                case Next():
                    current_conditions_root.add_child(new_condition)
                    new_conditions_root = current_conditions_root
                case _:
                    new_conditions_root = Next((current_conditions_root, new_condition))
        case _:
            raise ValueError(
                f"Invalid edge type: {condition_edge_type}, expected one of: {RDREdge.Alternative}, {RDREdge.Next}"
            )
    return new_conditions_root


# %% Aggregators


def max(
    variable: Selectable[T],
    key: Optional[Callable] = None,
    default: Optional[T] = None,
    distinct: bool = False,
) -> Union[T, Max[T]]:
    """
    Maps the variable values to their maximum value.

    :param variable: The variable for which the maximum value is to be found.
    :param key: A function that extracts a comparison key from each variable value.
    :param default: The value returned when the iterable is empty.
    :param distinct: Whether to only consider distinct values.
    :return: A Max object that can be evaluated to find the maximum value.
    """
    return Max(
        variable, _key_function_=key, _default_value_=default, _distinct_=distinct
    )


def min(
    variable: Selectable[T],
    key: Optional[Callable] = None,
    default: Optional[T] = None,
    distinct: bool = False,
) -> Union[T, Min[T]]:
    """
    Maps the variable values to their minimum value.

    :param variable: The variable for which the minimum value is to be found.
    :param key: A function that extracts a comparison key from each variable value.
    :param default: The value returned when the iterable is empty.
    :param distinct: Whether to only consider distinct values.
    :return: A Min object that can be evaluated to find the minimum value.
    """
    return Min(
        variable, _key_function_=key, _default_value_=default, _distinct_=distinct
    )


def sum(
    variable: Union[T, Selectable[T]],
    key: Optional[Callable] = None,
    default: Optional[T] = None,
    distinct: bool = False,
) -> Union[T, Sum]:
    """
    Computes the sum of values produced by the given variable.

    :param variable: The variable for which the sum is calculated.
    :param key: A function that extracts a comparison key from each variable value.
    :param default: The value returned when the iterable is empty.
    :param distinct: Whether to only consider distinct values.
    :return: A Sum object that can be evaluated to find the sum of values.
    """
    return Sum(
        variable, _key_function_=key, _default_value_=default, _distinct_=distinct
    )


def average(
    variable: Union[Selectable[T], Any],
    key: Optional[Callable] = None,
    default: Optional[T] = None,
    distinct: bool = False,
) -> Union[T, Average]:
    """
    Computes the sum of values produced by the given variable.

    :param variable: The variable for which the sum is calculated.
    :param key: A function that extracts a comparison key from each variable value.
    :param default: The value returned when the iterable is empty.
    :param distinct: Whether to only consider distinct values.
    :return: A Sum object that can be evaluated to find the sum of values.
    """
    return Average(
        variable, _key_function_=key, _default_value_=default, _distinct_=distinct
    )


def count(
    variable: Optional[Selectable[T]] = None, distinct: bool = False
) -> Union[T, Count[T]]:
    """
    Count the number of values produced by the given variable.

    :param variable: The variable for which the count is calculated, if not given, the count of all results (by group)
     is returned.
    :param distinct: Whether to only consider distinct values.
    :return: A Count object that can be evaluated to count the number of values.
    """
    return Count(variable, _distinct_=distinct)


def distinct(
    expression: T,
    *on: Any,
) -> T:
    """
    Indicate that the result of the expression should be distinct.
    """
    match expression:
        case Query():
            return expression.distinct(*on)
        case ResultQuantifier():
            return expression._child_.distinct(*on)
        case Selectable():
            return entity(expression).distinct(*on)
        case _:
            raise UnsupportedExpressionTypeForDistinct(type(expression))
