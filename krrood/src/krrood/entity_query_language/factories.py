"""
User interface (grammar & vocabulary) for entity query language.
"""

from __future__ import annotations

import inspect
import operator
from inspect import isclass
from uuid import UUID

from typing_extensions import Iterable, List, overload

from krrood.entity_query_language.core.base_expressions import (
    SymbolicExpression,
    TruthValueOperator,
    OperationResult,
)
from krrood.entity_query_language.core.mapped_variable import (
    FlatVariable,
    CanBehaveLikeAVariable,
    Attribute,
)
from krrood.entity_query_language.core.variable import (
    DomainType,
    Literal,
    ExternallySetVariable,
)
from krrood.entity_query_language.cache_data import InstanceFilteredDomain
from krrood.entity_query_language.enums import DomainSource
from krrood.entity_query_language.exceptions import UnsupportedExpressionTypeForDistinct
from krrood.entity_query_language.operators.aggregators import (
    Max,
    Min,
    Sum,
    Average,
    Count,
    CountAll,
    CountRange,
    Mode,
    MultiMode,
)
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.concatenation import Concatenation
from krrood.entity_query_language.operators.conditionals import CaseWhen
from krrood.entity_query_language.operators.core_logical_operators import (
    chained_logic,
    AND,
    OR,
)
from krrood.entity_query_language.operators.logical_quantifiers import ForAll, Exists
from krrood.entity_query_language.predicate import *  # type: ignore
from krrood.entity_query_language.predicate import symbolic_function
from krrood.entity_query_language.query.match import (
    Match,
)
from krrood.entity_query_language.query.quantifiers import (
    ResultQuantificationConstraint,
    An,
    The,
    ResultQuantifier,
)
from krrood.entity_query_language.query.query import Entity, SetOf, Query
from krrood.entity_query_language.rules.conclusion import Add
from krrood.entity_query_language.rules.conclusion_selector import (
    Refinement,
    Alternative,
    Next,
)
from krrood.entity_query_language.utils import is_iterable
from krrood.symbol_graph.symbol_graph import Symbol, SymbolGraph

ConditionType = Union[SymbolicExpression, bool, Predicate, TruthValueOperator]
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


# %% Variable Declaration


def variable(
    type_: Type[T],
    domain: Optional[DomainType] = None,
) -> Union[T, Selectable[T], Variable[T]]:
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
    :return: A Variable that can be queried for.
    """
    # Determine the domain source
    if is_iterable(domain):
        domain = InstanceFilteredDomain(type_, domain)
    elif domain is None and issubclass(type_, Symbol):
        domain = SymbolGraph().get_instances_of_type(type_)
    else:
        domain = domain

    result = Variable(
        _type_=type_,
        _domain_=domain,
    )

    return result


def deduced_variable(
    type_: Optional[Type[T]] = None,
) -> Union[Type[T], ExternallySetVariable[T]]:
    """
    Create a variable that is inferred through deductive reasoning.

    :param type_: The type of the variable values.
    :return: An instance of `ExternallySetVariable[T]`.
    """
    return ExternallySetVariable(_type_=type_, _domain_source_=DomainSource.DEDUCTION)


def variable_from(
    domain: Union[Iterable[T], Selectable[T], T],
) -> Union[T, Variable[T]]:
    """
    Create a variable from a given domain.

    :param domain: An iterable or selectable expression to use as the variable's domain.
    :return: A variable that can be queried for.
    """
    return Variable(_domain_=domain)


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


def flat_variable(
    var: Union[CanBehaveLikeAVariable[T], Iterable[T]],
) -> Union[FlatVariable[T], T]:
    """
    Flatten a nested iterable domain into individual items while preserving the parent
    bindings.

    This returns a DomainMapping that, when evaluated, yields one solution per inner
    element (similar to SQL UNNEST), keeping existing variable bindings intact.
    """
    return FlatVariable(var)


# %% Logical Operators


def and_(*conditions: ConditionType) -> ConditionType:
    """
    Logical conjunction of conditions.

    :param conditions: One or more conditions to combine.
    :type conditions: SymbolicExpression | bool
    :return: An AND operator joining the conditions.
    :rtype: SymbolicExpression
    """
    return chained_logic(AND, *conditions)


def or_(*conditions: ConditionType) -> ConditionType:
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
        operand = Literal(_value_=operand)
    return operand._invert_()


def for_all(
    universal_variable: Union[CanBehaveLikeAVariable[T], T],
    condition: ConditionType,
) -> ForAll:
    """
    A universal on variable that finds all sets of variable bindings (values) that
    satisfy the condition for **every** value of the universal_variable.

    :param universal_variable: The universal on variable that the condition must satisfy
        for all its values.
    :param condition: A SymbolicExpression or bool representing a condition that must be
        satisfied.
    :return: A SymbolicExpression that can be evaluated producing every set that
        satisfies the condition.
    """
    return ForAll(universal_variable, condition)


def exists(
    universal_variable: Union[CanBehaveLikeAVariable[T], T],
    condition: ConditionType,
) -> Exists:
    """
    A universal on variable that finds all sets of variable bindings (values) that
    satisfy the condition for **any** value of the universal_variable.

    :param universal_variable: The universal on variable that the condition must satisfy
        for any of its values.
    :param condition: A SymbolicExpression or bool representing a condition that must be
        satisfied.
    :return: A SymbolicExpression that can be evaluated producing every set that
        satisfies the condition.
    """
    return Exists(universal_variable, condition)


# %% Result Quantifiers


def _quantify_or_build_match(
    arg: Union[T, Query, Type[T], Callable[..., T]],
    quantifier_type: Type[ResultQuantifier],
    quantification: Optional[ResultQuantificationConstraint] = None,
    *,
    target_type: Optional[Type[T]] = None,
) -> Union[T, Query, Match[T]]:
    """
    Shared implementation for :py:func:`an` and :py:func:`the`.

    The behaviour is selected by the runtime type of ``arg``:

    * If ``arg`` is a :class:`~krrood.entity_query_language.core.base_expressions.SymbolicExpression`
      (an entity, a set expression, a variable or an attribute), it is quantified with
      ``quantifier_type``. Raw selectables that are not already a
      :class:`~krrood.entity_query_language.query.query.Query` are first wrapped with
      :py:func:`entity`.
    * Otherwise ``arg`` is treated as a type (or a callable factory) and a structural
      :class:`~krrood.entity_query_language.query.match.Match` is built — selectable by default
      and generative-ready through a
      :class:`~krrood.entity_query_language.backends.GenerativeBackend`. Restrict the search to
      specific instances with :meth:`~krrood.entity_query_language.query.match.Match.from_`.

    :param arg: An entity/set/variable/attribute to quantify, or a type/callable to match.
    :param quantifier_type: The result quantifier to apply (``An`` or ``The``).
    :param quantification: Optional quantification constraint (quantify path only).
    :param target_type: Optional explicit type for callable factories (match path only).
    :return: A quantified query, or a ``Match`` builder.
    """
    if isinstance(arg, SymbolicExpression):
        if not isinstance(arg, Query):
            arg = entity(arg)
        return arg._quantify_(quantifier_type, quantification_constraint=quantification)

    match_ = Match(factory=arg, type_=target_type)
    match_._quantifier_type_ = quantifier_type
    return match_


@overload
def an(
    entity_: Type[T],
    quantification: None = ...,
    *,
    target_type: None = ...,
) -> Match[T]: ...


@overload
def an(
    entity_: Callable[..., T],
    quantification: None = ...,
    *,
    target_type: Type[T] = ...,
) -> Match[T]: ...


@overload
def an(
    entity_: T,
    quantification: Optional[ResultQuantificationConstraint] = ...,
    *,
    target_type: None = ...,
) -> T: ...


def an(
    entity_,
    quantification=None,
    *,
    target_type=None,
):
    """
    Select all values satisfying the given description.

    Depending on ``entity_`` this either quantifies an existing symbolic expression with the
    ``An`` quantifier (zero or more results), or builds a structural ``Match`` when ``entity_``
    is a type or a callable factory. See :py:func:`_quantify_or_build_match` for details;
    restrict a match to specific instances with
    :meth:`~krrood.entity_query_language.query.match.Match.from_`.

    :param entity_: An entity/set/variable/attribute to quantify, or a type/callable to match.
    :param quantification: Optional quantification constraint (quantify path only).
    :param target_type: Optional explicit type for callable factories (match path only).
    :return: The applied quantifier or the constructed match.
    """
    return _quantify_or_build_match(
        entity_, An, quantification, target_type=target_type
    )


@overload
def a(
    entity_: Type[T],
    quantification: None = ...,
    *,
    target_type: None = ...,
) -> Match[T]: ...


@overload
def a(
    entity_: Callable[..., T],
    quantification: None = ...,
    *,
    target_type: Type[T] = ...,
) -> Match[T]: ...


@overload
def a(
    entity_: T,
    quantification: Optional[ResultQuantificationConstraint] = ...,
    *,
    target_type: None = ...,
) -> T: ...


def a(
    entity_,
    quantification=None,
    *,
    target_type=None,
):
    """
    Select all values satisfying the given description.

    This accommodates words not starting with a vowel; it delegates to :func:`an`. It is a real
    function (not an ``a = an`` alias) so its ``__name__`` is ``"a"``, which lets tools that key a
    namespace by ``__name__`` (e.g. the doctest harness) expose it under the name ``a``.

    :param entity_: An entity/set/variable/attribute to quantify, or a type/callable to match.
    :param quantification: Optional quantification constraint (quantify path only).
    :param target_type: Optional explicit type for callable factories (match path only).
    :return: The applied quantifier or the constructed match.
    """
    return an(entity_, quantification, target_type=target_type)


@overload
def the(
    entity_: Type[T],
    *,
    target_type: None = ...,
) -> Match[T]: ...


@overload
def the(
    entity_: Callable[..., T],
    *,
    target_type: Type[T] = ...,
) -> Match[T]: ...


@overload
def the(
    entity_: T,
    *,
    target_type: None = ...,
) -> T: ...


def the(
    entity_,
    *,
    target_type=None,
):
    """
    Select the unique value satisfying the given description.

    Behaves like :py:func:`an` but applies the ``The`` quantifier, which expects exactly one
    result when the expression is materialized (raising otherwise). Restrict a match to
    specific instances with :meth:`~krrood.entity_query_language.query.match.Match.from_`.

    :param entity_: An entity/set/variable/attribute to quantify, or a type/callable to match.
    :param target_type: Optional explicit type for callable factories (match path only).
    :return: The applied quantifier or the constructed match.
    """
    return _quantify_or_build_match(entity_, The, None, target_type=target_type)


# %% Rules


def add(variable: Any, value: Any) -> None:
    """
    Add a value to a variable.

    :param variable: The variable to which the value will be added.
    :param value: The value to be added to the variable.
    :return: None
    """
    Add(variable, value)


def inference(
    type_: Type[T],
) -> Union[Callable[[], Union[T, InstantiatedVariable[T]]]]:
    """
    This returns a factory function that creates a new variable of the given type and
    takes keyword arguments for the type constructor.

    :param type_: The type of the variable (i.e., The class you want to instantiate).
    :return: The factory function for creating a new variable.
    """
    return lambda **kwargs: InstantiatedVariable(
        _type_=type_,
        _kwargs_=kwargs,
    )


def refinement(*conditions: ConditionType) -> SymbolicExpression:
    """
    Add a refinement branch (ExceptIf node with its right the new conditions and its
    left the base/parent rule/query) to the current condition tree.

    Each provided condition is chained with AND, and the resulting branch is connected
    via ExceptIf to the current node, representing a refinement/specialization path.

    :param conditions: The refinement conditions. They are chained with AND.
    :returns: The newly created branch node for further chaining.
    """
    return Refinement.create_and_update_rule_tree(*conditions)


def alternative(*conditions: ConditionType) -> SymbolicExpression:
    """
    Add an alternative branch (logical ElseIf) to the current condition tree.

    Each provided condition is chained with AND, and the resulting branch is connected
    via ElseIf to the current node, representing an alternative path.

    :param conditions: Conditions to chain with AND and attach as an alternative.
    :returns: The newly created branch node for further chaining.
    """
    return Alternative.create_and_update_rule_tree(*conditions)


def next_rule(*conditions: ConditionType) -> SymbolicExpression:
    """
    Add a consequent rule that gets always executed after the current rule.

    Each provided condition is chained with AND, and the resulting branch is connected
    via Next to the current node, representing the next path.

    :param conditions: Conditions to chain with AND and attach as an alternative.
    :returns: The newly created branch node for further chaining.
    """
    return Next.create_and_update_rule_tree(*conditions)


# %% Conditionals


def case_when(
    condition: SymbolicExpression,
    then_value: SymbolicExpression,
    else_value: Optional[SymbolicExpression] = None,
) -> CaseWhen:
    """
    Create a CASE WHEN ... THEN ... ELSE ... END expression.

    .. code-block:: python

        action = variable(MoveAction, domain=[])
        query = an(set_of(
            min(case_when(action.polymorphic_type == 'PickUpActionDAO', action.database_id))
        ))

    :param condition: The condition to evaluate
    :param then_value: The value if condition is true
    :param else_value: The value if condition is false
    :return: A CaseWhen expression
    """
    return CaseWhen(condition, then_value, else_value)


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


def mode(
    variable: Selectable[T],
    default: Optional[T] = None,
) -> Union[T, Mode[T]]:
    """
    Calculate and return the first mode from the variable values.

    The mode is the most common value in the iterable. It is found by counting the
    occurrences of each value and returning the one with the highest count. If there are
    multiple values with the same highest count, the first one encountered is returned.
    This is an aggregation function, thus the query will be fully evaluated before the
    result is returned.

    :param variable: The variable for which the mode value is to be found.
    :param default: The value returned when the iterable is empty.
    :return: A Max object that can be evaluated to find the mode value.
    """
    return Mode(variable, _default_value_=default)


def multimode(
    variable: Selectable[T],
    default: Optional[T] = None,
) -> Union[T, MultiMode[T]]:
    """
    Calculate and return all mode values from the variable values.

    Similar to :py:func:`krrood.entity_query_language.factories.mode` but returns all values that have the same mode value (i.e., all values that have the same highest count).

    :param variable: The variable for which the mode value is to be found.
    :param default: The value returned when the iterable is empty.
    :return: A Max object that can be evaluated to find the mode value.
    """
    return MultiMode(variable, _default_value_=default)


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


def count(variable: Selectable[T], distinct: bool = False) -> Union[T, Count[T]]:
    """
    Count the number of values produced by the given variable.

    :param variable: The variable for which the count is calculated.
    :param distinct: Whether to only consider distinct values.
    :return: A Count object that can be evaluated to count the number of values.
    """
    return Count(variable, _distinct_=distinct)


def count_all(distinct: bool = False) -> Union[T, Count[T]]:
    """
    Count all results (by group).

    :param distinct: Whether to only consider distinct values.
    :return: A Count object that can be evaluated to count the number of values.
    """
    return CountAll(_distinct_=distinct)


def count_range(
    variable: Selectable[T], distinct: bool = False
) -> Union[T, CountRange[T]]:
    """
    Count values produced by the given variable and return a closed interval reflecting
    uncertainty.

    Concrete (non-``...``) values set the lower bound; ``...`` (Ellipsis) values are
    added to the upper bound to represent items whose category is unknown.

    :param variable: The variable whose values are counted.
    :param distinct: Whether to only consider distinct values.
    :return: A CountRange that evaluates to a SimpleInterval ``[concrete_count,
        concrete_count + ellipsis_count]``.
    """
    return CountRange(variable, _distinct_=distinct)


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


def get_conditioned_statements(
    statement, condition: Callable[OperationResult, bool]
) -> List[SymbolicExpression]:
    """
    Iterates over all sub-statements of the statement and returns all statements that
    satisfy the condition.

    :param statement: The statement to iterate over.
    :param condition: The condition to evaluate each sub-statement against.
    :return: A list of sub-statements that satisfy the condition.
    """
    condition_results = []
    for node in [
        s
        for s in statement._children_
        if not isinstance(s, (Variable, inspect.Attribute))
    ]:
        node_result = node.evaluate()
        if condition(node_result):
            condition_results.append(node)
    if statement in condition_results:
        condition_results.remove(statement)

    return condition_results


def get_false_statements(statement: SymbolicExpression) -> List[SymbolicExpression]:
    """
    The false statements of all statements of this condition.

    :return: The false statements of all statements of this condition.
    """
    return get_conditioned_statements(statement, lambda x: not x == [])


def get_true_statements(statement: SymbolicExpression) -> List[SymbolicExpression]:
    """
    The true statements of all statements of this condition.

    :return: The true statements of this condition.
    """
    return get_conditioned_statements(statement, lambda x: x == [])


def evaluate_condition(condition: ConditionType) -> bool:
    """
    Evaluates the condition to True or False.

    :param condition: The condition to evaluate.
    :return: True if there is a possible solution, False otherwise.
    """
    if type(condition) is bool:
        return condition
    return any(condition.evaluate())


@symbolic_function
def node_id(node: SymbolicExpression) -> UUID:
    return node._id_


@symbolic_function
def node_descendants(node: SymbolicExpression) -> Iterable[SymbolicExpression]:
    return node._descendants_


@symbolic_function
def node_type(node: Selectable) -> Optional[Type]:
    return getattr(node, "_type_", None)


@symbolic_function
def node_children(node: CanBehaveLikeAVariable) -> Iterable[SymbolicExpression]:
    return node._children_


@symbolic_function
def attribute_owner_class(node: Attribute) -> Type:
    return node._owner_class_


@symbolic_function
def node_parents(node: SymbolicExpression) -> Iterable[SymbolicExpression]:
    return node._parents_


@symbolic_function
def issubclass_(cls: Type, cls_or_tuple: Type | Tuple[Type, ...]) -> bool:
    return issubclass(cls, cls_or_tuple)


@symbolic_function
def is_class(obj: Any) -> bool:
    return isclass(obj)


@symbolic_function
def type_(obj: Any) -> Type:
    return obj.__class__
