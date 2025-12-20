from __future__ import annotations

from .symbol_graph import SymbolGraph
from .utils import is_iterable, T

"""
User interface (grammar & vocabulary) for entity query language.
"""
import operator

from typing_extensions import (
    Any,
    Optional,
    Union,
    Iterable,
    Type,
    Callable,
    TYPE_CHECKING,
)

from .symbolic import (
    SymbolicExpression,
    Entity,
    SetOf,
    AND,
    Comparator,
    chained_logic,
    CanBehaveLikeAVariable,
    Variable,
    optimize_or,
    Flatten,
    ForAll,
    Exists,
    Literal,
    Selectable,
    DomainType,
)

from .predicate import (
    Predicate,
    # type: ignore
    Symbol,  # type: ignore
)

if TYPE_CHECKING:
    pass

ConditionType = Union[SymbolicExpression, bool, Predicate]
"""
The possible types for conditions.
"""


def entity(selected_variable: T) -> Entity[T]:
    """
    Create an entity descriptor for a selected variable.

    :param selected_variable: The variable to select in the result.
    :return: Entity descriptor.
    """
    return Entity(_selected_variables=[selected_variable])


def set_of(*selected_variables: Union[Selectable[T], Any]) -> SetOf:
    """
    Create a set descriptor for the selected variables.

    :param selected_variables: The variables to select in the result set.
    :return: Set descriptor.
    """
    return SetOf(_selected_variables=list(selected_variables))


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
    domain_source = _get_domain_source_from_domain_and_type_values(domain, type_)

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
    domain: DomainType,
    name: Optional[str] = None,
) -> Union[T, Selectable[T]]:
    """
    Similar to `variable` but constructed from a domain directly wihout specifying its type.
    """
    return Literal(data=domain, name=name)


def _get_domain_source_from_domain_and_type_values(
    domain: DomainType, type_: Type
) -> Optional[DomainType]:
    """
    Get the domain source from the domain and the type values.

    :param domain: The domain value.
    :param type_: The type of the variable.
    :return: The domain source as a From object.
    """
    if is_iterable(domain):
        domain = filter(lambda x: isinstance(x, type_), domain)
    elif domain is None and issubclass(type_, Symbol):
        domain = SymbolGraph().get_instances_of_type(type_)
    return domain


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
    return chained_logic(optimize_or, *conditions)


def not_(operand: ConditionType) -> SymbolicExpression:
    """
    A symbolic NOT operation that can be used to negate symbolic expressions.
    """
    if not isinstance(operand, SymbolicExpression):
        operand = Literal(operand)
    return operand._invert_()


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


def inference(
    type_: Type[T],
) -> Union[Type[T], Callable[[Any], Variable[T]]]:
    """
    This returns a factory function that creates a new variable of the given type and takes keyword arguments for the
    type constructor.

    :param type_: The type of the variable (i.e., The class you want to instantiate).
    :return: The factory function for creating a new variable.
    """
    return lambda **kwargs: Variable(
        _type_=type_, _name__=type_.__name__, _kwargs_=kwargs, _is_inferred_=True
    )
