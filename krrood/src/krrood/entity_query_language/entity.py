from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property


from .hashed_data import T
from .symbol_graph import SymbolGraph
from .utils import is_iterable

"""
User interface (grammar & vocabulary) for entity query language.
"""
import operator

from typing_extensions import (
    Any,
    Optional,
    Union,
    Iterable,
    Dict,
    Generic,
    Type,
    Tuple,
    List,
    Callable,
    TypeVar,
)

from .symbolic import (
    SymbolicExpression,
    Entity,
    SetOf,
    The,
    An,
    AND,
    Comparator,
    chained_logic,
    CanBehaveLikeAVariable,
    From,
    Variable,
    optimize_or,
    Flatten,
    ForAll,
    Exists,
    Literal,
    ResultQuantifier,
)
from .result_quantification_constraint import ResultQuantificationConstraint

from .predicate import (
    Predicate,
    # type: ignore
    Symbol,  # type: ignore
    HasType,
)

ConditionType = Union[SymbolicExpression, bool, Predicate]
"""
The possible types for conditions.
"""

EntityType = Union[
    SetOf[T], Entity[T], T, Iterable[T], Type[T]
]  # include Match[T] after moving match to a module @bass
"""
The possible types for entities.
"""


def an(
    entity_: EntityType,
    quantification: Optional[ResultQuantificationConstraint] = None,
) -> Union[An[T], T]:
    """
    Select a single element satisfying the given entity description.

    :param entity_: An entity or a set expression to quantify over.
    :param quantification: Optional quantification constraint.
    :return: A quantifier representing "an" element.
    :rtype: An[T]
    """
    return _quantify_entity(An, entity_, _quantification_constraint_=quantification)


a = an
"""
This is an alias to accommodate for words not starting with vowels.
"""


def the(
    entity_: EntityType,
) -> Union[The[T], T]:
    """
    Select the unique element satisfying the given entity description.

    :param entity_: An entity or a set expression to quantify over.
    :return: A quantifier representing "an" element.
    :rtype: The[T]
    """
    return _quantify_entity(The, entity_)


def _quantify_entity(
    quantifier: Type[ResultQuantifier], entity_: EntityType, **quantifier_kwargs
) -> Union[ResultQuantifier[T], T]:
    """
    Apply the given quantifier to the given entity.

    :param quantifier: The quantifier to apply.
    :param entity_: The entity to quantify.
    :param quantifier_kwargs: Keyword arguments to pass to the quantifier.
    :return: The quantified entity.
    """
    if isinstance(entity_, Match):
        entity_ = entity_.expression
    return quantifier(entity_, **quantifier_kwargs)


def entity(
    selected_variable: T,
    *properties: ConditionType,
) -> Entity[T]:
    """
    Create an entity descriptor from a selected variable and its properties.

    :param selected_variable: The variable to select in the result.
    :type selected_variable: T
    :param properties: Conditions that define the entity.
    :type properties: Union[SymbolicExpression, bool]
    :return: Entity descriptor.
    :rtype: Entity[T]
    """
    selected_variables, expression = _extract_variables_and_expression(
        [selected_variable], *properties
    )
    return Entity(selected_variables=selected_variables, _child_=expression)


def set_of(
    selected_variables: Iterable[T],
    *properties: ConditionType,
) -> SetOf[T]:
    """
    Create a set descriptor from selected variables and their properties.

    :param selected_variables: Iterable of variables to select in the result set.
    :type selected_variables: Iterable[T]
    :param properties: Conditions that define the set.
    :type properties: Union[SymbolicExpression, bool]
    :return: Set descriptor.
    :rtype: SetOf[T]
    """
    selected_variables, expression = _extract_variables_and_expression(
        selected_variables, *properties
    )
    return SetOf(selected_variables=selected_variables, _child_=expression)


def _extract_variables_and_expression(
    selected_variables: Iterable[T], *properties: ConditionType
) -> Tuple[List[T], SymbolicExpression]:
    """
    Extracts the variables and expressions from the selected variables.

    :param selected_variables: Iterable of variables to select in the result set.
    :param properties: Conditions on the selected variables.
    :return: Tuple of selected variables and expressions.
    """
    expression_list = list(properties)
    selected_variables = list(selected_variables)
    expression = None
    if len(expression_list) > 0:
        expression = (
            and_(*expression_list) if len(expression_list) > 1 else expression_list[0]
        )
    return selected_variables, expression


DomainType = Union[Iterable, None]


def let(
    type_: Type[T],
    domain: DomainType,
    name: Optional[str] = None,
) -> Union[T, CanBehaveLikeAVariable[T], Variable[T]]:
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
    :return: A Variable that can be queried for.
    """
    domain_source = _get_domain_source_from_domain_and_type_values(domain, type_)

    if name is None:
        name = type_.__name__

    result = Variable(
        _type_=type_,
        _domain_source_=domain_source,
        _name__=name,
    )

    return result


def _get_domain_source_from_domain_and_type_values(
    domain: DomainType, type_: Type
) -> Optional[From]:
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
    return From(domain)


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


def not_(operand: SymbolicExpression):
    """
    A symbolic NOT operation that can be used to negate symbolic expressions.
    """
    if not isinstance(operand, SymbolicExpression):
        operand = Literal(operand)
    return operand.__invert__()


def contains(container: Union[Iterable, CanBehaveLikeAVariable[T]], item: Any):
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


@dataclass
class Match(Generic[T]):
    """
    Construct a query that looks for the pattern provided by the type and the keyword arguments.
    """

    type_: Type[T]
    """
    The type of the variable.
    """
    kwargs: Dict[str, Any]
    """
    The keyword arguments to match against.
    """
    variable: CanBehaveLikeAVariable[T] = field(init=False)
    """
    The created variable from the type and kwargs.
    """
    conditions: List[ConditionType] = field(init=False, default_factory=list)
    """
    The conditions that define the match.
    """

    def _resolve(self, variable: Optional[CanBehaveLikeAVariable] = None):
        """
        Resolve the match by creating the variable and conditions expressions.

        :param variable: An optional pre-existing variable to use for the match; if not provided, a new variable will be created.
        :return:
        """
        self.variable = variable if variable else self._create_variable()
        for k, v in self.kwargs.items():
            attr = getattr(self.variable, k)
            if isinstance(v, Match):
                v._resolve(attr)
                self.conditions.append(HasType(attr, v.type_))
                self.conditions.extend(v.conditions)
            else:
                self.conditions.append(attr == v)

    def _create_variable(self) -> Variable[T]:
        """
        Create a variable with the given type.
        """
        return let(self.type_, None)

    @cached_property
    def expression(self) -> Entity[T]:
        """
        Return the entity expression corresponding to the match query.
        """
        self._resolve()
        return entity(self.variable, *self.conditions)


@dataclass
class MatchEntity(Match[T]):
    """
    A match that can also take a domain and should be used as the outermost match in a nested match statement.
    This is because the inner match statements derive their domain from the outer match as they are basically attributes
    of the outer match variable.
    """

    domain: DomainType
    """
    The domain to use for the variable created by the match.
    """

    def _create_variable(self) -> Variable[T]:
        """
        Create a variable with the given type and domain.
        """
        return let(self.type_, self.domain)


def match(type_: Type[T]) -> Union[Type[T], Callable[..., Match[T]]]:
    """
    This returns a factory function that creates a Match instance that looks for the pattern provided by the type and the
    keyword arguments.

    :param type_: The type of the variable (i.e., The class you want to instantiate).
    :return: The factory function for creating the match query.
    """

    def match_factory(**kwargs) -> Match[T]:
        return Match(type_, kwargs)

    return match_factory


def entity_matching(
    type_: Type[T], domain: DomainType
) -> Union[Type[T], Callable[..., MatchEntity[T]]]:
    """
    Same as :py:func:`krrood.entity_query_language.entity.match` but with a domain to use for the variable created
     by the match.

    :param type_: The type of the variable (i.e., The class you want to instantiate).
    :param domain: The domain used for the variable created by the match.
    :return: The factory function for creating the match query.
    """

    def match_factory(**kwargs) -> MatchEntity[T]:
        return MatchEntity(type_, kwargs, domain)

    return match_factory
