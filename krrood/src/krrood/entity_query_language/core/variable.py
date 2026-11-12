"""
This module defines variable and literal representations for the Entity Query Language.

It contains classes for simple variables, constant literals, and variables that are
instantiated from other expressions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from inspect import isclass

from typing_extensions import (
    Any,
    Iterable,
    Union as TypingUnion,
    Callable,
    Generic,
    Iterator,
    get_origin,
)

from krrood.class_diagrams.call_arguments import CallArguments
from krrood.class_diagrams.collection_kind import CollectionKind, DeclaredValueKind
from krrood.entity_query_language.core.base_expressions import (
    Bindings,
    OperationResult,
    SymbolicExpression,
    Selectable,
)
from krrood.entity_query_language.core.mapped_variable import (
    CanBehaveLikeAVariable,
    HasSymbolicOperations,
)
from krrood.entity_query_language.evaluation_context import EvaluationContext
from krrood.entity_query_language.ground_term import CallTerm, GroundTermExpression
from krrood.entity_query_language.cache_data import (
    InstanceFilteredDomain,
    ReEnterableLazyIterable,
)
from krrood.entity_query_language.enums import DomainSource
from krrood.class_diagrams.utils import get_type_hints_of_object
from krrood.entity_query_language.exceptions import (
    ResultTypeCannotBeDetermined,
    ResultTypeIsNotAType,
)
from krrood.entity_query_language.operators.set_operations import (
    MultiArityExpressionThatPerformsACartesianProduct,
)
from krrood.patterns.factory_and_kwargs import HasFactoryAndKwargs
from krrood.entity_query_language.utils import (
    T,
    is_iterable,
    make_list,
)
from krrood.entity_query_language._monitoring import (
    monitored,
)
from krrood.ormatic.utils import classproperty
from krrood.symbol_graph.symbol_graph import SymbolGraph

DomainType = Iterable[T] | HasSymbolicOperations
"""
The type of the domain used for the variable: its values, or an expression whose value,
for each of its bindings, holds them.
"""

# %% domains


@dataclass(kw_only=True)
class DomainValue(Generic[T]):
    """
    A value a variable takes, with the result it was found under.
    """

    value: T
    """
    The value the variable takes.
    """

    sources: OperationResult
    """
    The result the value was found under, whose bindings the variable's result keeps.
    """


class VariableDomain(ABC, Generic[T]):
    """
    The values a :class:`Variable` ranges over.
    """

    @abstractmethod
    def values_with_sources(
        self, *, sources: OperationResult
    ) -> Iterator[DomainValue[T]]:
        """
        :param sources: The bindings the variable is evaluated under.
        :return: Every value the variable takes under ``sources``.
        """

    @property
    @abstractmethod
    def listed_values(self) -> Iterable[T]:
        """
        The values that are known without evaluating any other expression, which are
        none for a domain that depends on another expression.
        """


@dataclass(kw_only=True)
class ListedValues(ReEnterableLazyIterable[T], VariableDomain[T]):
    """
    A domain of given values, each read as it is in every evaluation.
    """

    def values_with_sources(
        self, *, sources: OperationResult
    ) -> Iterator[DomainValue[T]]:
        return (DomainValue(value=value, sources=sources) for value in self)

    @property
    def listed_values(self) -> Iterable[T]:
        return self


@dataclass(kw_only=True)
class ExistingValues(ListedValues[T]):
    """
    A domain whose values are read, in every evaluation that iterates it, through the
    evaluation's :class:`~krrood.entity_query_language.ground_term.TermReader`, so a
    value that does not exist in the evaluation, such as an object a state removes, is
    left out.
    """

    def __iter__(self) -> Iterator[T]:
        return EvaluationContext.current_term_reader().existing_values(
            values=super().__iter__()
        )


@dataclass(kw_only=True)
class InstancesOfType(ExistingValues[T]):
    """
    The domain of every instance of a type in the evaluation that iterates it: the
    instances the live :class:`~krrood.symbol_graph.symbol_graph.SymbolGraph` holds that
    exist in the evaluation, then those the evaluation adds.
    """

    type_: type[T]
    """
    The type whose instances make up the domain.
    """

    def __post_init__(self):
        self.set_iterable(
            iterable=SymbolGraph().get_instances_of_type(type_=self.type_)
        )

    def __iter__(self) -> Iterator[T]:
        existing_ids = set()
        for instance in super().__iter__():
            existing_ids.add(id(instance))
            yield instance
        yield from (
            added
            for added in EvaluationContext.current_term_reader().added_instances_of(
                type_=self.type_
            )
            if id(added) not in existing_ids
        )


@dataclass(kw_only=True)
class ExpressionDomain(VariableDomain[T]):
    """
    The domain of the elements, for each binding of an expression, of the value it
    evaluates to, that are instances of a type.

    A value declared as one value, or of undeclared kind and not a collection, is its
    only element. A value that is ``None`` has no elements.
    """

    expression: CanBehaveLikeAVariable
    """
    The expression whose values hold the domain's values.
    """

    element_type: type[T] | Callable | None = None
    """
    The type the values are instances of, or no type filter when it is not a class.
    """

    def values_with_sources(
        self, *, sources: OperationResult
    ) -> Iterator[DomainValue[T]]:
        for result in self.expression._evaluate_(sources=sources):
            elements = self._elements_in_value_(value=result.value)
            if isclass(self.element_type):
                elements = InstanceFilteredDomain(
                    type_=self.element_type, domain=elements
                )
            yield from (
                DomainValue(value=element, sources=result) for element in elements
            )

    def _elements_in_value_(self, *, value: Any) -> Iterable[T]:
        """
        :param value: A value the expression evaluated to.
        :return: The elements of the value: the value's own when the expression is
            declared to hold a collection, or is of undeclared kind and the value is a
            collection, else the value alone.
        """
        if value is None:
            return ()
        declared_value_kind = self.expression._declared_value_kind_
        if declared_value_kind is DeclaredValueKind.COLLECTION:
            return value
        if declared_value_kind is DeclaredValueKind.UNDECLARED and is_iterable(value):
            return value
        return (value,)

    @property
    def declared_element_type(self) -> Any:
        """
        The type the expression is declared to hold its elements as.
        """
        return CollectionKind.declared_element_type(annotation=self.expression._type_)

    @property
    def listed_values(self) -> Iterable[T]:
        return ()


@dataclass(kw_only=True, eq=False, repr=False)
class CanHaveDomainSource(CanBehaveLikeAVariable[T], ABC):
    """
    A superclass for variables that can have a domain source.
    """

    _type_: type[T] | Callable = field(kw_only=True, default=None)
    """
    The values type of the variable.

    (The value of `T`)
    """

    _domain_source_: DomainSource | None = field(default=None, kw_only=True)
    """
    The source type of the domain (e.g., EXPLICIT, DEDUCED, ...etc.).
    """

    def __post_init__(self):
        self._var_ = self
        super().__post_init__()


@monitored
@dataclass(kw_only=True, eq=False, repr=False)
class Variable(CanHaveDomainSource[T]):
    """
    An atomic expression of EQL that has a domain of possible values.

    It can be evaluated to yield values from its domain. A domain that is an expression
    is read as an :class:`ExpressionDomain` of the variable's :attr:`_type_`, and a
    variable given no type takes the type the expression declares its elements as.
    """

    _domain_: DomainType | VariableDomain[T] = field(default_factory=list)
    """
    The domain as given, or as last given to :meth:`_update_domain_`.
    """

    _values_domain_: VariableDomain[T] = field(
        init=False, default_factory=ExistingValues, repr=False
    )
    """
    The values the variable ranges over, read from :attr:`_domain_`.
    """

    _domain_source_: DomainSource = field(init=False, default=DomainSource.EXPLICIT)
    """
    The source of the domain for Variable is always EXPLICIT.
    """

    def __post_init__(self):
        super().__post_init__()

        self._update_domain_(domain=self._domain_)

    @classproperty
    def _listed_values_type_(cls) -> type[ListedValues]:
        """
        The domain that given values are read as.
        """
        return ExistingValues

    def _update_domain_(self, *, domain: DomainType | VariableDomain[T]) -> None:
        """
        Range the variable over ``domain`` instead of its current domain.

        A value that is not a collection is the only value of the domain.
        """
        for domain_expression in tuple(self._children_):
            domain_expression._remove_parent_(parent=self)
        self._domain_ = domain
        if isinstance(domain, VariableDomain):
            self._values_domain_ = domain
        elif isinstance(domain, HasSymbolicOperations):
            self._values_domain_ = ExpressionDomain(
                expression=self._update_children_(domain)[0], element_type=self._type_
            )
            if self._type_ is None:
                self._type_ = self._values_domain_.declared_element_type
        else:
            self._values_domain_ = self._listed_values_type_()
            self._values_domain_.set_iterable(
                iterable=domain if is_iterable(domain) else [domain]
            )

    def _evaluate__(
        self,
        *,
        sources: OperationResult,
    ) -> Iterable[OperationResult]:
        """
        Yield an OperationResult for each value of the domain.
        """
        for domain_value in self._values_domain_.values_with_sources(sources=sources):
            bindings = domain_value.sources.bindings | {self._id_: domain_value.value}
            yield self._build_operation_result_(
                bindings=bindings, child_result=domain_value.sources
            )

    @property
    def _expression_read_from_(self) -> CanBehaveLikeAVariable | None:
        """
        Overrides :attr:`CanBehaveLikeAVariable._expression_read_from_`: the expression
        a domain that is one reads the variable's values from.
        """
        if isinstance(self._values_domain_, ExpressionDomain):
            return self._values_domain_.expression
        return None

    def _replace_child_field_(
        self, *, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        self._domain_ = new_child
        self._values_domain_ = ExpressionDomain(
            expression=new_child, element_type=self._values_domain_.element_type
        )

    @cached_property
    def _name_(self) -> str:
        if self._type_:
            return self._type_.__name__
        try:
            first_value = next(iter(self._values_domain_.listed_values))
            return f"{self.__class__.__name__}({type(first_value).__name__}, ...)"
        except StopIteration:
            return f"{self.__class__.__name__}()"


@dataclass(kw_only=True, eq=False, repr=False)
class Literal(Variable[T]):
    """
    Literals are variables that do not necessarily have a type but they must have a
    domain.
    """

    _value_: T = field(kw_only=True)
    """
    The value of the literal.
    """

    _domain_: list[T] = field(init=False, repr=False)
    """
    The domain of the literal.

    It is constructed from the `_value_` and is always a singleton iterable.
    """

    _name__: str | None = field(default=None, kw_only=True)
    """
    The name to use for the variable.
    """

    def __post_init__(
        self,
    ):
        self._domain_ = [self._value_]
        super().__post_init__()

    @classproperty
    def _listed_values_type_(cls) -> type[ListedValues]:
        """
        The literal's value is read even in an evaluation in which it does not exist,
        since a literal names one object rather than ranging over a domain.
        """
        return ListedValues

    @cached_property
    def _name_(self) -> str:
        if self._name__:
            return self._name__
        return super()._name_


@monitored
@dataclass(kw_only=True, eq=False, repr=False)
class InstantiatedVariable(
    MultiArityExpressionThatPerformsACartesianProduct,
    CanHaveDomainSource[T],
    HasFactoryAndKwargs[T],
    GroundTermExpression[CallTerm],
):
    """
    A variable which does not have an explicit domain, but builds its values by calling
    :attr:`_factory_` with the `_kwargs_` that are provided.

    The `_kwargs_` are variables that can be used to generate combinations of bindings
    to create instances for each combination. By definition this variable is inferred.
    It also represents Predicates and symbolic functions.
    """

    _factory_: Callable[..., T] = field(kw_only=True)
    """
    The callable that builds the values this variable binds, such as a type, a factory
    method or a symbolic function.
    """

    _declared_factory_: Callable[..., Any] | None = field(default=None, kw_only=True)
    """
    The callable as the query states it: the decorated function for a
    :func:`~krrood.entity_query_language.predicate.symbolic_function`, otherwise
    :attr:`_factory_` when not given.
    """

    _type_: type[T] | None = field(default=None, kw_only=True)
    """
    The type of the values this variable binds (the value of `T`), always a type.

    When not given, it is the type :attr:`_factory_` builds (see
    :meth:`~krrood.patterns.factory_and_kwargs.HasFactoryAndKwargs._type_built_by_`).
    """

    _child_variables_: dict[str, SymbolicExpression] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    A dictionary mapping child variable names to variables, these are from the _kwargs_
    dictionary.
    """

    _domain_source_: DomainSource = field(init=False, default=DomainSource.DEDUCTION)
    """
    The source of the domain for InstantiatedVariable is always DEDUCED.
    """

    def __post_init__(self):
        """
        :raises ResultTypeCannotBeDetermined: If no :attr:`_type_` is given and
            :attr:`_factory_` names no type it builds.
        :raises ResultTypeIsNotAType: If the given :attr:`_type_` is not a type.
        """
        if self._declared_factory_ is None:
            self._declared_factory_ = self._factory_
        if self._type_ is None:
            self._type_ = self._type_built_by_(factory=self._factory_)
        if self._type_ is None:
            raise ResultTypeCannotBeDetermined(factory=self._factory_)
        if not isinstance(self._type_, type):
            raise ResultTypeIsNotAType(result_type=self._type_)
        self._update_child_variables_from_kwargs_()
        self._operation_children_ = tuple(self._child_variables_.values())
        # This is done here as it uses `_operation_children_`
        super().__post_init__()

    def _update_child_variables_from_kwargs_(self):
        """
        Set the child variables from the kwargs dictionary.
        """
        for name, value in self._kwargs_.items():
            self._child_variables_[name] = SymbolicExpression._as_operand_(
                value=value, name=name
            )

    @property
    def _ground_term_operands_(self) -> tuple[SymbolicExpression, ...]:
        return tuple(self._child_variables_.values())

    def _ground_term_(self, *, operand_values: tuple[Any, ...]) -> CallTerm:
        return CallTerm.from_call(
            callee=self._factory_,
            arguments=CallArguments(
                keyword=dict(zip(self._child_variables_, operand_values))
            ),
        )

    def _evaluate__(
        self,
        *,
        sources: OperationResult,
    ) -> Iterable[OperationResult]:
        yield from self._instantiate_using_child_variables_and_yield_results_(
            sources=sources
        )

    def _instantiate_using_child_variables_and_yield_results_(
        self, *, sources: OperationResult
    ) -> Iterator[OperationResult]:
        """
        Create new instances of the variable type and using as keyword arguments the
        child variables values.
        """
        for child_result in self._evaluate_product_(sources=sources):
            # An argument without a binding is left out, so its default applies.
            arguments = {
                name: child_result.bindings[child._id_]
                for name, child in self._child_variables_.items()
                if child._id_ in child_result.bindings
            }
            instance = EvaluationContext.current_value_of_term(
                term=CallTerm.from_call(
                    callee=self._factory_, arguments=CallArguments(keyword=arguments)
                )
            ).value

            bindings = {self._id_: instance} | child_result.bindings
            result = self._build_operation_result_(
                bindings=bindings, child_result=child_result
            )
            result.previous_operation_result = child_result
            yield result

    def _replace_child_field_(
        self, *, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        MultiArityExpressionThatPerformsACartesianProduct._replace_child_field_(
            self, old_child=old_child, new_child=new_child
        )
        for name, child in self._child_variables_.items():
            if child._id_ == old_child._id_:
                self._child_variables_[name] = new_child

    @cached_property
    def _name_(self):
        return self._factory_.__name__

    def apply_mapping_on_external_root(self, *args, **kwargs: dict[str, Any]) -> Any:
        """
        Build a value with this variable's factory from the given arguments.

        :param args: Positional arguments for the factory.
        :param kwargs: Keyword arguments for the factory.
        :return: The built value.
        """
        return self._factory_(*args, **kwargs)


@dataclass(kw_only=True, eq=False, repr=False)
class ExternallySetVariable(CanHaveDomainSource[T]):
    """
    A variable that is externally set by another expression or another part of the
    application and can be used in the query language.
    """

    def _replace_child_field_(
        self, *, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        raise ValueError(f"class {self.__class__} does not have children")

    def _evaluate__(self, *, sources: OperationResult) -> Iterable[OperationResult]:
        """
        As this variable is externally set, it does not produce any results on its own,
        it just yields from an empty list to indicate that it has no results.

        It's important to note that this function will only be called when `_evaluate_`
        is called, and there's no value in the bindings `sources` for this variable.
        """
        yield from []
