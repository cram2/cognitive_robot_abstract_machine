"""
This module defines variable and literal representations for the Entity Query Language.

It contains classes for simple variables, constant literals, and variables that are
instantiated from other expressions.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from functools import cached_property

from typing_extensions import (
    Any,
    Iterable,
    Union as TypingUnion,
    Callable,
    Iterator,
    get_origin,
)

from krrood.class_diagrams.call_arguments import CallArguments
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
    ReEnterableIterable,
    ReEnterableLazyIterable,
)
from krrood.entity_query_language.enums import DomainSource
from krrood.class_diagrams.utils import get_type_hints_of_object
from krrood.entity_query_language.exceptions import (
    NoChildToReplace,
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
from krrood.symbol_graph.symbol_graph import SymbolGraph

DomainType = Iterable[T] | HasSymbolicOperations
"""
The type of the domain used for the variable: its values, or an expression whose value,
for each of its bindings, holds them.
"""

# %% domains


@dataclass
class InstancesOfType(ReEnterableIterable[T]):
    """
    The domain of every instance of a type in the evaluation that iterates it: the
    instances the evaluation's
    :class:`~krrood.entity_query_language.ground_term.TermReader` reads, which outside a
    state are those the live :class:`~krrood.symbol_graph.symbol_graph.SymbolGraph`
    holds.
    """

    type_: type[T]
    """
    The type whose instances make up the domain.
    """

    live_instances: ReEnterableLazyIterable[T] = field(
        init=False, default_factory=ReEnterableLazyIterable
    )
    """
    The instances of :attr:`type_` the live symbol graph holds, read once and kept.
    """

    def __post_init__(self):
        self.live_instances.set_iterable(
            SymbolGraph().get_instances_of_type(self.type_)
        )

    def __iter__(self) -> Iterator[T]:
        return EvaluationContext.current_term_reader().instances_of(
            self.type_, self.live_instances
        )


@dataclass(eq=False, repr=False)
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
@dataclass(eq=False, repr=False)
class Variable(CanHaveDomainSource[T]):
    """
    An atomic expression of EQL that has a domain of possible values.

    It can be evaluated to yield values from its domain.
    """

    _domain_: Iterable[T] = field(default_factory=list)
    """
    The original domain value.
    """

    _re_enterable_domain_generator_: ReEnterableIterable[T] = field(
        init=False, default_factory=ReEnterableLazyIterable, repr=False
    )
    """
    The re-enterable generator of values for this variable.

    This is created from the provided `_domain_`.
    """

    _domain_source_: DomainSource = field(init=False, default=DomainSource.EXPLICIT)
    """
    The source of the domain for Variable is always EXPLICIT.
    """

    def __post_init__(self):
        super().__post_init__()

        self._update_domain_(self._domain_)

    def _update_domain_(self, domain):
        """
        Set the domain, keeping a :class:`ReEnterableIterable` as it is and wrapping any
        other value in a :class:`ReEnterableLazyIterable`.
        """
        if isinstance(domain, ReEnterableIterable):
            self._re_enterable_domain_generator_ = domain
            return
        if not is_iterable(domain):
            domain = [domain]
        if not isinstance(
            self._re_enterable_domain_generator_, ReEnterableLazyIterable
        ):
            self._re_enterable_domain_generator_ = ReEnterableLazyIterable()
        self._re_enterable_domain_generator_.set_iterable(domain)

    def _evaluate__(
        self,
        sources: OperationResult,
    ) -> Iterable[OperationResult]:
        """
        Fetch values from the domain values and yield an OperationResult for each.
        """
        for v in self._re_enterable_domain_generator_:
            bindings = sources.bindings | {self._id_: v}
            yield self._build_operation_result_(bindings, sources)

    def _replace_child_field_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        raise NoChildToReplace(self, old_child, new_child)

    @cached_property
    def _name_(self) -> str:
        if self._type_:
            return self._type_.__name__
        try:
            first_value = next(iter(self._re_enterable_domain_generator_))
            return f"{self.__class__.__name__}({type(first_value).__name__}, ...)"
        except StopIteration:
            return f"{self.__class__.__name__}()"


@dataclass(eq=False, repr=False)
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

    @cached_property
    def _name_(self) -> str:
        if self._name__:
            return self._name__
        return super()._name_


@monitored
@dataclass(eq=False, repr=False)
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
            self._type_ = self._type_built_by_(self._factory_)
        if self._type_ is None:
            raise ResultTypeCannotBeDetermined(self._factory_)
        if not isinstance(self._type_, type):
            raise ResultTypeIsNotAType(self._type_)
        self._update_child_variables_from_kwargs_()
        self._operation_children_ = tuple(self._child_variables_.values())
        # This is done here as it uses `_operation_children_`
        super().__post_init__()

    def _update_child_variables_from_kwargs_(self):
        """
        Set the child variables from the kwargs dictionary.
        """
        for name, value in self._kwargs_.items():
            self._child_variables_[name] = SymbolicExpression._as_operand_(value, name)

    @property
    def _ground_term_operands_(self) -> tuple[SymbolicExpression, ...]:
        return tuple(self._child_variables_.values())

    def _ground_term_(self, *operand_values: Any) -> CallTerm:
        return CallTerm.from_call(
            self._factory_,
            CallArguments(keyword=dict(zip(self._child_variables_, operand_values))),
        )

    def _evaluate__(
        self,
        sources: OperationResult,
    ) -> Iterable[OperationResult]:
        yield from self._instantiate_using_child_variables_and_yield_results_(sources)

    def _instantiate_using_child_variables_and_yield_results_(
        self, sources: OperationResult
    ) -> Iterator[OperationResult]:
        """
        Create new instances of the variable type and using as keyword arguments the
        child variables values.
        """
        for child_result in self._evaluate_product_(sources):
            # An argument without a binding is left out, so its default applies.
            arguments = {
                name: child_result.bindings[child._id_]
                for name, child in self._child_variables_.items()
                if child._id_ in child_result.bindings
            }
            instance = EvaluationContext.current_value_of_term(
                CallTerm.from_call(self._factory_, CallArguments(keyword=arguments))
            ).value

            bindings = {self._id_: instance} | child_result.bindings
            result = self._build_operation_result_(bindings, child_result)
            result.previous_operation_result = child_result
            yield result

    def _replace_child_field_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        MultiArityExpressionThatPerformsACartesianProduct._replace_child_field_(
            self, old_child, new_child
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


@dataclass(eq=False, repr=False)
class ExternallySetVariable(CanHaveDomainSource[T]):
    """
    A variable that is externally set by another expression or another part of the
    application and can be used in the query language.
    """

    def _replace_child_field_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        raise ValueError(f"class {self.__class__} does not have children")

    def _evaluate__(self, sources: OperationResult) -> Iterable[OperationResult]:
        """
        As this variable is externally set, it does not produce any results on its own,
        it just yields from an empty list to indicate that it has no results.

        It's important to note that this function will only be called when `_evaluate_`
        is called, and there's no value in the bindings `sources` for this variable.
        """
        yield from []
