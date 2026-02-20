"""
This module defines variable and literal representations for the Entity Query Language.

It contains classes for simple variables, constant literals, and variables that are instantiated from other expressions.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from functools import cached_property

from typing_extensions import (
    Type,
    Any,
    Dict,
    Optional,
    Iterable,
    Union as TypingUnion,
    Union,
    Callable,
    Iterator,
)

from .base_expressions import (
    Bindings,
    OperationResult,
    SymbolicExpression,
    Selectable,
)
from .domain_mapping import CanBehaveLikeAVariable
from ..cache_data import ReEnterableLazyIterable
from ..operators.set_operations import MultiArityExpressionThatPerformsACartesianProduct
from ..utils import (
    T,
    is_iterable,
    make_list,
)


@dataclass(eq=False, repr=False)
class HasDomain(CanBehaveLikeAVariable[T], ABC):
    """
    A superclass for expressions that have a domain.
    """

    _domain_source_: Iterable[T] = field(default_factory=list)
    """
    The source for the variable domain.
    """
    _domain_: ReEnterableLazyIterable = field(
        init=False, default_factory=ReEnterableLazyIterable, repr=False
    )
    """
    The iterable domain of values for this variable.
    """

    def __post_init__(self):
        super().__post_init__()

        self._update_domain_(self._domain_source_)

        self._var_ = self

    def _update_domain_(self, domain):
        """
        Set the domain and ensure it is a lazy re-enterable iterable.
        """
        if isinstance(domain, ReEnterableLazyIterable):
            self._domain_ = domain
            return
        if not is_iterable(domain):
            domain = [domain]
        self._domain_.set_iterable(domain)

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        """
        Fetch values from the domain values and yield an OperationResult for each.
        """

        for v in self._domain_:
            bindings = {**sources, self._binding_id_: v}
            yield self._build_operation_result_and_update_truth_value_(bindings)

    def _replace_child_field_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        raise ValueError(f"class {self.__class__} does not have children")

    @property
    def _original_value_is_iterable_and_this_operation_preserves_that_(self):
        return is_iterable(next(iter(self._domain_), None))


@dataclass(eq=False, repr=False)
class Variable(HasDomain[T]):
    """
    A Variable that queries will assign. The Variable produces results of type `T`.
    """

    _type_: Union[Type, Callable] = field(kw_only=True)
    """
    The result type of the variable. (The value of `T`)
    """

    @cached_property
    def _name_(self):
        return self._type_.__name__


@dataclass(eq=False, repr=False)
class Literal(HasDomain[T]):
    """
    Literals are variables that do not necessarily have a type but they must have a domain.
    """

    _wrap_in_iterator_: bool = field(default=True, kw_only=True)
    """
    Whether to wrap the domain in an iterator.
    """

    def __post_init__(
        self,
    ):
        if self._wrap_in_iterator_:
            self._domain_source_ = [self._domain_source_]
        super().__post_init__()


@dataclass(eq=False, repr=False)
class InstantiatedVariable(
    MultiArityExpressionThatPerformsACartesianProduct, CanBehaveLikeAVariable[T]
):
    """
    A variable which does not have an explicit domain, but creates new instances using the `_type_` and `_kwargs_`
    that are provided. The `_kwargs_` are variables that can be used to generate combinations of bindings to create
    instances for each combination. By definition this variable is inferred. It also represents Predicates and symbolic
    functions.
    """

    _kwargs_: Dict[str, Any] = field(default_factory=dict)
    """
    The properties of the variable as keyword arguments.
    """
    _child_vars_: Dict[str, SymbolicExpression] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    A dictionary mapping child variable names to variables, these are from the _kwargs_ dictionary. 
    """
    _child_var_id_name_map_: Dict[int, str] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    A dictionary mapping child variable ids to their names. 
    """

    def __post_init__(self):
        self._is_inferred_ = True
        self._update_child_vars_from_kwargs_()
        self._operation_children_ = tuple(self._child_vars_.values())
        # This is done here as it uses `_operation_children_`
        super().__post_init__()

    def _update_child_vars_from_kwargs_(self):
        """
        Set the child variables from the kwargs dictionary.
        """
        for k, v in self._kwargs_.items():
            self._child_vars_[k] = (
                v if isinstance(v, SymbolicExpression) else Literal(v, name=k)
            )
            self._child_var_id_name_map_[self._child_vars_[k]._binding_id_] = k

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        yield from self._instantiate_using_child_vars_and_yield_results_(sources)

    def _instantiate_using_child_vars_and_yield_results_(
        self, sources: Bindings
    ) -> Iterator[OperationResult]:
        """
        Create new instances of the variable type and using as keyword arguments the child variables values.
        """
        for child_result in self._evaluate_product_(sources):
            # Build once: unwrapped hashed kwargs for already provided child vars
            kwargs = {
                self._child_var_id_name_map_[id_]: v
                for id_, v in child_result.bindings.items()
                if id_ in self._child_var_id_name_map_
            }
            instance = self._type_(**kwargs)
            bindings = {self._binding_id_: instance} | child_result.bindings
            result = self._build_operation_result_and_update_truth_value_(bindings)
            result.previous_operation_result = child_result
            yield result

    def _replace_child_field_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        MultiArityExpressionThatPerformsACartesianProduct._replace_child_field_(
            self, old_child, new_child
        )
        for k, v in self._child_vars_.items():
            if v is old_child:
                self._child_vars_[k] = new_child
                self._child_var_id_name_map_[self._child_vars_[k]._binding_id_] = k
                break


@dataclass(eq=False, repr=False)
class ExternallySetVariable(CanBehaveLikeAVariable[T]):
    """
    A variable that is externally set by another expression or another part of the application and can be used in the
     query language.
    """

    def _replace_child_field_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        raise ValueError(f"class {self.__class__} does not have children")

    def _evaluate__(self, sources: Bindings) -> Iterable[OperationResult]:
        raise ValueError(f"Variable {self._name_} should be externally set.")


@dataclass(eq=False, repr=False)
class ReasonedVariable(ExternallySetVariable):
    """
    A variable that reasoning infers (e.g., using rules or other inference mechanisms).
    """


DomainType = TypingUnion[Iterable[T], CanBehaveLikeAVariable[T], None]
"""
The type of the domain used for the variable.
"""
