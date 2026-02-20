"""
This module provides mechanisms for mapping symbolic expressions to object domains.

It contains classes for attribute access, indexing, and function calls on symbolic expressions.
"""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass, fields, field
from functools import cached_property

from typing_extensions import (
    List,
    Iterable,
    Any,
    Type,
    Optional,
    Tuple,
    Dict,
    TYPE_CHECKING,
)

from ..operators.comparator import Comparator
from ...class_diagrams.class_diagram import WrappedClass
from ...class_diagrams.failures import ClassIsUnMappedInClassDiagram
from ...class_diagrams.wrapped_field import WrappedField

from .base_expressions import (
    UnaryExpression,
    Bindings,
    OperationResult,
    Selectable,
    TruthValueOperator,
)
from ..utils import (
    T,
    merge_args_and_kwargs,
    convert_args_and_kwargs_into_a_hashable_key,
    is_iterable,
)
from ...symbol_graph.symbol_graph import SymbolGraph


@dataclass(eq=False, repr=False)
class CanBehaveLikeAVariable(Selectable[T], ABC):
    """
    This class adds the monitoring/tracking behavior on variables that tracks attribute access, calling,
    and comparison operations.
    """

    _known_mappings_: Dict[DomainMappingCacheItem, DomainMapping] = field(
        init=False, default_factory=dict
    )
    """
    A storage of created domain mappings to prevent recreating same mapping multiple times.
    """

    def _get_domain_mapping_(
        self, type_: Type[DomainMapping], *args, **kwargs
    ) -> DomainMapping:
        """
        Retrieves or creates a domain mapping instance based on the provided arguments.

        :param type_: The type of the domain mapping to retrieve or create.
        :param args: Positional arguments to pass to the domain mapping constructor.
        :param kwargs: Keyword arguments to pass to the domain mapping constructor.
        :return: The retrieved or created domain mapping instance.
        """
        cache_item = DomainMappingCacheItem(type_, self, args, kwargs)
        if cache_item in self._known_mappings_:
            return self._known_mappings_[cache_item]
        else:
            instance = type_(**cache_item.all_kwargs)
            self._known_mappings_[cache_item] = instance
            return instance

    def _get_domain_mapping_key_(self, type_: Type[DomainMapping], *args, **kwargs):
        """
        Generates a hashable key for the given type and arguments.

        :param type_: The type of the domain mapping.
        :param args: Positional arguments to pass to the domain mapping constructor.
        :param kwargs: Keyword arguments to pass to the domain mapping constructor.
        :return: The generated hashable key.
        """
        args = (self,) + args
        all_kwargs = merge_args_and_kwargs(type_, args, kwargs, ignore_first=True)
        return convert_args_and_kwargs_into_a_hashable_key(all_kwargs)

    def __getattr__(self, name: str) -> CanBehaveLikeAVariable[T]:
        # Prevent debugger/private attribute lookups from being interpreted as symbolic attributes
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute {name}"
            )
        return self._get_domain_mapping_(Attribute, name, self._type__)

    def __getitem__(self, key) -> CanBehaveLikeAVariable[T]:
        return self._get_domain_mapping_(Index, key)

    def __call__(self, *args, **kwargs) -> CanBehaveLikeAVariable[T]:
        return self._get_domain_mapping_(Call, args, kwargs)

    def __eq__(self, other) -> Comparator:
        return Comparator(self, other, operator.eq)

    def __ne__(self, other) -> Comparator:
        return Comparator(self, other, operator.ne)

    def __lt__(self, other) -> Comparator:
        return Comparator(self, other, operator.lt)

    def __le__(self, other) -> Comparator:
        return Comparator(self, other, operator.le)

    def __gt__(self, other) -> Comparator:
        return Comparator(self, other, operator.gt)

    def __ge__(self, other) -> Comparator:
        return Comparator(self, other, operator.ge)

    def __hash__(self):
        return super().__hash__()


@dataclass(eq=False, repr=False)
class DomainMapping(UnaryExpression, CanBehaveLikeAVariable[T], ABC):
    """
    A symbolic expression the maps the domain of symbolic variables.
    """

    _child_: CanBehaveLikeAVariable[T]
    """
    The child expression to apply the domain mapping to.
    """

    def __post_init__(self):
        super().__post_init__()
        self._var_ = self

    @cached_property
    def _type_(self):
        return self._child_._type_

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        """
        Apply the domain mapping to the child's values.
        """

        yield from (
            self._build_operation_result_and_update_truth_value_(
                child_result, mapped_value
            )
            for child_result in self._child_._evaluate_(sources, parent=self)
            for mapped_value in self._apply_mapping_(child_result.value)
        )

    def _build_operation_result_and_update_truth_value_(
        self, child_result: OperationResult, current_value: Any
    ) -> OperationResult:
        """
        Set the current truth value of the operation result, and build the operation result to be yielded.

        :param child_result: The current result from the child operation.
        :param current_value: The current value of this operation that is derived from the child result.
        :return: The operation result.
        """
        self._update_truth_value_(current_value)
        return OperationResult(
            {**child_result.bindings, self._binding_id_: current_value},
            self._is_false_,
            self,
        )

    @abstractmethod
    def _apply_mapping_(self, value: Any) -> Iterable[Any]:
        """
        Apply the domain mapping to a symbolic value.
        """
        pass


@dataclass(eq=False, repr=False)
class Attribute(DomainMapping):
    """
    A symbolic attribute that can be used to access attributes of symbolic variables.

    For instance, if Body.name is called, then the attribute name is "name" and `_owner_class_` is `Body`
    """

    _attribute_name_: str
    """
    The name of the attribute.
    """

    _owner_class_: Type
    """
    The class that owns this attribute.
    """

    @property
    def _original_value_is_iterable_and_this_operation_preserves_that_(self):
        if not self._wrapped_field_:
            return False
        return self._wrapped_field_.is_iterable

    @cached_property
    def _type_(self) -> Optional[Type]:
        """
        :return: The type of the accessed attribute.
        """

        if not is_dataclass(self._owner_class_):
            return None

        if self._attribute_name_ not in {f.name for f in fields(self._owner_class_)}:
            return None

        if self._wrapped_owner_class_:
            # try to get the type endpoint from a field
            try:
                return self._wrapped_field_.type_endpoint
            except (KeyError, AttributeError):
                return None
        else:
            wrapped_cls = WrappedClass(self._owner_class_)
            wrapped_cls._class_diagram = SymbolGraph().class_diagram
            wrapped_field = WrappedField(
                wrapped_cls,
                [
                    f
                    for f in fields(self._owner_class_)
                    if f.name == self._attribute_name_
                ][0],
            )
            try:
                return wrapped_field.type_endpoint
            except (AttributeError, RuntimeError):
                return None

    @cached_property
    def _wrapped_field_(self) -> Optional[WrappedField]:
        if self._wrapped_owner_class_ is None:
            return None
        return self._wrapped_owner_class_._wrapped_field_name_map_.get(
            self._attribute_name_, None
        )

    @cached_property
    def _wrapped_owner_class_(self):
        """
        :return: The owner class of the attribute from the symbol graph.
        """
        try:
            return SymbolGraph().class_diagram.get_wrapped_class(self._owner_class_)
        except ClassIsUnMappedInClassDiagram:
            return None

    def _apply_mapping_(self, value: Any) -> Iterable[Any]:
        yield getattr(value, self._attribute_name_)

    @property
    def _name_(self):
        return f"{self._child_._var_._name_}.{self._attribute_name_}"


@dataclass(eq=False, repr=False)
class Index(DomainMapping):
    """
    A symbolic indexing operation that can be used to access items of symbolic variables via [] operator.
    """

    _key_: Any
    """
    The key to index with.
    """

    def _apply_mapping_(self, value: Any) -> Iterable[Any]:
        yield value[self._key_]

    @property
    def _name_(self):
        return f"{self._child_._var_._name_}[{self._key_}]"


@dataclass(eq=False, repr=False)
class Call(DomainMapping):
    """
    A symbolic call that can be used to call methods on symbolic variables.
    """

    _args_: Tuple[Any, ...] = field(default_factory=tuple)
    """
    The arguments to call the method with.
    """
    _kwargs_: Dict[str, Any] = field(default_factory=dict)
    """
    The keyword arguments to call the method with.
    """

    def _apply_mapping_(self, value: Any) -> Iterable[Any]:
        if len(self._args_) > 0 or len(self._kwargs_) > 0:
            yield value(*self._args_, **self._kwargs_)
        else:
            yield value()

    @property
    def _name_(self):
        return f"{self._child_._var_._name_}()"


@dataclass(eq=False, repr=False)
class Flatten(DomainMapping):
    """
    Domain mapping that flattens an iterable-of-iterables into a single iterable of items.

    Given a child expression that evaluates to an iterable (e.g., Views.bodies), this mapping yields
    one solution per inner element while preserving the original bindings (e.g., the View instance),
    similar to UNNEST in SQL.
    """

    def _apply_mapping_(self, value: Iterable[Any]) -> Iterable[Any]:
        yield from value

    @cached_property
    def _name_(self):
        return f"Flatten({self._child_._name_})"

    @property
    def _original_value_is_iterable_and_this_operation_preserves_that_(self):
        """
        :return: False as Flatten loops inside the iterable yielding element by element.
        """
        return False


@dataclass
class DomainMappingCacheItem:
    """
    A cache item for domain mapping creation. To prevent recreating same mapping multiple times, mapping instances are
    stored in a dictionary with a hashable key. This class is used to generate the key for the dictionary that stores
    the mapping instances.
    """

    type: Type[DomainMapping]
    """
    The type of the domain mapping.
    """
    child: CanBehaveLikeAVariable
    """
    The child of the domain mapping (i.e. the original variable on which the domain mapping is applied).
    """
    args: Tuple[Any, ...] = field(default_factory=tuple)
    """
    Positional arguments to pass to the domain mapping constructor.
    """
    kwargs: Dict[str, Any] = field(default_factory=dict)
    """
    Keyword arguments to pass to the domain mapping constructor.
    """

    def __post_init__(self):
        self.args = (self.child,) + self.args

    @cached_property
    def all_kwargs(self):
        return merge_args_and_kwargs(
            self.type, self.args, self.kwargs, ignore_first=True
        )

    @cached_property
    def hashable_key(self):
        return (self.type,) + convert_args_and_kwargs_into_a_hashable_key(
            self.all_kwargs
        )

    def __hash__(self):
        return hash(self.hashable_key)

    def __eq__(self, other):
        return (
            isinstance(other, DomainMappingCacheItem)
            and self.hashable_key == other.hashable_key
        )
