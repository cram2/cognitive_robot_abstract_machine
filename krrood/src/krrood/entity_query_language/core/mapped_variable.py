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
    Iterable,
    Any,
    Type,
    Optional,
    Tuple,
    Dict,
)

from .base_expressions import (
    UnaryExpression,
    Bindings,
    OperationResult,
    Selectable,
)
from ..operators.comparator import Comparator
from ..utils import (
    T,
    merge_args_and_kwargs,
    convert_args_and_kwargs_into_hashable_key,
)

from ...symbol_graph.helpers import get_field_type_endpoint


@dataclass(eq=False, repr=False)
class CanBehaveLikeAVariable(Selectable[T], ABC):
    """
    This class adds the monitoring/tracking behavior on variables that tracks attribute access, calling,
    and comparison operations.
    """

    _known_mapped_variables_: Dict[MappedVariableCacheItem, MappedVariable] = field(
        init=False, default_factory=dict
    )
    """
    A storage of created MappedVariable instances to prevent recreating same mapping multiple times.
    """

    def _get_mapped_variable_(
        self, type_: Type[MappedVariable], *args, **kwargs
    ) -> MappedVariable:
        """
        Retrieves or creates a MappedVariable instance based on the provided arguments.

        :param type_: The type of the MappedVariable to retrieve or create.
        :param args: Positional arguments to pass to the MappedVariable constructor.
        :param kwargs: Keyword arguments to pass to the MappedVariable constructor.
        :return: The retrieved or created MappedVariable instance.
        """
        cache_item = MappedVariableCacheItem(type_, self, args, kwargs)
        if cache_item in self._known_mapped_variables_:
            return self._known_mapped_variables_[cache_item]
        else:
            instance = type_(**cache_item.all_kwargs)
            self._known_mapped_variables_[cache_item] = instance
            return instance

    def _get_mapped_variable_key_(self, type_: Type[MappedVariable], *args, **kwargs):
        """
        Generates a hashable key for the given type and arguments.

        :param type_: The type of the mapped variable to generate a key for, e.g., Attribute, Index, etc.
        :param args: Positional arguments to pass to the MappedVariable constructor.
        :param kwargs: Keyword arguments to pass to the MappedVariable constructor.
        :return: The generated hashable key.
        """
        args = (self,) + args
        all_kwargs = merge_args_and_kwargs(type_, args, kwargs, ignore_first=True)
        return convert_args_and_kwargs_into_hashable_key(all_kwargs)

    def __getattr__(self, name: str) -> CanBehaveLikeAVariable[T]:
        # Prevent debugger/private attribute lookups from being interpreted as symbolic attributes
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute {name}"
            )
        return self._get_mapped_variable_(Attribute, name)

    def __getitem__(self, key) -> CanBehaveLikeAVariable[T]:
        return self._get_mapped_variable_(Index, key)

    def __call__(self, *args, **kwargs) -> CanBehaveLikeAVariable[T]:
        return self._get_mapped_variable_(Call, args, kwargs)

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
class MappedVariable(UnaryExpression, CanBehaveLikeAVariable[T], ABC):
    """
    A symbolic expression the maps the values of symbolic variables.
    """

    _child_: CanBehaveLikeAVariable[T]
    """
    The child expression to apply the mapping to.
    """

    def __post_init__(self):
        self._var_ = self
        super().__post_init__()
        self._update_type_()

    def _update_type_(self) -> None:
        """
        Update the `_type_` attribute.
        """
        # Default implementation is that the type is the child type.
        self._type_ = self._child_._type_ if self._type_ is None else self._type_

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        """
        Apply the mapping to the child's values.
        """

        yield from (
            self._build_operation_result_and_update_truth_value_(
                child_result.bindings | {self._binding_id_: mapped_value}, child_result
            )
            for child_result in self._child_._evaluate_(sources, parent=self)
            for mapped_value in self._apply_mapping_(child_result.value)
        )

    @abstractmethod
    def _apply_mapping_(self, value: Any) -> Iterable[Any]:
        """
        Apply the mapping to a value from the child variable.
        """
        pass


@dataclass(eq=False, repr=False)
class Attribute(MappedVariable):
    """
    A symbolic attribute that can be used to access attributes of symbolic variables.

    For instance, if Body.name is called, then the attribute name is "name" and `_owner_class_` is `Body`
    """

    _attribute_name_: str
    """
    The name of the attribute.
    """

    @cached_property
    def _owner_class_(self) -> Optional[Type]:
        """
        The class that owns this attribute.
        """
        return self._child_._type_

    def _update_type_(self) -> None:
        """
        Update the `_type_` attribute with the type of the values of this attribute.
        """
        self._type_ = get_field_type_endpoint(self._owner_class_, self._attribute_name_)

    def _apply_mapping_(self, value: Any) -> Iterable[Any]:
        yield getattr(value, self._attribute_name_)

    @property
    def _name_(self):
        return f"{self._child_._name_}.{self._attribute_name_}"


@dataclass(eq=False, repr=False)
class Index(MappedVariable):
    """
    A variable that was created through collection indexing by a certain key on its child variable.
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
class Call(MappedVariable):
    """
    A variable created through a function call operation on its child variable.
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
class FlatVariable(MappedVariable):
    """
    A variable that is created from its child through a flattening operation that
     transforms the values of the child from an iterable-of-iterables into a single iterable of items.
     Note: It only unwraps one level of nesting.

    Given a child expression that evaluates to an iterable (e.g., Views.bodies), this mapping yields
    one solution per inner element while preserving the original bindings (e.g., the View instance),
    similar to UNNEST in SQL.
    """

    def _apply_mapping_(self, value: Iterable[Any]) -> Iterable[Any]:
        yield from value

    @cached_property
    def _name_(self):
        return f"Flatten({self._child_._name_})"


@dataclass
class MappedVariableCacheItem:
    """
    A cache item for mapped variable creation. To prevent recreating same mapped variable multiple times, mapping
     instances are stored in a dictionary with a hashable key. This class is used to generate the key for the dictionary
      that stores the mapped variable instances.
    """

    type: Type[MappedVariable]
    """
    The mapping type to create, e.g., Attribute, Index, etc.
    """
    child: CanBehaveLikeAVariable
    """
    The child of the mapping (i.e. the original variable on which the mapping is applied).
    """
    args: Tuple[Any, ...] = field(default_factory=tuple)
    """
    Positional arguments to pass to the mapping constructor.
    """
    kwargs: Dict[str, Any] = field(default_factory=dict)
    """
    Keyword arguments to pass to the mapping constructor.
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
        return (self.type,) + convert_args_and_kwargs_into_hashable_key(self.all_kwargs)

    def __hash__(self):
        return hash(self.hashable_key)

    def __eq__(self, other):
        return (
            isinstance(other, MappedVariableCacheItem)
            and self.hashable_key == other.hashable_key
        )
