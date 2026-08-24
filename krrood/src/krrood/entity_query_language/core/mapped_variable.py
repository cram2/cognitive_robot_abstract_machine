"""
This module provides mechanisms for mapping symbolic expressions to object domains.

It contains classes for attribute access, indexing, and function calls on symbolic
expressions.
"""

from __future__ import annotations

import operator
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import Self

from typing_extensions import (
    TYPE_CHECKING,
    Iterable,
    Any,
    Type,
    Optional,
    Tuple,
    Dict,
    List,
)

from krrood.class_diagrams.utils import get_type_hints_of_object
from krrood.entity_query_language.core.base_expressions import (
    UnaryExpression,
    Bindings,
    OperationResult,
    Selectable,
    SymbolicExpression,
    UnificationDict,
)
from krrood.entity_query_language.exceptions import (
    MultipleValuesAlongAccessPath,
    ReadOnlyMapping,
    SymbolicDunderAccessError,
)
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.math_operations import MathOperator
from krrood.entity_query_language.utils import (
    T,
    merge_args_and_kwargs,
    convert_args_and_kwargs_into_hashable_key,
)
from krrood.symbol_graph.helpers import get_field_type_endpoint

if TYPE_CHECKING:
    from krrood.entity_query_language.operators.arithmetic import (
        ArithmeticOperation,
    )


@dataclass(eq=False, repr=False)
class CanBehaveLikeAVariable(Selectable[T], ABC):
    """
    This class adds the monitoring/tracking behavior on variables that tracks attribute
    access, calling, and comparison operations.
    """

    _known_mapped_variables_: Dict[MappedVariableCacheItem, MappedVariable] = field(
        init=False, default_factory=dict
    )
    """
    A storage of created MappedVariable instances to prevent recreating same mapping
    multiple times.
    """

    __iter__ = None
    """
    Prevent iteration on this class.
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

        :param type_: The type of the mapped variable to generate a key for, e.g.,
            Attribute, Index, etc.
        :param args: Positional arguments to pass to the MappedVariable constructor.
        :param kwargs: Keyword arguments to pass to the MappedVariable constructor.
        :return: The generated hashable key.
        """
        args = (self,) + args
        all_kwargs = merge_args_and_kwargs(type_, args, kwargs, ignore_first=True)
        return convert_args_and_kwargs_into_hashable_key(all_kwargs)

    def __getattr__(self, name: str) -> Attribute[T]:
        # Dunder names are never symbolic attribute access. Mapping them would (a) let copy/pickle
        # and other machinery that probes optional dunder hooks recurse into endless variable
        # creation, and (b) blur language semantics. Access a dunder-named member symbolically via a
        # :func:`symbolic_function` instead. SymbolicDunderAccessError is an AttributeError so that
        # optional-hook probing still treats it as a missing attribute.
        if name.startswith("__") and name.endswith("__"):
            raise SymbolicDunderAccessError(name)
        return self._get_mapped_variable_(Attribute, name)

    def __dir__(self) -> List[str]:
        """
        Surface the wrapped value type's attributes for interactive completion.

        ``__getattr__`` already makes every non-dunder name a valid (symbolic)
        attribute, but completion engines list ``__dir__`` only — which would otherwise
        show just this expression's own members. We union those with the public
        attributes of the value type so e.g. ``case_variable.<tab>`` offers the case
        type's fields.

        ``_type_`` is read from ``__dict__`` directly (never ``getattr``, which routes
        through ``__getattr__`` and would return a :class:`MappedVariable` instead of
        ``None``). This does not affect attribute resolution in any way.
        """
        names = set(super().__dir__())
        type_ = self.__dict__.get("_type_")
        if isinstance(type_, type):
            names.update(
                name
                for name in dir(type_)
                if not (name.startswith("__") and name.endswith("__"))
            )
            for klass in type_.__mro__:
                names.update(getattr(klass, "__annotations__", {}).keys())
        return sorted(names)

    def __getitem__(self, key: Any) -> Index[T]:
        indexing = (
            IndexByExpression if isinstance(key, SymbolicExpression) else IndexByValue
        )
        return self._get_mapped_variable_(indexing, key)

    def __call__(self, *args, **kwargs) -> Call[T]:
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

    def _arithmetic_(
        self, other: Any, math_operator: MathOperator
    ) -> ArithmeticOperation:
        """
        Build a binary arithmetic operation with this variable as the left operand.

        :param other: The right operand.
        :param math_operator: The operator to apply.
        :return: The symbolic arithmetic operation.
        """
        from krrood.entity_query_language.operators.arithmetic import (
            ArithmeticOperation,
        )

        return ArithmeticOperation(self, other, math_operator)

    def _reflected_arithmetic_(
        self, other: Any, math_operator: MathOperator
    ) -> ArithmeticOperation:
        """
        Build a binary arithmetic operation with this variable as the right operand.

        This is used for the reflected dunders (``other <op> self``) so that the operand
        order is preserved for non-commutative operators such as subtraction and
        division.

        :param other: The left operand.
        :param math_operator: The operator to apply.
        :return: The symbolic arithmetic operation.
        """
        from krrood.entity_query_language.operators.arithmetic import (
            ArithmeticOperation,
        )

        return ArithmeticOperation(other, self, math_operator)

    def __add__(self, other) -> ArithmeticOperation:
        return self._arithmetic_(other, MathOperator.ADD)

    def __radd__(self, other) -> ArithmeticOperation:
        return self._reflected_arithmetic_(other, MathOperator.ADD)

    def __sub__(self, other) -> ArithmeticOperation:
        return self._arithmetic_(other, MathOperator.SUBTRACT)

    def __rsub__(self, other) -> ArithmeticOperation:
        return self._reflected_arithmetic_(other, MathOperator.SUBTRACT)

    def __mul__(self, other) -> ArithmeticOperation:
        return self._arithmetic_(other, MathOperator.MULTIPLY)

    def __rmul__(self, other) -> ArithmeticOperation:
        return self._reflected_arithmetic_(other, MathOperator.MULTIPLY)

    def __truediv__(self, other) -> ArithmeticOperation:
        return self._arithmetic_(other, MathOperator.DIVIDE)

    def __rtruediv__(self, other) -> ArithmeticOperation:
        return self._reflected_arithmetic_(other, MathOperator.DIVIDE)

    def __floordiv__(self, other) -> ArithmeticOperation:
        return self._arithmetic_(other, MathOperator.FLOOR_DIVIDE)

    def __rfloordiv__(self, other) -> ArithmeticOperation:
        return self._reflected_arithmetic_(other, MathOperator.FLOOR_DIVIDE)

    def __mod__(self, other) -> ArithmeticOperation:
        return self._arithmetic_(other, MathOperator.MODULO)

    def __rmod__(self, other) -> ArithmeticOperation:
        return self._reflected_arithmetic_(other, MathOperator.MODULO)

    def __pow__(self, other) -> ArithmeticOperation:
        return self._arithmetic_(other, MathOperator.POWER)

    def __rpow__(self, other) -> ArithmeticOperation:
        return self._reflected_arithmetic_(other, MathOperator.POWER)

    def __neg__(self) -> ArithmeticOperation:
        from krrood.entity_query_language.operators.arithmetic import (
            UnaryArithmeticOperation,
        )

        return UnaryArithmeticOperation(self, MathOperator.NEGATE)

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

    _rerooted_chains_: Dict[uuid.UUID, MappedVariable] = field(
        init=False, default_factory=dict
    )
    """
    This chain rebuilt on other roots, keyed by the identifier of each new root.

    Roots are identified rather than held, because comparing two symbolic expressions
    builds a comparison instead of answering whether they are the same.
    """

    def __post_init__(self):
        self._var_ = self
        super().__post_init__()
        self._update_type_()

    def _node_for_new_position_(self) -> MappedVariable:
        """:return: This variable itself — mapped variables are shared-identity singletons, not cloned."""
        return self

    def _update_type_(self) -> None:
        """
        Update the `_type_` attribute.
        """
        # Default implementation is that the type is the child type.
        self._type_ = self._child_._type_ if self._type_ is None else self._type_

    def _evaluate__(
        self,
        sources: OperationResult,
    ) -> Iterable[OperationResult]:
        """
        Apply the mapping to the child's values.
        """
        yield from (
            self._build_operation_result_(
                child_result.bindings | {self._id_: mapped_value}, child_result
            )
            for child_result in self._child_._evaluate_(sources)
            for mapped_value in self._apply_mapping_(
                child_result.value, sources=sources
            )
        )

    @abstractmethod
    def _apply_mapping_(
        self, value: Any, sources: Optional[OperationResult] = None
    ) -> Iterable[T]:
        """
        Apply the mapping to a value from the child variable.

        :param value: The value to map.
        :param sources: The bindings from the evaluation context.
        """
        pass

    @abstractmethod
    def _rebuild_on_(self, child: CanBehaveLikeAVariable) -> MappedVariable:
        """
        :param child: The expression to apply this mapping to.
        :return: This mapping applied to ``child``.
        """

    def _reroot_on_(
        self, root: CanBehaveLikeAVariable, replaced: SymbolicExpression
    ) -> CanBehaveLikeAVariable:
        """
        Rebuild this mapping chain on top of another expression.

        Each mapping is rebuilt once per root, so chains that share a mapping still
        share it afterwards and keep ranging over the same values.

        .. note::
            The root alone identifies a rebuild: ``replaced`` is either this chain's
            own base or a step within it, both fixed once the chain exists.

        :param root: The expression to place at the base of the rebuilt chain.
        :param replaced: The expression ``root`` takes the place of, which is this
            chain's own base unless a step within the chain names the new root.
        :return: The mappings applied after ``replaced``, in the same order, on top of
            ``root``.
        """
        if self._id_ == replaced._id_:
            return root
        if root._id_ in self._rerooted_chains_:
            return self._rerooted_chains_[root._id_]
        if isinstance(self._child_, MappedVariable):
            rebuilt_child = self._child_._reroot_on_(root, replaced)
        elif self._child_._id_ == replaced._id_:
            rebuilt_child = root
        else:
            rebuilt_child = self._child_
        self._rerooted_chains_[root._id_] = self._rebuild_on_(rebuilt_child)
        return self._rerooted_chains_[root._id_]

    @property
    def _access_path_(self) -> List[Self]:
        """
        :return: The access path of the variable as a list of operations.
        """
        current = self
        result = [current]
        while isinstance(current, MappedVariable):
            current = current._child_
            result.append(current)
        return result[:-1][::-1]

    @property
    def _chain_root_(self) -> CanBehaveLikeAVariable:
        """
        :return: The first non-:class:`MappedVariable` expression at the base of this
            mapping chain (e.g. the ``robot`` variable behind ``robot.arm.joint``).
        """
        current = self
        while isinstance(current, MappedVariable):
            current = current._child_
        return current

    def _set_external_root_instance_value_(self, instance: Any, value: Any):
        """
        Set the field of the instance at this access path to the given value.

        This modifies instance in-place.

        .. warning::
            This method is only for use cases where symbolic access is needed outside of EQLs query evaluation.


        :param instance: The instance to be updated.
        :param value: The value to set.
        """
        current = instance
        for domain_mapping in self._access_path_[:-1]:
            if not isinstance(domain_mapping, SingleValueMapping):
                raise MultipleValuesAlongAccessPath(self, domain_mapping)
            current = next(domain_mapping._apply_mapping_(current))

        self._set_child_instance_value_(current, value)

    def _set_child_instance_value_(self, instance: Any, value: Any):
        """
        Set the field of the instance using this operation to the given value.

        This modifies instance in-place.

        Two mappings name where their value is kept and so override this: an
        :class:`Attribute` names a field, and an :class:`IndexByValue` names the key an
        element is stored under. The other three do not, and a value cannot be written
        through any of them: a :class:`Call` computes its value, an
        :class:`IndexByExpression` names which elements to reach rather than where one
        of them lives, and a :class:`FlatVariable` reaches every element without naming
        any.

        Naming such a place is a separate question from how many values a mapping
        reaches. A call reaches exactly one value and still cannot be written through,
        because it computes that value rather than reading it from somewhere.

        :param instance: The instance to be updated.
        :param value: The value to set.
        :raises ReadOnlyMapping: If this mapping does not name where its value is kept.
        """
        raise ReadOnlyMapping(self)

    def apply_mapping_on_external_root(self, instance: Any) -> T:
        """
        Follow this chain from a value outside query evaluation, applying each mapping
        along the access path in turn.

        .. warning::
            This is only for use cases where symbolic access is needed outside of EQL's
            query evaluation.

        :param instance: The value to follow the chain from.
        :return: The value the chain leads to.
        :raises MultipleValuesAlongAccessPath: If a step reaches more than one value,
            leaving the rest of the chain without one value to follow.
        """
        current = instance
        for domain_mapping in self._access_path_:
            if not isinstance(domain_mapping, SingleValueMapping):
                raise MultipleValuesAlongAccessPath(self, domain_mapping)
            current = next(domain_mapping._apply_mapping_(current))
        return current

    def get_clean_name_from_mapped_variable(self) -> str:
        """
        Get a clean name from a mapped variable by joining its attribute names.

        :return: The clean name.
        """
        names = []
        for step in self._access_path_:
            if isinstance(step, Attribute):
                names.append(step._attribute_name_)
        return ".".join(names)


@dataclass(eq=False, repr=False)
class SingleValueMapping(MappedVariable[T], ABC):
    """
    A mapping that reaches a single value, fully determined by the expression it is
    applied to and its own arguments.

    Two occurrences of one therefore denote the same value and share the node
    :meth:`CanBehaveLikeAVariable._get_mapped_variable_` gives them.

    ..note::
        The guarantee is *at most* one value. An attribute reaches no value when the
        instance does not have that attribute, and an index by a value reaches no value
        when nothing is stored under that key.
    """


@dataclass(eq=False, repr=False)
class Attribute(SingleValueMapping[T]):
    """
    A symbolic attribute that can be used to access attributes of symbolic variables.

    For instance, if Body.name is called, then the attribute name is "name" and
    `_owner_class_` is `Body`
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

    def _apply_mapping_(
        self, value: Any, sources: Optional[OperationResult] = None
    ) -> Iterable[T]:
        if hasattr(value, self._attribute_name_):
            yield getattr(value, self._attribute_name_)

    def _rebuild_on_(self, child: CanBehaveLikeAVariable) -> MappedVariable:
        return child._get_mapped_variable_(
            Attribute, _attribute_name_=self._attribute_name_
        )

    @property
    def _name_(self):
        return f"{self._child_._name_}.{self._attribute_name_}"

    def _set_child_instance_value_(self, obj: Any, value: Any):
        setattr(obj, self._attribute_name_, value)


@dataclass(eq=False, repr=False)
class Index(MappedVariable[T], ABC):
    """
    A variable created by indexing its child by a key.

    What the key is decides how many values the indexing reaches, so the two kinds are
    separate: see :class:`IndexByValue` and :class:`IndexByExpression`.
    """

    _key_: Any
    """
    The key to index with.
    """

    @property
    def _name_(self):
        return f"{self._child_._var_._name_}[{repr(self._key_)}]"


@dataclass(eq=False, repr=False)
class IndexByValue(Index[T], SingleValueMapping[T]):
    """
    Indexing by a key that is a plain value, which reaches the one element stored under
    it.
    """

    def _apply_mapping_(
        self, value: Any, sources: Optional[OperationResult] = None
    ) -> Iterable[T]:
        try:
            yield value[self._key_]
        except IndexError:  # break iterator if the key does not exist
            return

    def _rebuild_on_(self, child: CanBehaveLikeAVariable) -> MappedVariable:
        return child._get_mapped_variable_(IndexByValue, _key_=self._key_)

    def _set_child_instance_value_(self, instance: Any, value: Any):
        instance[self._key_] = value


@dataclass(eq=False, repr=False)
class IndexByExpression(Index[T]):
    """
    Indexing by a key that is itself an expression, which reaches one element per value
    that expression takes.

    A row of a query is the exception: it is keyed by the expressions it binds, so
    indexing it by one of them reaches the single value bound to it.
    """

    def _apply_mapping_(
        self, value: Any, sources: Optional[OperationResult] = None
    ) -> Iterable[T]:
        try:
            if isinstance(value, UnificationDict) and self._key_ in value:
                yield value[self._key_]
                return
            for key in self._key_._evaluate_(sources):
                yield value[key.value]
        except IndexError:  # break iterator if the key does not exist
            return

    def _rebuild_on_(self, child: CanBehaveLikeAVariable) -> MappedVariable:
        return child._get_mapped_variable_(IndexByExpression, _key_=self._key_)


@dataclass(eq=False, repr=False)
class Call(SingleValueMapping[T]):
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

    def _apply_mapping_(
        self, value: Any, sources: Optional[OperationResult] = None
    ) -> Iterable[T]:
        if len(self._args_) > 0 or len(self._kwargs_) > 0:
            yield value(*self._args_, **self._kwargs_)
        else:
            yield value()

    def _rebuild_on_(self, child: CanBehaveLikeAVariable) -> MappedVariable:
        return child._get_mapped_variable_(
            Call, _args_=self._args_, _kwargs_=self._kwargs_
        )

    @property
    def _name_(self):
        return f"{self._child_._var_._name_}()"

    def _update_type_(self) -> None:
        if self._child_._type_ is None:
            return
        self._type_ = get_type_hints_of_object(self._child_._type_)["return"]


@dataclass(eq=False, repr=False)
class FlatVariable(MappedVariable[T]):
    """
    A variable that is created from its child through a flattening operation that
     transforms the values of the child from an iterable-of-iterables into a single iterable of items.
     Note: It only unwraps one level of nesting.

    Given a child expression that evaluates to an iterable (e.g., Views.bodies), this mapping yields
    one solution per inner element while preserving the original bindings (e.g., the View instance),
    similar to UNNEST in SQL.

    Unlike a :class:`SingleValueMapping`, it takes no argument naming which element it means, so
    the iteration belongs to the node itself: each flattening written is a variable of
    its own, and two of them range over the elements independently.
    """

    def _apply_mapping_(
        self, value: Iterable[T], sources: Optional[OperationResult] = None
    ) -> Iterable[T]:
        yield from value

    def _rebuild_on_(self, child: CanBehaveLikeAVariable) -> MappedVariable:
        # Not routed through the mapped-variable cache: a flattening is an iteration
        # variable, so each one written ranges over the elements on its own.
        return FlatVariable(child)

    @cached_property
    def _name_(self) -> str:
        return f"Flatten({self._child_._name_})"


@dataclass
class MappedVariableCacheItem:
    """
    A cache item for mapped variable creation.

    To prevent recreating same mapped variable multiple times, mapping instances are
    stored in a dictionary with a hashable key. This class is used to generate the key
    for the dictionary  that stores the mapped variable instances.
    """

    type: Type[MappedVariable]
    """
    The mapping type to create, e.g., Attribute, Index, etc.
    """

    child: CanBehaveLikeAVariable
    """
    The child of the mapping (i.e. the original variable on which the mapping is
    applied).
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
