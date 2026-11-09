"""
Ground terms - a function applied to concrete arguments - the expressions that read
them, and how an evaluation reads their values.
"""

from __future__ import annotations

import inspect
import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, is_dataclass
from functools import cached_property
from types import MemberDescriptorType, WrapperDescriptorType

from typing_extensions import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Hashable,
    Mapping,
    Sequence,
    TypeVar,
)

from krrood.class_diagrams.call_arguments import CallArguments
from krrood.class_diagrams.call_signature import CallSignature
from krrood.entity_query_language.core.bound_value import HasBoundValue
from krrood.patterns.interface import Interface
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.utils import ensure_hashable

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import SymbolicExpression

# %% ground terms


@dataclass(kw_only=True, frozen=True)
class TermValue:
    """
    The value a ground term has, where it has one.
    """

    value: Any
    """
    The value, which may itself be ``None``.
    """


@dataclass(kw_only=True, frozen=True, eq=False)
class ObjectIdentity:
    """
    The identity of an object, which no other object shares, however the two compare.
    """

    identified_object: Any
    """
    The object.
    """

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ObjectIdentity):
            return NotImplemented
        return self.identified_object is other.identified_object

    def __hash__(self) -> int:
        return id(self.identified_object)


@dataclass(kw_only=True, frozen=True)
class FunctionKey:
    """
    What identifies the function of a ground term, independently of its arguments.
    """

    term_type: type[GroundTerm]
    """
    The class of the ground term, which tells apart an attribute, an index and a call of
    the same function.
    """

    function: Hashable
    """
    What the ground term applies to its arguments.
    """


@dataclass(kw_only=True, frozen=True, eq=False)
class GroundTerm(ABC):
    """
    A function applied to concrete arguments.

    Two ground terms are equal when they apply the same function to the same arguments,
    however the arguments were passed and including defaults that were left out. The
    object a ground term belongs to - an attribute's instance, an index's container, a
    method's receiver - is compared by identity. Every other argument is compared with
    ``==`` where it is hashable, and by identity where it is not.

    >>> def area(width: float, height: float = 1.0) -> float:
    ...     return width * height
    >>> CallTerm(callee=area, arguments=CallArguments(positional=(2.0,))) == CallTerm(
    ...     callee=area, arguments=CallArguments(keyword={"width": 2.0, "height": 1.0})
    ... )
    True
    >>> CallTerm(callee=area, arguments=CallArguments(positional=(2.0,))).live_value()
    2.0
    """

    @property
    @abstractmethod
    def function(self) -> Hashable:
        """
        :return: What this ground term applies to its arguments, the same for every
            argument it is applied to.
        """

    @property
    @abstractmethod
    def normalized_arguments(self) -> tuple[Hashable, ...]:
        """
        :return: The arguments of the application, the same for every way of writing
            it.
        """

    @cached_property
    def function_key(self) -> FunctionKey:
        """
        :return: What identifies the function of this ground term, independently of its
            arguments.
        """
        return FunctionKey(term_type=type(self), function=self.function)

    @cached_property
    def _key_(self) -> tuple[Hashable, ...]:
        """
        :return: What identifies this ground term, independently of how its arguments
            were passed.
        """
        return self.function_key, self.normalized_arguments

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GroundTerm) and self._key_ == other._key_

    def __hash__(self) -> int:
        return hash(self._key_)

    def has_live_value(self) -> bool:
        """
        :return: Whether this ground term currently has a value.
        """
        return True

    def setting_runs_code(self) -> bool:
        """
        :return: Whether setting this ground term on a live object runs code of the
            object's class, which may change more than this ground term.
        """
        return False

    @abstractmethod
    def live_value(self) -> Any:
        """
        :return: The value this ground term currently has, where
            :meth:`has_live_value` holds.
        """

    def live_term_value(self) -> TermValue | None:
        """
        :return: The value this ground term currently has, or ``None`` when it has none.
        """
        if not self.has_live_value():
            return None
        return TermValue(value=self.live_value())


@dataclass(kw_only=True, frozen=True, eq=False)
class AttributeTerm(GroundTerm):
    """
    One attribute of an instance.
    """

    instance: Any
    """
    The instance whose attribute this is.
    """

    attribute_name: str
    """
    The name of the attribute.
    """

    @property
    def function(self) -> str:
        return self.attribute_name

    @property
    def normalized_arguments(self) -> tuple[Hashable, ...]:
        return (ObjectIdentity(identified_object=self.instance),)

    def has_live_value(self) -> bool:
        return hasattr(self.instance, self.attribute_name)

    def setting_runs_code(self) -> bool:
        """Setting runs code where the class overrides
        :meth:`object.__setattr__`, other than to freeze a dataclass, or where
        the attribute is a descriptor with a setter that is neither a slot nor
        a read-only property."""
        owner = type(self.instance)
        is_frozen_dataclass = is_dataclass(owner) and owner.__dataclass_params__.frozen
        if owner.__setattr__ is not object.__setattr__ and not is_frozen_dataclass:
            return True
        descriptor = inspect.getattr_static(owner, self.attribute_name, None)
        if isinstance(descriptor, MemberDescriptorType):
            return False
        if isinstance(descriptor, property):
            return descriptor.fset is not None
        return hasattr(type(descriptor), "__set__")

    def live_value(self) -> Any:
        return getattr(self.instance, self.attribute_name)


@dataclass(kw_only=True, frozen=True, eq=False)
class IndexTerm(GroundTerm):
    """
    The element a container stores under a key.
    """

    container: Any
    """
    The object that stores the element, such as a sequence or a mapping: anything that
    supports reading an element by key, whether or not it supports assigning one.
    """

    key: Any
    """
    The key the element is stored under.
    """

    @property
    def function(self) -> Callable[[Any, Any], Any]:
        return operator.getitem

    @property
    def normalized_arguments(self) -> tuple[Hashable, ...]:
        return ObjectIdentity(identified_object=self.container), ensure_hashable(
            self.key
        )

    def setting_runs_code(self) -> bool:
        """
        Setting runs code where the container's class defines its own item assignment
        instead of inheriting a builtin one, and never where it has no item assignment.
        """
        item_assignment = inspect.getattr_static(
            type(self.container), "__setitem__", None
        )
        return item_assignment is not None and not isinstance(
            item_assignment, WrapperDescriptorType
        )

    def has_live_value(self) -> bool:
        """
        A container that is neither a mapping nor a sequence is assumed to store every
        key.
        """
        if isinstance(self.container, Mapping):
            return self.key in self.container
        if isinstance(self.container, Sequence) and isinstance(self.key, int):
            return -len(self.container) <= self.key < len(self.container)
        return True

    def live_value(self) -> Any:
        return self.container[self.key]


@dataclass(kw_only=True, frozen=True, eq=False)
class CallTerm(GroundTerm):
    """
    The result of calling a function, method or class.

    A bound method is its function called with the receiver first, and a function
    decorated with :func:`~krrood.entity_query_language.predicate.symbolic_function` is
    the function it wraps.
    """

    callee: Callable[..., Any]
    """
    What is called.
    """

    arguments: CallArguments = field(default_factory=CallArguments)
    """
    The arguments of the call, as they were passed.
    """

    @classmethod
    def from_call(
        cls, *, callee: Callable[..., Any], arguments: CallArguments
    ) -> CallTerm:
        """
        :param callee: A function, a bound method, a class, or a function decorated
            with :func:`~krrood.entity_query_language.predicate.symbolic_function`.
        :param arguments: The arguments of the call.
        :return: The result of calling *callee* with *arguments*, as queries reach it.
        """
        if inspect.ismethod(callee):
            arguments = CallArguments(
                positional=(callee.__self__, *arguments.positional),
                keyword=arguments.keyword,
            )
            callee = callee.__func__
        if inspect.isclass(callee) and issubclass(callee, HasBoundValue):
            return BoundValueTerm(callee=callee, arguments=arguments)
        return CallTerm(callee=inspect.unwrap(callee), arguments=arguments)

    @property
    def function(self) -> Hashable:
        return ensure_hashable(self.callee)

    @property
    def normalized_arguments(self) -> tuple[Hashable, ...]:
        """
        Passing an argument by position or by name, or leaving out one that has a
        default, gives the same arguments.
        """
        arguments = CallSignature.from_callable(callee=self.callee).bind(
            arguments=self.arguments
        )
        normalized = arguments.key_by(value_form=ensure_hashable)
        if not self._first_argument_is_receiver(arguments=arguments):
            return normalized
        return ObjectIdentity(identified_object=arguments.positional[0]), *normalized[
            1:
        ]

    def _first_argument_is_receiver(self, *, arguments: CallArguments) -> bool:
        """
        :param arguments: The arguments of the call, bound to the signature.
        :return: Whether :attr:`callee` is a method the class of the first argument
            defines, so the first argument is its receiver.
        """
        if not (inspect.isfunction(self.callee) and arguments.positional):
            return False
        definition = inspect.getattr_static(
            type(arguments.positional[0]), self.callee.__name__, None
        )
        return (
            not isinstance(definition, staticmethod)
            and inspect.unwrap(definition) is self.callee
        )

    def live_value(self) -> Any:
        return self.callee(*self.arguments.positional, **self.arguments.keyword)


@dataclass(kw_only=True, frozen=True, eq=False)
class BoundValueTerm(CallTerm):
    """
    The bound value of applying a
    :class:`~krrood.entity_query_language.core.bound_value.HasBoundValue` class, such as a
    predicate or a symbolic function class.
    """

    callee: type[HasBoundValue]
    """
    The class that is applied.
    """

    def live_value(self) -> Any:
        return self.callee._bound_value_(
            **CallSignature.from_callable(callee=self.callee).bind_by_name(
                arguments=self.arguments
            )
        )


# %% expressions that read ground terms

GroundTermType = TypeVar("GroundTermType", bound=GroundTerm)


class GroundTermExpression(Generic[GroundTermType], SubClassSafeGeneric):
    """
    An expression whose value, for each binding of its operands, is the value of one
    ground term of the bound type.
    """

    @property
    @abstractmethod
    def _ground_term_operands_(self) -> tuple[SymbolicExpression, ...]:
        """
        :return: The expressions whose values are the arguments of the ground term.
        """

    @abstractmethod
    def _ground_term_(self, *, operand_values: tuple[Any, ...]) -> GroundTermType:
        """
        :param operand_values: One value of each of :attr:`_ground_term_operands_`, in
            the same order.
        :return: The ground term this expression reads for *operand_values*.
        """


# %% reading ground terms


class TermReader(Interface):
    """
    Reads the value of a ground term during an evaluation.
    """

    @abstractmethod
    def value_of_term(self, *, term: GroundTerm) -> TermValue | None:
        """
        :param term: The ground term to read.
        :return: The value of *term*, or ``None`` when it has none.
        """


@dataclass(kw_only=True)
class LiveTermReader(TermReader):
    """
    Reads the value a ground term currently has on the live objects.
    """

    def value_of_term(self, *, term: GroundTerm) -> TermValue | None:
        return term.live_term_value()
