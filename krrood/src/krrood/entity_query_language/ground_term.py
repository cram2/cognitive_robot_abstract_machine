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
    Iterator,
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


@dataclass(frozen=True)
class TermValue:
    """
    The value a ground term has, where it has one.
    """

    value: Any
    """
    The value, which may itself be ``None``.
    """


@dataclass(frozen=True, eq=False)
class ObjectIdentity:
    """
    The identity of an object, which no other object shares, however the two compare.
    """

    identified_object: Any
    """
    The object.
    """

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ObjectIdentity)
            and self.identified_object is other.identified_object
        )

    def __hash__(self) -> int:
        return id(self.identified_object)


@dataclass(frozen=True)
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


@dataclass(frozen=True, eq=False)
class GroundTerm(ABC):
    """
    A function applied to concrete arguments.

    Two ground terms are equal when they apply the same function to the same arguments,
    however the arguments were passed and including defaults that were left out. The
    object a ground term belongs to - an attribute's instance, an index's container, a
    method's receiver - is compared by identity, other hashable arguments by value and
    all others by identity.

    >>> def area(width: float, height: float = 1.0) -> float:
    ...     return width * height
    >>> CallTerm(area, CallArguments((2.0,))) == CallTerm(
    ...     area, CallArguments((), {"width": 2.0, "height": 1.0})
    ... )
    True
    >>> CallTerm(area, CallArguments((2.0,))).live_value()
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
        return FunctionKey(type(self), self.function)

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
        return TermValue(self.live_value())


@dataclass(frozen=True, eq=False)
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
        return (ObjectIdentity(self.instance),)

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


@dataclass(frozen=True, eq=False)
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
        return ObjectIdentity(self.container), ensure_hashable(self.key)

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


@dataclass(frozen=True, eq=False)
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
        cls, callee: Callable[..., Any], arguments: CallArguments
    ) -> CallTerm:
        """
        :param callee: A function, a bound method, a class, or a function decorated
            with :func:`~krrood.entity_query_language.predicate.symbolic_function`.
        :param arguments: The arguments of the call.
        :return: The result of calling *callee* with *arguments*, as queries reach it.
        """
        if inspect.ismethod(callee):
            arguments = CallArguments(
                (callee.__self__, *arguments.positional), arguments.keyword
            )
            callee = callee.__func__
        if inspect.isclass(callee) and issubclass(callee, HasBoundValue):
            return BoundValueTerm(callee, arguments)
        return CallTerm(inspect.unwrap(callee), arguments)

    @property
    def function(self) -> Hashable:
        return ensure_hashable(self.callee)

    @property
    def normalized_arguments(self) -> tuple[Hashable, ...]:
        """
        Passing an argument by position or by name, or leaving out one that has a
        default, gives the same arguments.
        """
        arguments = CallSignature.from_callable(self.callee).bind(self.arguments)
        normalized = arguments.key_by(ensure_hashable)
        if not self._is_method_of_first_argument(arguments):
            return normalized
        return ObjectIdentity(arguments.positional[0]), *normalized[1:]

    def _is_method_of_first_argument(self, arguments: CallArguments) -> bool:
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

    def depends_only_on_arguments(self) -> bool:
        """
        :return: Whether another call with equal arguments gives the same result, which a
            call that constructs a new object does not.
        """
        return not inspect.isclass(self.callee)


@dataclass(frozen=True, eq=False)
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

    def depends_only_on_arguments(self) -> bool:
        """
        Unlike :meth:`CallTerm.depends_only_on_arguments`, holds for every class: the
        bound value is what the class computes from its arguments.
        """
        return True

    def live_value(self) -> Any:
        return self.callee._bound_value_(
            **CallSignature.from_callable(self.callee).bind_by_name(self.arguments)
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
    def _ground_term_(self, *operand_values: Any) -> GroundTermType:
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
    def value_of(self, term: GroundTerm) -> TermValue | None:
        """
        :param term: The ground term to read.
        :return: The value of *term*, or ``None`` when it has none.
        """


@dataclass
class LiveTermReader(TermReader):
    """
    Reads the value a ground term currently has on the live objects.
    """

    def value_of(self, term: GroundTerm) -> TermValue | None:
        return term.live_term_value()


@dataclass
class CallResults(TermReader):
    """
    Reads ground terms with another reader, and reads each call once: a later call of
    the same callee that passes equal arguments the same way gets the value the first
    one read.

    .. warning:: A live object that changes while the calls are being read is not seen
        by the calls already made.
    """

    term_reader: TermReader
    """
    Reads every ground term whose value is not reused.
    """

    values: dict[Hashable, dict[CallArguments, TermValue | None]] = field(
        default_factory=dict
    )
    """
    The value read for each callee, by the arguments as they were passed, or ``None``
    where the call had none. A call that passes its arguments another way, such as
    leaving out a default, is read again.
    """

    def value_of(self, term: GroundTerm) -> TermValue | None:
        """
        Reads *term* with :attr:`term_reader`, reusing the value read before for the
        same callee and equal arguments. Never reused are a call that does not
        :meth:`CallTerm.depends_only_on_arguments`, and a value that is an iterator,
        which whoever reads it first consumes.
        """
        if not isinstance(term, CallTerm) or not term.depends_only_on_arguments():
            return self.term_reader.value_of(term)
        values_by_arguments = self.values.setdefault(term.function, {})
        if term.arguments in values_by_arguments:
            return values_by_arguments[term.arguments]
        term_value = self.term_reader.value_of(term)
        if term_value is not None and isinstance(term_value.value, Iterator):
            return term_value
        values_by_arguments[term.arguments] = term_value
        return term_value
