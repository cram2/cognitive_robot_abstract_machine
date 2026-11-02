"""
The arguments of one call, compared as a call compares them.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from functools import cached_property

from typing_extensions import Any, Callable, Hashable, Mapping

from krrood.patterns.caching import memoize
from krrood.utils import ensure_hashable, is_hashable

# %% call arguments


@dataclass(frozen=True, eq=False)
class CallArguments:
    """
    The positional and keyword arguments of one call.

    Two call arguments are equal when they hold equal values in the same places. A
    hashable value is compared by value, a tuple, list, set or dictionary by its type
    and contents, and any other value by identity, so two objects that compare equal
    but cannot be hashed are different arguments.

    >>> CallArguments((1, [2, 3]), {"scale": 2.0}) == CallArguments((1, [2, 3]), {"scale": 2.0})
    True
    """

    positional: tuple[Any, ...] = ()
    """
    The arguments passed by position.
    """

    keyword: Mapping[str, Any] = field(default_factory=dict)
    """
    The arguments passed by name.
    """

    @classmethod
    def bound_to(
        cls,
        callee: Callable[..., Any],
        positional: tuple[Any, ...],
        keyword: Mapping[str, Any],
    ) -> CallArguments:
        """
        :param callee: The function or class the arguments are passed to.
        :param positional: The arguments passed by position.
        :param keyword: The arguments passed by name.
        :return: The arguments *callee* receives, the same however they were passed and
            including the defaults that were left out.
        """
        bound_arguments = cls.bind(callee, positional, keyword)
        bound_arguments.apply_defaults()
        return cls(bound_arguments.args, bound_arguments.kwargs)

    @classmethod
    def named_by_parameter(
        cls,
        callee: Callable[..., Any],
        positional: tuple[Any, ...],
        keyword: Mapping[str, Any],
        without_first: bool = False,
    ) -> dict[str, Any]:
        """
        :param callee: The function the arguments are passed to.
        :param positional: The arguments passed by position.
        :param keyword: The arguments passed by name.
        :param without_first: Whether the first parameter of *callee* is not passed,
            as for the ``self`` of a method that is not bound.
        :return: Each argument that was passed, by the name of the parameter that
            receives it. Defaults that were left out are not included.
        """
        return dict(cls.bind(callee, positional, keyword, without_first).arguments)

    @classmethod
    def bind(
        cls,
        callee: Callable[..., Any],
        positional: tuple[Any, ...],
        keyword: Mapping[str, Any],
        without_first: bool = False,
    ) -> inspect.BoundArguments:
        """
        :param callee: The function or class the arguments are passed to.
        :param positional: The arguments passed by position.
        :param keyword: The arguments passed by name.
        :param without_first: Whether the first parameter of *callee* is not passed.
        :return: The arguments bound to the parameters of *callee*.
        :raises TypeError: If the arguments do not fit the parameters of *callee*.
        """
        signature = (
            cls.signature_of(callee, without_first)
            if is_hashable(callee)
            else cls.read_signature(callee, without_first)
        )
        return signature.bind(*positional, **keyword)

    @classmethod
    @memoize
    def signature_of(
        cls, callee: Callable[..., Any], without_first: bool = False
    ) -> inspect.Signature:
        """
        :param callee: A function or class that can be hashed.
        :param without_first: Whether the first parameter is left out of the result.
        :return: The signature of *callee*, read once.
        """
        return cls.read_signature(callee, without_first)

    @staticmethod
    def read_signature(
        callee: Callable[..., Any], without_first: bool
    ) -> inspect.Signature:
        """
        :param callee: A function or class.
        :param without_first: Whether the first parameter is left out of the result.
        :return: The signature of *callee*.
        """
        signature = inspect.signature(callee)
        if not without_first:
            return signature
        return signature.replace(parameters=list(signature.parameters.values())[1:])

    @cached_property
    def hashable_key(self) -> tuple[Hashable, ...]:
        """
        :return: What identifies these arguments, each compared by
            :meth:`hashable_form`.
        """
        return self.key_by(self.hashable_form)

    def key_by(self, value_form: Callable[[Any], Hashable]) -> tuple[Hashable, ...]:
        """
        :param value_form: What one argument is compared by.
        :return: The positional arguments in order, followed by the keyword arguments
            sorted by name, each compared by *value_form*.
        """
        return tuple(value_form(argument) for argument in self.positional) + (
            tuple(
                (name, value_form(value))
                for name, value in sorted(self.keyword.items())
            ),
        )

    @classmethod
    def hashable_form(cls, value: Any) -> Hashable:
        """
        :param value: One argument.
        :return: What the argument is compared by: the value itself where it can be
            hashed, the type and contents of a tuple, list, set or dictionary, and the
            identity of any other value.
        """
        if is_hashable(value):
            return value
        if type(value) in (tuple, list):
            return type(value), tuple(cls.hashable_form(element) for element in value)
        if type(value) is set:
            return set, frozenset(value)
        if type(value) is dict:
            return dict, frozenset(
                (key, cls.hashable_form(element)) for key, element in value.items()
            )
        return ensure_hashable(value)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, CallArguments) and self.hashable_key == other.hashable_key
        )

    def __hash__(self) -> int:
        return hash(self.hashable_key)
