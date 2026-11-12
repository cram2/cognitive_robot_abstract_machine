"""
The parameters a call passes its arguments to.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

from typing_extensions import Any, Callable

from krrood.class_diagrams.call_arguments import CallArguments
from krrood.class_diagrams.exceptions import ArgumentsPassableOnlyByPosition
from krrood.ormatic.utils import classproperty
from krrood.patterns.caching import memoize
from krrood.utils import is_hashable

# %% call signature


@dataclass(kw_only=True, frozen=True)
class CallSignature:
    """
    The parameters a call to one callee passes its arguments to, as its caller writes
    them: a class is called without ``self``, a method read from its class with it.

    >>> CallSignature.from_callable(callee=divmod).accepts_keyword(keyword="x")
    False
    """

    parameters: inspect.Signature
    """
    The parameters of the callee, in the order a call fills them.
    """

    @classmethod
    def from_callable(cls, *, callee: Callable[..., Any]) -> CallSignature:
        """
        :param callee: The function, method or class that is called.
        :return: The signature of *callee*, read once for each callee that can be
            hashed.
        """
        if not is_hashable(callee):
            return cls(parameters=inspect.signature(callee))
        return cls._from_hashable_callable_(callee=callee)

    @classmethod
    @memoize
    def _from_hashable_callable_(cls, *, callee: Callable[..., Any]) -> CallSignature:
        """
        :param callee: A callee that can be hashed.
        :return: The signature of *callee*.
        """
        return cls(parameters=inspect.signature(callee))

    @classproperty
    def kinds_passed_by_name(cls) -> frozenset[inspect._ParameterKind]:
        """
        :return: The kinds of parameter that take one argument, by its name.
        """
        return frozenset(
            {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        )

    def bind(self, *, arguments: CallArguments) -> CallArguments:
        """
        :param arguments: The arguments a caller passes to the callee.
        :return: The arguments the callee receives, the same however they were passed
            and including the defaults that were left out.
        :raises TypeError: If the arguments do not fit the parameters.
        """
        bound_arguments = self.parameters.bind(
            *arguments.positional, **arguments.keyword
        )
        bound_arguments.apply_defaults()
        return CallArguments(
            positional=bound_arguments.args, keyword=bound_arguments.kwargs
        )

    def bind_by_name(self, *, arguments: CallArguments) -> dict[str, Any]:
        """
        :param arguments: The arguments a caller passes to the callee.
        :return: Each argument that was passed, by the name it is passed by, so calling
            the callee with these alone makes the same call. Defaults that were left out
            are not included.
        :raises ArgumentsPassableOnlyByPosition: If an argument goes to a parameter that
            takes it only by position.
        :raises TypeError: If the arguments do not fit the parameters.
        """
        passed = self.parameters.bind(
            *arguments.positional, **arguments.keyword
        ).arguments
        kinds = {name: self.parameters.parameters[name].kind for name in passed.keys()}
        passed_only_by_position = tuple(
            name
            for name, kind in kinds.items()
            if kind not in self.kinds_passed_by_name
            and kind is not inspect.Parameter.VAR_KEYWORD
        )
        if passed_only_by_position:
            raise ArgumentsPassableOnlyByPosition(
                parameters=self.parameters, parameter_names=passed_only_by_position
            )
        keyword_arguments = {}
        for name, value in passed.items():
            if kinds[name] is inspect.Parameter.VAR_KEYWORD:
                keyword_arguments.update(value)
            else:
                keyword_arguments[name] = value
        return keyword_arguments

    def accepts_keyword(self, *, keyword: str) -> bool:
        """
        :param keyword: The name of a keyword argument.
        :return: Whether the callee takes *keyword*, as a parameter of that name or
            through arbitrary keywords (a ``**kwargs`` parameter).
        """
        return any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            or (
                parameter.name == keyword
                and parameter.kind in self.kinds_passed_by_name
            )
            for parameter in self.parameters.parameters.values()
        )
