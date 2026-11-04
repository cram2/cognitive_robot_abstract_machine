"""Things that stand for a callee and the arguments it is called with."""

from __future__ import annotations

from abc import ABC, abstractmethod

from typing_extensions import Any, Callable

from krrood.class_diagrams.call_arguments import CallArguments
from krrood.class_diagrams.call_signature import CallSignature

# %% call arguments of a callee


class HasCallArguments(ABC):
    """Stands for a callee and the arguments it is called with.

    The members are underscore-wrapped because hosts of this mixin may
    hand their public attribute namespace to another type.
    """

    @property
    @abstractmethod
    def _callee_(self) -> Callable[..., Any]:
        """
        :return: The function, method or class that is called.
        """

    @property
    @abstractmethod
    def _call_arguments_(self) -> CallArguments:
        """
        :return: The arguments as they were passed.
        """

    def _bound_call_arguments_(self) -> CallArguments:
        """
        :return: The arguments :attr:`_callee_` receives, the same however they were
            passed and including the defaults that were left out.
        :raises TypeError: If the arguments do not fit the parameters of the callee.
        """
        return CallSignature.of(self._callee_).bind(
            *self._call_arguments_.positional, **self._call_arguments_.keyword
        )
