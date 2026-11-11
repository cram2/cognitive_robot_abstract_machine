"""
Lambda function annotator for RoboKudo.

This module provides an annotator for executing arbitrary functions.
It supports:

* Dynamic function execution
* Custom function arguments
* Flexible parameter passing
* Generic function handling

The module is used for:

* Custom processing
* Dynamic behavior
* Testing and debugging
* Quick prototyping
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field

from krrood.ormatic.utils import classproperty
from py_trees.common import Status
from typing_extensions import Callable

from robokudo.annotators.core import BaseAnnotator
from robokudo.datastructures.annotator_descriptor import AnnotatorDescriptor
from robokudo.datastructures.annotator_parameters import AnnotatorParameters

# %% LambdaFunction parameters


@dataclass(kw_only=True)
class LambdaFunctionParameters(AnnotatorParameters):
    """Parameter container for function configuration."""

    function: Callable | None = None
    """Configured callable, or None to perform no operation."""

    function_arguments: tuple | None = None
    """Positional arguments passed to the configured callable."""

    function_keyword_arguments: dict | None = None
    """Keyword arguments passed to the configured callable."""


# %% LambdaFunction descriptor


@dataclass(kw_only=True)
class LambdaFunctionDescriptor(AnnotatorDescriptor[LambdaFunctionParameters]):
    """Declarative configuration of :class:`LambdaFunctionAnnotator`."""

    parameters: LambdaFunctionParameters = field(
        default_factory=LambdaFunctionParameters.from_defaults
    )
    """Effective settings initialized from application and annotator defaults."""


# %% LambdaFunction annotator


@dataclass(eq=False)
class LambdaFunctionAnnotator(BaseAnnotator[LambdaFunctionDescriptor]):
    """
    Annotator for executing arbitrary functions.

    This annotator executes a provided function with configurable arguments, allowing
    for dynamic behavior definition without creating new annotator classes.
    """

    name: str = __qualname__
    """Name of this node in the behaviour tree."""

    descriptor: LambdaFunctionDescriptor = field(
        default_factory=LambdaFunctionDescriptor
    )
    """Configuration supplied to this annotator instance."""

    @classproperty
    def Descriptor(cls) -> type[LambdaFunctionDescriptor]:
        """Specialize :meth:`robokudo.annotators.core.BaseAnnotator.Descriptor`."""
        return LambdaFunctionDescriptor

    def update(self) -> Status:
        """
        Execute the configured function.

        The function is called with the annotator instance as first argument, followed
        by any configured positional and keyword arguments.

        :return: SUCCESS status
        """
        function = self.descriptor.parameters.function

        if function:
            function_arguments = self.descriptor.parameters.function_arguments or []
            function_keyword_arguments = (
                self.descriptor.parameters.function_keyword_arguments or {}
            )

            function(self, *function_arguments, **function_keyword_arguments)

        return Status.SUCCESS
