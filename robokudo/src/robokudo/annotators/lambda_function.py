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

    func: Callable | None = None
    """Configured callable, or None to perform no operation."""

    func_args: tuple | None = None
    """Positional arguments passed to the configured callable."""

    func_kwargs: dict | None = None
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


class LambdaFunctionAnnotator(BaseAnnotator[LambdaFunctionDescriptor]):
    """
    Annotator for executing arbitrary functions.

    This annotator executes a provided function with configurable arguments, allowing
    for dynamic behavior definition without creating new annotator classes.
    """

    @classproperty
    def Descriptor(cls) -> type[LambdaFunctionDescriptor]:
        """Specialize :meth:`robokudo.annotators.core.BaseAnnotator.Descriptor`."""
        return LambdaFunctionDescriptor

    def __init__(
        self,
        name: str = "LambdaFunctionAnnotator",
        descriptor: LambdaFunctionDescriptor | None = None,
    ):
        """
        Initialize the lambda function annotator.

        :param name: Annotator name
        :param descriptor: Configuration descriptor
        """
        super().__init__(name, descriptor)
        self.logger.debug("%s.__init__()" % self.__class__.__name__)

    def update(self) -> Status:
        """
        Execute the configured function.

        The function is called with the annotator instance as first argument, followed
        by any configured positional and keyword arguments.

        :return: SUCCESS status
        """
        func = self.descriptor.parameters.func

        if func:
            func_args = self.descriptor.parameters.func_args or []
            func_kwargs = self.descriptor.parameters.func_kwargs or {}

            func(self, *func_args, **func_kwargs)

        return Status.SUCCESS
