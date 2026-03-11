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

import py_trees

import robokudo.annotators.core
from typing_extensions import Optional, Tuple, Dict, Callable


class LambdaFunctionAnnotator(robokudo.annotators.core.BaseAnnotator):
    """
    Annotator for executing arbitrary functions.

    This annotator executes a provided function with configurable arguments,
    allowing for dynamic behavior definition without creating new annotator classes.
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """Configuration descriptor for lambda function annotator."""

        class Parameters:
            """
            Parameter container for function configuration.

            :ivar func: Function to execute
            :type func: callable
            :ivar func_args: Positional arguments for function
            :type func_args: tuple
            :ivar func_kwargs: Keyword arguments for function
            :type func_kwargs: dict
            """

            def __init__(self) -> None:
                self.func: Optional[Callable] = None
                self.func_args: Optional[Tuple] = None
                self.func_kwargs: Optional[Dict] = None

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "LambdaFunctionAnnotator",
        descriptor: "LambdaFunctionAnnotator.Descriptor" = Descriptor(),
    ):
        """
        Initialize the lambda function annotator. Minimal one-time init!

        :param name: Annotator name, defaults to "LambdaFunctionAnnotator"
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        """
        super().__init__(name, descriptor)
        self.logger.debug("%s.__init__()" % self.__class__.__name__)

    def update(self) -> py_trees.common.Status:
        """
        Execute the configured function.

        The function is called with the annotator instance as first argument,
        followed by any configured positional and keyword arguments.

        :return: SUCCESS status
        """
        func = self.descriptor.parameters.func

        if func:
            func_args = self.descriptor.parameters.func_args or []
            func_kwargs = self.descriptor.parameters.func_kwargs or {}

            func(self, *func_args, **func_kwargs)

        return py_trees.common.Status.SUCCESS
