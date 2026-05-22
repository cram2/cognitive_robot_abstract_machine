"""
Filter annotator for RoboKudo.

This module provides an annotator for filtering annotations based on custom conditions.
It supports:

* Dynamic filtering through callable functions
* Custom filter arguments
* In-place annotation list modification
* Flexible condition evaluation

The module is used for:

* Annotation filtering
* Data preprocessing
* Result refinement
* Conditional processing
"""

from py_trees.common import Status
from typing_extensions import Optional, Tuple, Dict, Callable

from robokudo.annotators.core import BaseAnnotator


class FilterAnnotator(BaseAnnotator):
    """
    Annotator for applying filter conditions to annotations.

    This annotator applies a provided filter function to the current set of
    annotations, modifying the annotation list in-place based on the filter results.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Configuration descriptor for filter annotator.

        :ivar parameters: Filter parameters
        :type parameters: FilterAnnotator.Descriptor.Parameters
        """

        class Parameters:
            """Parameter container for filter configuration."""

            def __init__(self) -> None:
                self.func: Optional[Callable] = None
                """Filter function to apply"""

                self.func_args: Optional[Tuple] = None
                """Positional arguments for filter function"""

                self.func_kwargs: Optional[Dict] = None
                """Keyword arguments for filter function"""

        # overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "FilterAnnotator",
        descriptor: "FilterAnnotator.Descriptor" = Descriptor(),
    ) -> None:
        """Initialize the filter annotator.

        :param name: Annotator name, defaults to "FilterAnnotator"
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        """
        super().__init__(name, descriptor)
        self.logger.debug("%s.__init__()" % self.__class__.__name__)

    def update(self) -> Status:
        """
        Apply the filter function to current annotations.

        The filter function is applied to each annotation with the configured
        arguments. Only annotations that pass the filter are kept.

        :return: SUCCESS status
        """
        func = self.descriptor.parameters.func

        if func:
            func_args = self.descriptor.parameters.func_args or []
            func_kwargs = self.descriptor.parameters.func_kwargs or {}

            annotations = self.get_cas().annotations
            annotations = [
                annotation
                for annotation in annotations
                if func(annotation, *func_args, **func_kwargs)
            ]
            self.get_cas().annotations = annotations

        return Status.SUCCESS
