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

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field

from krrood.ormatic.utils import classproperty
from py_trees.common import Status
from typing_extensions import Callable

from robokudo.annotators.core import BaseAnnotator
from robokudo.datastructures.annotator_descriptor import AnnotatorDescriptor
from robokudo.datastructures.annotator_parameters import AnnotatorParameters

# %% Filter parameters


@dataclass(kw_only=True)
class FilterParameters(AnnotatorParameters):
    """Parameter container for filter configuration."""

    function: Callable | None = None
    """Filter function to apply"""

    function_arguments: tuple | None = None
    """Positional arguments for filter function"""

    function_keyword_arguments: dict | None = None
    """Keyword arguments for filter function"""


# %% Filter descriptor


@dataclass(kw_only=True)
class FilterDescriptor(AnnotatorDescriptor[FilterParameters]):
    """Declarative configuration of :class:`FilterAnnotator`."""

    parameters: FilterParameters = field(default_factory=FilterParameters.from_defaults)
    """Effective settings initialized from application and annotator defaults."""


# %% Filter annotator


class FilterAnnotator(BaseAnnotator[FilterDescriptor]):
    """
    Annotator for applying filter conditions to annotations.

    This annotator applies a provided filter function to the current set of
    annotations, modifying the annotation list in-place based on the filter results.
    """

    @classproperty
    def Descriptor(cls) -> type[FilterDescriptor]:
        """Specialize :meth:`robokudo.annotators.core.BaseAnnotator.Descriptor`."""
        return FilterDescriptor

    def __init__(
        self,
        name: str = "FilterAnnotator",
        descriptor: FilterDescriptor | None = None,
    ) -> None:
        """Initialize the filter annotator.

        :param name: Annotator name
        :param descriptor: Configuration descriptor
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
        function = self.descriptor.parameters.function

        if function:
            function_arguments = self.descriptor.parameters.function_arguments or []
            function_keyword_arguments = (
                self.descriptor.parameters.function_keyword_arguments or {}
            )

            temporary_annotations = self.get_cas().annotations
            temporary_annotations = [
                annotation
                for annotation in temporary_annotations
                if function(
                    annotation, *function_arguments, **function_keyword_arguments
                )
            ]
            self.get_cas().annotations = temporary_annotations

        return Status.SUCCESS
