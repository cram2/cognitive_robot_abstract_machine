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

    func: Callable | None = None
    """Filter function to apply"""

    func_args: tuple | None = None
    """Positional arguments for filter function"""

    func_kwargs: dict | None = None
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
        func = self.descriptor.parameters.func

        if func:
            func_args = self.descriptor.parameters.func_args or []
            func_kwargs = self.descriptor.parameters.func_kwargs or {}

            temporary_annotations = self.get_cas().annotations
            temporary_annotations = [
                annotation
                for annotation in temporary_annotations
                if func(annotation, *func_args, **func_kwargs)
            ]
            self.get_cas().annotations = temporary_annotations

        return Status.SUCCESS
