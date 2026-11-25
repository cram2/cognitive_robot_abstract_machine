"""
Annotators that inherit their configuration entirely from generic bindings.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field

from robokudo.annotators.blur import BlurDescriptor
from robokudo.annotators.cas_check import (
    CASCheckAnnotationTypeExists,
    CASCheckObjectHypothesisExists,
)
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.core import BaseAnnotator, ThreadedAnnotator
from robokudo.annotators.plane import PlaneDescriptor
from robokudo.types.annotation import Classification


# %% Generic configuration inheritance
@dataclass(kw_only=True, eq=False)
class DescriptorBindingAnnotator(BaseAnnotator[BlurDescriptor]):
    """
    A concrete binding without a descriptor field or discovery override.
    """


@dataclass(kw_only=True, eq=False)
class InheritedDescriptorBindingAnnotator(DescriptorBindingAnnotator):
    """
    An additional inheritance level preserving its parent's configuration.
    """


@dataclass(kw_only=True, eq=False)
class ThreadedDescriptorBindingAnnotator(ThreadedAnnotator[PlaneDescriptor]):
    """
    A threaded binding relying on inherited configuration initialization.
    """


# %% Qualified annotator names
@dataclass(kw_only=True)
class AnnotatorNamingScope:
    """
    Provide a scoped annotator class for qualified-name checks.
    """

    @dataclass(kw_only=True, eq=False)
    class Annotator(DescriptorBindingAnnotator):
        """
        An annotator whose qualified name includes its enclosing scope.
        """


# %% Inherited checker initialization
@dataclass(kw_only=True, eq=False)
class InheritedAnnotationChecker(CASCheckAnnotationTypeExists):
    """
    A checker inheriting its parent's annotation-type initialization.
    """


@dataclass(kw_only=True, eq=False)
class InheritedObjectHypothesisChecker(CASCheckObjectHypothesisExists):
    """
    An object checker inheriting its parent's specialized initialization.
    """


# %% Declarative initialization
@dataclass(kw_only=True, eq=False)
class DeclarativeAnnotationChecker(CASCheckAnnotationTypeExists):
    """A condition configured entirely by a fixed dataclass field."""

    annotation_type: type = field(default=Classification, init=False)
    """Annotation category satisfying the declarative condition."""


@dataclass(kw_only=True, eq=False)
class DeclarativeCollectionReader(CollectionReaderAnnotator):
    """A camera reader relying entirely on inherited post-initialization."""
