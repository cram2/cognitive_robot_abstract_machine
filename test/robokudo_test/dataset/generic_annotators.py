"""Annotators that inherit their configuration entirely from generic bindings."""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass

from robokudo.annotators.blur import BlurDescriptor
from robokudo.annotators.core import BaseAnnotator, ThreadedAnnotator
from robokudo.annotators.plane import PlaneDescriptor


# %% Generic configuration inheritance
@dataclass(kw_only=True, eq=False)
class DescriptorBindingAnnotator(BaseAnnotator[BlurDescriptor]):
    """A concrete binding without a descriptor field or discovery override."""

    name: str = __qualname__
    """Behaviour-tree label independent of the descriptor type."""


@dataclass(kw_only=True, eq=False)
class InheritedDescriptorBindingAnnotator(DescriptorBindingAnnotator):
    """An additional inheritance level preserving its parent's configuration."""


@dataclass(kw_only=True, eq=False)
class ThreadedDescriptorBindingAnnotator(ThreadedAnnotator[PlaneDescriptor]):
    """A threaded binding relying on inherited configuration initialization."""

    name: str = __qualname__
    """Behaviour-tree label independent of the descriptor type."""
