"""
A generic class parametrized with a class from another top-level package, so a
generated interface names that package only in the specialization's type arguments.
"""

from __future__ import annotations

from dataclasses import dataclass

from random_events.interval import SimpleInterval
from typing_extensions import Generic, TypeVar

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric

# %% the generic class

IntervalT = TypeVar("IntervalT", bound=SimpleInterval)
"""
The kind of interval a :class:`LabelledIntervals` is about.
"""


@dataclass(kw_only=True)
class LabelledIntervals(Generic[IntervalT], SubClassSafeGeneric):
    """
    A labelled collection over one kind of interval, without a field of that kind.
    """

    label: str
    """
    The name of the collection.
    """


# %% a field naming the specialization


@dataclass(kw_only=True)
class HolderOfSpecializedGeneric:
    """
    Holds a collection specialized with the interval from the other package.
    """

    intervals: LabelledIntervals[SimpleInterval]
    """
    The collection this holder keeps.
    """
