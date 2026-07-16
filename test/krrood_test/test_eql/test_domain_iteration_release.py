"""
Tests that a variable's lazy domain does not permanently pin objects a domain generator
only holds in its suspended frame.

Some domains build a heavy object as a local inside their ``__iter__`` (for example a
coraplex location that deep-copies the world to test candidate poses) and yield values
that do not reference it. Grounding consumes only the first value, leaving the domain
generator suspended with that heavy local alive. Because a query keeps its variables
reachable, the suspended frame must be released so the heavy object is collected once
the domain is no longer being iterated.
"""

from __future__ import annotations

import gc
import weakref

from dataclasses import dataclass, field
from typing_extensions import Iterator, List

from krrood.entity_query_language.cache_data import ReEnterableLazyIterable
from krrood.entity_query_language.factories import an, entity, variable


@dataclass
class FrameLocalHeavyObject:
    """
    Stand-in for a resource a domain generator holds only as a frame local (e.g. a deep-
    copied world).
    """

    payload: bytes = field(default_factory=lambda: bytes(2048))


@dataclass
class FrameBuildingLazyDomain:
    """
    A domain whose ``__iter__`` builds a heavy object as a frame local and yields values
    that do not reference it, mirroring a location generator that deep-copies the world
    only to validate candidates.
    """

    value_count: int
    built_object_references: List[weakref.ReferenceType] = field(default_factory=list)
    """
    Weak references to every heavy object built, so tests can observe their collection.
    """

    def __iter__(self) -> Iterator[int]:
        heavy_object = FrameLocalHeavyObject()
        self.built_object_references.append(weakref.ref(heavy_object))
        for value in range(self.value_count):
            yield value


def test_retained_variable_releases_partially_consumed_domain_frame():
    """
    A retained variable whose domain was only partially consumed must not keep the
    domain generator's frame (and its heavy local) alive.
    """
    domain = FrameBuildingLazyDomain(value_count=5)
    query = an(entity(variable(int, domain)))

    first_value = query.first()
    assert first_value == 0

    del first_value
    gc.collect()

    assert domain.built_object_references[0]() is None


def test_re_enterable_lazy_iterable_replays_after_source_release():
    """
    Releasing the underlying source must not truncate results: re-iterating still yields
    the full domain.
    """
    lazy_iterable = ReEnterableLazyIterable()
    lazy_iterable.set_iterable([0, 1, 2, 3])

    partial = []
    for value in lazy_iterable:
        partial.append(value)
        if len(partial) == 2:
            break
    assert partial == [0, 1]

    gc.collect()

    assert list(lazy_iterable) == [0, 1, 2, 3]
    assert list(lazy_iterable) == [0, 1, 2, 3]
