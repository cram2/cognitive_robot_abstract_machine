from __future__ import annotations

from krrood.utils import T, ensure_hashable

"""
Cache utilities.

This module provides caching datastructures and utilities.
"""
from dataclasses import dataclass, field
from itertools import islice
from typing_extensions import Dict, Generic, Iterable, Iterator, List, Optional


@dataclass
class SeenSet:
    """
    Coverage index for previously seen partial assignments.

    This replaces the linear scan with a trie-based index using a fixed key order.
    An assignment A is considered covered if there exists a stored constraint C
    such that C.items() is a subset of A.items().
    """

    keys: tuple = field(default_factory=tuple, repr=False)
    all_seen: bool = field(default=False, init=False)
    constraints: list = field(default_factory=list, init=False, repr=False)
    exact: set = field(default_factory=set, init=False, repr=False)

    def add(self, assignment: Dict) -> None:
        """
        Add a constraint (partial assignment) to the coverage index.
        """
        if self.all_seen:
            return
        if not assignment:
            # Empty constraint means everything is covered
            self.all_seen = True
            return

        if self.keys:
            assignment = {k: v for k, v in assignment.items() if k in self.keys}
        else:
            assignment = dict(assignment)

        # Maintain exact-match set only when all keys are present
        if self.keys and all(k in assignment for k in self.keys):
            self.exact.add(tuple(ensure_hashable(assignment[k]) for k in self.keys))

        self.constraints.append(assignment)

    def check(self, assignment: Dict) -> bool:
        """
        Return True if any stored constraint is a subset of the given assignment.
        Mirrors previous semantics: encountering an empty assignment flips all_seen
        but returns False the first time to allow population.
        """
        if self.all_seen:
            return True
        if not assignment:
            # First observation of empty assignment should not be considered covered
            # but should mark the index so later checks short-circuit.
            self.all_seen = True
            return False

        # Fast exact-key path when all keys are present
        if self.keys and all(k in assignment for k in self.keys):
            return self.exact_contains(assignment)

        # Fallback to coverage check using constraints
        for constraint in self.constraints:
            if all(
                (k in assignment) and (assignment[k] == v)
                for k, v in constraint.items()
            ):
                return True
        return False

    def exact_contains(self, assignment: Dict) -> bool:
        """
        Return True if the assignment contains all cache keys and the exact key tuple
        exists in the cache. This is an O(1) membership test and does not consult
        the coverage trie.
        """
        t = tuple(ensure_hashable(assignment[k]) for k in self.keys)
        if t in self.exact:
            return True
        return False

    def clear(self):
        self.all_seen = False
        self.constraints.clear()
        self.exact.clear()


@dataclass
class InstanceFilteredDomain(Generic[T]):
    """
    A re-iterable view over a domain that yields only the elements that are instances of a type.

    Unlike :func:`filter`, which is a one-shot iterator that eagerly holds ``iter(domain)`` for its
    whole lifetime, this creates a fresh underlying iterator on every iteration and holds none
    between them. That keeps :class:`ReEnterableLazyIterable` able to release and recreate its
    source, so a domain whose iterator only holds a heavy object in its suspended frame (for example
    a location generator that deep-copied the world) is not pinned for the variable's whole life.
    """

    type_: type
    """
    The type each yielded element must be an instance of.
    """
    domain: Iterable[T]
    """
    The underlying domain to filter.
    """

    def __iter__(self) -> Iterator[T]:
        return (element for element in self.domain if isinstance(element, self.type_))


@dataclass
class ReEnterableLazyIterable(Generic[T]):
    """
    A wrapper for an iterable that allows multiple iterations over its elements,
    materializing values as they are iterated over.
    """

    iterable: Optional[Iterator[T]] = field(default=None)
    """
    The live source iterator currently feeding new values, or ``None`` when it has been released. It
    is recreated on demand from :attr:`_source` so releasing it never loses values still to come.
    """
    materialized_values: List[T] = field(default_factory=list)
    """
    The materialized values of the iterable.
    """
    _source: Optional[Iterable[T]] = field(default=None, init=False, repr=False)
    """
    The original iterable, kept so a released source iterator can be recreated for a later iteration.
    """
    _source_is_re_iterable: bool = field(default=False, init=False)
    """
    Whether iterating :attr:`_source` yields a fresh iterator each time; only then can the live source
    be released early and recreated without losing values.
    """
    _source_exhausted: bool = field(default=False, init=False)
    """
    Whether the source has been fully materialized; once true, iterations serve only from the buffer.
    """
    _active_iteration_count: int = field(default=0, init=False, repr=False)
    """
    How many iterations are currently in flight, so the source is only released when none remain.
    """

    def set_iterable(self, iterable):
        """
        Set the iterable and wrap it in a generator.

        This is needed because of the weakref data we get from SymbolGraph. If we do `self.iterable = iterable` and
        weakref instances die, the iterable would have None values for them. But if we wrap it in a generator,
        they are actually removed, and the generator doesn't find them, which is the wanted behavior.
        """
        self._source = iterable
        # A source is re-iterable when ``iter(source)`` returns a fresh iterator rather than the
        # source itself (as one-shot iterators like generators and ``filter`` do). The probe iterator
        # is discarded without being advanced.
        self._source_is_re_iterable = iter(iterable) is not iterable
        self.iterable = (v for v in iterable)
        self.materialized_values = []
        self._source_exhausted = False

    def __iter__(self):
        """
        Iterate over the values, materializing them as they are iterated over. This allows multiple iterations over
        the iterable simultaneously, and it also allows for efficient access to previously materialized values.

        When a re-iterable source has no iteration left in flight its live iterator is released, so a source that only
        holds a heavy object in its suspended frame (for example a location generator that deep-copied the world)
        does not keep it alive; a later iteration recreates the source and skips the already-materialized values.

        :return: An iterator over the values.
        """
        index = 0
        self._active_iteration_count += 1
        try:
            while True:
                if index < len(self.materialized_values):
                    yield self.materialized_values[index]
                    index += 1
                    continue
                if self._source_exhausted:
                    return
                self._ensure_source()
                try:
                    value = next(self.iterable)
                except StopIteration:
                    self._source_exhausted = True
                    self._release_source()
                    return
                self.materialized_values.append(value)
                yield value
                index += 1
        finally:
            self._active_iteration_count -= 1
            self._release_source_if_idle()

    def _release_source_if_idle(self) -> None:
        """
        Release the live source once nothing is iterating it, but only when it can be recreated
        without losing values (a re-iterable source that is not yet exhausted).
        """
        if (
            self._active_iteration_count == 0
            and self._source_is_re_iterable
            and not self._source_exhausted
        ):
            self._release_source()

    def _ensure_source(self) -> None:
        """
        Recreate the live source iterator if it was released, skipping the values already materialized.
        """
        if self.iterable is not None:
            return
        recreated_source = (v for v in self._source)
        skip_count = len(self.materialized_values)
        # itertools' "consume" recipe: an empty slice starting at skip_count still forces the
        # underlying iterator to be advanced that many steps, entirely at C speed.
        next(islice(recreated_source, skip_count, skip_count), None)
        self.iterable = recreated_source

    def _release_source(self) -> None:
        """
        Close and drop the live source iterator so its suspended frame (and anything it holds) is freed.
        """
        released_source = self.iterable
        self.iterable = None
        if released_source is not None:
            released_source.close()

    def __bool__(self):
        """
        Return True if the iterable has values, False otherwise.
        """
        return bool(self.materialized_values) or (
            not self._source_exhausted and self._source is not None
        )
