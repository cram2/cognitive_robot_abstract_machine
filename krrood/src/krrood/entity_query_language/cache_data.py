from __future__ import annotations

from .utils import T

"""
Cache utilities.

This module provides caching datastructures and utilities.
"""
from dataclasses import dataclass, field
from typing_extensions import Dict, Generic, Iterable, List


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
            self.exact.add(tuple(assignment[k] for k in self.keys))

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
        t = tuple(assignment[k] for k in self.keys)
        if t in self.exact:
            return True
        return False

    def clear(self):
        self.all_seen = False
        self.constraints.clear()
        self.exact.clear()


@dataclass
class ReEnterableLazyIterable(Generic[T]):
    """
    A wrapper for an iterable that allows multiple iterations over its elements,
    materializing values as they are iterated over.
    """

    iterable: Iterable[T] = field(default_factory=list)
    """
    The iterable to wrap.
    """
    materialized_values: List[T] = field(default_factory=list)
    """
    The materialized values of the iterable.
    """

    def set_iterable(self, iterable):
        """
        Set the iterable and wrap it in a generator.

        This is needed because of the weakref data we get from SymbolGraph. If we do `self.iterable = iterable` and
        weakref instances die, the iterable would have None values for them. But if we wrap it in a generator,
        they are actually removed, and the generator doesn't find them, which is the wanted behavior.
        """
        self.iterable = (v for v in iterable)

    def __iter__(self):
        """
        Iterate over the values, materializing them as they are iterated over.

        :return: An iterator over the values.
        """
        yield from self.materialized_values
        for v in self.iterable:
            self.materialized_values.append(v)
            yield v

    def __bool__(self):
        """
        Return True if the iterable has values, False otherwise.
        """
        return bool(self.materialized_values) or bool(self.iterable)
